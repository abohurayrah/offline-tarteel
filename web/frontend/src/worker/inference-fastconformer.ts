/**
 * FastConformer CTC inference worker.
 *
 * Pipeline: audio → mel spectrogram → ONNX inference → CTC logprobs
 *           → constrained beam search (Quran trie) → verse match
 *
 * This replaces the Whisper-based inference worker with a CTC approach
 * that constrains decoding to only produce valid Quran text.
 */
import { computeMelSpectrogram } from "./mel";
import { CTCDecoder } from "./ctc-decode";
import { QuranTrie } from "./quran-trie";
import { stripUthmaniMarks, BPETokenizer, ForcedAligner } from "./forced-alignment";
import type { WordAlignment } from "./forced-alignment";
import { CTCVerseScorer } from "./ctc-verse-scorer";
import { QuranDB } from "../lib/quran-db";
import { RecitationTracker } from "../lib/tracker";
import type { TranscribeResult, CTCScoreFn } from "../lib/tracker";
import type { WorkerInbound, WorkerOutbound } from "../lib/types";
import { SAMPLE_RATE } from "../lib/types";
import * as ort from "onnxruntime-web/wasm";

// ─── State ──────────────────────────────────────────────────────────────────

let tracker: RecitationTracker | null = null;
let db: QuranDB | null = null;
let session: ort.InferenceSession | null = null;
let decoder: CTCDecoder | null = null;
let trie: QuranTrie | null = null;
let vocabJson: Record<string, string> | null = null;
let verseScorer: CTCVerseScorer | null = null;
let ctcScoreCallback: CTCScoreFn | undefined;

// Forced alignment state (active during tracking mode)
let faAligner: ForcedAligner | null = null;
let faVerse: { surah: number; ayah: number } | null = null;
let faLastProgressTime = 0;

// Concurrency guard
let busy = false;
let pendingChunks: Float32Array[] = [];

function post(msg: WorkerOutbound) {
  self.postMessage(msg);
}

// ─── ONNX inference (shared between transcription and forced alignment) ─────

async function runOnnx(audio: Float32Array): Promise<{
  logprobs: Float32Array;
  timeSteps: number;
  vocabSize: number;
}> {
  if (!session) throw new Error("Model not loaded");
  const { features, timeFrames } = await computeMelSpectrogram(audio);
  const inputTensor = new ort.Tensor("float32", features, [1, 80, timeFrames]);
  const lengthTensor = new ort.Tensor(
    "int64",
    BigInt64Array.from([BigInt(timeFrames)]),
    [1],
  );
  const feeds: Record<string, ort.Tensor> = {
    [session.inputNames[0]]: inputTensor,
    [session.inputNames[1]]: lengthTensor,
  };
  const results = await session.run(feeds);
  const outputTensor = results[session.outputNames[0]];
  const [, timeSteps, vocabSize] = outputTensor.dims as number[];
  return { logprobs: outputTensor.data as Float32Array, timeSteps, vocabSize };
}

// ─── CTC transcription ──────────────────────────────────────────────────────

async function transcribe(audio: Float32Array): Promise<TranscribeResult> {
  if (!decoder) throw new Error("Model not loaded");

  const { logprobs, timeSteps, vocabSize } = await runOnnx(audio);

  // Feed forced aligner if active (frame-accurate word tracking)
  if (faAligner && faVerse) {
    // FA timeout: abort after 5s with no progress to prevent stalls
    if (faLastProgressTime > 0 && Date.now() - faLastProgressTime > 5000) {
      faAligner = null;
      faVerse = null;
    }
  }
  if (faAligner && faVerse) {
    try {
      const { newWords, currentWordIdx } = faAligner.processFrames(logprobs, timeSteps);

      // Update FA progress timer
      if (newWords.length > 0) {
        faLastProgressTime = Date.now();
      }

      // Only emit word_aligned for words with meaningful confidence
      for (const w of newWords) {
        if (w.confidence > 0.1 && !isNaN(w.confidence)) {
          post({
            type: "word_aligned",
            surah: faVerse.surah,
            ayah: faVerse.ayah,
            word_index: w.wordIdx,
            total_words: faAligner.totalWords,
            confidence: w.confidence,
            cumulative_indices: Array.from(
              { length: w.wordIdx + 1 },
              (_, i) => i,
            ),
          });
        }
      }

      // Check verse completion — require meaningful confidence, not just position
      // The FA can report high currentWordIdx on garbage audio; guard against that
      if (currentWordIdx >= faAligner.totalWords - 1 && newWords.length > 0) {
        const avgConf = newWords.reduce((s, w) => s + w.confidence, 0) / newWords.length;
        if (avgConf > 0.3 && !isNaN(avgConf)) {
          const allWords = faAligner.finalize();
          const overallScore = allWords.reduce((s, w) => s + w.confidence, 0) / allWords.length;
          if (!isNaN(overallScore) && overallScore > 0.2) {
            const nextV = db?.getNextVerse(faVerse.surah, faVerse.ayah);
            post({
              type: "verse_complete",
              surah: faVerse.surah,
              ayah: faVerse.ayah,
              overall_score: overallScore,
              word_scores: allWords.map((w) => w.confidence),
              next_surah: nextV?.surah ?? faVerse.surah,
              next_ayah: nextV?.ayah ?? faVerse.ayah + 1,
            });
          }
          // Don't auto-advance FA to next verse — let the tracker handle
          // verse transitions via verse_match. Starting FA on a verse the
          // tracker hasn't confirmed causes cascading false completions.
          faAligner = null;
          faVerse = null;
        }
      }
    } catch {
      // FA failure is non-fatal — text-based tracking continues
    }
  }

  // Decode text (CTC greedy or constrained beam)
  // Use greedy decoding by default — it is more accurate for short verses.
  // Only use constrained beam search when the audio is long (>5s = >500 mel
  // frames) because the trie gets stuck on wrong paths for short verses
  // (e.g. 112:1 produced "قل هو الذي" instead of "قل هو الله أحد").
  let text: string;
  let rawTokens: string;

  const useConstrainedBeam = trie && timeSteps > 500;

  if (useConstrainedBeam) {
    const hypotheses = decoder.constrainedBeamSearch(
      logprobs,
      timeSteps,
      vocabSize,
      trie,
      { beamWidth: 10, topK: 20 },
    );
    if (hypotheses.length > 0) {
      text = hypotheses[0].text;
      rawTokens = hypotheses[0].rawTokens;
    } else {
      const greedy = decoder.decode(logprobs, timeSteps, vocabSize);
      text = greedy.text;
      rawTokens = greedy.rawTokens;
    }
  } else {
    const greedy = decoder.decode(logprobs, timeSteps, vocabSize);
    text = greedy.text;
    rawTokens = greedy.rawTokens;
  }

  return { text, rawTokens, logprobs, timeSteps, vocabSize };
}

// ─── Forced alignment control ───────────────────────────────────────────────

function startForcedAlignment(surah: number, ayah: number): void {
  if (!vocabJson || !decoder) return;
  const verse = db?.getVerse(surah, ayah);
  if (!verse) return;

  let targetText = stripUthmaniMarks(verse.text_clean || verse.text_uthmani);

  // Strip basmala from ayah 1 of surahs that have a separate basmala line.
  // The audio for these verses doesn't include the basmala (it's recited
  // separately), but text_clean prepends it. Without stripping, the FA
  // stalls at word 0 waiting for "بسم" that never comes.
  if (ayah === 1 && surah !== 1 && surah !== 9) {
    const bsm = stripUthmaniMarks("بسم الله الرحمن الرحيم");
    if (targetText.startsWith(bsm)) {
      targetText = targetText.slice(bsm.length).trim();
    }
  }
  const blankId = decoder.blankId;
  const vocabSize = decoder.vocabSize;

  try {
    faAligner = new ForcedAligner(targetText, vocabJson, blankId, vocabSize);
    faVerse = { surah, ayah };
    faLastProgressTime = Date.now();
  } catch {
    faAligner = null;
    faVerse = null;
    faLastProgressTime = 0;
  }
}

// ─── Audio processing with concurrency guard ────────────────────────────────

function concatFloat32Arrays(arrays: Float32Array[]): Float32Array {
  let totalLength = 0;
  for (const arr of arrays) totalLength += arr.length;
  const result = new Float32Array(totalLength);
  let offset = 0;
  for (const arr of arrays) {
    result.set(arr, offset);
    offset += arr.length;
  }
  return result;
}

async function processAudio(samples: Float32Array): Promise<void> {
  if (!tracker) return;
  if (busy) {
    pendingChunks.push(samples);
    return;
  }

  busy = true;
  try {
    const messages = await tracker.feed(samples);
    for (const m of messages) {
      post(m);
      // Start forced alignment when a verse is confirmed
      if (m.type === "verse_match") {
        startForcedAlignment(m.surah, m.ayah);
      }
    }

    while (pendingChunks.length > 0) {
      const queued = pendingChunks;
      pendingChunks = [];
      const combined = concatFloat32Arrays(queued);
      const msgs = await tracker.feed(combined);
      for (const m of msgs) {
        post(m);
        if (m.type === "verse_match") {
          startForcedAlignment(m.surah, m.ayah);
        }
      }
    }
  } finally {
    busy = false;
  }
}

// ─── Initialization ─────────────────────────────────────────────────────────

async function init() {
  try {
    // Configure ONNX Runtime for WASM
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    // Load FastConformer ONNX model
    post({ type: "loading_status", message: "Loading FastConformer model..." });
    post({ type: "loading", percent: 10 });

    session = await ort.InferenceSession.create("/fastconformer_ar_ctc_q8.onnx", {
      executionProviders: ["wasm"],
    });
    post({ type: "loading", percent: 50 });
    post({ type: "loading_status", message: "Model loaded." });

    // Load vocab for CTC decoder
    post({ type: "loading_status", message: "Loading vocabulary..." });
    const vocabRes = await fetch("/vocab.json");
    if (!vocabRes.ok) throw new Error(`vocab.json fetch failed: ${vocabRes.status}`);
    const vocabData = await vocabRes.json();
    vocabJson = vocabData;
    decoder = new CTCDecoder(vocabData);
    post({ type: "loading", percent: 60 });

    // Load QuranDB
    post({ type: "loading_status", message: "Loading Quran data..." });
    const quranRes = await fetch("/quran.json");
    if (!quranRes.ok) throw new Error(`quran.json fetch failed: ${quranRes.status}`);
    const quranData = await quranRes.json();
    db = new QuranDB(quranData);
    post({ type: "loading", percent: 65 });

    // Load disambiguation map (ambiguity-compact.json from paper research).
    // This enables the prefix-narrowing path in RecitationTracker which
    // mirrors the paper's progressive candidate narrowing algorithm.
    // The file is ~619 KB and loads asynchronously — failure is non-fatal.
    try {
      post({ type: "loading_status", message: "Loading disambiguation index..." });
      const disambigRes = await fetch("/ambiguity-compact.json");
      if (disambigRes.ok) {
        const disambigData = await disambigRes.json();
        db.loadDisambiguationMap(disambigData);
      }
    } catch {
      // Non-fatal: tracker falls back to Levenshtein-only matching
      console.warn("ambiguity-compact.json not available; prefix-narrowing disabled");
    }
    post({ type: "loading", percent: 70 });

    // Build Quran trie for constrained decoding
    post({ type: "loading_status", message: "Building search index..." });
    trie = new QuranTrie(vocabData);
    // Build trie from verse text — stripUthmaniMarks (shared helper) removes
    // Uthmani annotation marks that have no BPE tokens (small waw U+06E5,
    // small yaa U+06E6, rub el hizb U+06DE, sajdah U+06E9, etc.)
    const trieVerses = quranData.map((v: { text_clean?: string; text_uthmani: string; surah: number; ayah: number }) => ({
      text_norm: stripUthmaniMarks(v.text_clean || v.text_uthmani),
      surah: v.surah,
      ayah: v.ayah,
    }));
    trie.buildFromVerses(trieVerses);
    post({ type: "loading", percent: 80 });

    // ── CTC Viterbi verse scorer: pre-tokenize all 6,236 verses ──────────
    // This enables direct acoustic scoring of verse candidates against the
    // raw CTC logprob matrix, bypassing text decoding + Levenshtein distance.
    post({ type: "loading_status", message: "Pre-tokenizing verses for Viterbi scoring..." });
    const bpeTokenizer = new BPETokenizer(vocabData);
    verseScorer = new CTCVerseScorer(bpeTokenizer, decoder.blankId);

    // Pre-tokenize each verse and store token IDs on the QuranVerse object
    for (const verse of db.verses) {
      const verseText = stripUthmaniMarks(verse.text_clean || verse.text_uthmani);
      verse.bpe_token_ids = verseScorer.tokenizeVerse(verseText);
    }
    post({ type: "loading", percent: 90 });

    // Build the CTC scoring callback for the tracker.
    // This closure captures the verseScorer and db references.
    // Stored at module level so the reset handler can reuse it.
    ctcScoreCallback = (
      logprobs: Float32Array,
      timeSteps: number,
      vocabSize: number,
      candidateIndices: number[],
    ) => {
      if (!verseScorer || !db) return [];

      // Build candidates from pre-tokenized verse data
      const candidates = candidateIndices
        .filter((idx) => idx >= 0 && idx < db!.verses.length)
        .map((idx) => ({
          index: idx,
          tokenIds: db!.verses[idx].bpe_token_ids ?? [],
        }))
        .filter((c) => c.tokenIds.length > 0);

      if (candidates.length === 0) return [];

      return verseScorer.scoreVerses(logprobs, timeSteps, vocabSize, candidates)
        .map((r) => ({ index: r.index, score: r.score }));
    };

    // Warm up — first inference is slow due to WASM compilation
    post({ type: "loading_status", message: "Warming up model..." });
    const warmupAudio = new Float32Array(SAMPLE_RATE * 2);
    for (let i = 0; i < warmupAudio.length; i++) {
      warmupAudio[i] = (Math.random() - 0.5) * 0.001;
    }
    await transcribe(warmupAudio);
    post({ type: "loading", percent: 100 });

    // Create tracker with CTC Viterbi scoring callback
    tracker = new RecitationTracker(db, transcribe, ctcScoreCallback);
    post({ type: "ready" });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("Worker init failed:", message);
    post({ type: "error", message });
  }
}

// ─── Message handler ────────────────────────────────────────────────────────

self.onmessage = async (e: MessageEvent<WorkerInbound>) => {
  const msg = e.data;
  if (msg.type === "init") {
    await init();
  } else if (msg.type === "reset") {
    pendingChunks = [];
    busy = false;
    faAligner = null;
    faVerse = null;
    faLastProgressTime = 0;
    if (db) {
      tracker = new RecitationTracker(db, transcribe, ctcScoreCallback);
    }
  } else if (msg.type === "audio") {
    await processAudio(msg.samples);
  }
};
