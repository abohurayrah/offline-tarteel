import { loadModel } from "./model-cache";
import { computeMelSpectrogram } from "./mel";
import { CTCDecoder } from "./ctc-decode";
import { createSession, runInference } from "./session";
import { QuranDB } from "../lib/quran-db";
import { RecitationTracker } from "../lib/tracker";
import { ForcedAligner } from "./forced-alignment";
import type { TranscribeResult } from "../lib/tracker";
import type { WorkerInbound, WorkerOutbound, WordAlignedMessage, VerseCompleteMessage } from "../lib/types";
import {
  SAMPLE_RATE,
  FA_TRIGGER_SAMPLES,
  FA_VERSE_COMPLETE_THRESHOLD,
  SILENCE_RMS_THRESHOLD,
  TRACKING_SILENCE_SAMPLES,
} from "../lib/types";

const MODEL_URL = "/fastconformer_ar_ctc_q8.onnx";

let tracker: RecitationTracker | null = null;
let decoder: CTCDecoder | null = null;
let db: QuranDB | null = null;
let vocabJsonCache: Record<string, string> | null = null;

// Forced alignment state
let faAligner: ForcedAligner | null = null;
let faVerse: { surah: number; ayah: number } | null = null;
let faAudioBuffer = new Float32Array(0);
let faNewAudioCount = 0;
let faSilenceSamples = 0;
let faLastWordIdx = -1;

function post(msg: WorkerOutbound) {
  self.postMessage(msg);
}

async function transcribe(audio: Float32Array): Promise<TranscribeResult> {
  const { features, timeFrames } = await computeMelSpectrogram(audio);
  const numMels = 80;
  const { logprobs, timeSteps, vocabSize } = await runInference(
    features,
    numMels,
    timeFrames,
  );
  return decoder!.decode(logprobs, timeSteps, vocabSize);
}

/**
 * Run mel + ONNX inference and return raw logprobs (for forced alignment).
 */
async function inferLogprobs(audio: Float32Array): Promise<{
  logprobs: Float32Array;
  timeSteps: number;
  vocabSize: number;
}> {
  const { features, timeFrames } = await computeMelSpectrogram(audio);
  const numMels = 80;
  return runInference(features, numMels, timeFrames);
}

/**
 * Enter forced alignment mode for a specific verse.
 */
function enterFA(surah: number, ayah: number): void {
  if (!db || !vocabJsonCache || !decoder) return;

  const verse = db.getVerse(surah, ayah);
  if (!verse?.text_norm) {
    console.warn(`[FA-Worker] No verse text for ${surah}:${ayah}`);
    return;
  }

  // Get blank ID from decoder
  const blankId = (decoder as any).blankId ?? 1024;
  const vocabSize = (decoder as any).vocab?.size ?? 1025;

  faAligner = new ForcedAligner(verse.text_norm, vocabJsonCache, blankId, vocabSize);
  faVerse = { surah, ayah };
  faAudioBuffer = new Float32Array(0);
  faNewAudioCount = 0;
  faSilenceSamples = 0;
  faLastWordIdx = -1;

  console.log(`[FA-Worker] Entered FA mode for ${surah}:${ayah} (${faAligner.totalWords} words, ${faAligner.totalTokens} tokens)`);
}

/**
 * Exit forced alignment mode.
 */
function exitFA(reason: string): void {
  console.log(`[FA-Worker] Exiting FA: ${reason}`);
  faAligner = null;
  faVerse = null;
  faAudioBuffer = new Float32Array(0);
  faNewAudioCount = 0;
  faSilenceSamples = 0;
  faLastWordIdx = -1;
}

/**
 * Process audio in forced alignment mode.
 * Returns messages to send to main thread.
 */
async function handleFA(samples: Float32Array): Promise<WorkerOutbound[]> {
  if (!faAligner || !faVerse) return [];

  const messages: WorkerOutbound[] = [];

  // Silence detection
  let sumSq = 0;
  for (let i = 0; i < samples.length; i++) {
    sumSq += samples[i] * samples[i];
  }
  const rms = Math.sqrt(sumSq / samples.length);
  if (rms < SILENCE_RMS_THRESHOLD) {
    faSilenceSamples += samples.length;
    if (faSilenceSamples >= TRACKING_SILENCE_SAMPLES) {
      exitFA("extended silence");
      return messages;
    }
  } else {
    faSilenceSamples = 0;
  }

  // Append audio
  const newBuf = new Float32Array(faAudioBuffer.length + samples.length);
  newBuf.set(faAudioBuffer);
  newBuf.set(samples, faAudioBuffer.length);
  faAudioBuffer = newBuf;
  faNewAudioCount += samples.length;

  // Trim to max 10s
  const maxSamples = SAMPLE_RATE * 10;
  if (faAudioBuffer.length > maxSamples) {
    faAudioBuffer = faAudioBuffer.slice(-maxSamples);
  }

  // Wait for enough new audio
  if (faNewAudioCount < FA_TRIGGER_SAMPLES) return messages;
  faNewAudioCount = 0;

  // Run inference to get logprobs
  try {
    const { logprobs, timeSteps, vocabSize } = await inferLogprobs(faAudioBuffer.slice());

    // Feed logprobs to forced aligner
    // Reset aligner and feed full audio each time (simpler than incremental for now)
    faAligner.reset();
    const { currentWordIdx, allWords } = faAligner.processFrames(logprobs, timeSteps);

    // Emit word_aligned for each newly confirmed word
    const confirmedWords = allWords.filter((w) => w.confidence > 0);
    const cumulativeIndices = confirmedWords.map((w) => w.wordIndex);

    if (currentWordIdx > faLastWordIdx && currentWordIdx >= 0) {
      const currentWord = allWords.find((w) => w.wordIndex === currentWordIdx);
      const wordMsg: WordAlignedMessage = {
        type: "word_aligned",
        surah: faVerse.surah,
        ayah: faVerse.ayah,
        word_index: currentWordIdx,
        total_words: faAligner.totalWords,
        confidence: currentWord?.confidence ?? 0,
        cumulative_indices: cumulativeIndices,
      };
      messages.push(wordMsg);
      faLastWordIdx = currentWordIdx;

      console.log(
        `[FA-Worker] Word ${currentWordIdx}/${faAligner.totalWords}` +
        ` conf=${(currentWord?.confidence ?? 0).toFixed(3)}` +
        ` indices=[${cumulativeIndices.join(",")}]`,
      );
    }

    // Check if verse is complete
    const coverage = confirmedWords.length / faAligner.totalWords;
    const nearEnd = currentWordIdx >= faAligner.totalWords - 2;
    if (coverage >= FA_VERSE_COMPLETE_THRESHOLD && nearEnd) {
      const wordScores = allWords.map((w) => w.confidence);
      const overallScore = faAligner.getOverallScore();

      // Get next verse
      const nextV = db?.getNextVerse(faVerse.surah, faVerse.ayah);
      const nextSurah = nextV?.surah ?? faVerse.surah;
      const nextAyah = nextV?.ayah ?? faVerse.ayah + 1;

      const completeMsg: VerseCompleteMessage = {
        type: "verse_complete",
        surah: faVerse.surah,
        ayah: faVerse.ayah,
        overall_score: overallScore,
        word_scores: wordScores,
        next_surah: nextSurah,
        next_ayah: nextAyah,
      };
      messages.push(completeMsg);

      console.log(
        `[FA-Worker] Verse complete ${faVerse.surah}:${faVerse.ayah}` +
        ` score=${overallScore.toFixed(3)} → next ${nextSurah}:${nextAyah}`,
      );

      // Auto-advance to next verse
      exitFA("verse complete");
      if (nextV) {
        enterFA(nextSurah, nextAyah);
      }
    }
  } catch (err) {
    console.error("[FA-Worker] Inference error:", err);
  }

  return messages;
}

async function init() {
  try {
    // Load vocab
    post({ type: "loading_status", message: "Loading vocabulary..." });
    const vocabRes = await fetch("/vocab.json");
    if (!vocabRes.ok) throw new Error(`vocab.json fetch failed: ${vocabRes.status}`);
    const vocabJson = await vocabRes.json();
    vocabJsonCache = vocabJson;
    decoder = new CTCDecoder(vocabJson);

    // Load ONNX model
    post({ type: "loading_status", message: "Downloading model..." });
    const modelBuffer = await loadModel(
      MODEL_URL,
      (loaded, total) => {
        post({
          type: "loading",
          percent: total ? Math.round((loaded / total) * 100) : 0,
        });
      },
    );

    post({ type: "loading_status", message: "Creating inference session..." });
    await createSession(modelBuffer);

    // Load QuranDB (Arabic text data)
    post({ type: "loading_status", message: "Loading Quran data..." });
    const quranRes = await fetch("/quran.json");
    if (!quranRes.ok) throw new Error(`quran.json fetch failed: ${quranRes.status}`);
    const quranData = await quranRes.json();
    db = new QuranDB(quranData);

    // Warm up the ONNX runtime — first few inferences after session creation
    // produce unreliable outputs due to JIT compilation and buffer allocation.
    post({ type: "loading_status", message: "Warming up model..." });
    const warmupAudio = new Float32Array(SAMPLE_RATE * 2); // 2s of silence
    for (let i = 0; i < warmupAudio.length; i++) {
      warmupAudio[i] = (Math.random() - 0.5) * 0.001;
    }
    for (let pass = 0; pass < 3; pass++) {
      await transcribe(warmupAudio);
    }

    // Create tracker (Phase 1: Discovery)
    tracker = new RecitationTracker(db, transcribe);
    post({ type: "ready" });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("Worker init failed:", message);
    post({ type: "error", message });
  }
}

self.onmessage = async (e: MessageEvent<WorkerInbound>) => {
  const msg = e.data;
  if (msg.type === "init") {
    await init();
  } else if (msg.type === "reset") {
    exitFA("reset");
    if (db) {
      tracker = new RecitationTracker(db, transcribe);
    }
  } else if (msg.type === "audio") {
    if (!tracker) return;

    // If in FA mode, route audio to forced aligner
    if (faAligner) {
      const faMessages = await handleFA(msg.samples);
      for (const m of faMessages) {
        post(m);
      }
      return;
    }

    // Phase 1: Discovery mode (existing tracker)
    const messages = await tracker.feed(msg.samples);
    for (const m of messages) {
      post(m);

      // FA disabled — tracker's tracking mode handles word-level progress.
      // TODO: re-enable FA once we can seed it with tracker's audio buffer.
      // if (m.type === "verse_match" && m.confidence >= 0.45) {
      //   enterFA(m.surah, m.ayah);
      // }
    }
  }
};
