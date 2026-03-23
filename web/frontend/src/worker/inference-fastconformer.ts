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
import { QuranDB } from "../lib/quran-db";
import { RecitationTracker } from "../lib/tracker";
import type { TranscribeResult } from "../lib/tracker";
import type { WorkerInbound, WorkerOutbound } from "../lib/types";
import { SAMPLE_RATE } from "../lib/types";
import * as ort from "onnxruntime-web/wasm";

// ─── State ──────────────────────────────────────────────────────────────────

let tracker: RecitationTracker | null = null;
let db: QuranDB | null = null;
let session: ort.InferenceSession | null = null;
let decoder: CTCDecoder | null = null;
let trie: QuranTrie | null = null;

// Concurrency guard
let busy = false;
let pendingChunks: Float32Array[] = [];

function post(msg: WorkerOutbound) {
  self.postMessage(msg);
}

// ─── ONNX + CTC transcription ──────────────────────────────────────────────

async function transcribe(audio: Float32Array): Promise<TranscribeResult> {
  if (!session || !decoder) throw new Error("Model not loaded");

  // 1. Compute mel spectrogram (NeMo-compatible)
  const { features, timeFrames } = await computeMelSpectrogram(audio);

  // 2. Run ONNX inference → CTC logprobs
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
  const logprobs = outputTensor.data as Float32Array;
  const [, timeSteps, vocabSize] = outputTensor.dims as number[];

  // 3. Decode: use constrained beam search if trie is available, else greedy
  let text: string;
  let rawTokens: string;

  if (trie) {
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
      // Fallback to greedy if constrained search returns nothing
      const greedy = decoder.decode(logprobs, timeSteps, vocabSize);
      text = greedy.text;
      rawTokens = greedy.rawTokens;
    }
  } else {
    const greedy = decoder.decode(logprobs, timeSteps, vocabSize);
    text = greedy.text;
    rawTokens = greedy.rawTokens;
  }

  return { text, rawTokens };
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
    }

    while (pendingChunks.length > 0) {
      const queued = pendingChunks;
      pendingChunks = [];
      const combined = concatFloat32Arrays(queued);
      const msgs = await tracker.feed(combined);
      for (const m of msgs) {
        post(m);
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
    decoder = new CTCDecoder(vocabData);
    post({ type: "loading", percent: 60 });

    // Load QuranDB
    post({ type: "loading_status", message: "Loading Quran data..." });
    const quranRes = await fetch("/quran.json");
    if (!quranRes.ok) throw new Error(`quran.json fetch failed: ${quranRes.status}`);
    const quranData = await quranRes.json();
    db = new QuranDB(quranData);
    post({ type: "loading", percent: 70 });

    // Build Quran trie for constrained decoding
    post({ type: "loading_status", message: "Building search index..." });
    trie = new QuranTrie(vocabData);
    // Build trie from verse text — strip Uthmani annotation marks that have
    // no BPE tokens (small waw U+06E5, small yaa U+06E6, rub el hizb U+06DE,
    // sajdah U+06E9, and other diacritics the model never outputs)
    const stripForBPE = (text: string) =>
      text.replace(/[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]/g, "");
    const trieVerses = quranData.map((v: { text_clean?: string; text_uthmani: string; surah: number; ayah: number }) => ({
      text_norm: stripForBPE(v.text_clean || v.text_uthmani),
      surah: v.surah,
      ayah: v.ayah,
    }));
    trie.buildFromVerses(trieVerses);
    post({ type: "loading", percent: 85 });

    // Warm up — first inference is slow due to WASM compilation
    post({ type: "loading_status", message: "Warming up model..." });
    const warmupAudio = new Float32Array(SAMPLE_RATE * 2);
    for (let i = 0; i < warmupAudio.length; i++) {
      warmupAudio[i] = (Math.random() - 0.5) * 0.001;
    }
    await transcribe(warmupAudio);
    post({ type: "loading", percent: 100 });

    // Create tracker
    tracker = new RecitationTracker(db, transcribe);
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
    if (db) {
      tracker = new RecitationTracker(db, transcribe);
    }
  } else if (msg.type === "audio") {
    await processAudio(msg.samples);
  }
};
