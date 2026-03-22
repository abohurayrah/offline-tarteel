import { loadWhisper, transcribe as whisperTranscribe } from "./whisper-transcriber";
import { QuranDB } from "../lib/quran-db";
import { RecitationTracker } from "../lib/tracker";
import type { TranscribeResult } from "../lib/tracker";
import type { WorkerInbound, WorkerOutbound } from "../lib/types";
import { SAMPLE_RATE } from "../lib/types";

let tracker: RecitationTracker | null = null;
let db: QuranDB | null = null;

// Concurrency guard: queue audio while inference is running
let busy = false;
let pendingChunks: Float32Array[] = [];

function post(msg: WorkerOutbound) {
  self.postMessage(msg);
}

async function transcribe(audio: Float32Array, prompt?: string): Promise<TranscribeResult> {
  const text = await whisperTranscribe(audio, prompt);
  return { text, rawTokens: "" };
}

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
    // Queue audio for processing after current inference completes
    pendingChunks.push(samples);
    return;
  }

  busy = true;
  try {
    const messages = await tracker.feed(samples);
    for (const m of messages) {
      post(m);
    }

    // Process any audio that arrived during inference
    while (pendingChunks.length > 0) {
      const queued = pendingChunks;
      pendingChunks = [];
      // Combine all queued chunks into one feed call
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

async function init() {
  try {
    post({ type: "loading_status", message: "Loading Whisper model..." });
    await loadWhisper((progress) => {
      if (progress.status === "progress" && progress.progress != null) {
        post({ type: "loading", percent: Math.round(progress.progress) });
      }
      if (progress.status === "ready") {
        post({ type: "loading_status", message: "Model loaded." });
      }
    });

    post({ type: "loading_status", message: "Loading Quran data..." });
    const quranRes = await fetch("/quran.json");
    if (!quranRes.ok) throw new Error(`quran.json fetch failed: ${quranRes.status}`);
    const quranData = await quranRes.json();
    db = new QuranDB(quranData);

    post({ type: "loading_status", message: "Warming up model..." });
    const warmupAudio = new Float32Array(SAMPLE_RATE * 2);
    for (let i = 0; i < warmupAudio.length; i++) {
      warmupAudio[i] = (Math.random() - 0.5) * 0.001;
    }
    await transcribe(warmupAudio);

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
    // Wait for any in-flight processing to complete
    // Clear pending chunks to prevent stale audio from being processed
    pendingChunks = [];
    busy = false;
    if (db) {
      tracker = new RecitationTracker(db, transcribe);
    }
  } else if (msg.type === "audio") {
    await processAudio(msg.samples);
  }
};
