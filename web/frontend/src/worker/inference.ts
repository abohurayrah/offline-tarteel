import { loadWhisper, transcribe as whisperTranscribe } from "./whisper-transcriber";
import { QuranDB } from "../lib/quran-db";
import { RecitationTracker } from "../lib/tracker";
import type { TranscribeResult } from "../lib/tracker";
import type { WorkerInbound, WorkerOutbound } from "../lib/types";
import { SAMPLE_RATE } from "../lib/types";

let tracker: RecitationTracker | null = null;
let db: QuranDB | null = null;

function post(msg: WorkerOutbound) {
  self.postMessage(msg);
}

/**
 * Adapter: Whisper text output → TranscribeResult expected by RecitationTracker.
 */
async function transcribe(audio: Float32Array): Promise<TranscribeResult> {
  const text = await whisperTranscribe(audio);
  return { text, rawTokens: "" };
}

async function init() {
  try {
    // Load Whisper model via transformers.js
    post({ type: "loading_status", message: "Loading Whisper model..." });
    await loadWhisper((progress) => {
      if (progress.status === "progress" && progress.progress != null) {
        post({ type: "loading", percent: Math.round(progress.progress) });
      }
      if (progress.status === "ready") {
        post({ type: "loading_status", message: "Model loaded." });
      }
    });

    // Load QuranDB (Arabic text data)
    post({ type: "loading_status", message: "Loading Quran data..." });
    const quranRes = await fetch("/quran.json");
    if (!quranRes.ok) throw new Error(`quran.json fetch failed: ${quranRes.status}`);
    const quranData = await quranRes.json();
    db = new QuranDB(quranData);

    // Warm up — first inference is slow due to WASM compilation
    post({ type: "loading_status", message: "Warming up model..." });
    const warmupAudio = new Float32Array(SAMPLE_RATE * 2);
    for (let i = 0; i < warmupAudio.length; i++) {
      warmupAudio[i] = (Math.random() - 0.5) * 0.001;
    }
    await transcribe(warmupAudio);

    // Create tracker
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
    if (db) {
      tracker = new RecitationTracker(db, transcribe);
    }
  } else if (msg.type === "audio") {
    if (!tracker) return;
    const messages = await tracker.feed(msg.samples);
    for (const m of messages) {
      post(m);
    }
  }
};
