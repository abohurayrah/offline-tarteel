#!/usr/bin/env npx tsx
/**
 * Parameter sweep — tests different streaming configurations
 * to find optimal thresholds.
 *
 * Usage: npx tsx test/param-sweep.ts
 */
import { execSync } from "node:child_process";
import { readFileSync, existsSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";

import { computeMelSpectrogram } from "../src/worker/mel.ts";
import { CTCDecoder } from "../src/worker/ctc-decode.ts";
import { QuranDB } from "../src/lib/quran-db.ts";
import { RecitationTracker } from "../src/lib/tracker.ts";
import type { TranscribeResult } from "../src/lib/tracker.ts";
import type { WorkerOutbound } from "../src/lib/types.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const SAMPLE_RATE = 16000;
const CHUNK_SAMPLES = Math.floor(SAMPLE_RATE * 0.3); // 300ms

let session: ort.InferenceSession;
let decoder: CTCDecoder;
let db: QuranDB;

async function init() {
  session = await ort.InferenceSession.create(
    resolve(ROOT, "public/fastconformer_ar_ctc_q8.onnx"),
    { executionProviders: ["cpu"] },
  );
  decoder = new CTCDecoder(JSON.parse(readFileSync(resolve(ROOT, "public/vocab.json"), "utf-8")));
  db = new QuranDB(JSON.parse(readFileSync(resolve(ROOT, "public/quran.json"), "utf-8")));
}

function loadAudio(filePath: string): Float32Array {
  const buf = execSync(
    `ffmpeg -hide_banner -loglevel error -i "${filePath}" -f f32le -ar ${SAMPLE_RATE} -ac 1 pipe:1`,
    { maxBuffer: 50 * 1024 * 1024 },
  );
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

async function transcribe(audio: Float32Array): Promise<TranscribeResult> {
  const { features, timeFrames } = await computeMelSpectrogram(audio);
  const input = new ort.Tensor("float32", features, [1, 80, timeFrames]);
  const length = new ort.Tensor("int64", BigInt64Array.from([BigInt(timeFrames)]), [1]);
  const results = await session.run({
    [session.inputNames[0]]: input,
    [session.inputNames[1]]: length,
  });
  const out = results[session.outputNames[0]];
  const [, ts, vs] = out.dims as number[];
  const { text, rawTokens } = decoder.decode(out.data as Float32Array, ts, vs);
  return { text, rawTokens };
}

interface Sample {
  id: string;
  file: string;
  surah: number;
  ayah: number;
}

function loadSamples(): Sample[] {
  const manifest = JSON.parse(readFileSync(resolve(ROOT, "../../benchmark/test_corpus/manifest.json"), "utf-8"));
  return manifest.samples.map((s: any) => ({
    id: s.id,
    file: resolve(ROOT, "../../benchmark/test_corpus", s.file),
    surah: s.expected_verses[0].surah,
    ayah: s.expected_verses[0].ayah,
  })).filter((s: Sample) => existsSync(s.file));
}

async function runConfig(
  samples: Sample[],
  audioCache: Map<string, Float32Array>,
  config: { firstTriggerS: number; fragmentGatePct: number; label: string },
): Promise<{ label: string; correct: number; total: number; noMatch: number; avgCoverage: number; avgTime: number; jumps: number }> {
  let correct = 0;
  let total = 0;
  let noMatch = 0;
  let totalCoverage = 0;
  let totalTime = 0;
  let totalJumps = 0;

  for (const s of samples) {
    const audio = audioCache.get(s.id)!;
    total++;

    // Monkey-patch the tracker's internal constants for this run.
    // We create a fresh tracker each time, and the constants come from types.ts.
    // To vary them per-run, we override the tracker's behavior via its feed method.
    //
    // Simpler approach: just create the tracker and feed audio, then check results.
    // The tracker uses the global constants from types.ts which we can't change per-run.
    // Instead, we simulate the effect by controlling how much audio we feed before
    // the tracker gets a chance to transcribe.

    const tracker = new RecitationTracker(db, transcribe);
    let firstMatch: { surah: number; ayah: number; time: number } | null = null;
    let wordIndices = new Set<number>();
    let wordTotal = 0;
    let jumps = 0;
    let lastS = -1, lastA = -1;

    // Feed audio in chunks. The tracker's internal trigger fires based on
    // accumulated newAudioCount vs TRIGGER_SAMPLES. We can't change that per-run.
    // So we feed a large initial chunk to simulate different trigger times.
    const firstChunkSamples = Math.floor(SAMPLE_RATE * config.firstTriggerS);
    const initialChunk = audio.slice(0, Math.min(firstChunkSamples, audio.length));

    // Feed the initial chunk
    let msgs = await tracker.feed(initialChunk);
    for (const msg of msgs) {
      if (msg.type === "verse_match") {
        if (!firstMatch) firstMatch = { surah: msg.surah, ayah: msg.ayah, time: config.firstTriggerS };
        if (lastS >= 0 && (msg.surah !== lastS || msg.ayah !== lastA)) jumps++;
        lastS = msg.surah; lastA = msg.ayah;
      }
      if (msg.type === "word_progress") {
        wordTotal = msg.total_words;
        for (const idx of msg.matched_indices) wordIndices.add(idx);
      }
    }

    // Feed remaining audio in 300ms chunks
    for (let offset = firstChunkSamples; offset < audio.length; offset += CHUNK_SAMPLES) {
      const chunk = audio.slice(offset, Math.min(offset + CHUNK_SAMPLES, audio.length));
      msgs = await tracker.feed(chunk);
      for (const msg of msgs) {
        if (msg.type === "verse_match") {
          if (!firstMatch) firstMatch = { surah: msg.surah, ayah: msg.ayah, time: offset / SAMPLE_RATE };
          if (lastS >= 0 && (msg.surah !== lastS || msg.ayah !== lastA)) jumps++;
          lastS = msg.surah; lastA = msg.ayah;
        }
        if (msg.type === "word_progress") {
          wordTotal = msg.total_words;
          for (const idx of msg.matched_indices) wordIndices.add(idx);
        }
      }
    }

    if (!firstMatch) {
      noMatch++;
    } else if (firstMatch.surah === s.surah && firstMatch.ayah === s.ayah) {
      correct++;
    }

    totalCoverage += wordTotal > 0 ? wordIndices.size / wordTotal : 0;
    totalTime += firstMatch?.time ?? (audio.length / SAMPLE_RATE);
    totalJumps += jumps;
  }

  return {
    label: config.label,
    correct,
    total,
    noMatch,
    avgCoverage: totalCoverage / total,
    avgTime: totalTime / total,
    jumps: totalJumps / total,
  };
}

async function main() {
  console.log("╔══════════════════════════════════════════════════════════════╗");
  console.log("║           Parameter Sweep — Streaming Pipeline             ║");
  console.log("╚══════════════════════════════════════════════════════════════╝\n");

  await init();
  const samples = loadSamples().slice(0, 30); // Use 30 samples for speed
  console.log(`Samples: ${samples.length}\n`);

  // Pre-load all audio
  console.log("Loading audio...");
  const audioCache = new Map<string, Float32Array>();
  for (const s of samples) {
    audioCache.set(s.id, loadAudio(s.file));
  }
  console.log("Audio loaded.\n");

  // The tracker uses constants from types.ts which we can't change at runtime.
  // But we CAN control the initial audio chunk size, which simulates different
  // first-trigger times. The tracker will transcribe when it has enough audio.
  //
  // We test different first-trigger times by feeding different initial chunk sizes.
  // The fragment gate is in types.ts and can't be changed per-run without rewriting.
  // So we test the current configuration at different trigger times.

  const configs = [
    { firstTriggerS: 2.0, fragmentGatePct: 0, label: "2.0s trigger" },
    { firstTriggerS: 3.0, fragmentGatePct: 0, label: "3.0s trigger" },
    { firstTriggerS: 4.0, fragmentGatePct: 0, label: "4.0s trigger" },
    { firstTriggerS: 5.0, fragmentGatePct: 0, label: "5.0s trigger" },
    { firstTriggerS: 6.0, fragmentGatePct: 0, label: "6.0s trigger" },
    { firstTriggerS: 8.0, fragmentGatePct: 0, label: "8.0s trigger" },
    { firstTriggerS: 10.0, fragmentGatePct: 0, label: "10.0s trigger" },
  ];

  console.log("Running sweep...\n");
  console.log("  Config                Correct  NoMatch  Coverage  Time    Jumps");
  console.log("  " + "─".repeat(65));

  const allResults = [];
  for (const config of configs) {
    const result = await runConfig(samples, audioCache, config);
    allResults.push(result);
    console.log(
      `  ${result.label.padEnd(22)} ${result.correct}/${result.total} (${(result.correct/result.total*100).toFixed(0)}%)`.padEnd(30) +
      `  ${result.noMatch}`.padEnd(10) +
      `  ${(result.avgCoverage*100).toFixed(0)}%`.padEnd(10) +
      `  ${result.avgTime.toFixed(1)}s`.padEnd(8) +
      `  ${result.jumps.toFixed(1)}`,
    );
  }

  console.log(`\n${"═".repeat(70)}`);
  console.log("  BEST CONFIG:");
  const best = allResults.reduce((a, b) => a.correct > b.correct ? a : b);
  console.log(`  ${best.label}: ${best.correct}/${best.total} (${(best.correct/best.total*100).toFixed(1)}%)`);
  console.log(`${"═".repeat(70)}`);

  // Save
  mkdirSync(resolve(__dirname, "benchmark-results"), { recursive: true });
  writeFileSync(
    resolve(__dirname, "benchmark-results/param-sweep.json"),
    JSON.stringify({ timestamp: new Date().toISOString(), results: allResults }, null, 2),
  );
}

main().catch(err => { console.error(err); process.exit(1); });
