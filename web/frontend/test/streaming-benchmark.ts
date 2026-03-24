#!/usr/bin/env npx tsx
/**
 * Streaming Pipeline Benchmark — simulates real-time recitation.
 *
 * Feeds audio in 300ms chunks through the FULL pipeline:
 *   audio chunks → mel → ONNX → CTC decode → RecitationTracker → messages
 *
 * Measures:
 *   - Discovery accuracy: correct verse identified?
 *   - Time to first match: how many seconds before verse_match?
 *   - Word coverage: what % of verse words tracked?
 *   - False jumps: wrong verse transitions?
 *   - Verse stability: does it stay on the right verse?
 *
 * Usage:
 *   npx tsx test/streaming-benchmark.ts                    # 54-sample corpus
 *   npx tsx test/streaming-benchmark.ts --source=audio-samples --sample=100
 *   npx tsx test/streaming-benchmark.ts --source=expanded --sample=50
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
import type { TranscribeResult, CTCScoreFn } from "../src/lib/tracker.ts";
import type { WorkerOutbound } from "../src/lib/types.ts";
import { SAMPLE_RATE } from "../src/lib/types.ts";
import { CTCVerseScorer } from "../src/worker/ctc-verse-scorer.ts";
import { BPETokenizer } from "../src/worker/forced-alignment.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");

// CLI args
const SOURCE = process.argv.find(a => a.startsWith("--source="))?.split("=")[1] ?? "corpus";
const SAMPLE_LIMIT = parseInt(process.argv.find(a => a.startsWith("--sample="))?.split("=")[1] ?? "0");
const CHUNK_MS = 300;
const CHUNK_SAMPLES = Math.floor(SAMPLE_RATE * CHUNK_MS / 1000);

// ─── Setup ──────────────────────────────────────────────────────────────────

let session: ort.InferenceSession;
let decoder: CTCDecoder;
let db: QuranDB;
let ctcScoreFn: CTCScoreFn | undefined;

async function init() {
  console.log("Loading model...");
  session = await ort.InferenceSession.create(
    resolve(ROOT, "public/fastconformer_ar_ctc_q8.onnx"),
    { executionProviders: ["cpu"] },
  );
  const vocabJson = JSON.parse(readFileSync(resolve(ROOT, "public/vocab.json"), "utf-8"));
  decoder = new CTCDecoder(vocabJson);
  const quranData = JSON.parse(readFileSync(resolve(ROOT, "public/quran.json"), "utf-8"));
  db = new QuranDB(quranData);

  // Load disambiguation map for prefix-narrowing (mirrors the worker init path)
  const disambigPath = resolve(ROOT, "public/ambiguity-compact.json");
  if (existsSync(disambigPath)) {
    const disambigData = JSON.parse(readFileSync(disambigPath, "utf-8"));
    db.loadDisambiguationMap(disambigData);
    console.log("Disambiguation map loaded.");
  } else {
    console.warn("ambiguity-compact.json not found; prefix-narrowing disabled.");
  }

  // Set up CTC Viterbi verse scorer
  try {
    const tokenizer = new BPETokenizer(vocabJson);
    const scorer = new CTCVerseScorer(tokenizer, decoder.blankId);

    // Pre-tokenize all verses
    const verses = db.getAllVerses();
    let tokenized = 0;
    for (const v of verses) {
      const text = v.text_clean || v.text_uthmani;
      if (text) {
        v.bpe_token_ids = scorer.tokenizeVerse(text);
        tokenized++;
      }
    }
    console.log(`Pre-tokenized ${tokenized} verses for Viterbi scoring.`);

    // Build the scoring callback
    ctcScoreFn = (logprobs, timeSteps, vocabSize, candidateIndices) => {
      const candidates = candidateIndices.map(idx => {
        const v = verses[idx];
        return { index: idx, tokenIds: v?.bpe_token_ids ?? [] };
      }).filter(c => c.tokenIds.length > 0);

      if (!candidates.length) return [];
      return scorer.scoreVerses(logprobs, timeSteps, vocabSize, candidates);
    };
    console.log("CTC Viterbi scorer ready.");
  } catch (e: any) {
    console.warn(`Viterbi scorer setup failed: ${e.message}. Falling back to text matching.`);
  }

  console.log("Ready.\n");
}

function loadAudio(filePath: string): Float32Array {
  const buf = execSync(
    `ffmpeg -hide_banner -loglevel error -i "${filePath}" -f f32le -ar ${SAMPLE_RATE} -ac 1 pipe:1`,
    { maxBuffer: 50 * 1024 * 1024 },
  );
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

// ─── Transcribe function (same as inference worker) ─────────────────────────

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
  const logprobs = out.data as Float32Array;
  const { text, rawTokens } = decoder.decode(logprobs, ts, vs);
  return { text, rawTokens, logprobs, timeSteps: ts, vocabSize: vs };
}

// ─── Streaming simulation ───────────────────────────────────────────────────

interface StreamingResult {
  id: string;
  expectedSurah: number;
  expectedAyah: number;
  // Discovery
  firstMatchSurah: number | null;
  firstMatchAyah: number | null;
  firstMatchConfidence: number;
  firstMatchTime: number; // seconds
  discoveryCorrect: boolean;
  // Word tracking
  wordsCovered: number;
  totalWords: number;
  wordCoverage: number;
  // Stability
  verseJumps: number;
  totalMessages: number;
  // Timing
  audioDuration: number;
  processingTime: number;
}

async function simulateStreaming(
  audio: Float32Array,
  expectedSurah: number,
  expectedAyah: number,
  id: string,
): Promise<StreamingResult> {
  const tracker = new RecitationTracker(db, transcribe, ctcScoreFn);
  const allMessages: WorkerOutbound[] = [];
  const t0 = performance.now();

  let firstMatch: { surah: number; ayah: number; confidence: number; time: number } | null = null;
  let wordIndices = new Set<number>();
  let totalWords = 0;
  let jumps = 0;
  let lastSurah = -1;
  let lastAyah = -1;

  // Feed audio in 300ms chunks
  for (let offset = 0; offset < audio.length; offset += CHUNK_SAMPLES) {
    const chunk = audio.slice(offset, Math.min(offset + CHUNK_SAMPLES, audio.length));
    const messages = await tracker.feed(chunk);

    for (const msg of messages) {
      allMessages.push(msg);

      if (msg.type === "verse_match") {
        if (!firstMatch) {
          firstMatch = {
            surah: msg.surah,
            ayah: msg.ayah,
            confidence: msg.confidence,
            time: (offset + chunk.length) / SAMPLE_RATE,
          };
        }
        if (lastSurah >= 0 && (msg.surah !== lastSurah || msg.ayah !== lastAyah)) {
          jumps++;
        }
        lastSurah = msg.surah;
        lastAyah = msg.ayah;
      }

      if (msg.type === "word_progress") {
        totalWords = msg.total_words;
        for (const idx of msg.matched_indices) {
          wordIndices.add(idx);
        }
      }
    }
  }

  // End-of-utterance flush: give the tracker one final chance to match
  // with ALL accumulated audio (essentially non-streaming accuracy)
  if (!firstMatch) {
    const flushMsgs = await tracker.flush();
    for (const msg of flushMsgs) {
      allMessages.push(msg);
      if (msg.type === "verse_match" && !firstMatch) {
        firstMatch = {
          surah: msg.surah,
          ayah: msg.ayah,
          confidence: msg.confidence,
          time: audio.length / SAMPLE_RATE,
        };
        lastSurah = msg.surah;
        lastAyah = msg.ayah;
      }
    }
  }

  const processingTime = (performance.now() - t0) / 1000;
  const audioDuration = audio.length / SAMPLE_RATE;

  return {
    id,
    expectedSurah,
    expectedAyah,
    firstMatchSurah: firstMatch?.surah ?? null,
    firstMatchAyah: firstMatch?.ayah ?? null,
    firstMatchConfidence: firstMatch?.confidence ?? 0,
    firstMatchTime: firstMatch?.time ?? audioDuration,
    discoveryCorrect: firstMatch?.surah === expectedSurah && firstMatch?.ayah === expectedAyah,
    wordsCovered: wordIndices.size,
    totalWords,
    wordCoverage: totalWords > 0 ? wordIndices.size / totalWords : 0,
    verseJumps: jumps,
    totalMessages: allMessages.length,
    audioDuration,
    processingTime,
  };
}

// ─── Load samples ───────────────────────────────────────────────────────────

interface Sample {
  id: string;
  file: string;
  surah: number;
  ayah: number;
}

function loadSamples(): Sample[] {
  const samples: Sample[] = [];

  if (SOURCE === "corpus" || SOURCE === "all") {
    const manifest = JSON.parse(readFileSync(resolve(ROOT, "../../benchmark/test_corpus/manifest.json"), "utf-8"));
    for (const s of manifest.samples) {
      samples.push({
        id: s.id,
        file: resolve(ROOT, "../../benchmark/test_corpus", s.file),
        surah: s.expected_verses[0].surah,
        ayah: s.expected_verses[0].ayah,
      });
    }
  }

  if (SOURCE === "audio-samples" || SOURCE === "all") {
    const manifest = JSON.parse(readFileSync(resolve(ROOT, "test/audio-samples/manifest.json"), "utf-8"));
    for (const s of manifest) {
      if (s.type !== "full") continue;
      samples.push({
        id: `${s.surah}_${s.ayah}_${s.reciter}`,
        file: resolve(ROOT, "test/audio-samples", s.file),
        surah: s.surah,
        ayah: s.ayah,
      });
    }
  }

  if (SOURCE === "expanded") {
    const manifest = JSON.parse(readFileSync(resolve(ROOT, "../../benchmark/test_corpus_expanded/manifest.json"), "utf-8"));
    for (const s of manifest.samples) {
      samples.push({
        id: s.id,
        file: resolve(ROOT, "../../benchmark/test_corpus_expanded", s.file),
        surah: s.expected_verses[0].surah,
        ayah: s.expected_verses[0].ayah,
      });
    }
  }

  return SAMPLE_LIMIT > 0 ? samples.slice(0, SAMPLE_LIMIT) : samples;
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main() {
  console.log("╔══════════════════════════════════════════════════════════════╗");
  console.log("║        Streaming Pipeline Benchmark                        ║");
  console.log("╚══════════════════════════════════════════════════════════════╝\n");

  await init();
  const samples = loadSamples();
  console.log(`Source: ${SOURCE}, Samples: ${samples.length}\n`);

  const results: StreamingResult[] = [];

  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    if (!existsSync(s.file)) continue;

    try {
      const audio = loadAudio(s.file);
      const result = await simulateStreaming(audio, s.surah, s.ayah, s.id);
      results.push(result);

      const icon = result.discoveryCorrect ? "✓" : "✗";
      process.stdout.write(
        `  ${icon} ${result.id.slice(0, 25).padEnd(25)} ` +
        `${result.expectedSurah}:${result.expectedAyah} → ` +
        `${result.firstMatchSurah ?? "?"}:${result.firstMatchAyah ?? "?"} ` +
        `t=${result.firstMatchTime.toFixed(1)}s ` +
        `words=${result.wordsCovered}/${result.totalWords} ` +
        `(${(result.wordCoverage * 100).toFixed(0)}%) ` +
        `jumps=${result.verseJumps}\n`,
      );
    } catch (e: any) {
      process.stdout.write(`  E ${s.id.slice(0, 25).padEnd(25)} ERROR: ${e.message?.slice(0, 50)}\n`);
    }

    if ((i + 1) % 20 === 0) {
      const correct = results.filter(r => r.discoveryCorrect).length;
      const avgCoverage = results.reduce((s, r) => s + r.wordCoverage, 0) / results.length;
      const avgTime = results.reduce((s, r) => s + r.firstMatchTime, 0) / results.length;
      console.log(
        `\n  [${i + 1}/${samples.length}] Discovery: ${correct}/${results.length} ` +
        `(${(correct / results.length * 100).toFixed(1)}%) ` +
        `Avg coverage: ${(avgCoverage * 100).toFixed(0)}% ` +
        `Avg time-to-match: ${avgTime.toFixed(1)}s\n`,
      );
    }
  }

  // ═══════════════════════ REPORT ═══════════════════════════════════════
  const total = results.length;
  const correct = results.filter(r => r.discoveryCorrect).length;
  const avgCoverage = results.reduce((s, r) => s + r.wordCoverage, 0) / total;
  const avgTime = results.reduce((s, r) => s + r.firstMatchTime, 0) / total;
  const avgJumps = results.reduce((s, r) => s + r.verseJumps, 0) / total;
  const noMatch = results.filter(r => r.firstMatchSurah === null).length;

  console.log(`\n${"═".repeat(70)}`);
  console.log(`  STREAMING BENCHMARK RESULTS`);
  console.log(`${"═".repeat(70)}`);
  console.log(`  Discovery accuracy: ${correct}/${total} (${(correct / total * 100).toFixed(1)}%)`);
  console.log(`  No match:           ${noMatch}`);
  console.log(`  Avg word coverage:  ${(avgCoverage * 100).toFixed(1)}%`);
  console.log(`  Avg time to match:  ${avgTime.toFixed(1)}s`);
  console.log(`  Avg verse jumps:    ${avgJumps.toFixed(2)}`);
  console.log(`${"═".repeat(70)}`);

  // Save
  mkdirSync(resolve(__dirname, "benchmark-results"), { recursive: true });
  const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const outPath = resolve(__dirname, `benchmark-results/streaming-${SOURCE}-${ts}.json`);
  writeFileSync(outPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    source: SOURCE,
    summary: { total, correct, accuracy: correct / total, avgCoverage, avgTime, avgJumps, noMatch },
    results,
  }, null, 2));
  console.log(`\nSaved: ${outPath}`);
}

main().catch(err => { console.error("Fatal:", err); process.exit(1); });
