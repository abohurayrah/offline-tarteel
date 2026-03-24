#!/usr/bin/env npx tsx
/**
 * Adverse Audio Conditions Test
 *
 * Tests the pipeline under real-world degradation scenarios:
 *   1. Pure silence (5s zeros)           -- should produce NO verse_match
 *   2. Very quiet audio (10% amplitude)  -- should still work (RMS normalization)
 *   3. Short clip (0.5s)                 -- should return no match
 *   4. Background noise at various SNRs  -- should not produce false positives
 *   5. Clipped audio (first/last 20%)    -- test partial verse recognition
 *
 * KEY INVARIANT: FALSE POSITIVES must be ZERO.
 *   Returning no match is always preferable to returning the wrong verse.
 *
 * Both non-streaming (single-shot inference) and streaming (chunked feed)
 * pipelines are tested for each condition.
 *
 * Usage:
 *   npx tsx test/adverse-conditions.ts
 *   npx tsx test/adverse-conditions.ts --section=1   # silence only
 *   npx tsx test/adverse-conditions.ts --section=4   # noise only
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
import { SAMPLE_RATE } from "../src/lib/types.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const BENCHMARK_DIR = resolve(ROOT, "../../benchmark/test_corpus");
const RESULTS_DIR = resolve(__dirname, "benchmark-results");

const SECTION_FILTER = process.argv
  .find((a) => a.startsWith("--section="))
  ?.split("=")[1] ?? null;

const CHUNK_MS = 300;
const CHUNK_SAMPLES = Math.floor(SAMPLE_RATE * CHUNK_MS / 1000);

// ─── Audio helpers (Float32Array operations) ────────────────────────────────

function loadAudio(filePath: string): Float32Array {
  const buf = execSync(
    `ffmpeg -hide_banner -loglevel error -i "${filePath}" -f f32le -ar ${SAMPLE_RATE} -ac 1 pipe:1`,
    { maxBuffer: 50 * 1024 * 1024 },
  );
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

/** Create pure silence (all zeros). */
function createSilence(seconds: number): Float32Array {
  return new Float32Array(Math.floor(seconds * SAMPLE_RATE));
}

/** Scale amplitude by a factor (e.g. 0.1 = 10% volume). */
function scaleAmplitude(audio: Float32Array, factor: number): Float32Array {
  const out = new Float32Array(audio.length);
  for (let i = 0; i < audio.length; i++) {
    out[i] = audio[i] * factor;
  }
  return out;
}

/** Take only the first N seconds of audio. */
function takeFirst(audio: Float32Array, seconds: number): Float32Array {
  const samples = Math.min(Math.floor(seconds * SAMPLE_RATE), audio.length);
  return audio.slice(0, samples);
}

/** Remove first and last percentages of audio (simulate clipping). */
function clipEdges(audio: Float32Array, cutFraction: number): Float32Array {
  const cutSamples = Math.floor(audio.length * cutFraction);
  return audio.slice(cutSamples, audio.length - cutSamples);
}

/**
 * Add Gaussian noise at a target SNR (in dB).
 *
 * SNR_dB = 10 * log10(P_signal / P_noise)
 * => P_noise = P_signal / 10^(SNR_dB/10)
 * => noise_rms = signal_rms / 10^(SNR_dB/20)
 */
function addNoise(audio: Float32Array, snrDb: number): Float32Array {
  // Compute signal RMS
  let sumSq = 0;
  for (let i = 0; i < audio.length; i++) {
    sumSq += audio[i] * audio[i];
  }
  const signalRms = Math.sqrt(sumSq / audio.length);

  // Target noise RMS
  const noiseRms = signalRms / Math.pow(10, snrDb / 20);

  // Generate Gaussian noise using Box-Muller transform
  const out = new Float32Array(audio.length);
  for (let i = 0; i < audio.length; i++) {
    // Box-Muller: generate two uniform randoms, produce one normal
    const u1 = Math.random();
    const u2 = Math.random();
    const gaussian = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    out[i] = audio[i] + gaussian * noiseRms;
  }
  return out;
}

/** Compute RMS of an audio buffer. */
function computeRms(audio: Float32Array): number {
  let sumSq = 0;
  for (let i = 0; i < audio.length; i++) {
    sumSq += audio[i] * audio[i];
  }
  return Math.sqrt(sumSq / audio.length);
}

// ─── ONNX + pipeline setup ─────────────────────────────────────────────────

let session: ort.InferenceSession;
let decoder: CTCDecoder;
let db: QuranDB;

async function init(): Promise<void> {
  const modelPath = resolve(ROOT, "public/fastconformer_ar_ctc_q8.onnx");
  const vocabPath = resolve(ROOT, "public/vocab.json");
  console.log("Loading ONNX model...");
  const t0 = performance.now();
  session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ["cpu"],
  });
  console.log(`  Model loaded in ${(performance.now() - t0).toFixed(0)}ms`);

  const vocabJson = JSON.parse(readFileSync(vocabPath, "utf-8"));
  decoder = new CTCDecoder(vocabJson);

  const quranData = JSON.parse(
    readFileSync(resolve(ROOT, "public/quran.json"), "utf-8"),
  );
  db = new QuranDB(quranData);

  // Load disambiguation map (mirrors the worker init path)
  const disambigPath = resolve(ROOT, "public/ambiguity-compact.json");
  if (existsSync(disambigPath)) {
    const disambigData = JSON.parse(readFileSync(disambigPath, "utf-8"));
    db.loadDisambiguationMap(disambigData);
    console.log("  Disambiguation map loaded.");
  }
  console.log(`  QuranDB: ${db.totalVerses} verses\n`);
}

// Non-streaming transcription (single-shot)
async function transcribeNonStreaming(audio: Float32Array): Promise<TranscribeResult> {
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

// Non-streaming: transcribe + match
async function nonStreamingPipeline(
  audio: Float32Array,
): Promise<{ text: string; match: any }> {
  const { text } = await transcribeNonStreaming(audio);
  const match = db.matchVerse(text, 0.2, 6);
  return { text, match };
}

// Streaming: feed through RecitationTracker in 300ms chunks
async function streamingPipeline(
  audio: Float32Array,
): Promise<{ messages: WorkerOutbound[]; verseMatches: WorkerOutbound[] }> {
  const tracker = new RecitationTracker(db, transcribeNonStreaming);
  const allMessages: WorkerOutbound[] = [];

  for (let offset = 0; offset < audio.length; offset += CHUNK_SAMPLES) {
    const chunk = audio.slice(offset, Math.min(offset + CHUNK_SAMPLES, audio.length));
    const messages = await tracker.feed(chunk);
    allMessages.push(...messages);
  }

  const verseMatches = allMessages.filter((m) => m.type === "verse_match");
  return { messages: allMessages, verseMatches };
}

// ─── Test result tracking ───────────────────────────────────────────────────

interface TestResult {
  section: number;
  name: string;
  pipeline: "non-streaming" | "streaming";
  passed: boolean;
  falsePositive: boolean;
  details: string;
  transcript?: string;
  matchSurah?: number | null;
  matchAyah?: number | null;
  matchScore?: number;
  expectedSurah?: number;
  expectedAyah?: number;
  timeMs: number;
}

const allResults: TestResult[] = [];
let falsePositiveCount = 0;

function record(r: TestResult): void {
  allResults.push(r);
  if (r.falsePositive) falsePositiveCount++;
  const status = r.falsePositive
    ? "FALSE_POS"
    : r.passed
      ? "PASS"
      : "FAIL";
  const pipeLabel = r.pipeline === "streaming" ? "strm" : "full";
  console.log(`    [${status}] (${pipeLabel}) ${r.name} (${r.timeMs.toFixed(0)}ms)`);
  console.log(`           ${r.details}`);
  if (r.transcript !== undefined) {
    console.log(`           transcript: "${r.transcript.slice(0, 80)}"`);
  }
  if (r.matchSurah !== undefined) {
    const matchStr = r.matchSurah !== null ? `${r.matchSurah}:${r.matchAyah}` : "null";
    console.log(`           match: ${matchStr} (score=${(r.matchScore ?? 0).toFixed(3)})`);
  }
}

function shouldRun(section: number): boolean {
  return !SECTION_FILTER || SECTION_FILTER === String(section);
}

// ─── Source audio for tests ─────────────────────────────────────────────────

interface SourceAudio {
  label: string;
  file: string;
  surah: number;
  ayah: number;
  audio: Float32Array;
}

function loadSourceAudio(): SourceAudio[] {
  const sources: SourceAudio[] = [];
  const candidates = [
    { file: "002255.mp3", surah: 2, ayah: 255, label: "Ayat al-Kursi (2:255)" },
    { file: "001002.mp3", surah: 1, ayah: 2, label: "Al-Fatiha (1:2)" },
    { file: "112001.mp3", surah: 112, ayah: 1, label: "Al-Ikhlas (112:1)" },
    { file: "long_059_023.wav", surah: 59, ayah: 23, label: "Al-Hashr (59:23)" },
    { file: "retasy_004.wav", surah: 1, ayah: 3, label: "Retasy (1:3)" },
  ];

  for (const c of candidates) {
    const audioPath = resolve(BENCHMARK_DIR, c.file);
    if (existsSync(audioPath)) {
      try {
        const audio = loadAudio(audioPath);
        sources.push({ ...c, audio });
        console.log(`  Loaded: ${c.label} (${(audio.length / SAMPLE_RATE).toFixed(1)}s, RMS=${computeRms(audio).toFixed(4)})`);
      } catch {
        console.log(`  SKIP: ${c.label} (load error)`);
      }
    } else {
      console.log(`  SKIP: ${c.label} (not found)`);
    }
  }
  return sources;
}

// ─── SECTION 1: Pure silence ────────────────────────────────────────────────

async function section1_silence(): Promise<void> {
  console.log("\n" + "=".repeat(70));
  console.log("  SECTION 1: Pure silence (5 seconds of zeros)");
  console.log("  Expected: NO verse_match from either pipeline");
  console.log("=".repeat(70));

  const silentAudio = createSilence(5.0);
  console.log(`  Audio: ${(silentAudio.length / SAMPLE_RATE).toFixed(1)}s, RMS=${computeRms(silentAudio).toFixed(6)}\n`);

  // Non-streaming
  {
    const t0 = performance.now();
    try {
      const { text, match } = await nonStreamingPipeline(silentAudio);
      const timeMs = performance.now() - t0;
      const noMatch = match === null;
      const emptyText = text.trim().length === 0;
      const passed = noMatch || emptyText;

      record({
        section: 1,
        name: "silence_5s",
        pipeline: "non-streaming",
        passed,
        falsePositive: match !== null && match.score >= 0.4,
        details: `${noMatch ? "No match (correct)" : emptyText ? "Empty transcript (correct)" : `UNEXPECTED match: ${match.surah}:${match.ayah}`}`,
        transcript: text,
        matchSurah: match?.surah ?? null,
        matchAyah: match?.ayah ?? null,
        matchScore: match?.score ?? 0,
        timeMs,
      });
    } catch (e: any) {
      record({
        section: 1,
        name: "silence_5s",
        pipeline: "non-streaming",
        passed: true,
        falsePositive: false,
        details: `Pipeline error on silence (acceptable): ${e.message?.slice(0, 60)}`,
        timeMs: performance.now() - t0,
      });
    }
  }

  // Streaming
  {
    const t0 = performance.now();
    try {
      const { verseMatches } = await streamingPipeline(silentAudio);
      const timeMs = performance.now() - t0;
      const noMatch = verseMatches.length === 0;

      record({
        section: 1,
        name: "silence_5s",
        pipeline: "streaming",
        passed: noMatch,
        falsePositive: verseMatches.length > 0,
        details: noMatch
          ? "No verse_match emitted (correct)"
          : `UNEXPECTED: ${verseMatches.length} verse_match(es) emitted`,
        matchSurah: verseMatches.length > 0 ? (verseMatches[0] as any).surah : null,
        matchAyah: verseMatches.length > 0 ? (verseMatches[0] as any).ayah : null,
        matchScore: verseMatches.length > 0 ? (verseMatches[0] as any).confidence : 0,
        timeMs,
      });
    } catch (e: any) {
      record({
        section: 1,
        name: "silence_5s",
        pipeline: "streaming",
        passed: true,
        falsePositive: false,
        details: `Pipeline error on silence (acceptable): ${e.message?.slice(0, 60)}`,
        timeMs: performance.now() - t0,
      });
    }
  }
}

// ─── SECTION 2: Very quiet audio (10% amplitude) ───────────────────────────

async function section2_quietAudio(sources: SourceAudio[]): Promise<void> {
  console.log("\n" + "=".repeat(70));
  console.log("  SECTION 2: Very quiet audio (10% amplitude)");
  console.log("  Expected: Should still match correctly if RMS normalization works");
  console.log("=".repeat(70));

  for (const src of sources) {
    const quietAudio = scaleAmplitude(src.audio, 0.1);
    const originalRms = computeRms(src.audio);
    const quietRms = computeRms(quietAudio);
    console.log(`\n  -- ${src.label} (original RMS=${originalRms.toFixed(4)}, quiet RMS=${quietRms.toFixed(4)}) --`);

    // Non-streaming
    {
      const t0 = performance.now();
      try {
        const { text, match } = await nonStreamingPipeline(quietAudio);
        const timeMs = performance.now() - t0;

        const correctMatch = match !== null && match.surah === src.surah && match.ayah === src.ayah;
        const noMatch = match === null;
        const wrongMatch = match !== null && (match.surah !== src.surah || match.ayah !== src.ayah);

        record({
          section: 2,
          name: `quiet_${src.surah}:${src.ayah}`,
          pipeline: "non-streaming",
          passed: correctMatch,
          falsePositive: wrongMatch,
          details: correctMatch
            ? "Correct match despite 10% amplitude"
            : noMatch
              ? "No match (normalization may not compensate enough)"
              : `WRONG match: expected ${src.surah}:${src.ayah}, got ${match.surah}:${match.ayah}`,
          transcript: text,
          matchSurah: match?.surah ?? null,
          matchAyah: match?.ayah ?? null,
          matchScore: match?.score ?? 0,
          expectedSurah: src.surah,
          expectedAyah: src.ayah,
          timeMs,
        });
      } catch (e: any) {
        record({
          section: 2,
          name: `quiet_${src.surah}:${src.ayah}`,
          pipeline: "non-streaming",
          passed: false,
          falsePositive: false,
          details: `ERROR: ${e.message?.slice(0, 80)}`,
          timeMs: performance.now() - t0,
        });
      }
    }

    // Streaming
    {
      const t0 = performance.now();
      try {
        const { verseMatches } = await streamingPipeline(quietAudio);
        const timeMs = performance.now() - t0;

        const firstMatch = verseMatches.length > 0 ? (verseMatches[0] as any) : null;
        const correctMatch = firstMatch !== null &&
          firstMatch.surah === src.surah && firstMatch.ayah === src.ayah;
        const noMatch = verseMatches.length === 0;
        const wrongMatch = firstMatch !== null &&
          (firstMatch.surah !== src.surah || firstMatch.ayah !== src.ayah);

        record({
          section: 2,
          name: `quiet_${src.surah}:${src.ayah}`,
          pipeline: "streaming",
          passed: correctMatch,
          falsePositive: wrongMatch,
          details: correctMatch
            ? "Correct match in streaming mode"
            : noMatch
              ? "No match (streaming pipeline rejected quiet audio)"
              : `WRONG match: expected ${src.surah}:${src.ayah}, got ${firstMatch.surah}:${firstMatch.ayah}`,
          matchSurah: firstMatch?.surah ?? null,
          matchAyah: firstMatch?.ayah ?? null,
          matchScore: firstMatch?.confidence ?? 0,
          expectedSurah: src.surah,
          expectedAyah: src.ayah,
          timeMs,
        });
      } catch (e: any) {
        record({
          section: 2,
          name: `quiet_${src.surah}:${src.ayah}`,
          pipeline: "streaming",
          passed: false,
          falsePositive: false,
          details: `ERROR: ${e.message?.slice(0, 80)}`,
          timeMs: performance.now() - t0,
        });
      }
    }
  }
}

// ─── SECTION 3: Short clip (0.5 seconds) ───────────────────────────────────

async function section3_shortClip(sources: SourceAudio[]): Promise<void> {
  console.log("\n" + "=".repeat(70));
  console.log("  SECTION 3: Short clip (first 0.5 seconds of a verse)");
  console.log("  Expected: No match (too short to identify)");
  console.log("=".repeat(70));

  for (const src of sources) {
    const shortAudio = takeFirst(src.audio, 0.5);
    const durationMs = (shortAudio.length / SAMPLE_RATE * 1000).toFixed(0);
    console.log(`\n  -- ${src.label} (${durationMs}ms clip) --`);

    // Non-streaming
    {
      const t0 = performance.now();
      try {
        const { text, match } = await nonStreamingPipeline(shortAudio);
        const timeMs = performance.now() - t0;

        const noMatch = match === null;
        const emptyText = text.trim().length === 0;
        // A correct match on 0.5s is surprising but acceptable; a WRONG match is a false positive
        const correctMatch = match !== null && match.surah === src.surah && match.ayah === src.ayah;
        const wrongMatch = match !== null && (match.surah !== src.surah || match.ayah !== src.ayah);

        record({
          section: 3,
          name: `short_${src.surah}:${src.ayah}`,
          pipeline: "non-streaming",
          passed: noMatch || emptyText || correctMatch,
          falsePositive: wrongMatch,
          details: noMatch
            ? "No match (correct for 0.5s clip)"
            : emptyText
              ? "Empty transcript (correct)"
              : correctMatch
                ? `Surprisingly correct match on 0.5s (score=${match.score.toFixed(3)})`
                : `FALSE POSITIVE: wrong match ${match.surah}:${match.ayah} on 0.5s clip`,
          transcript: text,
          matchSurah: match?.surah ?? null,
          matchAyah: match?.ayah ?? null,
          matchScore: match?.score ?? 0,
          expectedSurah: src.surah,
          expectedAyah: src.ayah,
          timeMs,
        });
      } catch (e: any) {
        // Errors on very short audio are expected (too few mel frames)
        const isExpected = e.message?.includes("length") ||
          e.message?.includes("dimension") ||
          e.message?.includes("invalid");
        record({
          section: 3,
          name: `short_${src.surah}:${src.ayah}`,
          pipeline: "non-streaming",
          passed: isExpected,
          falsePositive: false,
          details: `${isExpected ? "Expected error" : "ERROR"}: ${e.message?.slice(0, 80)}`,
          timeMs: performance.now() - t0,
        });
      }
    }

    // Streaming: 0.5s is only ~2 chunks, way below FIRST_TRIGGER_SAMPLES (2s)
    {
      const t0 = performance.now();
      try {
        const { verseMatches } = await streamingPipeline(shortAudio);
        const timeMs = performance.now() - t0;

        const noMatch = verseMatches.length === 0;
        const firstMatch = verseMatches.length > 0 ? (verseMatches[0] as any) : null;
        const wrongMatch = firstMatch !== null &&
          (firstMatch.surah !== src.surah || firstMatch.ayah !== src.ayah);

        record({
          section: 3,
          name: `short_${src.surah}:${src.ayah}`,
          pipeline: "streaming",
          passed: noMatch,
          falsePositive: wrongMatch,
          details: noMatch
            ? "No verse_match (correct -- not enough audio for trigger)"
            : `UNEXPECTED verse_match on 0.5s`,
          matchSurah: firstMatch?.surah ?? null,
          matchAyah: firstMatch?.ayah ?? null,
          matchScore: firstMatch?.confidence ?? 0,
          expectedSurah: src.surah,
          expectedAyah: src.ayah,
          timeMs,
        });
      } catch (e: any) {
        record({
          section: 3,
          name: `short_${src.surah}:${src.ayah}`,
          pipeline: "streaming",
          passed: true,
          falsePositive: false,
          details: `Pipeline error on short clip (acceptable): ${e.message?.slice(0, 60)}`,
          timeMs: performance.now() - t0,
        });
      }
    }
  }
}

// ─── SECTION 4: Background noise at various SNRs ───────────────────────────

async function section4_backgroundNoise(sources: SourceAudio[]): Promise<void> {
  console.log("\n" + "=".repeat(70));
  console.log("  SECTION 4: Background noise at various SNR levels");
  console.log("  30dB = quiet room, 15dB = noisy room, 5dB = very noisy");
  console.log("  KEY: false positives (wrong match) must be ZERO");
  console.log("=".repeat(70));

  const snrLevels = [
    { snr: 30, label: "30dB (quiet room)", shouldWork: true },
    { snr: 15, label: "15dB (noisy room)", shouldWork: true },
    { snr: 5,  label: "5dB (very noisy)",  shouldWork: false },
  ];

  for (const src of sources) {
    for (const level of snrLevels) {
      const noisyAudio = addNoise(src.audio, level.snr);
      const noisyRms = computeRms(noisyAudio);
      console.log(`\n  -- ${src.label} @ SNR ${level.label} (RMS=${noisyRms.toFixed(4)}) --`);

      // Non-streaming
      {
        const t0 = performance.now();
        try {
          const { text, match } = await nonStreamingPipeline(noisyAudio);
          const timeMs = performance.now() - t0;

          const correctMatch = match !== null && match.surah === src.surah && match.ayah === src.ayah;
          const noMatch = match === null;
          const wrongMatch = match !== null && (match.surah !== src.surah || match.ayah !== src.ayah);

          // Pass conditions:
          // - Correct match = always pass
          // - No match at low SNR = acceptable
          // - Wrong match = FAIL (false positive)
          const passed = correctMatch || (noMatch && !level.shouldWork);
          // Even at "should work" SNR, no-match is not a false positive
          const fp = wrongMatch;

          record({
            section: 4,
            name: `noise_${level.snr}dB_${src.surah}:${src.ayah}`,
            pipeline: "non-streaming",
            passed: correctMatch || noMatch, // no-match is acceptable at any SNR
            falsePositive: fp,
            details: correctMatch
              ? `Correct match at SNR ${level.snr}dB`
              : noMatch
                ? `No match at SNR ${level.snr}dB${level.shouldWork ? " (degraded)" : " (expected)"}`
                : `FALSE POSITIVE at SNR ${level.snr}dB: expected ${src.surah}:${src.ayah}, got ${match.surah}:${match.ayah}`,
            transcript: text,
            matchSurah: match?.surah ?? null,
            matchAyah: match?.ayah ?? null,
            matchScore: match?.score ?? 0,
            expectedSurah: src.surah,
            expectedAyah: src.ayah,
            timeMs,
          });
        } catch (e: any) {
          record({
            section: 4,
            name: `noise_${level.snr}dB_${src.surah}:${src.ayah}`,
            pipeline: "non-streaming",
            passed: false,
            falsePositive: false,
            details: `ERROR: ${e.message?.slice(0, 80)}`,
            timeMs: performance.now() - t0,
          });
        }
      }

      // Streaming
      {
        const t0 = performance.now();
        try {
          const { verseMatches } = await streamingPipeline(noisyAudio);
          const timeMs = performance.now() - t0;

          const firstMatch = verseMatches.length > 0 ? (verseMatches[0] as any) : null;
          const correctMatch = firstMatch !== null &&
            firstMatch.surah === src.surah && firstMatch.ayah === src.ayah;
          const noMatch = verseMatches.length === 0;
          const wrongMatch = firstMatch !== null &&
            (firstMatch.surah !== src.surah || firstMatch.ayah !== src.ayah);

          record({
            section: 4,
            name: `noise_${level.snr}dB_${src.surah}:${src.ayah}`,
            pipeline: "streaming",
            passed: correctMatch || noMatch,
            falsePositive: wrongMatch,
            details: correctMatch
              ? `Correct streaming match at SNR ${level.snr}dB`
              : noMatch
                ? `No streaming match at SNR ${level.snr}dB`
                : `FALSE POSITIVE (streaming) at SNR ${level.snr}dB: expected ${src.surah}:${src.ayah}, got ${firstMatch.surah}:${firstMatch.ayah}`,
            matchSurah: firstMatch?.surah ?? null,
            matchAyah: firstMatch?.ayah ?? null,
            matchScore: firstMatch?.confidence ?? 0,
            expectedSurah: src.surah,
            expectedAyah: src.ayah,
            timeMs,
          });
        } catch (e: any) {
          record({
            section: 4,
            name: `noise_${level.snr}dB_${src.surah}:${src.ayah}`,
            pipeline: "streaming",
            passed: false,
            falsePositive: false,
            details: `ERROR: ${e.message?.slice(0, 80)}`,
            timeMs: performance.now() - t0,
          });
        }
      }
    }
  }
}

// ─── SECTION 5: Clipped audio (first/last 20% cut) ─────────────────────────

async function section5_clippedAudio(sources: SourceAudio[]): Promise<void> {
  console.log("\n" + "=".repeat(70));
  console.log("  SECTION 5: Clipped audio (first/last 20% cut off)");
  console.log("  Tests partial verse recognition with missing start/end");
  console.log("=".repeat(70));

  for (const src of sources) {
    const clippedAudio = clipEdges(src.audio, 0.2);
    const originalDur = (src.audio.length / SAMPLE_RATE).toFixed(1);
    const clippedDur = (clippedAudio.length / SAMPLE_RATE).toFixed(1);
    console.log(`\n  -- ${src.label} (${originalDur}s -> ${clippedDur}s after 20% trim) --`);

    // Non-streaming
    {
      const t0 = performance.now();
      try {
        const { text, match } = await nonStreamingPipeline(clippedAudio);
        const timeMs = performance.now() - t0;

        const correctMatch = match !== null && match.surah === src.surah && match.ayah === src.ayah;
        const correctSurah = match !== null && match.surah === src.surah;
        const noMatch = match === null;
        const wrongMatch = match !== null && match.surah !== src.surah;

        record({
          section: 5,
          name: `clipped_${src.surah}:${src.ayah}`,
          pipeline: "non-streaming",
          // Exact match or same-surah is acceptable for clipped audio
          passed: correctMatch || correctSurah || noMatch,
          falsePositive: wrongMatch,
          details: correctMatch
            ? "Exact match on clipped audio"
            : correctSurah
              ? `Same surah (${match.surah}:${match.ayah}) on clipped audio`
              : noMatch
                ? "No match on clipped audio (60% of verse)"
                : `FALSE POSITIVE: wrong surah ${match.surah}:${match.ayah}`,
          transcript: text,
          matchSurah: match?.surah ?? null,
          matchAyah: match?.ayah ?? null,
          matchScore: match?.score ?? 0,
          expectedSurah: src.surah,
          expectedAyah: src.ayah,
          timeMs,
        });
      } catch (e: any) {
        record({
          section: 5,
          name: `clipped_${src.surah}:${src.ayah}`,
          pipeline: "non-streaming",
          passed: false,
          falsePositive: false,
          details: `ERROR: ${e.message?.slice(0, 80)}`,
          timeMs: performance.now() - t0,
        });
      }
    }

    // Streaming
    {
      const t0 = performance.now();
      try {
        const { verseMatches } = await streamingPipeline(clippedAudio);
        const timeMs = performance.now() - t0;

        const firstMatch = verseMatches.length > 0 ? (verseMatches[0] as any) : null;
        const correctMatch = firstMatch !== null &&
          firstMatch.surah === src.surah && firstMatch.ayah === src.ayah;
        const correctSurah = firstMatch !== null && firstMatch.surah === src.surah;
        const noMatch = verseMatches.length === 0;
        const wrongMatch = firstMatch !== null && firstMatch.surah !== src.surah;

        record({
          section: 5,
          name: `clipped_${src.surah}:${src.ayah}`,
          pipeline: "streaming",
          passed: correctMatch || correctSurah || noMatch,
          falsePositive: wrongMatch,
          details: correctMatch
            ? "Exact streaming match on clipped audio"
            : correctSurah
              ? `Same surah streaming match (${firstMatch.surah}:${firstMatch.ayah})`
              : noMatch
                ? "No streaming match on clipped audio"
                : `FALSE POSITIVE (streaming): wrong surah ${firstMatch.surah}:${firstMatch.ayah}`,
          matchSurah: firstMatch?.surah ?? null,
          matchAyah: firstMatch?.ayah ?? null,
          matchScore: firstMatch?.confidence ?? 0,
          expectedSurah: src.surah,
          expectedAyah: src.ayah,
          timeMs,
        });
      } catch (e: any) {
        record({
          section: 5,
          name: `clipped_${src.surah}:${src.ayah}`,
          pipeline: "streaming",
          passed: false,
          falsePositive: false,
          details: `ERROR: ${e.message?.slice(0, 80)}`,
          timeMs: performance.now() - t0,
        });
      }
    }
  }
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  console.log("+" + "=".repeat(68) + "+");
  console.log("|  ADVERSE AUDIO CONDITIONS TEST                                   |");
  console.log("|  Testing pipeline robustness under real-world degradation         |");
  console.log("+" + "=".repeat(68) + "+\n");

  await init();

  console.log("Loading source audio from benchmark corpus...");
  const sources = loadSourceAudio();
  if (sources.length === 0) {
    console.error("No source audio files found. Cannot run tests.");
    process.exit(1);
  }
  console.log(`  ${sources.length} source files loaded.\n`);

  // Run sections
  if (shouldRun(1)) await section1_silence();
  if (shouldRun(2)) await section2_quietAudio(sources);
  if (shouldRun(3)) await section3_shortClip(sources);
  if (shouldRun(4)) await section4_backgroundNoise(sources);
  if (shouldRun(5)) await section5_clippedAudio(sources);

  // ═══════════════════════ SUMMARY ═══════════════════════════════════════
  console.log(`\n${"=".repeat(70)}`);
  console.log("  ADVERSE CONDITIONS TEST SUMMARY");
  console.log(`${"=".repeat(70)}\n`);

  const sectionLabels: Record<number, string> = {
    1: "Pure silence",
    2: "Very quiet audio (10%)",
    3: "Short clip (0.5s)",
    4: "Background noise",
    5: "Clipped audio (20% cut)",
  };

  let totalPassed = 0;
  let totalTests = 0;
  let totalFP = 0;

  for (let s = 1; s <= 5; s++) {
    const sr = allResults.filter((r) => r.section === s);
    if (sr.length === 0) continue;

    const passed = sr.filter((r) => r.passed).length;
    const fp = sr.filter((r) => r.falsePositive).length;
    totalPassed += passed;
    totalTests += sr.length;
    totalFP += fp;

    const nsResults = sr.filter((r) => r.pipeline === "non-streaming");
    const stResults = sr.filter((r) => r.pipeline === "streaming");
    const nsPassed = nsResults.filter((r) => r.passed).length;
    const stPassed = stResults.filter((r) => r.passed).length;

    console.log(`  Section ${s} (${sectionLabels[s]}):`);
    console.log(`    Non-streaming: ${nsPassed}/${nsResults.length} passed`);
    console.log(`    Streaming:     ${stPassed}/${stResults.length} passed`);
    if (fp > 0) {
      console.log(`    *** FALSE POSITIVES: ${fp} ***`);
    }
  }

  console.log(`\n${"~".repeat(70)}`);
  console.log(`  OVERALL: ${totalPassed}/${totalTests} passed (${totalTests > 0 ? ((totalPassed / totalTests) * 100).toFixed(1) : 0}%)`);
  console.log(`  FALSE POSITIVES: ${totalFP} ${totalFP === 0 ? "(ZERO -- target met)" : "*** VIOLATIONS ***"}`);
  console.log(`${"~".repeat(70)}`);

  // Detailed false positive report
  const fpResults = allResults.filter((r) => r.falsePositive);
  if (fpResults.length > 0) {
    console.log("\n  FALSE POSITIVE DETAILS:");
    for (const fp of fpResults) {
      console.log(`    [S${fp.section}] ${fp.name} (${fp.pipeline})`);
      console.log(`      Expected: ${fp.expectedSurah}:${fp.expectedAyah}`);
      console.log(`      Got:      ${fp.matchSurah}:${fp.matchAyah} (score=${(fp.matchScore ?? 0).toFixed(3)})`);
      console.log(`      Transcript: "${fp.transcript?.slice(0, 60)}"`);
    }
  }

  // Robustness by pipeline
  console.log("\n  BY PIPELINE:");
  for (const pType of ["non-streaming", "streaming"] as const) {
    const pr = allResults.filter((r) => r.pipeline === pType);
    const passed = pr.filter((r) => r.passed).length;
    const fp = pr.filter((r) => r.falsePositive).length;
    console.log(`    ${pType.padEnd(15)} ${passed}/${pr.length} passed, ${fp} false positives`);
  }

  // Robustness by degradation type for non-streaming (section 4 breakdown)
  const noiseResults = allResults.filter((r) => r.section === 4 && r.pipeline === "non-streaming");
  if (noiseResults.length > 0) {
    console.log("\n  NOISE ROBUSTNESS (non-streaming):");
    for (const snr of [30, 15, 5]) {
      const snrR = noiseResults.filter((r) => r.name.includes(`${snr}dB`));
      const correct = snrR.filter((r) => r.matchSurah !== null && !r.falsePositive && r.passed).length;
      const noMatch = snrR.filter((r) => r.matchSurah === null).length;
      const fp = snrR.filter((r) => r.falsePositive).length;
      console.log(`    SNR ${String(snr).padStart(2)}dB: ${correct}/${snrR.length} correct, ${noMatch} no-match, ${fp} false-pos`);
    }
  }

  console.log(`\n${"=".repeat(70)}\n`);

  // Save results
  mkdirSync(RESULTS_DIR, { recursive: true });
  const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const outPath = resolve(RESULTS_DIR, `adverse-conditions-${ts}.json`);
  writeFileSync(
    outPath,
    JSON.stringify(
      {
        timestamp: new Date().toISOString(),
        summary: {
          total: totalTests,
          passed: totalPassed,
          rate: totalTests > 0 ? totalPassed / totalTests : 0,
          false_positives: totalFP,
          target_met: totalFP === 0,
        },
        results: allResults,
      },
      null,
      2,
    ),
  );
  console.log(`Results saved: ${outPath}`);

  // Exit with error code if any false positives
  if (totalFP > 0) {
    console.error(`\nFATAL: ${totalFP} false positive(s) detected. This violates the zero-FP invariant.`);
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
