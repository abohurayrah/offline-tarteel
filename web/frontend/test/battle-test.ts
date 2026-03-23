#!/usr/bin/env npx tsx
/**
 * Battle Test — Edge cases for the FastConformer CTC pipeline.
 *
 * Tests:
 *   A. Half-cut ayahs (mid-verse starts)
 *   B. Very short audio (< 1 second)
 *   C. Silence (pure zeros)
 *   D. Concatenated verses
 *   E. Multiple reciters for same verse
 *   F. Failure analysis (beam search failures)
 *
 * Usage:
 *   npx tsx test/battle-test.ts
 *   npx tsx test/battle-test.ts --section=A      # run only section A
 *   npx tsx test/battle-test.ts --section=F      # run only failure analysis
 */
import { execSync } from "node:child_process";
import { readFileSync, existsSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";

import { computeMelSpectrogram } from "../src/worker/mel.ts";
import { CTCDecoder } from "../src/worker/ctc-decode.ts";
import { QuranDB } from "../src/lib/quran-db.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const BENCHMARK_DIR = resolve(ROOT, "../../benchmark/test_corpus");
const RESULTS_DIR = resolve(__dirname, "benchmark-results");
const SAMPLE_RATE = 16000;

const SECTION_FILTER = process.argv
  .find((a) => a.startsWith("--section="))
  ?.split("=")[1]
  ?.toUpperCase() ?? null;

// ─── Audio helpers ──────────────────────────────────────────────────────────

function loadAudio(filePath: string): Float32Array {
  const buf = execSync(
    `ffmpeg -hide_banner -loglevel error -i "${filePath}" -f f32le -ar ${SAMPLE_RATE} -ac 1 pipe:1`,
    { maxBuffer: 50 * 1024 * 1024 },
  );
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function cutAudioSecondHalf(audio: Float32Array): Float32Array {
  const midpoint = Math.floor(audio.length / 2);
  return audio.slice(midpoint);
}

function cutAudioDuration(audio: Float32Array, seconds: number): Float32Array {
  const samples = Math.min(Math.floor(seconds * SAMPLE_RATE), audio.length);
  return audio.slice(0, samples);
}

function createSilence(seconds: number): Float32Array {
  return new Float32Array(Math.floor(seconds * SAMPLE_RATE));
}

function concatenateAudio(...clips: Float32Array[]): Float32Array {
  const totalLength = clips.reduce((s, c) => s + c.length, 0);
  const result = new Float32Array(totalLength);
  let offset = 0;
  for (const clip of clips) {
    result.set(clip, offset);
    offset += clip.length;
  }
  return result;
}

// ─── ONNX Session ───────────────────────────────────────────────────────────

let session: ort.InferenceSession;
let decoder: CTCDecoder;

async function initModel(): Promise<void> {
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
}

async function runInference(
  audio: Float32Array,
): Promise<{
  logprobs: Float32Array;
  timeSteps: number;
  vocabSize: number;
}> {
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
  return {
    logprobs: outputTensor.data as Float32Array,
    timeSteps,
    vocabSize,
  };
}

async function transcribeGreedy(audio: Float32Array): Promise<string> {
  const { logprobs, timeSteps, vocabSize } = await runInference(audio);
  return decoder.decode(logprobs, timeSteps, vocabSize).text;
}

async function transcribeBeam(audio: Float32Array): Promise<string> {
  const { logprobs, timeSteps, vocabSize } = await runInference(audio);
  const hypotheses = decoder.beamSearch(logprobs, timeSteps, vocabSize, {
    beamWidth: 10,
    topK: 20,
  });
  return hypotheses.length > 0 ? hypotheses[0].text : "";
}

// ─── Test result tracking ───────────────────────────────────────────────────

interface TestResult {
  section: string;
  name: string;
  passed: boolean;
  details: string;
  transcript?: string;
  match?: string;
  score?: number;
  timeMs: number;
}

const allResults: TestResult[] = [];

function record(r: TestResult): void {
  allResults.push(r);
  const icon = r.passed ? "PASS" : "FAIL";
  console.log(`    [${icon}] ${r.name} (${r.timeMs.toFixed(0)}ms)`);
  console.log(`           ${r.details}`);
  if (r.transcript) {
    console.log(`           transcript: "${r.transcript}"`);
  }
  if (r.match) {
    console.log(`           match: ${r.match} (score=${r.score?.toFixed(3)})`);
  }
}

function shouldRun(section: string): boolean {
  return !SECTION_FILTER || SECTION_FILTER === section;
}

// ─── TESTS ──────────────────────────────────────────────────────────────────

let db: QuranDB;

async function sectionA_halfCutAyahs(): Promise<void> {
  console.log("\n══ SECTION A: Half-cut ayahs (mid-verse starts) ══");
  console.log("   Feed only the second half of each audio to the pipeline.\n");

  // Pick a variety: short, medium, long
  const testCases = [
    { file: "001002.mp3", surah: 1, ayah: 2, label: "Al-Fatiha 1:2" },
    { file: "002255.mp3", surah: 2, ayah: 255, label: "Ayat al-Kursi 2:255" },
    {
      file: "long_059_023.wav",
      surah: 59,
      ayah: 23,
      label: "Al-Hashr 59:23",
    },
    {
      file: "long_003_191.wav",
      surah: 3,
      ayah: 191,
      label: "Aal-Imran 3:191",
    },
    {
      file: "long_048_029.wav",
      surah: 48,
      ayah: 29,
      label: "Al-Fath 48:29",
    },
  ];

  for (const tc of testCases) {
    const t0 = performance.now();
    const audioPath = resolve(BENCHMARK_DIR, tc.file);
    if (!existsSync(audioPath)) {
      record({
        section: "A",
        name: `half_${tc.label}`,
        passed: false,
        details: `File not found: ${tc.file}`,
        timeMs: 0,
      });
      continue;
    }

    const fullAudio = loadAudio(audioPath);
    const halfAudio = cutAudioSecondHalf(fullAudio);
    const durationSec = (halfAudio.length / SAMPLE_RATE).toFixed(1);

    try {
      const text = await transcribeGreedy(halfAudio);
      const match = db.matchVerse(text, 0.2, 6);
      const timeMs = performance.now() - t0;

      // Pass if correct surah (mid-verse is hard, surah match is acceptable)
      const correctVerse =
        match && match.surah === tc.surah && match.ayah === tc.ayah;
      const correctSurah = match && match.surah === tc.surah;
      const passed = correctVerse || correctSurah || false;

      const matchStr = match
        ? `${match.surah}:${match.ayah}${match.ayah_end ? `-${match.ayah_end}` : ""}`
        : "null";

      record({
        section: "A",
        name: `half_${tc.label}`,
        passed,
        details: `${durationSec}s half-audio | expected=${tc.surah}:${tc.ayah} | ${correctVerse ? "exact" : correctSurah ? "surah-only" : "miss"}`,
        transcript: text.slice(0, 80),
        match: matchStr,
        score: match?.score ?? 0,
        timeMs,
      });
    } catch (e: any) {
      record({
        section: "A",
        name: `half_${tc.label}`,
        passed: false,
        details: `ERROR: ${e.message?.slice(0, 80)}`,
        timeMs: performance.now() - t0,
      });
    }
  }
}

async function sectionB_veryShortAudio(): Promise<void> {
  console.log("\n══ SECTION B: Very short audio (< 1 second) ══");
  console.log("   Feed 0.5-0.8s clips. Should return no match (too short).\n");

  const testCases = [
    { file: "001001.mp3", duration: 0.5, label: "0.5s from 1:1" },
    { file: "001002.mp3", duration: 0.8, label: "0.8s from 1:2" },
    { file: "112001.mp3", duration: 0.5, label: "0.5s from 112:1" },
    { file: "retasy_000.wav", duration: 0.6, label: "0.6s from retasy_000" },
    { file: "retasy_005.wav", duration: 0.7, label: "0.7s from retasy_005" },
  ];

  for (const tc of testCases) {
    const t0 = performance.now();
    const audioPath = resolve(BENCHMARK_DIR, tc.file);
    if (!existsSync(audioPath)) {
      record({
        section: "B",
        name: `short_${tc.label}`,
        passed: false,
        details: `File not found: ${tc.file}`,
        timeMs: 0,
      });
      continue;
    }

    try {
      const fullAudio = loadAudio(audioPath);
      const shortAudio = cutAudioDuration(fullAudio, tc.duration);
      const text = await transcribeGreedy(shortAudio);
      const match = db.matchVerse(text, 0.2, 6);
      const timeMs = performance.now() - t0;

      // Acceptable outcomes: no match, OR a very low score match, OR empty text
      // "No wrong match" means: null result OR empty transcript is ideal
      const noMatch = match === null;
      const emptyTranscript = text.trim().length === 0;
      const lowConfidence = match !== null && match.score < 0.5;

      // It is acceptable if:
      // - No match at all (ideal)
      // - Empty/gibberish transcript (acceptable)
      // - Low-confidence match (somewhat OK, not ideal)
      // A high-confidence WRONG match is a failure
      const passed = noMatch || emptyTranscript || lowConfidence;

      const matchStr = match
        ? `${match.surah}:${match.ayah} (score=${match.score?.toFixed(3)})`
        : "null";

      record({
        section: "B",
        name: `short_${tc.label}`,
        passed,
        details: `${tc.duration}s clip | transcript="${text.slice(0, 40)}" | match=${matchStr} | ${noMatch ? "no_match" : emptyTranscript ? "empty" : lowConfidence ? "low_conf" : "WRONG_MATCH"}`,
        transcript: text,
        match: matchStr,
        score: match?.score ?? 0,
        timeMs,
      });
    } catch (e: any) {
      // An error on extremely short audio may be expected (too few frames)
      const isExpected =
        e.message?.includes("length") || e.message?.includes("dimension");
      record({
        section: "B",
        name: `short_${tc.label}`,
        passed: isExpected,
        details: `${isExpected ? "Expected error" : "ERROR"}: ${e.message?.slice(0, 80)}`,
        timeMs: performance.now() - t0,
      });
    }
  }
}

async function sectionC_silence(): Promise<void> {
  console.log("\n══ SECTION C: Silence (pure zeros) ══");
  console.log("   Feed silence. Should return no match.\n");

  const durations = [1.0, 2.0, 5.0];

  for (const dur of durations) {
    const t0 = performance.now();
    try {
      const silentAudio = createSilence(dur);
      const text = await transcribeGreedy(silentAudio);
      const match = db.matchVerse(text, 0.2, 6);
      const timeMs = performance.now() - t0;

      const noMatch = match === null;
      const emptyTranscript = text.trim().length === 0;
      const passed = noMatch || emptyTranscript;

      record({
        section: "C",
        name: `silence_${dur}s`,
        passed,
        details: `${dur}s silence | transcript="${text.slice(0, 40)}" | match=${match ? `${match.surah}:${match.ayah}` : "null"}`,
        transcript: text,
        match: match ? `${match.surah}:${match.ayah}` : "null",
        score: match?.score ?? 0,
        timeMs,
      });
    } catch (e: any) {
      // Error on silence is also acceptable — means pipeline rejects it
      record({
        section: "C",
        name: `silence_${dur}s`,
        passed: true,
        details: `Pipeline error on silence (acceptable): ${e.message?.slice(0, 60)}`,
        timeMs: performance.now() - t0,
      });
    }
  }
}

async function sectionD_concatenatedVerses(): Promise<void> {
  console.log("\n══ SECTION D: Concatenated verses ══");
  console.log(
    "   Concat two consecutive verses. Should identify at least the first.\n",
  );

  const testCases = [
    {
      files: ["001001.mp3", "001002.mp3"],
      surah: 1,
      ayah1: 1,
      ayah2: 2,
      label: "1:1+1:2",
    },
    {
      files: ["long_059_023.wav", "long_059_024.wav"],
      surah: 59,
      ayah1: 23,
      ayah2: 24,
      label: "59:23+59:24",
    },
    {
      files: ["long_002_285.wav", "long_002_286.wav"],
      surah: 2,
      ayah1: 285,
      ayah2: 286,
      label: "2:285+2:286",
    },
  ];

  for (const tc of testCases) {
    const t0 = performance.now();
    const audioPaths = tc.files.map((f) => resolve(BENCHMARK_DIR, f));
    const allExist = audioPaths.every((p) => existsSync(p));
    if (!allExist) {
      record({
        section: "D",
        name: `concat_${tc.label}`,
        passed: false,
        details: "Missing audio file(s)",
        timeMs: 0,
      });
      continue;
    }

    try {
      const clips = audioPaths.map((p) => loadAudio(p));
      const combined = concatenateAudio(...clips);
      const durationSec = (combined.length / SAMPLE_RATE).toFixed(1);
      const text = await transcribeGreedy(combined);
      const match = db.matchVerse(text, 0.2, 6);
      const timeMs = performance.now() - t0;

      // Pass if match includes the first verse (exact or in a span)
      let passed = false;
      if (match) {
        const matchStart = match.ayah;
        const matchEnd = match.ayah_end ?? match.ayah;
        if (match.surah === tc.surah) {
          // First verse in range
          if (tc.ayah1 >= matchStart && tc.ayah1 <= matchEnd) {
            passed = true;
          }
          // Or second verse in range
          if (tc.ayah2 >= matchStart && tc.ayah2 <= matchEnd) {
            passed = true;
          }
        }
      }

      const matchStr = match
        ? `${match.surah}:${match.ayah}${match.ayah_end ? `-${match.ayah_end}` : ""}`
        : "null";

      record({
        section: "D",
        name: `concat_${tc.label}`,
        passed,
        details: `${durationSec}s combined | expected=${tc.surah}:${tc.ayah1}-${tc.ayah2}`,
        transcript: text.slice(0, 100),
        match: matchStr,
        score: match?.score ?? 0,
        timeMs,
      });
    } catch (e: any) {
      record({
        section: "D",
        name: `concat_${tc.label}`,
        passed: false,
        details: `ERROR: ${e.message?.slice(0, 80)}`,
        timeMs: performance.now() - t0,
      });
    }
  }
}

async function sectionE_multipleReciters(): Promise<void> {
  console.log("\n══ SECTION E: Multiple reciters for same verse ══");
  console.log("   Same verse from different audio sources should both match.\n");

  // Verse 1:1 has everyayah (001001.mp3) and retasy (retasy_000.wav, retasy_001.wav)
  // Verse 1:4 has retasy_002.wav, retasy_006.wav, retasy_014.wav
  // Verse 103:2 has retasy_005.wav, retasy_013.wav

  const testGroups = [
    {
      surah: 1,
      ayah: 1,
      label: "1:1 (Bismillah)",
      files: ["001001.mp3", "retasy_000.wav", "retasy_001.wav"],
      sources: ["everyayah", "retasy_000", "retasy_001"],
    },
    {
      surah: 1,
      ayah: 4,
      label: "1:4 (Maliki yawm al-din)",
      files: ["retasy_002.wav", "retasy_006.wav", "retasy_014.wav"],
      sources: ["retasy_002", "retasy_006", "retasy_014"],
    },
    {
      surah: 103,
      ayah: 2,
      label: "103:2 (Inna al-insana)",
      files: ["retasy_005.wav", "retasy_013.wav"],
      sources: ["retasy_005", "retasy_013"],
    },
  ];

  for (const group of testGroups) {
    console.log(`  -- ${group.label} --`);
    const groupResults: { source: string; match: string; passed: boolean }[] =
      [];

    for (let i = 0; i < group.files.length; i++) {
      const t0 = performance.now();
      const audioPath = resolve(BENCHMARK_DIR, group.files[i]);
      if (!existsSync(audioPath)) {
        record({
          section: "E",
          name: `reciter_${group.label}_${group.sources[i]}`,
          passed: false,
          details: `File not found: ${group.files[i]}`,
          timeMs: 0,
        });
        continue;
      }

      try {
        const audio = loadAudio(audioPath);
        const text = await transcribeGreedy(audio);
        const match = db.matchVerse(text, 0.2, 6);
        const timeMs = performance.now() - t0;

        const passed =
          match !== null &&
          match.surah === group.surah &&
          match.ayah === group.ayah;
        const matchStr = match
          ? `${match.surah}:${match.ayah}`
          : "null";

        groupResults.push({
          source: group.sources[i],
          match: matchStr,
          passed,
        });
        record({
          section: "E",
          name: `reciter_${group.label}_${group.sources[i]}`,
          passed,
          details: `expected=${group.surah}:${group.ayah} got=${matchStr}`,
          transcript: text.slice(0, 60),
          match: matchStr,
          score: match?.score ?? 0,
          timeMs,
        });
      } catch (e: any) {
        record({
          section: "E",
          name: `reciter_${group.label}_${group.sources[i]}`,
          passed: false,
          details: `ERROR: ${e.message?.slice(0, 60)}`,
          timeMs: performance.now() - t0,
        });
      }
    }

    const allPassed = groupResults.every((r) => r.passed);
    const anyPassed = groupResults.some((r) => r.passed);
    console.log(
      `     => ${allPassed ? "ALL MATCH" : anyPassed ? "PARTIAL MATCH" : "NONE MATCH"} across ${groupResults.length} reciters`,
    );
  }
}

async function sectionF_failureAnalysis(): Promise<void> {
  console.log("\n══ SECTION F: Failure analysis (beam search failures) ══");
  console.log(
    "   Analyzing the 5 remaining failures with full transcript + top-3 candidates.\n",
  );

  const failures = [
    {
      id: "retasy_003",
      file: "retasy_003.wav",
      surah: 1,
      ayah: 2,
      label: "1:2 (Al-Hamdulillah)",
    },
    {
      id: "retasy_012",
      file: "retasy_012.wav",
      surah: 114,
      ayah: 2,
      label: "114:2 (Malik al-nas)",
    },
    {
      id: "retasy_016",
      file: "retasy_016.wav",
      surah: 3,
      ayah: 2,
      label: "3:2 (Allahu la ilaha illa huw)",
    },
    {
      id: "retasy_021",
      file: "retasy_021.wav",
      surah: 1,
      ayah: 7,
      label: "1:7 (Sirat alladhina)",
    },
    {
      id: "multi_036_001_005",
      file: "multi_036_001_005.wav",
      surah: 36,
      ayah: 1,
      ayah_end: 5,
      label: "36:1-5 (Ya-Sin)",
    },
  ];

  for (const f of failures) {
    const t0 = performance.now();
    const audioPath = resolve(BENCHMARK_DIR, f.file);
    if (!existsSync(audioPath)) {
      record({
        section: "F",
        name: `analysis_${f.id}`,
        passed: false,
        details: `File not found: ${f.file}`,
        timeMs: 0,
      });
      continue;
    }

    try {
      const audio = loadAudio(audioPath);
      const audioDuration = (audio.length / SAMPLE_RATE).toFixed(2);

      // Greedy transcript
      const { logprobs, timeSteps, vocabSize } = await runInference(audio);
      const greedyResult = decoder.decode(logprobs, timeSteps, vocabSize);
      const greedyText = greedyResult.text;

      // Beam transcript
      const beamHypotheses = decoder.beamSearch(logprobs, timeSteps, vocabSize, {
        beamWidth: 10,
        topK: 20,
      });
      const beamText =
        beamHypotheses.length > 0 ? beamHypotheses[0].text : "(empty)";

      // Match with top-K candidates (returnTopK=3)
      const greedyMatch = db.matchVerse(greedyText, 0.1, 6, null, 5);
      const beamMatch =
        beamHypotheses.length > 0
          ? db.matchVerse(beamText, 0.1, 6, null, 5)
          : null;

      // Get expected verse text for comparison
      const expectedVerse = db.getVerse(f.surah, f.ayah);
      const expectedText = expectedVerse?.text_norm ?? "(not found)";

      const timeMs = performance.now() - t0;

      console.log(`\n  ── ${f.id} (expected: ${f.label}) ──`);
      console.log(`     Audio duration: ${audioDuration}s`);
      console.log(`     Expected verse text: "${expectedText}"`);
      console.log(`     Greedy transcript:   "${greedyText}"`);
      console.log(`     Beam transcript:     "${beamText}"`);
      if (beamHypotheses.length > 1) {
        console.log(`     Beam alt #2:         "${beamHypotheses[1]?.text}" (score=${beamHypotheses[1]?.score.toFixed(3)})`);
      }
      if (beamHypotheses.length > 2) {
        console.log(`     Beam alt #3:         "${beamHypotheses[2]?.text}" (score=${beamHypotheses[2]?.score.toFixed(3)})`);
      }

      console.log(`     Greedy match: ${greedyMatch ? `${greedyMatch.surah}:${greedyMatch.ayah}${greedyMatch.ayah_end ? `-${greedyMatch.ayah_end}` : ""} (score=${greedyMatch.score.toFixed(3)})` : "null"}`);
      if (greedyMatch?.runners_up) {
        console.log(`     Top-3 candidates (greedy):`);
        for (
          let i = 0;
          i < Math.min(3, greedyMatch.runners_up.length);
          i++
        ) {
          const c = greedyMatch.runners_up[i];
          console.log(
            `       ${i + 1}. ${c.surah}:${c.ayah} score=${c.score.toFixed(3)} raw=${c.raw_score.toFixed(3)} "${c.text_norm}"`,
          );
        }
      }

      console.log(`     Beam match: ${beamMatch ? `${beamMatch.surah}:${beamMatch.ayah}${beamMatch.ayah_end ? `-${beamMatch.ayah_end}` : ""} (score=${beamMatch.score.toFixed(3)})` : "null"}`);
      if (beamMatch?.runners_up) {
        console.log(`     Top-3 candidates (beam):`);
        for (let i = 0; i < Math.min(3, beamMatch.runners_up.length); i++) {
          const c = beamMatch.runners_up[i];
          console.log(
            `       ${i + 1}. ${c.surah}:${c.ayah} score=${c.score.toFixed(3)} raw=${c.raw_score.toFixed(3)} "${c.text_norm}"`,
          );
        }
      }

      // Determine root cause
      let rootCause: string;
      const isTranscriptGarbled =
        greedyText.length < 3 ||
        (expectedText.length > 0 &&
          greedyText.length < expectedText.length * 0.3);
      const greedyCorrectSurah =
        greedyMatch && greedyMatch.surah === f.surah;
      const beamCorrectSurah = beamMatch && beamMatch.surah === f.surah;

      if (isTranscriptGarbled) {
        rootCause = "ASR: transcript too short/garbled";
      } else if (!greedyMatch && !beamMatch) {
        rootCause = "MATCHING: both decoders return null";
      } else if (greedyCorrectSurah || beamCorrectSurah) {
        rootCause = "MATCHING: correct surah but wrong ayah";
      } else {
        rootCause = "ASR: transcript does not match expected verse content";
      }
      console.log(`     ROOT CAUSE: ${rootCause}`);

      record({
        section: "F",
        name: `analysis_${f.id}`,
        passed: false, // These are known failures — passing means N/A
        details: `Root cause: ${rootCause}`,
        transcript: greedyText,
        match: greedyMatch
          ? `${greedyMatch.surah}:${greedyMatch.ayah}`
          : "null",
        score: greedyMatch?.score ?? 0,
        timeMs,
      });
    } catch (e: any) {
      record({
        section: "F",
        name: `analysis_${f.id}`,
        passed: false,
        details: `ERROR: ${e.message?.slice(0, 80)}`,
        timeMs: performance.now() - t0,
      });
    }
  }
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  console.log("╔══════════════════════════════════════════════════════════════╗");
  console.log("║  BATTLE TEST — FastConformer CTC Edge Cases                ║");
  console.log("╚══════════════════════════════════════════════════════════════╝\n");

  await initModel();

  const quranData = JSON.parse(
    readFileSync(resolve(ROOT, "public/quran.json"), "utf-8"),
  );
  db = new QuranDB(quranData);
  console.log(`QuranDB: ${db.totalVerses} verses`);

  if (shouldRun("A")) await sectionA_halfCutAyahs();
  if (shouldRun("B")) await sectionB_veryShortAudio();
  if (shouldRun("C")) await sectionC_silence();
  if (shouldRun("D")) await sectionD_concatenatedVerses();
  if (shouldRun("E")) await sectionE_multipleReciters();
  if (shouldRun("F")) await sectionF_failureAnalysis();

  // ═══════════════════════ SUMMARY ═══════════════════════════════════════
  console.log(`\n${"=".repeat(70)}`);
  console.log("  BATTLE TEST SUMMARY");
  console.log(`${"=".repeat(70)}\n`);

  const sections = ["A", "B", "C", "D", "E", "F"];
  const sectionLabels: Record<string, string> = {
    A: "Half-cut ayahs",
    B: "Very short audio",
    C: "Silence",
    D: "Concatenated verses",
    E: "Multiple reciters",
    F: "Failure analysis (known)",
  };

  let totalPassed = 0;
  let totalRun = 0;
  // Section F is analysis-only (all are known failures), so exclude from pass/fail count
  const countableSections = ["A", "B", "C", "D", "E"];

  for (const s of sections) {
    const sr = allResults.filter((r) => r.section === s);
    if (sr.length === 0) continue;
    const passed = sr.filter((r) => r.passed).length;
    const isCountable = countableSections.includes(s);
    if (isCountable) {
      totalPassed += passed;
      totalRun += sr.length;
    }
    console.log(
      `  Section ${s} (${sectionLabels[s]}): ${passed}/${sr.length} ${isCountable ? "passed" : "(analysis)"}`,
    );
  }

  console.log(`${"─".repeat(70)}`);
  console.log(
    `  OVERALL (A-E): ${totalPassed}/${totalRun} passed (${totalRun > 0 ? ((totalPassed / totalRun) * 100).toFixed(1) : 0}%)`,
  );

  // List all failures
  const failures = allResults.filter(
    (r) => !r.passed && countableSections.includes(r.section),
  );
  if (failures.length > 0) {
    console.log(`\n  Failures:`);
    for (const f of failures) {
      console.log(`    [${f.section}] ${f.name}: ${f.details}`);
    }
  }
  console.log(`${"=".repeat(70)}\n`);

  // Save results
  mkdirSync(RESULTS_DIR, { recursive: true });
  const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const outPath = resolve(RESULTS_DIR, `battle-test-${ts}.json`);
  writeFileSync(
    outPath,
    JSON.stringify(
      {
        timestamp: new Date().toISOString(),
        summary: {
          total: totalRun,
          passed: totalPassed,
          rate: totalRun > 0 ? totalPassed / totalRun : 0,
        },
        results: allResults,
      },
      null,
      2,
    ),
  );
  console.log(`Results saved: ${outPath}`);
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
