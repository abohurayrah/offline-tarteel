#!/usr/bin/env npx tsx
/**
 * Whisper ASR Benchmark — runs the current Whisper pipeline against the
 * curated 54-sample test corpus.
 *
 * Usage:
 *   npx tsx test/benchmark-whisper.ts
 *   npx tsx test/benchmark-whisper.ts --sample=3     # Run only first N samples
 *   npx tsx test/benchmark-whisper.ts --category=short
 *   npx tsx test/benchmark-whisper.ts --source=retasy
 *
 * Measures:
 *   - Per-sample: Whisper transcription + QuranDB matching
 *   - Categories: correct, equivalent, adjacent (+/-1), wrong, no_match
 *   - Accuracy by category (short/medium/long/multi) and source (everyayah/retasy/user)
 *   - Average inference time
 *   - Saves JSON results to test/benchmark-results/whisper-{timestamp}.json
 */

import { execSync } from "node:child_process";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { pipeline, env } from "@huggingface/transformers";
import type { AutomaticSpeechRecognitionOutput } from "@huggingface/transformers";

import { QuranDB } from "../src/lib/quran-db.ts";
import { ratio } from "../src/lib/levenshtein.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const BENCHMARK_DIR = resolve(ROOT, "../../benchmark/test_corpus");
const RESULTS_DIR = resolve(__dirname, "benchmark-results");
const SAMPLE_RATE = 16000;

// ---- CLI args ---------------------------------------------------------------

const SAMPLE_LIMIT = parseInt(
  process.argv.find((a) => a.startsWith("--sample="))?.split("=")[1] ?? "0",
);
const FILTER_CATEGORY =
  process.argv.find((a) => a.startsWith("--category="))?.split("=")[1] ?? null;
const FILTER_SOURCE =
  process.argv.find((a) => a.startsWith("--source="))?.split("=")[1] ?? null;

// ---- Types ------------------------------------------------------------------

interface BenchmarkSample {
  id: string;
  file: string;
  surah: number;
  ayah: number;
  ayah_end: number | null;
  category: string;
  source: string;
  expected_verses: { surah: number; ayah: number }[];
}

type EvalResult = "correct" | "equiv" | "adjacent" | "wrong" | "no_match";

interface SampleResult {
  id: string;
  category: string;
  source: string;
  expected: string;
  got: string;
  eval: EvalResult;
  score: number;
  transcript: string;
  timeMs: number;
}

// ---- Audio loading via ffmpeg ------------------------------------------------

function loadAudio(filePath: string): Float32Array {
  const buf = execSync(
    `ffmpeg -hide_banner -loglevel error -i "${filePath}" -f f32le -ar ${SAMPLE_RATE} -ac 1 pipe:1`,
    { maxBuffer: 50 * 1024 * 1024 },
  );
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

// ---- Arabic normalization (mirrors quran-db.ts) -----------------------------

function normalizeArabic(text: string): string {
  text = text.replace(/\u2581/g, " ");
  text = text.replace(/\uFEFF/g, "");
  text = text.replace(
    /[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]/g,
    "",
  );
  text = text.replace(/[أإآٱ]/g, "ا");
  text = text.replace(/ة/g, "ه");
  text = text.replace(/ى/g, "ي");
  text = text.replace(/ـ/g, "");
  text = text.replace(/[،؟.!:]/g, "");
  text = text.replace(/\s+/g, " ").trim();
  return text;
}

// ---- Equivalence check (text-based, for scoring) ----------------------------

function areEquivalentByText(
  db: QuranDB,
  s1: number,
  a1: number,
  s2: number,
  a2: number,
): boolean {
  const v1 = db.getVerse(s1, a1);
  const v2 = db.getVerse(s2, a2);
  if (!v1 || !v2) return false;
  if (s1 === s2 && a1 === a2) return true;

  const t1 = normalizeArabic(v1.text_norm_no_bsm ?? v1.text_norm ?? v1.text_clean);
  const t2 = normalizeArabic(v2.text_norm_no_bsm ?? v2.text_norm ?? v2.text_clean);
  const t1ns = t1.replace(/ /g, "");
  const t2ns = t2.replace(/ /g, "");

  if (t1ns === t2ns) return true;
  return ratio(t1ns, t2ns) > 0.95;
}

// ---- Whisper pipeline -------------------------------------------------------

type ASRPipeline = Awaited<
  ReturnType<typeof pipeline<"automatic-speech-recognition">>
>;
let asr: ASRPipeline | null = null;

async function initWhisper(): Promise<void> {
  const modelPath = resolve(ROOT, "public/models/whisper-quran");

  // Configure transformers.js for local model loading in Node
  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  env.localModelPath = resolve(ROOT, "public/models");

  console.log("Loading Whisper model from", modelPath, "...");
  const t0 = performance.now();

  asr = await pipeline("automatic-speech-recognition", modelPath, {
    dtype: {
      encoder_model: "fp32",
      decoder_model_merged: "q8",
    },
    device: "cpu",
    local_files_only: true,
  });

  console.log(`  Model loaded in ${(performance.now() - t0).toFixed(0)}ms`);
}

async function transcribe(audio: Float32Array): Promise<string> {
  if (!asr) throw new Error("Whisper model not loaded");
  const result = (await asr(audio)) as AutomaticSpeechRecognitionOutput;
  return result.text.trim();
}

// ---- Evaluation logic -------------------------------------------------------

function evaluate(
  db: QuranDB,
  sample: BenchmarkSample,
  match: Record<string, any> | null,
): EvalResult {
  if (!match) return "no_match";

  const matchStart = match.ayah as number;
  const matchEnd = (match.ayah_end as number | undefined) ?? matchStart;
  const matchSurah = match.surah as number;

  let anyCorrect = false;
  let anyEquiv = false;
  let anyAdjacent = false;

  for (const expected of sample.expected_verses) {
    if (matchSurah === expected.surah) {
      // Check if expected ayah falls within the matched span
      if (expected.ayah >= matchStart && expected.ayah <= matchEnd) {
        anyCorrect = true;
      } else if (
        areEquivalentByText(db, expected.surah, expected.ayah, matchSurah, matchStart)
      ) {
        anyEquiv = true;
      } else if (
        Math.abs(expected.ayah - matchStart) <= 1 ||
        Math.abs(expected.ayah - matchEnd) <= 1
      ) {
        anyAdjacent = true;
      }
    }
  }

  // Also check the first expected verse directly for single-verse samples
  const firstExpected = sample.expected_verses[0];
  if (
    matchSurah === firstExpected.surah &&
    firstExpected.ayah >= matchStart &&
    firstExpected.ayah <= matchEnd
  ) {
    anyCorrect = true;
  }

  if (anyCorrect) return "correct";
  if (anyEquiv) return "equiv";
  if (anyAdjacent) return "adjacent";
  return "wrong";
}

// ---- Main -------------------------------------------------------------------

async function main() {
  console.log(
    "================================================================",
  );
  console.log(
    "  Whisper Benchmark -- Curated Test Corpus",
  );
  console.log(
    "================================================================\n",
  );

  // Load manifest
  const manifestPath = resolve(BENCHMARK_DIR, "manifest.json");
  if (!existsSync(manifestPath)) {
    console.error(`Manifest not found: ${manifestPath}`);
    console.error(
      "Expected benchmark corpus at ../../benchmark/test_corpus/ relative to frontend root.",
    );
    process.exit(1);
  }

  const manifest: { samples: BenchmarkSample[] } = JSON.parse(
    readFileSync(manifestPath, "utf-8"),
  );

  // Filter samples
  let samples = manifest.samples;
  if (FILTER_CATEGORY) {
    samples = samples.filter((s) => s.category === FILTER_CATEGORY);
    console.log(`Filtering by category: ${FILTER_CATEGORY}`);
  }
  if (FILTER_SOURCE) {
    samples = samples.filter((s) => s.source === FILTER_SOURCE);
    console.log(`Filtering by source: ${FILTER_SOURCE}`);
  }
  if (SAMPLE_LIMIT > 0) {
    samples = samples.slice(0, SAMPLE_LIMIT);
    console.log(`Limiting to first ${SAMPLE_LIMIT} samples`);
  }

  console.log(`Benchmark corpus: ${samples.length} samples\n`);

  // Load Whisper model
  await initWhisper();

  // Warm up with a short silent clip
  console.log("Warming up model...");
  const warmup = new Float32Array(SAMPLE_RATE * 2);
  for (let i = 0; i < warmup.length; i++) {
    warmup[i] = (Math.random() - 0.5) * 0.001;
  }
  await transcribe(warmup);
  console.log("Warmup complete.\n");

  // Load Quran DB
  const quranData = JSON.parse(
    readFileSync(resolve(ROOT, "public/quran.json"), "utf-8"),
  );
  const db = new QuranDB(quranData);
  console.log(`Quran DB: ${db.totalVerses} verses\n`);

  // Run benchmark
  const categories = ["short", "medium", "long", "multi"];
  const results: SampleResult[] = [];
  let totalInferenceMs = 0;

  for (const cat of categories) {
    const catSamples = samples.filter((s) => s.category === cat);
    if (catSamples.length === 0) continue;

    console.log(`\n-- ${cat.toUpperCase()} (${catSamples.length} samples) --`);

    for (const sample of catSamples) {
      const audioPath = resolve(BENCHMARK_DIR, sample.file);
      if (!existsSync(audioPath)) {
        console.log(`  SKIP ${sample.id} -- file not found: ${sample.file}`);
        continue;
      }

      try {
        // Load audio
        const audio = loadAudio(audioPath);

        // Transcribe
        const t0 = performance.now();
        const text = await transcribe(audio);
        const inferenceMs = performance.now() - t0;
        totalInferenceMs += inferenceMs;

        // Match verse
        const match = db.matchVerse(text);

        // Evaluate
        const evalResult = evaluate(db, sample, match);

        // Format output
        const matchStr = match
          ? `${match.surah}:${match.ayah}${match.ayah_end ? `-${match.ayah_end}` : ""}`
          : "null";
        const expectedStr =
          sample.expected_verses.length === 1
            ? `${sample.expected_verses[0].surah}:${sample.expected_verses[0].ayah}`
            : `${sample.surah}:${sample.ayah}-${sample.ayah_end}`;
        const icon =
          evalResult === "correct"
            ? "OK"
            : evalResult === "equiv"
              ? "~="
              : evalResult === "adjacent"
                ? "+1"
                : evalResult === "no_match"
                  ? "--"
                  : "XX";

        // Span coverage info for multi-verse samples
        let spanInfo = "";
        if (sample.expected_verses.length > 1 && match) {
          const matchStart = match.ayah as number;
          const matchEnd = (match.ayah_end as number | undefined) ?? matchStart;
          const covered = sample.expected_verses.filter(
            (ev) =>
              match.surah === ev.surah &&
              ev.ayah >= matchStart &&
              ev.ayah <= matchEnd,
          ).length;
          spanInfo = ` [span: ${covered}/${sample.expected_verses.length}]`;
        }

        console.log(
          `  [${icon}] ${sample.id.padEnd(25)} expected=${expectedStr.padEnd(10)} got=${matchStr.padEnd(10)} score=${((match?.score as number) ?? 0).toFixed(3)} ${inferenceMs.toFixed(0)}ms${spanInfo}`,
        );

        if (evalResult !== "correct" && evalResult !== "equiv") {
          console.log(
            `    transcript: "${text.slice(0, 100)}${text.length > 100 ? "..." : ""}"`,
          );
        }

        results.push({
          id: sample.id,
          category: sample.category,
          source: sample.source,
          expected: expectedStr,
          got: matchStr,
          eval: evalResult,
          score: (match?.score as number) ?? 0,
          transcript: text,
          timeMs: inferenceMs,
        });
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        console.log(
          `  [ER] ${sample.id.padEnd(25)} ERROR: ${msg.slice(0, 80)}`,
        );
      }
    }
  }

  // ===========================================================================
  // FINAL REPORT
  // ===========================================================================

  console.log(`\n${"=".repeat(70)}`);
  console.log("  WHISPER BENCHMARK RESULTS");
  console.log(`${"=".repeat(70)}\n`);

  // Per-category summary
  for (const cat of categories) {
    const catResults = results.filter((r) => r.category === cat);
    if (catResults.length === 0) continue;

    const correct = catResults.filter(
      (r) => r.eval === "correct" || r.eval === "equiv",
    ).length;
    const lenient = catResults.filter(
      (r) =>
        r.eval === "correct" ||
        r.eval === "equiv" ||
        r.eval === "adjacent",
    ).length;
    const total = catResults.length;
    const avgTime =
      catResults.reduce((s, r) => s + r.timeMs, 0) / total;

    console.log(
      `  ${cat.padEnd(8)} ${((correct / total) * 100).toFixed(1)}% strict (${correct}/${total})  ${((lenient / total) * 100).toFixed(1)}% lenient (${lenient}/${total})  avg=${avgTime.toFixed(0)}ms`,
    );
  }

  // Overall
  const total = results.length;
  const correct = results.filter(
    (r) => r.eval === "correct" || r.eval === "equiv",
  ).length;
  const lenient = results.filter(
    (r) =>
      r.eval === "correct" ||
      r.eval === "equiv" ||
      r.eval === "adjacent",
  ).length;
  const wrong = results.filter((r) => r.eval === "wrong").length;
  const noMatch = results.filter((r) => r.eval === "no_match").length;
  const avgTime = results.reduce((s, r) => s + r.timeMs, 0) / total;
  const avgInference = totalInferenceMs / total;

  console.log(`${"-".repeat(70)}`);
  console.log(
    `  OVERALL:        ${correct}/${total} (${((correct / total) * 100).toFixed(1)}%) correct/equiv`,
  );
  console.log(
    `  LENIENT:        ${lenient}/${total} (${((lenient / total) * 100).toFixed(1)}%) including +/-1 ayah`,
  );
  console.log(
    `  FALSE POSITIVE: ${wrong}/${total} (${((wrong / total) * 100).toFixed(1)}%) wrong verse`,
  );
  console.log(
    `  NO MATCH:       ${noMatch}/${total} (${((noMatch / total) * 100).toFixed(1)}%)`,
  );
  console.log(`  AVG INFERENCE:  ${avgInference.toFixed(0)}ms per sample`);
  console.log(`  AVG TOTAL:      ${avgTime.toFixed(0)}ms per sample`);

  // By source
  console.log(`\n  By source:`);
  for (const source of ["everyayah", "retasy", "user"]) {
    const srcResults = results.filter((r) => r.source === source);
    if (srcResults.length === 0) continue;
    const ok = srcResults.filter(
      (r) => r.eval === "correct" || r.eval === "equiv",
    ).length;
    const srcLenient = srcResults.filter(
      (r) =>
        r.eval === "correct" ||
        r.eval === "equiv" ||
        r.eval === "adjacent",
    ).length;
    console.log(
      `    ${source.padEnd(12)} ${ok}/${srcResults.length} (${((ok / srcResults.length) * 100).toFixed(1)}%) strict | ${srcLenient}/${srcResults.length} (${((srcLenient / srcResults.length) * 100).toFixed(1)}%) lenient`,
    );
  }

  // By eval result breakdown
  console.log(`\n  Result breakdown:`);
  const evalCounts: Record<string, number> = {};
  for (const r of results) {
    evalCounts[r.eval] = (evalCounts[r.eval] ?? 0) + 1;
  }
  for (const [evalType, count] of Object.entries(evalCounts).sort(
    (a, b) => b[1] - a[1],
  )) {
    console.log(
      `    ${evalType.padEnd(12)} ${count} (${((count / total) * 100).toFixed(1)}%)`,
    );
  }

  // Failures detail
  const failures = results.filter(
    (r) => r.eval === "wrong" || r.eval === "no_match",
  );
  if (failures.length > 0) {
    console.log(`\n  Failures (${failures.length}):`);
    for (const f of failures) {
      console.log(
        `    ${f.id}: expected=${f.expected} got=${f.got} (score=${f.score.toFixed(3)})`,
      );
      console.log(
        `      transcript: "${f.transcript.slice(0, 120)}"`,
      );
    }
  }

  console.log(`\n${"=".repeat(70)}\n`);

  // Save results to JSON
  if (!existsSync(RESULTS_DIR)) {
    mkdirSync(RESULTS_DIR, { recursive: true });
  }

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const outputPath = resolve(RESULTS_DIR, `whisper-${timestamp}.json`);

  const report = {
    timestamp: new Date().toISOString(),
    model: "tarteel-ai/whisper-tiny-ar-quran",
    config: {
      encoder: "fp32",
      decoder: "q8",
      device: "cpu",
      sampleLimit: SAMPLE_LIMIT || null,
      filterCategory: FILTER_CATEGORY,
      filterSource: FILTER_SOURCE,
    },
    summary: {
      total,
      correct,
      equiv: results.filter((r) => r.eval === "equiv").length,
      adjacent: results.filter((r) => r.eval === "adjacent").length,
      wrong,
      no_match: noMatch,
      strict_accuracy: correct / total,
      lenient_accuracy: lenient / total,
      false_positive_rate: wrong / total,
      avg_inference_ms: avgInference,
      avg_total_ms: avgTime,
    },
    by_category: Object.fromEntries(
      categories
        .map((cat) => {
          const catResults = results.filter((r) => r.category === cat);
          if (catResults.length === 0) return null;
          const ok = catResults.filter(
            (r) => r.eval === "correct" || r.eval === "equiv",
          ).length;
          return [
            cat,
            {
              total: catResults.length,
              correct: ok,
              accuracy: ok / catResults.length,
              avg_ms:
                catResults.reduce((s, r) => s + r.timeMs, 0) /
                catResults.length,
            },
          ];
        })
        .filter(Boolean) as [string, Record<string, number>][],
    ),
    by_source: Object.fromEntries(
      ["everyayah", "retasy", "user"]
        .map((src) => {
          const srcResults = results.filter((r) => r.source === src);
          if (srcResults.length === 0) return null;
          const ok = srcResults.filter(
            (r) => r.eval === "correct" || r.eval === "equiv",
          ).length;
          return [
            src,
            {
              total: srcResults.length,
              correct: ok,
              accuracy: ok / srcResults.length,
            },
          ];
        })
        .filter(Boolean) as [string, Record<string, number>][],
    ),
    results,
  };

  writeFileSync(outputPath, JSON.stringify(report, null, 2));
  console.log(`Results saved to: ${outputPath}`);
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
