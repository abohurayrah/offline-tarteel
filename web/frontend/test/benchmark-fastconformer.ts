#!/usr/bin/env npx tsx
/**
 * FastConformer CTC Benchmark — with improved QuranDB + constrained beam search.
 *
 * Usage:
 *   npx tsx test/benchmark-fastconformer.ts                    # greedy
 *   npx tsx test/benchmark-fastconformer.ts --constrained      # constrained beam search
 *   npx tsx test/benchmark-fastconformer.ts --beam-search      # unconstrained beam search
 *   npx tsx test/benchmark-fastconformer.ts --sample=5         # first N samples
 */
import { execSync } from "node:child_process";
import { readFileSync, existsSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";

import { computeMelSpectrogram } from "../src/worker/mel.ts";
import { CTCDecoder } from "../src/worker/ctc-decode.ts";
import { QuranTrie } from "../src/worker/quran-trie.ts";
import { QuranDB } from "../src/lib/quran-db.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const BENCHMARK_DIR = resolve(ROOT, "../../benchmark/test_corpus");
const RESULTS_DIR = resolve(__dirname, "benchmark-results");
const SAMPLE_RATE = 16000;

const CONSTRAINED = process.argv.includes("--constrained");
const BEAM_SEARCH = process.argv.includes("--beam-search");
const SAMPLE_LIMIT = parseInt(
  process.argv.find((a) => a.startsWith("--sample="))?.split("=")[1] ?? "0",
);

// ─── Audio + ONNX ────────────────────────────────────────────────────────────

function loadAudio(filePath: string): Float32Array {
  const buf = execSync(
    `ffmpeg -hide_banner -loglevel error -i "${filePath}" -f f32le -ar ${SAMPLE_RATE} -ac 1 pipe:1`,
    { maxBuffer: 50 * 1024 * 1024 },
  );
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

let session: ort.InferenceSession;
let decoder: CTCDecoder;
let trie: QuranTrie | null = null;

async function initModel(): Promise<void> {
  const modelPath = resolve(ROOT, "public/fastconformer_ar_ctc_q8.onnx");
  const vocabPath = resolve(ROOT, "public/vocab.json");
  console.log("Loading ONNX model...");
  const t0 = performance.now();
  session = await ort.InferenceSession.create(modelPath, { executionProviders: ["cpu"] });
  console.log(`  Model loaded in ${(performance.now() - t0).toFixed(0)}ms`);

  const vocabJson = JSON.parse(readFileSync(vocabPath, "utf-8"));
  decoder = new CTCDecoder(vocabJson);

  if (CONSTRAINED) {
    console.log("Building Quran trie for constrained decoding...");
    const quranData = JSON.parse(readFileSync(resolve(ROOT, "public/quran.json"), "utf-8"));
    trie = new QuranTrie(vocabJson);
    const trieVerses = quranData.map((v: any) => ({
      text_norm: v.text_clean || v.text_uthmani,
      surah: v.surah,
      ayah: v.ayah,
    }));
    trie.buildFromVerses(trieVerses);
    console.log(`  Trie built: ${trie.nodeCount} nodes`);
  }
}

async function runInference(audio: Float32Array): Promise<{ logprobs: Float32Array; timeSteps: number; vocabSize: number }> {
  const { features, timeFrames } = await computeMelSpectrogram(audio);
  const inputTensor = new ort.Tensor("float32", features, [1, 80, timeFrames]);
  const lengthTensor = new ort.Tensor("int64", BigInt64Array.from([BigInt(timeFrames)]), [1]);
  const feeds: Record<string, ort.Tensor> = {
    [session.inputNames[0]]: inputTensor,
    [session.inputNames[1]]: lengthTensor,
  };
  const results = await session.run(feeds);
  const outputTensor = results[session.outputNames[0]];
  const [, timeSteps, vocabSize] = outputTensor.dims as number[];
  return { logprobs: outputTensor.data as Float32Array, timeSteps, vocabSize };
}

async function transcribe(audio: Float32Array): Promise<string> {
  const { logprobs, timeSteps, vocabSize } = await runInference(audio);

  if (CONSTRAINED && trie) {
    const hypotheses = decoder.constrainedBeamSearch(logprobs, timeSteps, vocabSize, trie, {
      beamWidth: 10, topK: 20,
    });
    if (hypotheses.length > 0) return hypotheses[0].text;
  }

  if (BEAM_SEARCH) {
    const hypotheses = decoder.beamSearch(logprobs, timeSteps, vocabSize, {
      beamWidth: 10, topK: 20,
    });
    if (hypotheses.length > 0) return hypotheses[0].text;
  }

  return decoder.decode(logprobs, timeSteps, vocabSize).text;
}

// ─── Main ────────────────────────────────────────────────────────────────────

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

async function main() {
  const mode = CONSTRAINED ? "CONSTRAINED" : BEAM_SEARCH ? "BEAM" : "GREEDY";
  console.log("╔══════════════════════════════════════════════════════════════╗");
  console.log(`║  FastConformer CTC + Improved QuranDB — ${mode.padEnd(12)}        ║`);
  console.log("╚══════════════════════════════════════════════════════════════╝\n");

  const manifestPath = resolve(BENCHMARK_DIR, "manifest.json");
  if (!existsSync(manifestPath)) {
    console.error(`Manifest not found: ${manifestPath}`);
    process.exit(1);
  }

  const manifest: { samples: BenchmarkSample[] } = JSON.parse(readFileSync(manifestPath, "utf-8"));
  let samples = manifest.samples;
  if (SAMPLE_LIMIT > 0) samples = samples.slice(0, SAMPLE_LIMIT);
  console.log(`Benchmark: ${samples.length} samples, mode: ${mode}`);

  await initModel();

  // Use our IMPROVED QuranDB (with all matching improvements)
  const quranData = JSON.parse(readFileSync(resolve(ROOT, "public/quran.json"), "utf-8"));
  const db = new QuranDB(quranData);
  console.log(`QuranDB: ${db.totalVerses} verses\n`);

  const categories = ["short", "medium", "long", "multi"];
  const results: {
    sample: BenchmarkSample;
    eval: string;
    transcript: string;
    match: any;
    timeMs: number;
  }[] = [];

  for (const cat of categories) {
    const catSamples = samples.filter((s) => s.category === cat);
    if (catSamples.length === 0) continue;

    console.log(`\n── ${cat.toUpperCase()} (${catSamples.length} samples) ──`);

    for (const sample of catSamples) {
      const audioPath = resolve(BENCHMARK_DIR, sample.file);
      if (!existsSync(audioPath)) {
        console.log(`  SKIP ${sample.id}`);
        continue;
      }

      const t0 = performance.now();
      try {
        const audio = loadAudio(audioPath);
        const text = await transcribe(audio);

        // Use our improved QuranDB.matchVerse with all improvements
        const match = db.matchVerse(text, 0.2, 6);

        const timeMs = performance.now() - t0;

        let evalResult = "no_match";
        if (match) {
          const matchStart = match.ayah;
          const matchEnd = match.ayah_end ?? match.ayah;

          for (const expected of sample.expected_verses) {
            if (match.surah === expected.surah) {
              if (expected.ayah >= matchStart && expected.ayah <= matchEnd) {
                evalResult = "correct";
                break;
              } else if (Math.abs(expected.ayah - matchStart) <= 1) {
                evalResult = "adjacent";
              }
            }
          }
          if (evalResult === "no_match") evalResult = "wrong";
        }

        const icon = evalResult === "correct" ? "✓" : evalResult === "adjacent" ? "±" : "✗";
        const matchStr = match
          ? `${match.surah}:${match.ayah}${match.ayah_end ? `-${match.ayah_end}` : ""}`
          : "null";
        const expectedStr =
          sample.expected_verses.length === 1
            ? `${sample.expected_verses[0].surah}:${sample.expected_verses[0].ayah}`
            : `${sample.surah}:${sample.ayah}-${sample.ayah_end}`;

        console.log(
          `  ${icon} ${sample.id.padEnd(25)} expected=${expectedStr.padEnd(10)} got=${matchStr.padEnd(10)} score=${(match?.score ?? 0).toFixed(3)} ${timeMs.toFixed(0)}ms`,
        );

        if (evalResult !== "correct") {
          console.log(`    transcript: "${text.slice(0, 80)}${text.length > 80 ? "..." : ""}"`);
        }

        results.push({ sample, eval: evalResult, transcript: text, match, timeMs });
      } catch (e: any) {
        console.log(`  E ${sample.id.padEnd(25)} ERROR: ${e.message?.slice(0, 60)}`);
      }
    }
  }

  // ═══════════════════════ REPORT ═══════════════════════════════════════
  console.log(`\n${"═".repeat(70)}`);
  console.log(`  FASTCONFORMER + IMPROVED QURANDB (${mode})`);
  console.log(`${"═".repeat(70)}\n`);

  for (const cat of categories) {
    const cr = results.filter((r) => r.sample.category === cat);
    if (cr.length === 0) continue;
    const ok = cr.filter((r) => r.eval === "correct").length;
    const avg = cr.reduce((s, r) => s + r.timeMs, 0) / cr.length;
    const bar = "█".repeat(Math.round((ok / cr.length) * 20)) + "░".repeat(20 - Math.round((ok / cr.length) * 20));
    console.log(`  ${cat.padEnd(8)} ${bar} ${((ok / cr.length) * 100).toFixed(1)}% (${ok}/${cr.length}) avg=${avg.toFixed(0)}ms`);
  }

  const total = results.length;
  const correct = results.filter((r) => r.eval === "correct").length;
  const avgTime = results.reduce((s, r) => s + r.timeMs, 0) / total;

  console.log(`${"─".repeat(70)}`);
  console.log(`  OVERALL: ${correct}/${total} (${((correct / total) * 100).toFixed(1)}%) correct`);
  console.log(`  Average time: ${avgTime.toFixed(0)}ms per sample`);

  console.log(`\n  By source:`);
  for (const source of ["everyayah", "retasy", "user"]) {
    const sr = results.filter((r) => r.sample.source === source);
    if (sr.length === 0) continue;
    const ok = sr.filter((r) => r.eval === "correct").length;
    console.log(`    ${source.padEnd(12)} ${ok}/${sr.length} (${((ok / sr.length) * 100).toFixed(1)}%)`);
  }

  const failures = results.filter((r) => r.eval !== "correct");
  if (failures.length > 0) {
    console.log(`\n  Failures (${failures.length}):`);
    for (const f of failures) {
      const exp = f.sample.expected_verses.map((v) => `${v.surah}:${v.ayah}`).join(", ");
      const got = f.match ? `${f.match.surah}:${f.match.ayah}${f.match.ayah_end ? `-${f.match.ayah_end}` : ""}` : "null";
      console.log(`    ${f.sample.id}: expected=[${exp}] got=${got} (${(f.match?.score ?? 0).toFixed(3)})`);
    }
  }

  // Save results
  mkdirSync(RESULTS_DIR, { recursive: true });
  const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const outPath = resolve(RESULTS_DIR, `fastconformer-${mode.toLowerCase()}-${ts}.json`);
  writeFileSync(outPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    model: "fastconformer_ar_ctc_q8",
    mode,
    summary: { total, correct, accuracy: correct / total, avg_time_ms: avgTime },
    results: results.map((r) => ({
      id: r.sample.id, category: r.sample.category, source: r.sample.source,
      expected: r.sample.expected_verses.map((v) => `${v.surah}:${v.ayah}`).join(","),
      got: r.match ? `${r.match.surah}:${r.match.ayah}${r.match.ayah_end ? `-${r.match.ayah_end}` : ""}` : "null",
      eval: r.eval, score: r.match?.score ?? 0, transcript: r.transcript, timeMs: r.timeMs,
    })),
  }, null, 2));
  console.log(`\nResults saved: ${outPath}`);
  console.log(`${"═".repeat(70)}\n`);
}

main().catch((err) => { console.error("Fatal:", err); process.exit(1); });
