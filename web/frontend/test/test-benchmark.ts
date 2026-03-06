#!/usr/bin/env npx tsx
/**
 * Benchmark test — runs the pipeline against the curated benchmark corpus.
 *
 * Usage:
 *   npx tsx test/test-benchmark.ts
 *   npx tsx test/test-benchmark.ts --beam-search
 *   npx tsx test/test-benchmark.ts --beam-search --beam-width=15
 */

import { execSync } from "node:child_process";
import { readFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";

import { computeMelSpectrogram } from "../src/worker/mel.ts";
import { CTCDecoder, Hypothesis } from "../src/worker/ctc-decode.ts";
import { ratio, fragmentScore } from "../src/lib/levenshtein.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const BENCHMARK_DIR = resolve(ROOT, "../../benchmark/test_corpus");
const SAMPLE_RATE = 16000;

const ENABLE_BEAM_SEARCH = process.argv.includes("--beam-search");
const BEAM_WIDTH = parseInt(process.argv.find(a => a.startsWith("--beam-width="))?.split("=")[1] ?? "10");

// ─── Arabic normalization ────────────────────────────────────────────────────

function normalizeArabic(text: string): string {
  text = text.replace(/\u2581/g, " ");
  text = text.replace(/\uFEFF/g, "");
  text = text.replace(/[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]/g, "");
  text = text.replace(/[أإآٱ]/g, "ا");
  text = text.replace(/ة/g, "ه");
  text = text.replace(/ى/g, "ي");
  text = text.replace(/ـ/g, "");
  text = text.replace(/[،؟.!:]/g, "");
  text = text.replace(/\s+/g, " ").trim();
  return text;
}

// ─── QuranDB (same as test-pipeline-full.ts) ────────────────────────────────

interface ArabicVerse {
  surah: number;
  ayah: number;
  text_norm: string;
  text_norm_ns: string;
  text_norm_no_bsm: string | null;
  text_norm_no_bsm_ns: string | null;
}

const MUQATTAAT_SURAHS = new Set([2,3,7,10,11,12,13,14,15,19,20,26,27,28,29,30,31,32,36,38,40,41,42,43,44,45,46,50,68]);

class ArabicQuranDB {
  verses: ArabicVerse[];
  private _trigramIndex: Map<string, number[]> = new Map();
  private _bySurah: Map<number, ArabicVerse[]> = new Map();
  private _byRef: Map<string, number> = new Map();
  private _equivByText: Map<string, number[]> = new Map();
  private _equivByIdx: Map<number, number[]> = new Map();

  private static BSM_NORM = normalizeArabic("بسم الله الرحمن الرحيم");

  constructor(data: any[]) {
    this.verses = data.map(v => {
      const norm = normalizeArabic(v.text_clean || v.text_uthmani);
      const ns = norm.replace(/ /g, "");
      let noBsm: string | null = null;
      let noBsmNs: string | null = null;
      if (v.ayah === 1 && v.surah !== 1 && v.surah !== 9) {
        const stripped = norm.replace(ArabicQuranDB.BSM_NORM, "").trim();
        if (stripped.length > 0) { noBsm = stripped; noBsmNs = stripped.replace(/ /g, ""); }
      }
      return { surah: v.surah, ayah: v.ayah, text_norm: norm, text_norm_ns: ns,
               text_norm_no_bsm: noBsm, text_norm_no_bsm_ns: noBsmNs };
    });

    for (let i = 0; i < this.verses.length; i++) {
      const v = this.verses[i];
      this._byRef.set(`${v.surah}:${v.ayah}`, i);
      const arr = this._bySurah.get(v.surah) ?? [];
      arr.push(v);
      this._bySurah.set(v.surah, arr);
    }

    this._buildTrigramIndex();
    this._buildEquivalenceGroups();
  }

  private _buildTrigramIndex(): void {
    for (let idx = 0; idx < this.verses.length; idx++) {
      const v = this.verses[idx];
      const seen = new Set<string>();
      for (const text of [v.text_norm_ns, v.text_norm_no_bsm_ns]) {
        if (!text) continue;
        for (let i = 0; i <= text.length - 3; i++) {
          const tri = text.slice(i, i + 3);
          if (seen.has(tri)) continue;
          seen.add(tri);
          const arr = this._trigramIndex.get(tri);
          if (arr) arr.push(idx);
          else this._trigramIndex.set(tri, [idx]);
        }
      }
    }
  }

  private _buildEquivalenceGroups(): void {
    for (let idx = 0; idx < this.verses.length; idx++) {
      const v = this.verses[idx];
      const effectiveText = v.text_norm_no_bsm_ns ?? v.text_norm_ns;
      if (!this._equivByText.has(effectiveText)) {
        this._equivByText.set(effectiveText, []);
      }
      this._equivByText.get(effectiveText)!.push(idx);
    }
    const fullTextGroups = new Map<string, number[]>();
    for (let idx = 0; idx < this.verses.length; idx++) {
      const ns = this.verses[idx].text_norm_ns;
      if (!fullTextGroups.has(ns)) fullTextGroups.set(ns, []);
      fullTextGroups.get(ns)!.push(idx);
    }
    for (let idx = 0; idx < this.verses.length; idx++) {
      const equivs = new Set<number>();
      const v = this.verses[idx];
      const effectiveText = v.text_norm_no_bsm_ns ?? v.text_norm_ns;
      for (const i of this._equivByText.get(effectiveText) ?? []) equivs.add(i);
      for (const i of fullTextGroups.get(v.text_norm_ns) ?? []) equivs.add(i);
      if (v.text_norm_no_bsm_ns) {
        for (const i of fullTextGroups.get(v.text_norm_no_bsm_ns) ?? []) equivs.add(i);
      }
      if (equivs.size > 1) {
        this._equivByIdx.set(idx, [...equivs]);
      }
    }
  }

  areEquivalent(idx1: number, idx2: number): boolean {
    if (idx1 === idx2) return true;
    const group = this._equivByIdx.get(idx1);
    return group ? group.includes(idx2) : false;
  }

  areEquivalentByRef(s1: number, a1: number, s2: number, a2: number): boolean {
    const i1 = this._byRef.get(`${s1}:${a1}`);
    const i2 = this._byRef.get(`${s2}:${a2}`);
    if (i1 === undefined || i2 === undefined) return false;
    if (this.areEquivalent(i1, i2)) return true;
    const v1 = this.verses[i1];
    const v2 = this.verses[i2];
    const t1 = v1.text_norm_no_bsm_ns ?? v1.text_norm_ns;
    const t2 = v2.text_norm_no_bsm_ns ?? v2.text_norm_ns;
    return ratio(t1, t2) > 0.95;
  }

  private _getCandidates(text: string, maxCandidates = 300): Set<number> {
    const noSpace = text.replace(/ /g, "");
    if (noSpace.length < 3) {
      const candidates = new Set<number>();
      for (let idx = 0; idx < this.verses.length; idx++) {
        const v = this.verses[idx];
        if (v.text_norm_ns.length <= noSpace.length + 5 ||
            (v.ayah === 1 && MUQATTAAT_SURAHS.has(v.surah)) ||
            (v.text_norm_no_bsm_ns && v.text_norm_no_bsm_ns.length <= noSpace.length + 5)) {
          candidates.add(idx);
        }
      }
      if (candidates.size < 20) return new Set(this.verses.map((_, i) => i));
      return candidates;
    }
    const queryTrigrams = new Set<string>();
    for (let i = 0; i <= noSpace.length - 3; i++) queryTrigrams.add(noSpace.slice(i, i + 3));
    const hits = new Map<number, number>();
    for (const tri of queryTrigrams) {
      const posting = this._trigramIndex.get(tri);
      if (!posting) continue;
      for (const idx of posting) hits.set(idx, (hits.get(idx) ?? 0) + 1);
    }
    const sorted = [...hits.entries()].sort((a, b) => b[1] - a[1]);
    const candidates = new Set<number>();
    for (let i = 0; i < Math.min(sorted.length, maxCandidates); i++) candidates.add(sorted[i][0]);
    return candidates;
  }

  private static _smartScore(textNs: string, verseNs: string): number {
    const lr = textNs.length / verseNs.length;
    if (lr >= 0.7 && lr <= 1.3) return ratio(textNs, verseNs);
    if (lr < 0.7) {
      const frag = fragmentScore(textNs, verseNs);
      const r = ratio(textNs, verseNs);
      return Math.max(frag, 0.6 * frag + 0.4 * r);
    }
    return ratio(textNs, verseNs);
  }

  matchVerse(transcribedText: string): { surah: number; ayah: number; score: number; ayah_end?: number } | null {
    const normT = normalizeArabic(transcribedText);
    const normTns = normT.replace(/ /g, "");
    if (!normTns || normTns.length < 3) return null;

    const candidates = this._getCandidates(normT, 300);
    const scored: [number, number][] = [];

    for (const idx of candidates) {
      const v = this.verses[idx];
      let score = ArabicQuranDB._smartScore(normTns, v.text_norm_ns);
      score = Math.max(score, ratio(normT, v.text_norm));
      if (v.text_norm_no_bsm_ns) score = Math.max(score, ArabicQuranDB._smartScore(normTns, v.text_norm_no_bsm_ns));
      if (v.text_norm_no_bsm) score = Math.max(score, ratio(normT, v.text_norm_no_bsm));
      scored.push([idx, score]);
    }
    scored.sort((a, b) => {
      const diff = b[1] - a[1];
      if (Math.abs(diff) < 0.001) {
        const lenA = this.verses[a[0]].text_norm_ns.length;
        const lenB = this.verses[b[0]].text_norm_ns.length;
        return Math.abs(lenA - normTns.length) - Math.abs(lenB - normTns.length);
      }
      return diff;
    });

    // Pass 2: Multi-ayah spans
    const pass2Surahs = new Set<number>();
    for (let i = 0; i < Math.min(scored.length, 20); i++) pass2Surahs.add(this.verses[scored[i][0]].surah);

    let bestScore = scored.length > 0 ? scored[0][1] : 0;
    let bestIdx = scored.length > 0 ? scored[0][0] : -1;
    let bestSpanEnd: number | undefined;

    for (const s of pass2Surahs) {
      const verses = this._bySurah.get(s)!;
      for (let i = 0; i < verses.length; i++) {
        for (let span = 2; span <= 6; span++) {  // Support up to 6-verse spans for benchmark
          if (i + span > verses.length) break;
          const chunk = verses.slice(i, i + span);
          const firstText = chunk[0].text_norm_no_bsm ?? chunk[0].text_norm;
          const combined = [firstText, ...chunk.slice(1).map(c => c.text_norm)].join(" ");
          const combinedNs = combined.replace(/ /g, "");
          let score = ArabicQuranDB._smartScore(normTns, combinedNs);
          score = Math.max(score, ratio(normT, combined));
          if (score > bestScore + 0.001 ||
              (score > bestScore - 0.001 && bestSpanEnd === undefined &&
               Math.abs(combinedNs.length - normTns.length) < Math.abs(this.verses[bestIdx]?.text_norm_ns.length - normTns.length))) {
            bestScore = score;
            bestIdx = this.verses.indexOf(chunk[0]);
            bestSpanEnd = chunk[chunk.length - 1].ayah;
          }
        }
      }
    }

    if (bestIdx >= 0 && bestScore >= 0.2) {
      const result: { surah: number; ayah: number; score: number; ayah_end?: number } = {
        surah: this.verses[bestIdx].surah, ayah: this.verses[bestIdx].ayah, score: bestScore,
      };
      if (bestSpanEnd !== undefined && bestSpanEnd !== result.ayah) result.ayah_end = bestSpanEnd;
      return result;
    }
    return null;
  }
}

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

async function initModel(): Promise<void> {
  const modelPath = resolve(ROOT, "public/fastconformer_ar_ctc_q8.onnx");
  const vocabPath = resolve(ROOT, "public/vocab.json");
  console.log("Loading ONNX model...");
  const t0 = performance.now();
  session = await ort.InferenceSession.create(modelPath, { executionProviders: ["cpu"] });
  console.log(`  Model loaded in ${(performance.now() - t0).toFixed(0)}ms`);
  decoder = new CTCDecoder(JSON.parse(readFileSync(vocabPath, "utf-8")));
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
  const { text } = decoder.decode(logprobs, timeSteps, vocabSize);
  return text;
}

const FALLBACK_THRESHOLD = 0.85;
const BEAM_MARGIN = 0.05;

async function transcribeBeamAndMatch(
  audio: Float32Array,
  db: ArabicQuranDB,
): Promise<{ text: string; match: { surah: number; ayah: number; score: number; ayah_end?: number } | null; usedBeam: boolean }> {
  const { logprobs, timeSteps, vocabSize } = await runInference(audio);
  const { text: greedyText } = decoder.decode(logprobs, timeSteps, vocabSize);
  const greedyMatch = db.matchVerse(greedyText);
  const greedyScore = greedyMatch?.score ?? 0;

  if (greedyScore >= FALLBACK_THRESHOLD) {
    return { text: greedyText, match: greedyMatch, usedBeam: false };
  }

  const hypotheses = decoder.beamSearch(logprobs, timeSteps, vocabSize, {
    beamWidth: BEAM_WIDTH, topK: 20,
  });

  let bestText = greedyText;
  let bestMatch = greedyMatch;
  let bestScore = greedyScore;
  let usedBeam = false;

  for (const hyp of hypotheses) {
    if (hyp.text === greedyText) continue;
    const match = db.matchVerse(hyp.text);
    const matchScore = match?.score ?? 0;
    if (matchScore > bestScore + BEAM_MARGIN) {
      bestText = hyp.text;
      bestMatch = match;
      bestScore = matchScore;
      usedBeam = true;
    }
  }

  return { text: bestText, match: bestMatch, usedBeam };
}

// ─── Benchmark types ─────────────────────────────────────────────────────────

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

// ─── Main ────────────────────────────────────────────────────────────────────

async function main() {
  console.log("╔══════════════════════════════════════════════════════════════╗");
  console.log("║         Benchmark Test — Curated Test Corpus               ║");
  console.log("╚══════════════════════════════════════════════════════════════╝\n");

  const manifestPath = resolve(BENCHMARK_DIR, "manifest.json");
  if (!existsSync(manifestPath)) {
    console.error(`Manifest not found: ${manifestPath}`);
    process.exit(1);
  }

  const manifest: { samples: BenchmarkSample[] } = JSON.parse(readFileSync(manifestPath, "utf-8"));
  console.log(`Benchmark corpus: ${manifest.samples.length} samples`);

  await initModel();

  const quranData = JSON.parse(readFileSync(resolve(ROOT, "public/quran.json"), "utf-8"));
  const db = new ArabicQuranDB(quranData);
  console.log(`Quran DB: ${db.verses.length} verses`);
  console.log(`Mode: ${ENABLE_BEAM_SEARCH ? `beam search (width=${BEAM_WIDTH})` : "greedy"}`);
  console.log();

  // Group by category
  const categories = ["short", "medium", "long", "multi"];
  const results: {
    sample: BenchmarkSample;
    eval: EvalResult;
    transcript: string;
    match: { surah: number; ayah: number; score: number; ayah_end?: number } | null;
    usedBeam: boolean;
    timeMs: number;
  }[] = [];

  let beamUpgrades = 0;

  for (const cat of categories) {
    const samples = manifest.samples.filter(s => s.category === cat);
    if (samples.length === 0) continue;

    console.log(`\n── ${cat.toUpperCase()} (${samples.length} samples) ──`);

    for (const sample of samples) {
      const audioPath = resolve(BENCHMARK_DIR, sample.file);
      if (!existsSync(audioPath)) {
        console.log(`  SKIP ${sample.id} — file not found: ${sample.file}`);
        continue;
      }

      const t0 = performance.now();
      try {
        const audio = loadAudio(audioPath);
        let text: string;
        let match: { surah: number; ayah: number; score: number; ayah_end?: number } | null;
        let usedBeam = false;

        if (ENABLE_BEAM_SEARCH) {
          const result = await transcribeBeamAndMatch(audio, db);
          text = result.text;
          match = result.match;
          usedBeam = result.usedBeam;
          if (usedBeam) beamUpgrades++;
        } else {
          text = await transcribe(audio);
          match = db.matchVerse(text);
        }

        const timeMs = performance.now() - t0;

        // Evaluate: check if match covers any of the expected verses
        let evalResult: EvalResult = "no_match";
        if (match) {
          const matchStart = match.ayah;
          const matchEnd = match.ayah_end ?? match.ayah;

          // Check each expected verse
          let anyCorrect = false;
          let anyEquiv = false;
          let anyAdjacent = false;

          for (const expected of sample.expected_verses) {
            if (match.surah === expected.surah) {
              if (expected.ayah >= matchStart && expected.ayah <= matchEnd) {
                anyCorrect = true;
              } else if (db.areEquivalentByRef(expected.surah, expected.ayah, match.surah, matchStart)) {
                anyEquiv = true;
              } else if (Math.abs(expected.ayah - matchStart) <= 1 || Math.abs(expected.ayah - matchEnd) <= 1) {
                anyAdjacent = true;
              }
            }
          }

          // For single-verse: check first expected verse directly
          const firstExpected = sample.expected_verses[0];
          if (match.surah === firstExpected.surah && firstExpected.ayah >= matchStart && firstExpected.ayah <= matchEnd) {
            anyCorrect = true;
          }

          if (anyCorrect) evalResult = "correct";
          else if (anyEquiv) evalResult = "equiv";
          else if (anyAdjacent) evalResult = "adjacent";
          else evalResult = "wrong";
        }

        // For multi-verse samples: also check if the span covers ALL expected verses
        let spanCoverage = "";
        if (sample.expected_verses.length > 1 && match) {
          const matchStart = match.ayah;
          const matchEnd = match.ayah_end ?? match.ayah;
          const covered = sample.expected_verses.filter(
            ev => match!.surah === ev.surah && ev.ayah >= matchStart && ev.ayah <= matchEnd
          ).length;
          spanCoverage = ` [span: ${covered}/${sample.expected_verses.length}]`;
        }

        // Display
        const icon = evalResult === "correct" ? "✓" :
                     evalResult === "equiv" ? "~" :
                     evalResult === "adjacent" ? "±" : "✗";
        const matchStr = match ? `${match.surah}:${match.ayah}${match.ayah_end ? `-${match.ayah_end}` : ""}` : "null";
        const expectedStr = sample.expected_verses.length === 1
          ? `${sample.expected_verses[0].surah}:${sample.expected_verses[0].ayah}`
          : `${sample.surah}:${sample.ayah}-${sample.ayah_end}`;
        const beamTag = usedBeam ? " [beam]" : "";
        console.log(`  ${icon} ${sample.id.padEnd(25)} expected=${expectedStr.padEnd(10)} got=${matchStr.padEnd(10)} score=${(match?.score ?? 0).toFixed(3)} ${timeMs.toFixed(0)}ms${beamTag}${spanCoverage}`);

        if (evalResult !== "correct" && evalResult !== "equiv") {
          console.log(`    transcript: "${text.slice(0, 80)}${text.length > 80 ? "..." : ""}"`);
        }

        results.push({ sample, eval: evalResult, transcript: text, match, usedBeam, timeMs });
      } catch (e: any) {
        console.log(`  E ${sample.id.padEnd(25)} ERROR: ${e.message?.slice(0, 60)}`);
      }
    }
  }

  // ═══════════════════════ FINAL REPORT ═══════════════════════════════════════

  console.log(`\n${"═".repeat(70)}`);
  console.log("  BENCHMARK RESULTS");
  console.log(`${"═".repeat(70)}\n`);

  // Per-category summary
  for (const cat of categories) {
    const catResults = results.filter(r => r.sample.category === cat);
    if (catResults.length === 0) continue;

    const correct = catResults.filter(r => r.eval === "correct" || r.eval === "equiv").length;
    const lenient = catResults.filter(r => r.eval === "correct" || r.eval === "equiv" || r.eval === "adjacent").length;
    const total = catResults.length;
    const avgTime = catResults.reduce((s, r) => s + r.timeMs, 0) / total;
    const bar = "█".repeat(Math.round(correct / total * 20)) + "░".repeat(20 - Math.round(correct / total * 20));
    console.log(`  ${cat.padEnd(8)} ${bar} ${((correct/total)*100).toFixed(1)}% (${correct}/${total}) avg=${avgTime.toFixed(0)}ms`);
  }

  // Overall
  const total = results.length;
  const correct = results.filter(r => r.eval === "correct" || r.eval === "equiv").length;
  const lenient = results.filter(r => r.eval === "correct" || r.eval === "equiv" || r.eval === "adjacent").length;
  const avgTime = results.reduce((s, r) => s + r.timeMs, 0) / total;

  console.log(`${"─".repeat(70)}`);
  console.log(`  OVERALL: ${correct}/${total} (${((correct/total)*100).toFixed(1)}%) correct`);
  console.log(`  Lenient: ${lenient}/${total} (${((lenient/total)*100).toFixed(1)}%) including ±1 ayah`);
  console.log(`  Average time: ${avgTime.toFixed(0)}ms per sample`);
  if (ENABLE_BEAM_SEARCH) {
    console.log(`  Beam upgrades: ${beamUpgrades}`);
  }

  // By source
  console.log(`\n  By source:`);
  for (const source of ["everyayah", "retasy", "user"]) {
    const srcResults = results.filter(r => r.sample.source === source);
    if (srcResults.length === 0) continue;
    const ok = srcResults.filter(r => r.eval === "correct" || r.eval === "equiv").length;
    console.log(`    ${source.padEnd(12)} ${ok}/${srcResults.length} (${((ok/srcResults.length)*100).toFixed(1)}%)`);
  }

  // Failures detail
  const failures = results.filter(r => r.eval === "wrong" || r.eval === "no_match");
  if (failures.length > 0) {
    console.log(`\n  Failures (${failures.length}):`);
    for (const f of failures) {
      const expectedStr = f.sample.expected_verses.map(v => `${v.surah}:${v.ayah}`).join(", ");
      const matchStr = f.match ? `${f.match.surah}:${f.match.ayah}${f.match.ayah_end ? `-${f.match.ayah_end}` : ""}` : "null";
      console.log(`    ${f.sample.id}: expected=[${expectedStr}] got=${matchStr} (${(f.match?.score ?? 0).toFixed(3)})`);
      console.log(`      "${f.transcript.slice(0, 100)}"`);
    }
  }

  console.log(`\n${"═".repeat(70)}\n`);
}

main().catch(err => { console.error("Fatal:", err); process.exit(1); });
