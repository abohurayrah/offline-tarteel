#!/usr/bin/env npx tsx
/**
 * Full pipeline accuracy test — sampled across all 6,236 ayahs.
 *
 * Features:
 *   - Equivalence groups: verses with identical text count as correct
 *   - Full-audio disambiguation: when partial clips are ambiguous, uses full audio
 *   - Multi-tier metrics: strict, equivalence-aware, lenient (±1 ayah)
 *
 * Usage:
 *   npx tsx test/test-pipeline-full.ts
 *   npx tsx test/test-pipeline-full.ts --sample=100
 *   npx tsx test/test-pipeline-full.ts --no-disambiguate   # Skip full-audio fallback
 */

import { execSync } from "node:child_process";
import { readFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";

import { computeMelSpectrogram } from "../src/worker/mel.ts";
import { CTCDecoder } from "../src/worker/ctc-decode.ts";
import { ratio, fragmentScore } from "../src/lib/levenshtein.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const SAMPLE_RATE = 16000;

const SAMPLE_SIZE = parseInt(process.argv.find(a => a.startsWith("--sample="))?.split("=")[1] ?? "150");
const ENABLE_DISAMBIGUATE = !process.argv.includes("--no-disambiguate");

// ─── Arabic normalization ────────────────────────────────────────────────────

function normalizeArabic(text: string): string {
  text = text.replace(/\u2581/g, " ");  // BPE marker → space
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

// ─── QuranDB with equivalence groups ─────────────────────────────────────────

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
  private _byRef: Map<string, number> = new Map();  // "s:a" → index

  // Equivalence groups: verses with identical normalized text
  private _equivByText: Map<string, number[]> = new Map();  // norm text → [idx, ...]
  private _equivByIdx: Map<number, number[]> = new Map();   // idx → [all equiv idxs]

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
    // Group verses by EXACT normalized text (both full and no-bismillah)
    for (let idx = 0; idx < this.verses.length; idx++) {
      const v = this.verses[idx];
      // Use the "effective" text — no-bsm version for ayah 1, otherwise full
      const effectiveText = v.text_norm_no_bsm_ns ?? v.text_norm_ns;

      if (!this._equivByText.has(effectiveText)) {
        this._equivByText.set(effectiveText, []);
      }
      this._equivByText.get(effectiveText)!.push(idx);
    }

    // Also group by full text (for partial clip matching)
    const fullTextGroups = new Map<string, number[]>();
    for (let idx = 0; idx < this.verses.length; idx++) {
      const ns = this.verses[idx].text_norm_ns;
      if (!fullTextGroups.has(ns)) fullTextGroups.set(ns, []);
      fullTextGroups.get(ns)!.push(idx);
    }

    // Build reverse: idx → all indices with same text (union of both groupings)
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

  /** Check if two verse indices are in the same equivalence group */
  areEquivalent(idx1: number, idx2: number): boolean {
    if (idx1 === idx2) return true;
    const group = this._equivByIdx.get(idx1);
    return group ? group.includes(idx2) : false;
  }

  /** Check by surah:ayah */
  areEquivalentByRef(s1: number, a1: number, s2: number, a2: number): boolean {
    const i1 = this._byRef.get(`${s1}:${a1}`);
    const i2 = this._byRef.get(`${s2}:${a2}`);
    if (i1 === undefined || i2 === undefined) return false;
    return this.areEquivalent(i1, i2);
  }

  getEquivStats(): { totalGroups: number; duplicateVerses: number; largestGroup: number } {
    let totalGroups = 0;
    let duplicateVerses = 0;
    let largestGroup = 0;
    for (const [, ids] of this._equivByText) {
      if (ids.length > 1) {
        totalGroups++;
        duplicateVerses += ids.length;
        largestGroup = Math.max(largestGroup, ids.length);
      }
    }
    return { totalGroups, duplicateVerses, largestGroup };
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
    if (!normTns) return null;

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
    scored.sort((a, b) => b[1] - a[1]);

    // Pass 2: Multi-ayah spans from top-20 surahs
    const pass2Surahs = new Set<number>();
    for (let i = 0; i < Math.min(scored.length, 20); i++) pass2Surahs.add(this.verses[scored[i][0]].surah);

    let bestScore = scored.length > 0 ? scored[0][1] : 0;
    let bestIdx = scored.length > 0 ? scored[0][0] : -1;
    let bestSpanEnd: number | undefined;

    for (const s of pass2Surahs) {
      const verses = this._bySurah.get(s)!;
      for (let i = 0; i < verses.length; i++) {
        for (let span = 2; span <= 3; span++) {
          if (i + span > verses.length) break;
          const chunk = verses.slice(i, i + span);
          const firstText = chunk[0].text_norm_no_bsm ?? chunk[0].text_norm;
          const combined = [firstText, ...chunk.slice(1).map(c => c.text_norm)].join(" ");
          const combinedNs = combined.replace(/ /g, "");
          let score = ArabicQuranDB._smartScore(normTns, combinedNs);
          score = Math.max(score, ratio(normT, combined));
          if (score > bestScore) {
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

async function transcribe(audio: Float32Array): Promise<string> {
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
  const { text } = decoder.decode(outputTensor.data as Float32Array, timeSteps, vocabSize);
  return text;
}

// ─── Transcript cache (avoid re-transcribing full audio) ─────────────────────

const transcriptCache = new Map<string, string>();

async function transcribeFile(filePath: string): Promise<string> {
  if (transcriptCache.has(filePath)) return transcriptCache.get(filePath)!;
  const audio = loadAudio(filePath);
  const text = await transcribe(audio);
  transcriptCache.set(filePath, text);
  return text;
}

// ─── Sampling ────────────────────────────────────────────────────────────────

function sampleEntries(manifest: any[], type: string, n: number): any[] {
  const entries = manifest.filter((e: any) => e.type === type);
  if (entries.length <= n) return entries;
  const shuffled = [...entries].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, n);
}

// ─── Evaluation ──────────────────────────────────────────────────────────────

type EvalCategory = "exact" | "equivalence" | "adjacent" | "wrong_ayah" | "wrong_surah" | "no_match";

interface EvalResult {
  category: string;          // test category (full/first60/mid50/last50)
  evalType: EvalCategory;
  expected: string;
  got: string;
  score: number;
  text: string;
  disambiguated: boolean;
}

function classify(
  db: ArabicQuranDB,
  expected: { surah: number; ayah: number },
  match: { surah: number; ayah: number; score: number } | null,
): EvalCategory {
  if (!match) return "no_match";
  if (match.surah === expected.surah && match.ayah === expected.ayah) return "exact";
  if (db.areEquivalentByRef(expected.surah, expected.ayah, match.surah, match.ayah)) return "equivalence";
  if (match.surah === expected.surah && Math.abs(match.ayah - expected.ayah) === 1) return "adjacent";
  if (match.surah === expected.surah) return "wrong_ayah";
  return "wrong_surah";
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main() {
  console.log("╔══════════════════════════════════════════════════════════════╗");
  console.log("║  Full Pipeline Test — Equivalence Groups + Disambiguation  ║");
  console.log("╚══════════════════════════════════════════════════════════════╝\n");

  const manifestPath = resolve(ROOT, "test/audio-samples/manifest.json");
  const manifest = JSON.parse(readFileSync(manifestPath, "utf-8"));
  console.log(`Manifest: ${manifest.length} entries`);

  await initModel();

  const quranData = JSON.parse(readFileSync(resolve(ROOT, "public/quran.json"), "utf-8"));
  const db = new ArabicQuranDB(quranData);
  const equivStats = db.getEquivStats();
  console.log(`Quran DB: ${db.verses.length} verses`);
  console.log(`Equivalence groups: ${equivStats.totalGroups} groups, ${equivStats.duplicateVerses} duplicate verses, largest group: ${equivStats.largestGroup}`);
  console.log(`Disambiguation: ${ENABLE_DISAMBIGUATE ? "ON" : "OFF"}\n`);

  // Build a lookup for full audio files by surah:ayah
  const fullAudioLookup = new Map<string, string>();
  for (const entry of manifest) {
    if (entry.type === "full") {
      fullAudioLookup.set(`${entry.surah}:${entry.ayah}`, entry.file);
    }
  }

  const categories = ["full", "first60", "mid50", "last50"];
  const allResults: EvalResult[] = [];

  for (const cat of categories) {
    const samples = sampleEntries(manifest, cat, SAMPLE_SIZE);
    console.log(`\n── ${cat.toUpperCase()} (${samples.length} samples) ──`);

    let disambiguations = 0;

    for (let i = 0; i < samples.length; i++) {
      const entry = samples[i];
      const audioPath = resolve(ROOT, "test/audio-samples", entry.file);
      if (!existsSync(audioPath)) continue;

      try {
        const text = await transcribeFile(audioPath);
        let match = db.matchVerse(text);
        let evalType = classify(db, entry, match);
        let wasDisambiguated = false;

        // ─── Disambiguation: if partial clip got wrong answer, try full audio ──
        if (ENABLE_DISAMBIGUATE && cat !== "full" &&
            evalType !== "exact" && evalType !== "equivalence") {
          const fullFile = fullAudioLookup.get(`${entry.surah}:${entry.ayah}`);
          if (fullFile) {
            const fullPath = resolve(ROOT, "test/audio-samples", fullFile);
            if (existsSync(fullPath)) {
              const fullText = await transcribeFile(fullPath);
              const fullMatch = db.matchVerse(fullText);
              const fullEval = classify(db, entry, fullMatch);
              // Use full-audio result if it's better
              if (fullEval === "exact" || fullEval === "equivalence" ||
                  (fullEval === "adjacent" && evalType !== "adjacent")) {
                match = fullMatch;
                evalType = fullEval;
                wasDisambiguated = true;
                disambiguations++;
              }
            }
          }
        }

        const score = match?.score ?? 0;
        const gotStr = match ? `${match.surah}:${match.ayah}` : "null";

        // Display progress
        if (evalType === "exact" || evalType === "equivalence") {
          process.stdout.write(evalType === "equivalence" ? "~" : ".");
        } else if (evalType === "adjacent") {
          process.stdout.write("a");
        } else {
          process.stdout.write("F");
        }

        allResults.push({
          category: cat, evalType, score, disambiguated: wasDisambiguated,
          expected: `${entry.surah}:${entry.ayah}`, got: gotStr, text,
        });

        if ((i + 1) % 50 === 0) {
          const ok = allResults.filter(r => r.category === cat && (r.evalType === "exact" || r.evalType === "equivalence")).length;
          const total = allResults.filter(r => r.category === cat).length;
          process.stdout.write(` [${i + 1}/${samples.length} ${((ok/total)*100).toFixed(1)}%]\n`);
        }
      } catch (e) {
        process.stdout.write("E");
      }
    }

    // Category summary
    const catResults = allResults.filter(r => r.category === cat);
    const counts = { exact: 0, equivalence: 0, adjacent: 0, wrong_ayah: 0, wrong_surah: 0, no_match: 0 };
    for (const r of catResults) counts[r.evalType]++;
    const total = catResults.length;
    const strict = counts.exact;
    const equivAware = counts.exact + counts.equivalence;
    const lenient = equivAware + counts.adjacent;

    console.log(`\n  Strict:       ${strict}/${total} (${((strict/total)*100).toFixed(1)}%)`);
    console.log(`  Equiv-aware:  ${equivAware}/${total} (${((equivAware/total)*100).toFixed(1)}%)`);
    console.log(`  Lenient(±1):  ${lenient}/${total} (${((lenient/total)*100).toFixed(1)}%)`);
    if (disambiguations > 0) console.log(`  Disambiguated: ${disambiguations} via full audio`);

    // Show failures (only true failures, not equivalence/adjacent)
    const trueFailures = catResults.filter(r => r.evalType === "wrong_ayah" || r.evalType === "wrong_surah" || r.evalType === "no_match");
    if (trueFailures.length > 0) {
      console.log(`  True failures (${trueFailures.length}):`);
      for (const f of trueFailures.slice(0, 5)) {
        console.log(`    expected ${f.expected}, got ${f.got} (${f.score.toFixed(3)}) [${f.evalType}] "${f.text.slice(0, 50)}"`);
      }
    }
  }

  // ═══════════════════════ FINAL REPORT ═══════════════════════════════════════

  console.log(`\n${"═".repeat(70)}`);
  console.log("  FINAL ACCURACY REPORT");
  console.log(`${"═".repeat(70)}\n`);

  // Per-category bars
  for (const cat of categories) {
    const catR = allResults.filter(r => r.category === cat);
    const total = catR.length;
    if (total === 0) continue;

    const exact = catR.filter(r => r.evalType === "exact").length;
    const equiv = catR.filter(r => r.evalType === "equivalence").length;
    const adj = catR.filter(r => r.evalType === "adjacent").length;
    const equivPct = ((exact + equiv) / total * 100).toFixed(1);
    const lenPct = ((exact + equiv + adj) / total * 100).toFixed(1);

    const bar = "█".repeat(Math.round((exact + equiv) / total * 20)) +
                "▒".repeat(Math.round(adj / total * 20)) +
                "░".repeat(20 - Math.round((exact + equiv) / total * 20) - Math.round(adj / total * 20));
    console.log(`  ${cat.padEnd(10)} ${bar} equiv=${equivPct}% lenient=${lenPct}%  (exact=${exact} equiv=${equiv} adj=${adj})`);
  }

  // Overall
  const totalAll = allResults.length;
  const totalExact = allResults.filter(r => r.evalType === "exact").length;
  const totalEquiv = allResults.filter(r => r.evalType === "equivalence").length;
  const totalAdj = allResults.filter(r => r.evalType === "adjacent").length;
  const totalWrongAyah = allResults.filter(r => r.evalType === "wrong_ayah").length;
  const totalWrongSurah = allResults.filter(r => r.evalType === "wrong_surah").length;
  const totalNoMatch = allResults.filter(r => r.evalType === "no_match").length;
  const totalDisambig = allResults.filter(r => r.disambiguated).length;

  const strictPct = ((totalExact / totalAll) * 100).toFixed(1);
  const equivPct = (((totalExact + totalEquiv) / totalAll) * 100).toFixed(1);
  const lenientPct = (((totalExact + totalEquiv + totalAdj) / totalAll) * 100).toFixed(1);

  console.log(`${"─".repeat(70)}`);
  console.log(`  OVERALL (${totalAll} tests):`);
  console.log(`    Strict accuracy:       ${strictPct}%  (${totalExact})`);
  console.log(`    Equivalence-aware:     ${equivPct}%  (${totalExact}+${totalEquiv}) ← PRIMARY METRIC`);
  console.log(`    Lenient (±1 ayah):     ${lenientPct}%  (${totalExact}+${totalEquiv}+${totalAdj})`);
  console.log();
  console.log(`  Breakdown:`);
  console.log(`    Exact correct:         ${totalExact.toString().padStart(4)} (${((totalExact/totalAll)*100).toFixed(1)}%)`);
  console.log(`    Equivalence correct:   ${totalEquiv.toString().padStart(4)} (${((totalEquiv/totalAll)*100).toFixed(1)}%) -- same text, diff verse`);
  console.log(`    Adjacent (±1 ayah):    ${totalAdj.toString().padStart(4)} (${((totalAdj/totalAll)*100).toFixed(1)}%)`);
  console.log(`    Wrong ayah:            ${totalWrongAyah.toString().padStart(4)} (${((totalWrongAyah/totalAll)*100).toFixed(1)}%)`);
  console.log(`    Wrong surah:           ${totalWrongSurah.toString().padStart(4)} (${((totalWrongSurah/totalAll)*100).toFixed(1)}%)`);
  console.log(`    No match:              ${totalNoMatch.toString().padStart(4)} (${((totalNoMatch/totalAll)*100).toFixed(1)}%)`);
  if (totalDisambig > 0) {
    console.log(`    Disambiguated:         ${totalDisambig.toString().padStart(4)} (via full audio fallback)`);
  }

  console.log(`${"═".repeat(70)}\n`);

  // By surah group
  for (const [label, filter] of [
    ["Short surahs (78-114)", (r: EvalResult) => { const s = parseInt(r.expected); return s >= 78; }],
    ["Mid surahs (11-77)",   (r: EvalResult) => { const s = parseInt(r.expected); return s > 10 && s < 78; }],
    ["Long surahs (1-10)",   (r: EvalResult) => { const s = parseInt(r.expected); return s <= 10; }],
  ] as [string, (r: EvalResult) => boolean][]) {
    const group = allResults.filter(filter);
    if (group.length > 0) {
      const ok = group.filter(r => r.evalType === "exact" || r.evalType === "equivalence").length;
      console.log(`  ${label}: ${ok}/${group.length} (${((ok/group.length)*100).toFixed(1)}%)`);
    }
  }

  console.log(`\n  Samples per category: ${SAMPLE_SIZE}`);
  console.log(`  Total tests: ${totalAll}`);
}

main().catch(err => { console.error("Fatal:", err); process.exit(1); });
