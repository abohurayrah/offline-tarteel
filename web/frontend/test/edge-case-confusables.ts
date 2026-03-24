#!/usr/bin/env npx tsx
/**
 * Edge-Case Confusable Verses Test
 *
 * Tests the hardest disambiguation cases in the Quran recognition pipeline:
 *   1. Al-Ikhlas 112:1    "قل هو الله أحد"         (4 words, very common)
 *   2. Al-Falaq 113:1     "قل أعوذ برب الفلق"      (5 words)
 *   3. An-Nas 114:1       "قل أعوذ برب الناس"      (5 words — confusable with 113:1!)
 *   4. Al-Fatiha 1:1      "بسم الله الرحمن الرحيم"  (4 words — 113 duplicates)
 *   5. Al-Kawthar 108:1   "إنا أعطيناك الكوثر"     (3 words)
 *
 * For each verse, tests:
 *   A. Non-streaming transcription (greedy + beam search)
 *   B. Verse matching (QuranDB.matchVerse)
 *   C. Streaming tracker simulation (300ms chunks)
 *   D. Confusable pair disambiguation (113:1 vs 114:1)
 *
 * Usage:
 *   npx tsx test/edge-case-confusables.ts
 */
import { execSync } from "node:child_process";
import { readFileSync, existsSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";

import { computeMelSpectrogram } from "../src/worker/mel.ts";
import { CTCDecoder } from "../src/worker/ctc-decode.ts";
import { QuranTrie } from "../src/worker/quran-trie.ts";
import { stripUthmaniMarks } from "../src/worker/forced-alignment.ts";
import { QuranDB, normalizeArabic } from "../src/lib/quran-db.ts";
import { RecitationTracker } from "../src/lib/tracker.ts";
import type { TranscribeResult } from "../src/lib/tracker.ts";
import type { WorkerOutbound } from "../src/lib/types.ts";
import { SAMPLE_RATE } from "../src/lib/types.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const BENCHMARK_DIR = resolve(ROOT, "../../benchmark/test_corpus");
const EXPANDED_DIR = resolve(ROOT, "../../benchmark/test_corpus_expanded");
const RESULTS_DIR = resolve(__dirname, "benchmark-results");
const CHUNK_MS = 300;
const CHUNK_SAMPLES = Math.floor(SAMPLE_RATE * CHUNK_MS / 1000);

// ─── Verse definitions ───────────────────────────────────────────────────────

interface TestVerse {
  id: string;
  surah: number;
  ayah: number;
  text_ar: string;
  word_count: number;
  audioFiles: string[];       // primary + fallback audio files
  confusableWith?: { surah: number; ayah: number }[];
  notes: string;
}

const TEST_VERSES: TestVerse[] = [
  {
    id: "ikhlas_1",
    surah: 112, ayah: 1,
    text_ar: "قل هو الله أحد",
    word_count: 4,
    audioFiles: [
      resolve(BENCHMARK_DIR, "112001.mp3"),
      resolve(EXPANDED_DIR, "tarteel_131.wav"),
      resolve(EXPANDED_DIR, "tarteel_137.wav"),
    ],
    notes: "Very common short verse. Unique opening 'هو الله' should disambiguate quickly.",
  },
  {
    id: "falaq_1",
    surah: 113, ayah: 1,
    text_ar: "قل أعوذ برب الفلق",
    word_count: 5,
    audioFiles: [
      resolve(EXPANDED_DIR, "tarteel_139.wav"),
      resolve(EXPANDED_DIR, "tarteel_148.wav"),
    ],
    confusableWith: [{ surah: 114, ayah: 1 }],
    notes: "Shares 'قل أعوذ برب' with 114:1. Only 'الفلق' vs 'الناس' disambiguates.",
  },
  {
    id: "nas_1",
    surah: 114, ayah: 1,
    text_ar: "قل أعوذ برب الناس",
    word_count: 5,
    audioFiles: [
      resolve(EXPANDED_DIR, "tarteel_149.wav"),
      resolve(EXPANDED_DIR, "tarteel_155.wav"),
    ],
    confusableWith: [{ surah: 113, ayah: 1 }],
    notes: "Shares 'قل أعوذ برب' with 113:1. Critical confusable pair.",
  },
  {
    id: "fatiha_1",
    surah: 1, ayah: 1,
    text_ar: "بسم الله الرحمن الرحيم",
    word_count: 4,
    audioFiles: [
      resolve(BENCHMARK_DIR, "001001.mp3"),
      resolve(EXPANDED_DIR, "tarteel_000.wav"),
      resolve(EXPANDED_DIR, "tarteel_012.wav"),
    ],
    notes: "113 duplicates across Quran (bismillah). Disambiguation depends on context.",
  },
  {
    id: "kawthar_1",
    surah: 108, ayah: 1,
    text_ar: "إنا أعطيناك الكوثر",
    word_count: 3,
    audioFiles: [
      resolve(BENCHMARK_DIR, "108001.mp3"),
    ],
    notes: "Only 3 words. Unique vocabulary ('الكوثر') should make it identifiable.",
  },
];

// ─── Audio helpers ───────────────────────────────────────────────────────────

function loadAudio(filePath: string): Float32Array {
  const buf = execSync(
    `ffmpeg -hide_banner -loglevel error -i "${filePath}" -f f32le -ar ${SAMPLE_RATE} -ac 1 pipe:1`,
    { maxBuffer: 50 * 1024 * 1024 },
  );
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function audioDuration(audio: Float32Array): number {
  return audio.length / SAMPLE_RATE;
}

// ─── Model setup ─────────────────────────────────────────────────────────────

let session: ort.InferenceSession;
let decoder: CTCDecoder;
let trie: QuranTrie;
let db: QuranDB;

async function initModel(): Promise<void> {
  const modelPath = resolve(ROOT, "public/fastconformer_ar_ctc_q8.onnx");
  const vocabPath = resolve(ROOT, "public/vocab.json");
  const quranPath = resolve(ROOT, "public/quran.json");
  const disambigPath = resolve(ROOT, "public/ambiguity-compact.json");

  console.log("Loading ONNX model...");
  const t0 = performance.now();
  session = await ort.InferenceSession.create(modelPath, { executionProviders: ["cpu"] });
  console.log(`  Model loaded in ${(performance.now() - t0).toFixed(0)}ms`);

  const vocabJson = JSON.parse(readFileSync(vocabPath, "utf-8"));
  decoder = new CTCDecoder(vocabJson);

  const quranData = JSON.parse(readFileSync(quranPath, "utf-8"));
  db = new QuranDB(quranData);

  // Load disambiguation map
  if (existsSync(disambigPath)) {
    const disambigData = JSON.parse(readFileSync(disambigPath, "utf-8"));
    db.loadDisambiguationMap(disambigData);
    console.log("  Disambiguation map loaded.");
  }

  // Build trie
  trie = new QuranTrie(vocabJson);
  const trieVerses = quranData.map((v: any) => ({
    text_norm: stripUthmaniMarks(v.text_clean || v.text_uthmani),
    surah: v.surah,
    ayah: v.ayah,
  }));
  trie.buildFromVerses(trieVerses);
  console.log(`  Trie built: ${trie.nodeCount} nodes`);
}

async function runInference(audio: Float32Array): Promise<{
  logprobs: Float32Array;
  timeSteps: number;
  vocabSize: number;
}> {
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

// ─── Transcription functions ─────────────────────────────────────────────────

async function transcribeGreedy(audio: Float32Array): Promise<{ text: string; rawTokens: string }> {
  const { logprobs, timeSteps, vocabSize } = await runInference(audio);
  return decoder.decode(logprobs, timeSteps, vocabSize);
}

async function transcribeBeam(audio: Float32Array): Promise<{ text: string; rawTokens: string; score: number }[]> {
  const { logprobs, timeSteps, vocabSize } = await runInference(audio);
  return decoder.beamSearch(logprobs, timeSteps, vocabSize, { beamWidth: 10, topK: 20 });
}

async function transcribeConstrained(audio: Float32Array): Promise<{ text: string; rawTokens: string; score: number }[]> {
  const { logprobs, timeSteps, vocabSize } = await runInference(audio);
  return decoder.constrainedBeamSearch(logprobs, timeSteps, vocabSize, trie, { beamWidth: 10, topK: 20 });
}

// Wrapper for RecitationTracker (greedy decode, same as streaming-benchmark)
async function transcribeForTracker(audio: Float32Array): Promise<TranscribeResult> {
  const { logprobs, timeSteps, vocabSize } = await runInference(audio);
  const { text, rawTokens } = decoder.decode(logprobs, timeSteps, vocabSize);
  return { text, rawTokens };
}

// ─── Test framework ──────────────────────────────────────────────────────────

interface TranscriptionResult {
  method: string;
  text: string;
  rawTokens: string;
  score?: number;
  normalized: string;
  matchesExpected: boolean;
}

interface MatchResult {
  method: string;
  matchedSurah: number | null;
  matchedAyah: number | null;
  matchScore: number;
  isCorrect: boolean;
  runnersUp: { surah: number; ayah: number; score: number }[];
}

interface StreamResult {
  firstMatchSurah: number | null;
  firstMatchAyah: number | null;
  firstMatchTime: number;
  firstMatchConfidence: number;
  isCorrect: boolean;
  wordsCovered: number;
  totalWords: number;
  wordCoverage: number;
  verseJumps: number;
  totalMessages: number;
  messageTypes: Record<string, number>;
  audioDuration: number;
  processingTime: number;
}

interface VerseTestResult {
  verse: TestVerse;
  audioFile: string;
  audioDuration: number;
  transcriptions: TranscriptionResult[];
  matching: MatchResult[];
  streaming: StreamResult;
  disambiguationLength: number;
  isAmbiguous: boolean;
}

// ─── Section A: Non-streaming transcription ──────────────────────────────────

async function testTranscription(
  audio: Float32Array,
  verse: TestVerse,
): Promise<TranscriptionResult[]> {
  const results: TranscriptionResult[] = [];
  const expectedNorm = normalizeArabic(verse.text_ar);

  // Greedy
  const greedy = await transcribeGreedy(audio);
  const greedyNorm = normalizeArabic(greedy.text);
  results.push({
    method: "greedy",
    text: greedy.text,
    rawTokens: greedy.rawTokens,
    normalized: greedyNorm,
    matchesExpected: greedyNorm === expectedNorm,
  });

  // Beam search (unconstrained)
  const beamResults = await transcribeBeam(audio);
  if (beamResults.length > 0) {
    const bestBeam = beamResults[0];
    const beamNorm = normalizeArabic(bestBeam.text);
    results.push({
      method: "beam_search",
      text: bestBeam.text,
      rawTokens: bestBeam.rawTokens,
      score: bestBeam.score,
      normalized: beamNorm,
      matchesExpected: beamNorm === expectedNorm,
    });
  }

  // Constrained beam search (Quran trie)
  const constrainedResults = await transcribeConstrained(audio);
  if (constrainedResults.length > 0) {
    const bestConstrained = constrainedResults[0];
    const constrainedNorm = normalizeArabic(bestConstrained.text);
    results.push({
      method: "constrained_beam",
      text: bestConstrained.text,
      rawTokens: bestConstrained.rawTokens,
      score: bestConstrained.score,
      normalized: constrainedNorm,
      matchesExpected: constrainedNorm === expectedNorm,
    });
  }

  return results;
}

// ─── Section B: Verse matching ───────────────────────────────────────────────

async function testMatching(
  transcriptions: TranscriptionResult[],
  verse: TestVerse,
): Promise<MatchResult[]> {
  const results: MatchResult[] = [];

  for (const t of transcriptions) {
    const match = db.matchVerse(t.normalized, null) as any;
    const runnersUp = (match?.runners_up ?? [])
      .slice(0, 5)
      .map((r: any) => ({ surah: r.surah, ayah: r.ayah, score: r.score }));

    results.push({
      method: t.method,
      matchedSurah: match?.surah ?? null,
      matchedAyah: match?.ayah ?? null,
      matchScore: match?.score ?? 0,
      isCorrect: match?.surah === verse.surah && match?.ayah === verse.ayah,
      runnersUp,
    });
  }

  return results;
}

// ─── Section C: Streaming simulation ─────────────────────────────────────────

async function testStreaming(
  audio: Float32Array,
  verse: TestVerse,
): Promise<StreamResult> {
  const tracker = new RecitationTracker(db, transcribeForTracker);
  const allMessages: WorkerOutbound[] = [];
  const t0 = performance.now();

  let firstMatch: { surah: number; ayah: number; confidence: number; time: number } | null = null;
  let wordIndices = new Set<number>();
  let totalWords = 0;
  let jumps = 0;
  let lastSurah = -1;
  let lastAyah = -1;
  const msgCounts: Record<string, number> = {};

  for (let offset = 0; offset < audio.length; offset += CHUNK_SAMPLES) {
    const chunk = audio.slice(offset, Math.min(offset + CHUNK_SAMPLES, audio.length));
    const messages = await tracker.feed(chunk);

    for (const msg of messages) {
      allMessages.push(msg);
      msgCounts[msg.type] = (msgCounts[msg.type] ?? 0) + 1;

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

  const processingTime = (performance.now() - t0) / 1000;

  return {
    firstMatchSurah: firstMatch?.surah ?? null,
    firstMatchAyah: firstMatch?.ayah ?? null,
    firstMatchTime: firstMatch?.time ?? audioDuration(audio),
    firstMatchConfidence: firstMatch?.confidence ?? 0,
    isCorrect: firstMatch?.surah === verse.surah && firstMatch?.ayah === verse.ayah,
    wordsCovered: wordIndices.size,
    totalWords,
    wordCoverage: totalWords > 0 ? wordIndices.size / totalWords : 0,
    verseJumps: jumps,
    totalMessages: allMessages.length,
    messageTypes: msgCounts,
    audioDuration: audioDuration(audio),
    processingTime,
  };
}

// ─── Section D: Confusable pair analysis ─────────────────────────────────────

interface ConfusablePairResult {
  verse1: { surah: number; ayah: number; text: string };
  verse2: { surah: number; ayah: number; text: string };
  // Test 113:1 audio -> does it match 113 or 114?
  audio113_greedy: string;
  audio113_matchedCorrectly: boolean;
  audio113_confuserScore: number;
  // Test 114:1 audio -> does it match 114 or 113?
  audio114_greedy: string;
  audio114_matchedCorrectly: boolean;
  audio114_confuserScore: number;
  // Disambiguation analysis
  sharedPrefixWords: number;
  disambiguatingWord: string;
  disambigLength113: number;
  disambigLength114: number;
}

async function testConfusablePair(): Promise<ConfusablePairResult | null> {
  const falaq = TEST_VERSES.find(v => v.surah === 113)!;
  const nas = TEST_VERSES.find(v => v.surah === 114)!;

  // Find working audio for each
  const audio113File = falaq.audioFiles.find(f => existsSync(f));
  const audio114File = nas.audioFiles.find(f => existsSync(f));
  if (!audio113File || !audio114File) {
    console.log("  [SKIP] Missing audio for confusable pair test");
    return null;
  }

  const audio113 = loadAudio(audio113File);
  const audio114 = loadAudio(audio114File);

  // Greedy transcription
  const greedy113 = await transcribeGreedy(audio113);
  const greedy114 = await transcribeGreedy(audio114);

  // Match each
  const match113 = db.matchVerse(normalizeArabic(greedy113.text), null) as any;
  const match114 = db.matchVerse(normalizeArabic(greedy114.text), null) as any;

  // Find confuser score (how highly the wrong verse scores)
  const runners113 = (match113?.runners_up ?? []) as any[];
  const confuser113Score = runners113.find(
    (r: any) => r.surah === 114 && r.ayah === 1,
  )?.score ?? 0;

  const runners114 = (match114?.runners_up ?? []) as any[];
  const confuser114Score = runners114.find(
    (r: any) => r.surah === 113 && r.ayah === 1,
  )?.score ?? 0;

  // Disambiguation analysis
  const disambig113 = db.getDisambiguationLength(113, 1);
  const disambig114 = db.getDisambiguationLength(114, 1);

  // Calculate shared prefix
  const words113 = normalizeArabic(falaq.text_ar).split(" ");
  const words114 = normalizeArabic(nas.text_ar).split(" ");
  let shared = 0;
  for (let i = 0; i < Math.min(words113.length, words114.length); i++) {
    if (words113[i] === words114[i]) shared++;
    else break;
  }

  return {
    verse1: { surah: 113, ayah: 1, text: falaq.text_ar },
    verse2: { surah: 114, ayah: 1, text: nas.text_ar },
    audio113_greedy: greedy113.text,
    audio113_matchedCorrectly: match113?.surah === 113 && match113?.ayah === 1,
    audio113_confuserScore: confuser113Score,
    audio114_greedy: greedy114.text,
    audio114_matchedCorrectly: match114?.surah === 114 && match114?.ayah === 1,
    audio114_confuserScore: confuser114Score,
    sharedPrefixWords: shared,
    disambiguatingWord: shared < words113.length ? words113[shared] + " vs " + words114[shared] : "N/A",
    disambigLength113: disambig113,
    disambigLength114: disambig114,
  };
}

// ─── Main ────────────────────────────────────────────────────────────────────

function printSectionHeader(title: string) {
  console.log("\n" + "=".repeat(72));
  console.log(` ${title}`);
  console.log("=".repeat(72));
}

function printSubHeader(title: string) {
  console.log(`\n--- ${title} ${"─".repeat(Math.max(0, 60 - title.length))}`);
}

async function main() {
  console.log("Edge-Case Confusable Verses Test");
  console.log("================================\n");

  await initModel();

  const allResults: VerseTestResult[] = [];
  let totalTests = 0;
  let totalPass = 0;

  // ─── Per-verse tests ────────────────────────────────────────────────────
  for (const verse of TEST_VERSES) {
    printSectionHeader(`${verse.id}: ${verse.surah}:${verse.ayah} — "${verse.text_ar}"`);
    console.log(`  Notes: ${verse.notes}`);
    console.log(`  Word count: ${verse.word_count}`);

    // Disambiguation info
    const disambigLen = db.getDisambiguationLength(verse.surah, verse.ayah);
    const isAmbiguous = db.isAmbiguousInIsolation(verse.surah, verse.ayah);
    console.log(`  Disambiguation length: ${disambigLen === -1 ? "NEVER (ambiguous in isolation)" : disambigLen + " words"}`);
    if (isAmbiguous) {
      console.log("  ** WARNING: This verse cannot be uniquely identified from its opening words alone **");
    }

    // Find first available audio file
    const audioFile = verse.audioFiles.find(f => existsSync(f));
    if (!audioFile) {
      console.log("  [SKIP] No audio file found");
      continue;
    }

    const audio = loadAudio(audioFile);
    console.log(`  Audio: ${audioFile.split("/").pop()} (${audioDuration(audio).toFixed(2)}s)`);

    // A. Transcription tests
    printSubHeader("A. Transcription (greedy / beam / constrained)");
    const transcriptions = await testTranscription(audio, verse);

    for (const t of transcriptions) {
      const status = t.matchesExpected ? "PASS" : "FAIL";
      totalTests++;
      if (t.matchesExpected) totalPass++;
      console.log(`  [${status}] ${t.method.padEnd(18)} -> "${t.text}"`);
      console.log(`       normalized: "${t.normalized}"`);
      if (t.score !== undefined) {
        console.log(`       score: ${t.score.toFixed(4)}`);
      }
      if (!t.matchesExpected) {
        console.log(`       expected:   "${normalizeArabic(verse.text_ar)}"`);
      }
    }

    // B. Verse matching tests
    printSubHeader("B. Verse matching (from each transcription)");
    const matching = await testMatching(transcriptions, verse);

    for (const m of matching) {
      const status = m.isCorrect ? "PASS" : "FAIL";
      totalTests++;
      if (m.isCorrect) totalPass++;
      const matchStr = m.matchedSurah
        ? `${m.matchedSurah}:${m.matchedAyah} (score=${m.matchScore.toFixed(3)})`
        : "NO MATCH";
      console.log(`  [${status}] ${m.method.padEnd(18)} -> ${matchStr}`);
      if (m.runnersUp.length > 1) {
        console.log(`       runners-up: ${m.runnersUp.slice(1, 4).map(r => `${r.surah}:${r.ayah}(${r.score.toFixed(2)})`).join(", ")}`);
      }
    }

    // C. Streaming test
    printSubHeader("C. Streaming simulation (300ms chunks)");
    const streaming = await testStreaming(audio, verse);

    const streamStatus = streaming.isCorrect ? "PASS" : "FAIL";
    totalTests++;
    if (streaming.isCorrect) totalPass++;
    console.log(`  [${streamStatus}] First match: ${streaming.firstMatchSurah}:${streaming.firstMatchAyah}`);
    console.log(`       Time to match: ${streaming.firstMatchTime.toFixed(2)}s / ${streaming.audioDuration.toFixed(2)}s audio`);
    console.log(`       Confidence: ${streaming.firstMatchConfidence.toFixed(2)}`);
    console.log(`       Word coverage: ${streaming.wordsCovered}/${streaming.totalWords} (${(streaming.wordCoverage * 100).toFixed(0)}%)`);
    console.log(`       Verse jumps: ${streaming.verseJumps}`);
    console.log(`       Message types: ${JSON.stringify(streaming.messageTypes)}`);
    console.log(`       Processing time: ${streaming.processingTime.toFixed(2)}s (${(streaming.processingTime / streaming.audioDuration).toFixed(1)}x realtime)`);

    // Early commit analysis for streaming
    if (streaming.isCorrect && streaming.firstMatchTime < 2.0) {
      console.log("       ** FAST COMMIT: matched in under 2 seconds **");
    } else if (!streaming.isCorrect && streaming.firstMatchSurah !== null) {
      console.log(`       ** WRONG COMMIT: expected ${verse.surah}:${verse.ayah}, got ${streaming.firstMatchSurah}:${streaming.firstMatchAyah} **`);
    } else if (streaming.firstMatchSurah === null) {
      console.log("       ** NO MATCH: tracker never identified the verse **");
    }

    allResults.push({
      verse,
      audioFile,
      audioDuration: audioDuration(audio),
      transcriptions,
      matching,
      streaming,
      disambiguationLength: disambigLen,
      isAmbiguous,
    });
  }

  // ─── Section D: Confusable pair deep dive ───────────────────────────────
  printSectionHeader("D. Confusable Pair: Al-Falaq 113:1 vs An-Nas 114:1");
  const confusable = await testConfusablePair();

  if (confusable) {
    console.log(`\n  Shared prefix: ${confusable.sharedPrefixWords} words`);
    console.log(`  Disambiguating word: ${confusable.disambiguatingWord}`);
    console.log(`  Disambiguation length (113:1): ${confusable.disambigLength113 === -1 ? "never unique" : confusable.disambigLength113 + " words"}`);
    console.log(`  Disambiguation length (114:1): ${confusable.disambigLength114 === -1 ? "never unique" : confusable.disambigLength114 + " words"}`);

    console.log(`\n  113:1 audio (Al-Falaq):`);
    console.log(`    Greedy transcription: "${confusable.audio113_greedy}"`);
    console.log(`    Matched correctly: ${confusable.audio113_matchedCorrectly ? "YES" : "NO"}`);
    console.log(`    Confuser (114:1) score: ${confusable.audio113_confuserScore.toFixed(3)}`);

    totalTests++;
    if (confusable.audio113_matchedCorrectly) totalPass++;

    console.log(`\n  114:1 audio (An-Nas):`);
    console.log(`    Greedy transcription: "${confusable.audio114_greedy}"`);
    console.log(`    Matched correctly: ${confusable.audio114_matchedCorrectly ? "YES" : "NO"}`);
    console.log(`    Confuser (113:1) score: ${confusable.audio114_confuserScore.toFixed(3)}`);

    totalTests++;
    if (confusable.audio114_matchedCorrectly) totalPass++;

    const canDistinguish = confusable.audio113_matchedCorrectly && confusable.audio114_matchedCorrectly;
    console.log(`\n  *** System can distinguish 113:1 from 114:1: ${canDistinguish ? "YES" : "NO"} ***`);
  }

  // ─── Summary ────────────────────────────────────────────────────────────
  printSectionHeader("SUMMARY");
  console.log(`\n  Total tests: ${totalTests}`);
  console.log(`  Passed: ${totalPass} (${(totalPass / totalTests * 100).toFixed(0)}%)`);
  console.log(`  Failed: ${totalTests - totalPass}`);

  console.log("\n  Per-verse summary:");
  for (const r of allResults) {
    const txPass = r.transcriptions.filter(t => t.matchesExpected).length;
    const txTotal = r.transcriptions.length;
    const matchPass = r.matching.filter(m => m.isCorrect).length;
    const matchTotal = r.matching.length;
    const streamOk = r.streaming.isCorrect ? "OK" : "FAIL";

    console.log(
      `    ${r.verse.surah}:${r.verse.ayah} ${r.verse.id.padEnd(12)} ` +
      `tx=${txPass}/${txTotal} match=${matchPass}/${matchTotal} stream=${streamOk} ` +
      `words=${r.streaming.wordsCovered}/${r.streaming.totalWords} ` +
      `time=${r.streaming.firstMatchTime.toFixed(1)}s ` +
      `disambig=${r.disambiguationLength === -1 ? "ambig" : r.disambiguationLength + "w"}`,
    );
  }

  // Save results
  if (!existsSync(RESULTS_DIR)) mkdirSync(RESULTS_DIR, { recursive: true });
  const outPath = resolve(
    RESULTS_DIR,
    `edge-case-confusables-${new Date().toISOString().replace(/[:.]/g, "-")}.json`,
  );
  writeFileSync(outPath, JSON.stringify({ allResults, confusable, totalTests, totalPass }, null, 2));
  console.log(`\n  Results saved to: ${outPath}`);
}

main().catch(console.error);
