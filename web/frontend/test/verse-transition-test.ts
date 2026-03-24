#!/usr/bin/env npx tsx
/**
 * Verse-to-Verse Transition Test
 *
 * Tests the streaming pipeline's ability to track sequential recitation
 * through entire surahs by concatenating individual verse audio files
 * with silence gaps and feeding them through RecitationTracker.
 *
 * Test cases:
 *   1. Al-Fatiha 1:1-1:7   (7 verses, foundational surah)
 *   2. Al-Ikhlas 112:1-4   (4 short verses)
 *   3. An-Nas 114:1-6      (6 verses, should not confuse with Al-Falaq)
 *   4. Mid-surah: 2:255    (Ayat al-Kursi, single verse mid-surah start)
 *
 * Usage:
 *   npx tsx test/verse-transition-test.ts
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
const CORPUS_DIR = resolve(ROOT, "../../benchmark/test_corpus");
const EXPANDED_DIR = resolve(ROOT, "../../benchmark/test_corpus_expanded");

const CHUNK_MS = 300;
const CHUNK_SAMPLES = Math.floor(SAMPLE_RATE * CHUNK_MS / 1000);
const SILENCE_GAP_MS = 500;
const SILENCE_GAP_SAMPLES = Math.floor(SAMPLE_RATE * SILENCE_GAP_MS / 1000);

// ─── Setup ──────────────────────────────────────────────────────────────────

let session: ort.InferenceSession;
let decoder: CTCDecoder;
let db: QuranDB;

async function init() {
  console.log("Loading model...");
  session = await ort.InferenceSession.create(
    resolve(ROOT, "public/fastconformer_ar_ctc_q8.onnx"),
    { executionProviders: ["cpu"] },
  );
  decoder = new CTCDecoder(JSON.parse(readFileSync(resolve(ROOT, "public/vocab.json"), "utf-8")));
  db = new QuranDB(JSON.parse(readFileSync(resolve(ROOT, "public/quran.json"), "utf-8")));

  const disambigPath = resolve(ROOT, "public/ambiguity-compact.json");
  if (existsSync(disambigPath)) {
    const disambigData = JSON.parse(readFileSync(disambigPath, "utf-8"));
    db.loadDisambiguationMap(disambigData);
    console.log("Disambiguation map loaded.");
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

// ─── Audio Concatenation ────────────────────────────────────────────────────

function concatFloat32(...arrays: Float32Array[]): Float32Array {
  const totalLength = arrays.reduce((sum, a) => sum + a.length, 0);
  const result = new Float32Array(totalLength);
  let offset = 0;
  for (const a of arrays) {
    result.set(a, offset);
    offset += a.length;
  }
  return result;
}

function createSilence(samples: number): Float32Array {
  return new Float32Array(samples);
}

function concatenateVerseAudio(audioFiles: Float32Array[]): Float32Array {
  const parts: Float32Array[] = [];
  for (let i = 0; i < audioFiles.length; i++) {
    parts.push(audioFiles[i]);
    if (i < audioFiles.length - 1) {
      parts.push(createSilence(SILENCE_GAP_SAMPLES));
    }
  }
  return concatFloat32(...parts);
}

// ─── Test Infrastructure ────────────────────────────────────────────────────

interface VerseEvent {
  type: "verse_match" | "word_progress" | "raw_transcript" | "candidate_list" | string;
  surah?: number;
  ayah?: number;
  confidence?: number;
  word_index?: number;
  total_words?: number;
  matched_indices?: number[];
  time_offset_s: number;   // time in the concatenated audio stream
  text?: string;
}

interface TransitionResult {
  verse_ref: string;        // "surah:ayah"
  discovered: boolean;
  discovery_time_s: number; // time from start of this verse's audio to verse_match
  word_coverage: number;    // fraction of words tracked
  words_matched: number;
  total_words: number;
  false_matches: string[];  // wrong verse_match events
}

interface TestCaseResult {
  name: string;
  expected_verses: Array<{ surah: number; ayah: number }>;
  transitions: TransitionResult[];
  all_events: VerseEvent[];
  total_audio_duration_s: number;
  total_processing_time_s: number;
  all_verses_found: boolean;
  all_verses_in_order: boolean;
  wrong_surah_matches: string[];
}

async function runTransitionTest(
  testName: string,
  verseSpecs: Array<{ surah: number; ayah: number; file: string }>,
): Promise<TestCaseResult> {
  console.log(`\n${"=".repeat(70)}`);
  console.log(`  TEST: ${testName}`);
  console.log(`  Verses: ${verseSpecs.map(v => `${v.surah}:${v.ayah}`).join(" -> ")}`);
  console.log(`${"=".repeat(70)}`);

  // Load and concatenate audio
  const verseAudios: Float32Array[] = [];
  const verseBoundaries: Array<{ surah: number; ayah: number; start_s: number; end_s: number }> = [];
  let cumulativeSamples = 0;

  for (let i = 0; i < verseSpecs.length; i++) {
    const spec = verseSpecs[i];
    if (!existsSync(spec.file)) {
      console.error(`  MISSING: ${spec.file}`);
      continue;
    }
    const audio = loadAudio(spec.file);
    const startS = cumulativeSamples / SAMPLE_RATE;
    verseAudios.push(audio);
    cumulativeSamples += audio.length;
    const endS = cumulativeSamples / SAMPLE_RATE;

    verseBoundaries.push({ surah: spec.surah, ayah: spec.ayah, start_s: startS, end_s: endS });

    console.log(`  Loaded ${spec.surah}:${spec.ayah} (${(audio.length / SAMPLE_RATE).toFixed(1)}s) [${startS.toFixed(1)}s - ${endS.toFixed(1)}s]`);

    // Add silence gap (counted in cumulative)
    if (i < verseSpecs.length - 1) {
      cumulativeSamples += SILENCE_GAP_SAMPLES;
    }
  }

  const concatenated = concatenateVerseAudio(verseAudios);
  const totalDuration = concatenated.length / SAMPLE_RATE;
  console.log(`  Total concatenated audio: ${totalDuration.toFixed(1)}s\n`);

  // Feed through streaming pipeline
  const tracker = new RecitationTracker(db, transcribe);
  const allEvents: VerseEvent[] = [];
  const t0 = performance.now();

  for (let offset = 0; offset < concatenated.length; offset += CHUNK_SAMPLES) {
    const chunk = concatenated.slice(offset, Math.min(offset + CHUNK_SAMPLES, concatenated.length));
    const messages = await tracker.feed(chunk);
    const timeS = (offset + chunk.length) / SAMPLE_RATE;

    for (const msg of messages) {
      const event: VerseEvent = {
        type: msg.type,
        time_offset_s: timeS,
      };

      if (msg.type === "verse_match") {
        event.surah = msg.surah;
        event.ayah = msg.ayah;
        event.confidence = msg.confidence;
        console.log(`  [${timeS.toFixed(1)}s] VERSE_MATCH: ${msg.surah}:${msg.ayah} (conf=${msg.confidence.toFixed(2)})`);
      } else if (msg.type === "word_progress") {
        event.surah = msg.surah;
        event.ayah = msg.ayah;
        event.word_index = msg.word_index;
        event.total_words = msg.total_words;
        event.matched_indices = msg.matched_indices;
        // Only print every few word progress events to keep output manageable
        if (msg.word_index === 1 || msg.word_index === msg.total_words || msg.word_index % 3 === 0) {
          console.log(`  [${timeS.toFixed(1)}s] WORD_PROGRESS: ${msg.surah}:${msg.ayah} word ${msg.word_index}/${msg.total_words} (${msg.matched_indices.length} matched)`);
        }
      } else if (msg.type === "raw_transcript") {
        event.text = msg.text;
        event.confidence = msg.confidence;
        console.log(`  [${timeS.toFixed(1)}s] RAW_TRANSCRIPT: "${msg.text.slice(0, 50)}..." (conf=${msg.confidence.toFixed(2)})`);
      } else if (msg.type === "candidate_list") {
        // Summarize candidates
        const top3 = msg.candidates.slice(0, 3).map(c => `${c.surah}:${c.ayah}(${c.score.toFixed(2)})`).join(", ");
        console.log(`  [${timeS.toFixed(1)}s] CANDIDATES: ${top3}`);
      }

      allEvents.push(event);
    }
  }

  const processingTime = (performance.now() - t0) / 1000;

  // Analyze results per expected verse
  const transitions: TransitionResult[] = [];
  const verseMatchEvents = allEvents.filter(e => e.type === "verse_match");
  const wordProgressEvents = allEvents.filter(e => e.type === "word_progress");

  // Track which verse_match events correspond to expected verses
  const matchedEventIndices = new Set<number>();
  const wrongSurahMatches: string[] = [];

  for (const expected of verseSpecs) {
    const boundary = verseBoundaries.find(b => b.surah === expected.surah && b.ayah === expected.ayah);
    if (!boundary) {
      transitions.push({
        verse_ref: `${expected.surah}:${expected.ayah}`,
        discovered: false,
        discovery_time_s: -1,
        word_coverage: 0,
        words_matched: 0,
        total_words: 0,
        false_matches: [],
      });
      continue;
    }

    // Find first verse_match for this verse
    const matchEvent = verseMatchEvents.find(
      (e, idx) => e.surah === expected.surah && e.ayah === expected.ayah && !matchedEventIndices.has(idx)
    );
    const matchIdx = matchEvent ? verseMatchEvents.indexOf(matchEvent) : -1;
    if (matchIdx >= 0) matchedEventIndices.add(matchIdx);

    // Find word progress for this verse
    const verseWordEvents = wordProgressEvents.filter(
      e => e.surah === expected.surah && e.ayah === expected.ayah
    );
    const maxWordIndex = verseWordEvents.length > 0
      ? Math.max(...verseWordEvents.map(e => e.word_index ?? 0))
      : 0;
    const totalWords = verseWordEvents.length > 0
      ? verseWordEvents[verseWordEvents.length - 1].total_words ?? 0
      : 0;
    const allMatchedIndices = new Set<number>();
    for (const e of verseWordEvents) {
      if (e.matched_indices) {
        for (const idx of e.matched_indices) allMatchedIndices.add(idx);
      }
    }

    // Find false matches (events during this verse's time window that match wrong verse)
    const falseMatches: string[] = [];
    for (const e of verseMatchEvents) {
      if (e.time_offset_s >= boundary.start_s && e.time_offset_s <= boundary.end_s + 1.0) {
        if (e.surah !== expected.surah || e.ayah !== expected.ayah) {
          falseMatches.push(`${e.surah}:${e.ayah}`);
        }
      }
    }

    const discoveryTime = matchEvent
      ? matchEvent.time_offset_s - boundary.start_s
      : -1;

    transitions.push({
      verse_ref: `${expected.surah}:${expected.ayah}`,
      discovered: !!matchEvent,
      discovery_time_s: discoveryTime,
      word_coverage: totalWords > 0 ? allMatchedIndices.size / totalWords : 0,
      words_matched: allMatchedIndices.size,
      total_words: totalWords,
      false_matches: falseMatches,
    });
  }

  // Check for wrong surah matches (for An-Nas vs Al-Falaq confusion)
  for (const e of verseMatchEvents) {
    const isExpected = verseSpecs.some(v => v.surah === e.surah && v.ayah === e.ayah);
    // Also accept next verse (tracker may auto-advance to next verse)
    const isNext = verseSpecs.some(v => v.surah === e.surah && e.ayah === v.ayah + 1);
    if (!isExpected && !isNext) {
      wrongSurahMatches.push(`${e.surah}:${e.ayah}@${e.time_offset_s?.toFixed(1)}s`);
    }
  }

  // Check ordering
  const matchOrder = verseMatchEvents
    .filter(e => verseSpecs.some(v => v.surah === e.surah && v.ayah === e.ayah))
    .map(e => `${e.surah}:${e.ayah}`);
  const expectedOrder = verseSpecs.map(v => `${v.surah}:${v.ayah}`);
  // Check if matchOrder is a subsequence of expectedOrder
  let orderIdx = 0;
  let allInOrder = true;
  for (const matched of matchOrder) {
    const foundAt = expectedOrder.indexOf(matched, orderIdx);
    if (foundAt < 0) {
      allInOrder = false;
      break;
    }
    orderIdx = foundAt + 1;
  }

  const allFound = transitions.every(t => t.discovered);

  const result: TestCaseResult = {
    name: testName,
    expected_verses: verseSpecs.map(v => ({ surah: v.surah, ayah: v.ayah })),
    transitions,
    all_events: allEvents,
    total_audio_duration_s: totalDuration,
    total_processing_time_s: processingTime,
    all_verses_found: allFound,
    all_verses_in_order: allInOrder,
    wrong_surah_matches: wrongSurahMatches,
  };

  // Print summary
  console.log(`\n  ${"─".repeat(60)}`);
  console.log(`  SUMMARY: ${testName}`);
  console.log(`  ${"─".repeat(60)}`);
  const foundCount = transitions.filter(t => t.discovered).length;
  console.log(`  Verses discovered: ${foundCount}/${transitions.length}`);
  console.log(`  All in order:      ${allInOrder ? "YES" : "NO"}`);
  console.log(`  Processing time:   ${processingTime.toFixed(1)}s (audio: ${totalDuration.toFixed(1)}s, RTF: ${(processingTime / totalDuration).toFixed(2)}x)`);

  if (wrongSurahMatches.length > 0) {
    console.log(`  WRONG SURAH MATCHES: ${wrongSurahMatches.join(", ")}`);
  }

  console.log();
  for (const t of transitions) {
    const icon = t.discovered ? "OK" : "MISS";
    const timeStr = t.discovery_time_s >= 0 ? `${t.discovery_time_s.toFixed(1)}s` : "N/A";
    const coverageStr = t.total_words > 0
      ? `${t.words_matched}/${t.total_words} (${(t.word_coverage * 100).toFixed(0)}%)`
      : "N/A";
    const falseStr = t.false_matches.length > 0 ? ` FALSE=[${t.false_matches.join(",")}]` : "";
    console.log(`    [${icon}] ${t.verse_ref.padEnd(8)} discovery=${timeStr.padEnd(6)} words=${coverageStr}${falseStr}`);
  }

  return result;
}

// ─── Test Case Definitions ──────────────────────────────────────────────────

function getExpandedFile(surah: number, ayah: number): string | null {
  // Load the expanded manifest to find the FIRST file for this verse
  const manifest = JSON.parse(readFileSync(resolve(EXPANDED_DIR, "manifest.json"), "utf-8"));
  for (const s of manifest.samples) {
    if (s.surah === surah && s.ayah === ayah) {
      return resolve(EXPANDED_DIR, s.file);
    }
  }
  return null;
}

function getCorpusFile(surah: number, ayah: number): string | null {
  // Try everyayah naming convention
  const padded = String(surah).padStart(3, "0") + String(ayah).padStart(3, "0");
  const mp3 = resolve(CORPUS_DIR, `${padded}.mp3`);
  if (existsSync(mp3)) return mp3;
  const wav = resolve(CORPUS_DIR, `${padded}.wav`);
  if (existsSync(wav)) return wav;
  return null;
}

function findVerseFile(surah: number, ayah: number): string {
  // Try corpus first, then expanded
  const corpus = getCorpusFile(surah, ayah);
  if (corpus && existsSync(corpus)) return corpus;
  const expanded = getExpandedFile(surah, ayah);
  if (expanded && existsSync(expanded)) return expanded;
  throw new Error(`No audio file found for ${surah}:${ayah}`);
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main() {
  console.log(`${"#".repeat(70)}`);
  console.log(`#  Verse-to-Verse Transition Test`);
  console.log(`#  Chunk size: ${CHUNK_MS}ms, Silence gap: ${SILENCE_GAP_MS}ms`);
  console.log(`${"#".repeat(70)}\n`);

  await init();

  const results: TestCaseResult[] = [];

  // ── Test 1: Al-Fatiha 1:1 -> 1:7 ──
  {
    const verses = [];
    for (let ayah = 1; ayah <= 7; ayah++) {
      verses.push({ surah: 1, ayah, file: findVerseFile(1, ayah) });
    }
    results.push(await runTransitionTest("Al-Fatiha (1:1 -> 1:7)", verses));
  }

  // ── Test 2: Al-Ikhlas 112:1 -> 112:4 ──
  {
    const verses = [];
    for (let ayah = 1; ayah <= 4; ayah++) {
      verses.push({ surah: 112, ayah, file: findVerseFile(112, ayah) });
    }
    results.push(await runTransitionTest("Al-Ikhlas (112:1 -> 112:4)", verses));
  }

  // ── Test 3: An-Nas 114:1 -> 114:6 ──
  {
    const verses = [];
    for (let ayah = 1; ayah <= 6; ayah++) {
      verses.push({ surah: 114, ayah, file: findVerseFile(114, ayah) });
    }
    results.push(await runTransitionTest("An-Nas (114:1 -> 114:6)", verses));
  }

  // ── Test 4: Mid-surah start at Ayat al-Kursi (2:255) ──
  {
    const verses = [{ surah: 2, ayah: 255, file: findVerseFile(2, 255) }];
    results.push(await runTransitionTest("Ayat al-Kursi (2:255 mid-surah)", verses));
  }

  // ═══════════════════════ FINAL REPORT ═══════════════════════════════════

  console.log(`\n\n${"=".repeat(70)}`);
  console.log(`  VERSE-TO-VERSE TRANSITION TEST REPORT`);
  console.log(`${"=".repeat(70)}\n`);

  let totalVerses = 0;
  let totalFound = 0;
  let totalInOrder = 0;
  let totalWrongSurah = 0;

  for (const r of results) {
    const found = r.transitions.filter(t => t.discovered).length;
    const total = r.transitions.length;
    totalVerses += total;
    totalFound += found;
    if (r.all_verses_in_order) totalInOrder++;
    totalWrongSurah += r.wrong_surah_matches.length;

    const statusIcon = r.all_verses_found && r.all_verses_in_order ? "PASS" : "FAIL";
    const avgDiscovery = r.transitions
      .filter(t => t.discovered)
      .reduce((s, t) => s + t.discovery_time_s, 0) / Math.max(found, 1);
    const avgCoverage = r.transitions
      .filter(t => t.total_words > 0)
      .reduce((s, t) => s + t.word_coverage, 0) / Math.max(total, 1);

    console.log(`  [${statusIcon}] ${r.name}`);
    console.log(`         Verses found:   ${found}/${total}`);
    console.log(`         In order:       ${r.all_verses_in_order ? "YES" : "NO"}`);
    console.log(`         Avg discovery:  ${avgDiscovery.toFixed(1)}s`);
    console.log(`         Avg coverage:   ${(avgCoverage * 100).toFixed(0)}%`);
    console.log(`         Wrong surahs:   ${r.wrong_surah_matches.length}`);

    if (r.wrong_surah_matches.length > 0) {
      console.log(`         Wrong matches:  ${r.wrong_surah_matches.join(", ")}`);
    }

    // Show per-verse detail
    for (const t of r.transitions) {
      const icon = t.discovered ? "+" : "-";
      const timeStr = t.discovery_time_s >= 0 ? `${t.discovery_time_s.toFixed(1)}s` : "---";
      const covStr = t.total_words > 0
        ? `${(t.word_coverage * 100).toFixed(0)}%`
        : "---";
      console.log(`           ${icon} ${t.verse_ref.padEnd(8)} t=${timeStr.padEnd(6)} cov=${covStr}`);
    }
    console.log();
  }

  console.log(`  ${"─".repeat(50)}`);
  console.log(`  TOTALS:`);
  console.log(`    Verses discovered:  ${totalFound}/${totalVerses} (${(totalFound / totalVerses * 100).toFixed(1)}%)`);
  console.log(`    Tests in order:     ${totalInOrder}/${results.length}`);
  console.log(`    Wrong surah events: ${totalWrongSurah}`);
  console.log(`  ${"─".repeat(50)}`);

  // Save results
  mkdirSync(resolve(__dirname, "benchmark-results"), { recursive: true });
  const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const outPath = resolve(__dirname, `benchmark-results/verse-transitions-${ts}.json`);
  const saveData = results.map(r => ({
    name: r.name,
    expected_verses: r.expected_verses,
    transitions: r.transitions,
    total_audio_duration_s: r.total_audio_duration_s,
    total_processing_time_s: r.total_processing_time_s,
    all_verses_found: r.all_verses_found,
    all_verses_in_order: r.all_verses_in_order,
    wrong_surah_matches: r.wrong_surah_matches,
    event_counts: {
      verse_match: r.all_events.filter(e => e.type === "verse_match").length,
      word_progress: r.all_events.filter(e => e.type === "word_progress").length,
      raw_transcript: r.all_events.filter(e => e.type === "raw_transcript").length,
      candidate_list: r.all_events.filter(e => e.type === "candidate_list").length,
    },
  }));
  writeFileSync(outPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    config: { chunk_ms: CHUNK_MS, silence_gap_ms: SILENCE_GAP_MS },
    summary: { totalVerses, totalFound, totalInOrder, totalWrongSurah },
    tests: saveData,
  }, null, 2));
  console.log(`\nSaved: ${outPath}`);
}

main().catch(err => { console.error("Fatal:", err); process.exit(1); });
