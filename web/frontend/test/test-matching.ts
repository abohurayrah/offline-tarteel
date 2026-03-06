#!/usr/bin/env tsx
/**
 * Direct matching test harness for matchVerse()
 *
 * This harness tests the matching algorithm directly with pre-computed phoneme
 * strings, bypassing the ONNX model entirely. It simulates real streaming scenarios
 * by creating realistic phoneme fragments from actual verse phonemes.
 *
 * Usage:
 *   npx tsx test/test-matching.ts
 *   # or
 *   ./test/test-matching.ts
 *
 * Test Categories:
 *   - exact_full: Full verse phonemes (should score 1.0)
 *   - partial_start: First 40-70% of verse (early streaming)
 *   - partial_mid: Middle 40-60% of verse (THE HARD CASE - mid-verse entry)
 *   - noisy: Full verse with ~8-10% character errors (model noise)
 *   - short_verse: Very short verses (Al-Ikhlas, Al-Kawthar)
 *   - long_verse: Long verses (Ayat al-Kursi, Al-Baqarah 2:286)
 *   - continuation: Fragments with hints from previous verse (tests bonus system)
 *   - multi_verse: Consecutive verse spans (tests maxSpan logic)
 *
 * Success Criteria:
 *   - exact_full should be 100% (score=1.0)
 *   - partial_start should be >90% accurate
 *   - partial_mid should be >85% accurate (hardest case)
 *   - noisy should be >95% accurate (model errors shouldn't break matching)
 *   - continuation should get bonus boost and match correctly
 */

import { readFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import { QuranDB } from "../src/lib/quran-db";
import type { QuranVerse } from "../src/lib/types";

// ES module __dirname workaround
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ============================================================================
// Test Case Types
// ============================================================================

interface TestCase {
  name: string;
  input: string;
  expected_surah: number;
  expected_ayah: number;
  hint?: [number, number] | null;
  category: "exact_full" | "partial_start" | "partial_mid" | "noisy" |
            "short_verse" | "long_verse" | "continuation" | "multi_verse";
}

interface TestResult {
  name: string;
  category: string;
  pass: boolean;
  expected: string;
  got: string | null;
  score: number;
  details: string;
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Add noise to phoneme text by randomly replacing/deleting characters
 */
function addNoise(text: string, noiseLevel = 0.1): string {
  const chars = text.split("");
  const numChanges = Math.floor(chars.length * noiseLevel);

  for (let i = 0; i < numChanges; i++) {
    const idx = Math.floor(Math.random() * chars.length);
    const operation = Math.random();

    if (operation < 0.5 && chars[idx] !== " ") {
      // Substitute with random character
      const replacements = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ<>*^$";
      chars[idx] = replacements[Math.floor(Math.random() * replacements.length)];
    } else if (operation < 0.8) {
      // Delete character
      chars.splice(idx, 1);
    }
    // else: keep as is (effectively a no-op for this iteration)
  }

  return chars.join("");
}

/**
 * Extract a substring from the middle of text
 */
function extractMiddle(text: string, ratio = 0.5): string {
  const words = text.split(" ");
  if (words.length < 3) return text;

  const targetWords = Math.ceil(words.length * ratio);
  const startIdx = Math.floor((words.length - targetWords) / 2);

  return words.slice(startIdx, startIdx + targetWords).join(" ");
}

/**
 * Extract a prefix (first N% of words)
 */
function extractPrefix(text: string, ratio = 0.6): string {
  const words = text.split(" ");
  const targetWords = Math.ceil(words.length * ratio);
  return words.slice(0, targetWords).join(" ");
}

// ============================================================================
// Test Case Generation
// ============================================================================

function generateTestCases(quranData: QuranVerse[]): TestCase[] {
  const cases: TestCase[] = [];

  // Helper to find verse
  const findVerse = (surah: number, ayah: number) =>
    quranData.find(v => v.surah === surah && v.ayah === ayah);

  // -------------------------------------------------------------------------
  // EXACT FULL MATCHES (should be trivial 1.0 match)
  // -------------------------------------------------------------------------

  const v1_1 = findVerse(1, 1)!;
  cases.push({
    name: "Exact full: Al-Fatiha 1:1 (Bismillah)",
    input: v1_1.phonemes_joined,
    expected_surah: 1,
    expected_ayah: 1,
    category: "exact_full"
  });

  const v1_2 = findVerse(1, 2)!;
  cases.push({
    name: "Exact full: Al-Fatiha 1:2 (Al-Hamd)",
    input: v1_2.phonemes_joined,
    expected_surah: 1,
    expected_ayah: 2,
    category: "exact_full"
  });

  const v112_1 = findVerse(112, 1)!;
  cases.push({
    name: "Exact full: Al-Ikhlas 112:1",
    input: v112_1.phonemes_joined,
    expected_surah: 112,
    expected_ayah: 1,
    category: "exact_full"
  });

  // -------------------------------------------------------------------------
  // PARTIAL START (first 40-70% - simulates early streaming)
  // -------------------------------------------------------------------------

  const v2_255 = findVerse(2, 255)!; // Ayat al-Kursi - long verse
  cases.push({
    name: "Partial start: Ayat al-Kursi 2:255 (first 50%)",
    input: extractPrefix(v2_255.phonemes_joined, 0.5),
    expected_surah: 2,
    expected_ayah: 255,
    category: "partial_start"
  });

  cases.push({
    name: "Partial start: Ayat al-Kursi 2:255 (first 70%)",
    input: extractPrefix(v2_255.phonemes_joined, 0.7),
    expected_surah: 2,
    expected_ayah: 255,
    category: "partial_start"
  });

  const v67_1 = findVerse(67, 1)!; // Al-Mulk opener
  cases.push({
    name: "Partial start: Al-Mulk 67:1 (first 60%)",
    input: extractPrefix(v67_1.phonemes_joined, 0.6),
    expected_surah: 67,
    expected_ayah: 1,
    category: "partial_start"
  });

  const v18_1 = findVerse(18, 1)!; // Al-Kahf opener
  cases.push({
    name: "Partial start: Al-Kahf 18:1 (first 50%)",
    input: extractPrefix(v18_1.phonemes_joined, 0.5),
    expected_surah: 18,
    expected_ayah: 1,
    category: "partial_start"
  });

  // -------------------------------------------------------------------------
  // PARTIAL MID (middle 40-60% - THE HARD CASE)
  // -------------------------------------------------------------------------

  cases.push({
    name: "Partial mid: Ayat al-Kursi 2:255 (middle 50%)",
    input: extractMiddle(v2_255.phonemes_joined, 0.5),
    expected_surah: 2,
    expected_ayah: 255,
    category: "partial_mid"
  });

  const v2_286 = findVerse(2, 286)!; // End of Baqarah
  cases.push({
    name: "Partial mid: Al-Baqarah 2:286 (middle 50%)",
    input: extractMiddle(v2_286.phonemes_joined, 0.5),
    expected_surah: 2,
    expected_ayah: 286,
    category: "partial_mid"
  });

  cases.push({
    name: "Partial mid: Al-Kahf 18:1 (middle 40%)",
    input: extractMiddle(v18_1.phonemes_joined, 0.4),
    expected_surah: 18,
    expected_ayah: 1,
    category: "partial_mid"
  });

  const v3_26 = findVerse(3, 26)!; // Qul Allahumma Malik al-Mulk
  cases.push({
    name: "Partial mid: Ali 'Imran 3:26 (middle 50%)",
    input: extractMiddle(v3_26.phonemes_joined, 0.5),
    expected_surah: 3,
    expected_ayah: 26,
    category: "partial_mid"
  });

  // -------------------------------------------------------------------------
  // NOISY (full phonemes with ~10% character substitutions/deletions)
  // -------------------------------------------------------------------------

  cases.push({
    name: "Noisy: Al-Fatiha 1:1 (~10% noise)",
    input: addNoise(v1_1.phonemes_joined, 0.1),
    expected_surah: 1,
    expected_ayah: 1,
    category: "noisy"
  });

  cases.push({
    name: "Noisy: Al-Fatiha 1:2 (~10% noise)",
    input: addNoise(v1_2.phonemes_joined, 0.1),
    expected_surah: 1,
    expected_ayah: 2,
    category: "noisy"
  });

  const v112_2 = findVerse(112, 2)!;
  cases.push({
    name: "Noisy: Al-Ikhlas 112:2 (~10% noise)",
    input: addNoise(v112_2.phonemes_joined, 0.1),
    expected_surah: 112,
    expected_ayah: 2,
    category: "noisy"
  });

  cases.push({
    name: "Noisy: Ayat al-Kursi 2:255 (~8% noise)",
    input: addNoise(v2_255.phonemes_joined, 0.08),
    expected_surah: 2,
    expected_ayah: 255,
    category: "noisy"
  });

  // -------------------------------------------------------------------------
  // SHORT VERSES
  // -------------------------------------------------------------------------

  const v108_1 = findVerse(108, 1)!; // Al-Kawthar
  cases.push({
    name: "Short verse: Al-Kawthar 108:1",
    input: v108_1.phonemes_joined,
    expected_surah: 108,
    expected_ayah: 1,
    category: "short_verse"
  });

  const v112_3 = findVerse(112, 3)!;
  cases.push({
    name: "Short verse: Al-Ikhlas 112:3",
    input: v112_3.phonemes_joined,
    expected_surah: 112,
    expected_ayah: 3,
    category: "short_verse"
  });

  const v112_4 = findVerse(112, 4)!;
  cases.push({
    name: "Short verse: Al-Ikhlas 112:4",
    input: v112_4.phonemes_joined,
    expected_surah: 112,
    expected_ayah: 4,
    category: "short_verse"
  });

  // -------------------------------------------------------------------------
  // LONG VERSES
  // -------------------------------------------------------------------------

  cases.push({
    name: "Long verse: Ayat al-Kursi 2:255 (full)",
    input: v2_255.phonemes_joined,
    expected_surah: 2,
    expected_ayah: 255,
    category: "long_verse"
  });

  cases.push({
    name: "Long verse: Al-Baqarah 2:286 (full)",
    input: v2_286.phonemes_joined,
    expected_surah: 2,
    expected_ayah: 286,
    category: "long_verse"
  });

  cases.push({
    name: "Long verse: Al-Kahf 18:1 (full)",
    input: v18_1.phonemes_joined,
    expected_surah: 18,
    expected_ayah: 1,
    category: "long_verse"
  });

  // -------------------------------------------------------------------------
  // CONTINUATION (with hints - should boost next verse)
  // -------------------------------------------------------------------------

  cases.push({
    name: "Continuation: Al-Fatiha 1:2 with hint from 1:1",
    input: v1_2.phonemes_joined,
    expected_surah: 1,
    expected_ayah: 2,
    hint: [1, 1],
    category: "continuation"
  });

  cases.push({
    name: "Continuation: Al-Ikhlas 112:2 with hint from 112:1",
    input: v112_2.phonemes_joined,
    expected_surah: 112,
    expected_ayah: 2,
    hint: [112, 1],
    category: "continuation"
  });

  const v55_13 = findVerse(55, 13)!; // Ar-Rahman refrain
  const v55_14 = findVerse(55, 14)!;
  cases.push({
    name: "Continuation: Ar-Rahman 55:14 with hint from 55:13",
    input: v55_14.phonemes_joined,
    expected_surah: 55,
    expected_ayah: 14,
    hint: [55, 13],
    category: "continuation"
  });

  // Test partial with continuation hint
  cases.push({
    name: "Continuation: Al-Fatiha 1:2 (first 50%) with hint from 1:1",
    input: extractPrefix(v1_2.phonemes_joined, 0.5),
    expected_surah: 1,
    expected_ayah: 2,
    hint: [1, 1],
    category: "continuation"
  });

  // -------------------------------------------------------------------------
  // MULTI-VERSE SPANS
  // -------------------------------------------------------------------------

  // Al-Ikhlas 112:1-2 (consecutive verses)
  const ikhlas_1_2 = v112_1.phonemes_joined + " " + v112_2.phonemes_joined;
  cases.push({
    name: "Multi-verse: Al-Ikhlas 112:1-2",
    input: ikhlas_1_2,
    expected_surah: 112,
    expected_ayah: 1,
    category: "multi_verse"
  });

  // Al-Fatiha 1:1-2
  const fatiha_1_2 = v1_1.phonemes_joined + " " + v1_2.phonemes_joined;
  cases.push({
    name: "Multi-verse: Al-Fatiha 1:1-2",
    input: fatiha_1_2,
    expected_surah: 1,
    expected_ayah: 1,
    category: "multi_verse"
  });

  // Al-Ikhlas 112:3-4 (very short verses)
  const ikhlas_3_4 = v112_3.phonemes_joined + " " + v112_4.phonemes_joined;
  cases.push({
    name: "Multi-verse: Al-Ikhlas 112:3-4",
    input: ikhlas_3_4,
    expected_surah: 112,
    expected_ayah: 3,
    category: "multi_verse"
  });

  return cases;
}

// ============================================================================
// Test Runner
// ============================================================================

function runTests(db: QuranDB, testCases: TestCase[]): TestResult[] {
  const results: TestResult[] = [];

  console.log(`Running ${testCases.length} test cases...\n`);
  console.log("=".repeat(80));

  for (const testCase of testCases) {
    const { name, input, expected_surah, expected_ayah, hint, category } = testCase;

    // Call matchVerse with default threshold and maxSpan
    const threshold = 0.3;
    const maxSpan = 3;
    const result = db.matchVerse(input, threshold, maxSpan, hint ?? null, 0);

    const expected = `${expected_surah}:${expected_ayah}`;
    const got = result ? `${result.surah}:${result.ayah}` : null;
    const pass = got === expected;
    const score = result?.score ?? 0.0;

    let details = "";
    if (result) {
      if (result.ayah_end) {
        details = `matched ${result.surah}:${result.ayah}-${result.ayah_end}, score=${score.toFixed(3)}`;
      } else {
        details = `matched ${result.surah}:${result.ayah}, score=${score.toFixed(3)}`;
      }
      if (result.bonus && result.bonus > 0) {
        details += `, bonus=${result.bonus.toFixed(3)}`;
      }
    } else {
      details = "no match (below threshold)";
    }

    results.push({ name, category, pass, expected, got, score, details });

    // Print per-case result
    const statusIcon = pass ? "✓" : "✗";
    const statusColor = pass ? "\x1b[32m" : "\x1b[31m"; // Green or Red
    const resetColor = "\x1b[0m";

    console.log(`${statusColor}${statusIcon}${resetColor} [${category}] ${name}`);
    console.log(`  Expected: ${expected}, Got: ${got ?? "null"}, ${details}`);
    console.log();
  }

  return results;
}

// ============================================================================
// Statistics & Reporting
// ============================================================================

function printStatistics(results: TestResult[]) {
  console.log("=".repeat(80));
  console.log("TEST SUMMARY");
  console.log("=".repeat(80));

  const total = results.length;
  const passed = results.filter(r => r.pass).length;
  const failed = total - passed;
  const accuracy = total > 0 ? (passed / total) * 100 : 0;

  console.log(`Total:    ${total}`);
  console.log(`Passed:   ${passed} (${accuracy.toFixed(1)}%)`);
  console.log(`Failed:   ${failed}`);
  console.log();

  // Breakdown by category
  const categories = [...new Set(results.map(r => r.category))];
  console.log("ACCURACY BY CATEGORY:");
  console.log("-".repeat(80));

  for (const cat of categories) {
    const catResults = results.filter(r => r.category === cat);
    const catPassed = catResults.filter(r => r.pass).length;
    const catTotal = catResults.length;
    const catAccuracy = catTotal > 0 ? (catPassed / catTotal) * 100 : 0;

    console.log(`  ${cat.padEnd(20)} ${catPassed}/${catTotal} (${catAccuracy.toFixed(1)}%)`);
  }
  console.log();

  // Show failures
  const failures = results.filter(r => !r.pass);
  if (failures.length > 0) {
    console.log("FAILURES:");
    console.log("-".repeat(80));
    for (const fail of failures) {
      console.log(`  ${fail.name}`);
      console.log(`    Expected: ${fail.expected}, Got: ${fail.got ?? "null"}, ${fail.details}`);
    }
    console.log();
  }

  // Average scores by category
  console.log("AVERAGE SCORES BY CATEGORY:");
  console.log("-".repeat(80));
  for (const cat of categories) {
    const catResults = results.filter(r => r.category === cat);
    const avgScore = catResults.reduce((sum, r) => sum + r.score, 0) / catResults.length;
    console.log(`  ${cat.padEnd(20)} ${avgScore.toFixed(3)}`);
  }
  console.log();
}

// ============================================================================
// Main
// ============================================================================

function main() {
  console.log("Loading Quran phonemes database...");

  const dataPath = resolve(__dirname, "../public/quran_phonemes.json");
  const rawData = readFileSync(dataPath, "utf-8");
  const quranData: QuranVerse[] = JSON.parse(rawData);

  console.log(`Loaded ${quranData.length} verses.`);

  const db = new QuranDB(quranData);
  console.log(`QuranDB initialized: ${db.totalVerses} verses, ${db.surahCount} surahs\n`);

  // Generate test cases
  const testCases = generateTestCases(quranData);

  // Run tests
  const results = runTests(db, testCases);

  // Print statistics
  printStatistics(results);
}

main();
