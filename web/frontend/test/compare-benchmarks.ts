#!/usr/bin/env npx tsx
/**
 * Compare two benchmark result files side-by-side.
 *
 * Usage:
 *   npx tsx test/compare-benchmarks.ts test/benchmark-results/before.json test/benchmark-results/after.json
 *   npx tsx test/compare-benchmarks.ts  # compares the two most recent results
 */

import { readFileSync, readdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const RESULTS_DIR = resolve(__dirname, "benchmark-results");

interface BenchmarkResult {
  timestamp: string;
  model: string;
  summary: {
    total: number;
    correct: number;
    equiv: number;
    adjacent: number;
    wrong: number;
    no_match: number;
    strict_accuracy: number;
    lenient_accuracy: number;
    false_positive_rate: number;
    avg_inference_ms: number;
  };
  by_category: Record<string, { total: number; correct: number; accuracy: number; avg_ms: number }>;
  by_source: Record<string, { total: number; correct: number; accuracy: number }>;
  results: {
    id: string;
    category: string;
    source: string;
    expected: string;
    got: string;
    eval: string;
    score: number;
    transcript: string;
    timeMs: number;
  }[];
}

function loadResult(path: string): BenchmarkResult {
  return JSON.parse(readFileSync(path, "utf-8"));
}

function pct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

function delta(before: number, after: number): string {
  const d = after - before;
  const sign = d >= 0 ? "+" : "";
  const color = d > 0 ? "\x1b[32m" : d < 0 ? "\x1b[31m" : "\x1b[90m";
  return `${color}${sign}${(d * 100).toFixed(1)}%\x1b[0m`;
}

// Get file paths from args or find most recent two
let [fileA, fileB] = process.argv.slice(2);

if (!fileA || !fileB) {
  const files = readdirSync(RESULTS_DIR)
    .filter((f) => f.endsWith(".json"))
    .sort()
    .reverse();
  if (files.length < 2) {
    console.error("Need at least 2 benchmark results to compare. Run the benchmark twice first.");
    process.exit(1);
  }
  fileB = resolve(RESULTS_DIR, files[0]);
  fileA = resolve(RESULTS_DIR, files[1]);
  console.log(`Comparing: ${files[1]} (before) vs ${files[0]} (after)\n`);
}

const before = loadResult(fileA);
const after = loadResult(fileB);

// Overall summary
console.log("═".repeat(70));
console.log("  BENCHMARK COMPARISON");
console.log("═".repeat(70));
console.log();
console.log(`  Before: ${before.timestamp} (${before.model})`);
console.log(`  After:  ${after.timestamp} (${after.model})`);
console.log();

const metrics = [
  ["Strict accuracy", before.summary.strict_accuracy, after.summary.strict_accuracy],
  ["Lenient accuracy", before.summary.lenient_accuracy, after.summary.lenient_accuracy],
  ["False positive rate", before.summary.false_positive_rate, after.summary.false_positive_rate],
] as const;

for (const [name, b, a] of metrics) {
  console.log(`  ${name.padEnd(22)} ${pct(b).padEnd(8)} → ${pct(a).padEnd(8)} ${delta(b, a)}`);
}
console.log(`  ${"Avg inference".padEnd(22)} ${before.summary.avg_inference_ms.toFixed(0).padEnd(8)}ms → ${after.summary.avg_inference_ms.toFixed(0).padEnd(8)}ms`);

// By category
console.log();
console.log("  By category:");
const allCats = new Set([...Object.keys(before.by_category), ...Object.keys(after.by_category)]);
for (const cat of ["short", "medium", "long", "multi"]) {
  if (!allCats.has(cat)) continue;
  const b = before.by_category[cat]?.accuracy ?? 0;
  const a = after.by_category[cat]?.accuracy ?? 0;
  const bN = before.by_category[cat]?.total ?? 0;
  const aN = after.by_category[cat]?.total ?? 0;
  console.log(`    ${cat.padEnd(10)} ${pct(b).padEnd(8)} (${bN}) → ${pct(a).padEnd(8)} (${aN}) ${delta(b, a)}`);
}

// By source
console.log();
console.log("  By source:");
for (const src of ["everyayah", "retasy", "user"]) {
  const b = before.by_source[src]?.accuracy ?? 0;
  const a = after.by_source[src]?.accuracy ?? 0;
  console.log(`    ${src.padEnd(12)} ${pct(b).padEnd(8)} → ${pct(a).padEnd(8)} ${delta(b, a)}`);
}

// Per-sample changes
console.log();
console.log("  Changes:");
let improved = 0;
let regressed = 0;

for (const afterResult of after.results) {
  const beforeResult = before.results.find((r) => r.id === afterResult.id);
  if (!beforeResult) continue;

  const wasCorrect = beforeResult.eval === "correct" || beforeResult.eval === "equiv";
  const isCorrect = afterResult.eval === "correct" || afterResult.eval === "equiv";

  if (!wasCorrect && isCorrect) {
    improved++;
    console.log(`    \x1b[32m↑ FIXED\x1b[0m  ${afterResult.id}: ${beforeResult.eval}(${beforeResult.got}) → ${afterResult.eval}(${afterResult.got})`);
  } else if (wasCorrect && !isCorrect) {
    regressed++;
    console.log(`    \x1b[31m↓ REGRESSED\x1b[0m ${afterResult.id}: ${beforeResult.eval}(${beforeResult.got}) → ${afterResult.eval}(${afterResult.got})`);
  }
}

if (improved === 0 && regressed === 0) {
  console.log("    (no per-sample changes)");
}

console.log();
console.log(`  Summary: ${improved} improved, ${regressed} regressed`);
console.log("═".repeat(70));
