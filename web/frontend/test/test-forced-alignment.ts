#!/usr/bin/env npx tsx
/**
 * Forced Alignment Integration Test
 *
 * Tests the ForcedAligner end-to-end with real audio files:
 *   audio file -> ffmpeg decode -> mel spectrogram -> ONNX inference -> CTC logprobs
 *   -> ForcedAligner (Viterbi DP) -> word-level alignment
 *
 * Test cases:
 *   1. 1:1  — Bismillah (4 words, clear everyayah audio)
 *   2. 2:255 — Ayat al-Kursi (long verse, ~50 words)
 *   3. 112:1 — Al-Ikhlas (short, 4 words)
 *   4. Tarteel user recording (noisy real-world audio)
 *
 * Usage:
 *   npx tsx test/test-forced-alignment.ts
 */
import { execSync } from "node:child_process";
import { readFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";

import { computeMelSpectrogram } from "../src/worker/mel.ts";
import { CTCDecoder } from "../src/worker/ctc-decode.ts";
import {
  ForcedAligner,
  BPETokenizer,
  stripUthmaniMarks,
} from "../src/worker/forced-alignment.ts";
import type { WordAlignment } from "../src/worker/forced-alignment.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const CORPUS = resolve(ROOT, "../../benchmark/test_corpus");
const EXPANDED = resolve(ROOT, "../../benchmark/test_corpus_expanded");
const SAMPLE_RATE = 16000;

// ── Setup ────────────────────────────────────────────────────────────────────

let session: ort.InferenceSession;
let decoder: CTCDecoder;
let vocabJson: Record<string, string>;
let quranData: Array<{
  surah: number;
  ayah: number;
  text_uthmani: string;
  text_clean?: string;
}>;

function loadAudio(filePath: string): Float32Array {
  const buf = execSync(
    `ffmpeg -hide_banner -loglevel error -i "${filePath}" -f f32le -ar ${SAMPLE_RATE} -ac 1 pipe:1`,
    { maxBuffer: 50 * 1024 * 1024 },
  );
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

async function initModel(): Promise<void> {
  const modelPath = resolve(ROOT, "public/fastconformer_ar_ctc_q8.onnx");
  const vocabPath = resolve(ROOT, "public/vocab.json");
  const quranPath = resolve(ROOT, "public/quran.json");

  console.log("Loading ONNX model...");
  const t0 = performance.now();
  session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ["cpu"],
  });
  console.log(`  Model loaded in ${(performance.now() - t0).toFixed(0)}ms`);

  vocabJson = JSON.parse(readFileSync(vocabPath, "utf-8"));
  decoder = new CTCDecoder(vocabJson);
  quranData = JSON.parse(readFileSync(quranPath, "utf-8"));
  console.log(`  Vocab size: ${decoder.vocabSize}, Blank ID: ${decoder.blankId}`);
  console.log(`  Quran verses: ${quranData.length}\n`);
}

async function runOnnx(audio: Float32Array): Promise<{
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

// ── Helpers ──────────────────────────────────────────────────────────────────

function getVerseText(surah: number, ayah: number): string {
  const verse = quranData.find((v) => v.surah === surah && v.ayah === ayah);
  if (!verse) throw new Error(`Verse ${surah}:${ayah} not found`);
  return verse.text_clean || verse.text_uthmani;
}

function getExpectedWordCount(text: string): number {
  return stripUthmaniMarks(text).trim().split(/\s+/).length;
}

interface TestResult {
  id: string;
  surah: number;
  ayah: number;
  audioFile: string;
  audioDuration: number;
  expectedWords: number;
  alignedWords: number;
  skippedWords: number;
  allWordsAligned: boolean;
  completed: boolean;
  stalled: boolean;
  error: string | null;
  overallScore: number;
  greedyDecode: string;   // what CTC greedy actually recognized
  wordDetails: Array<{
    index: number;
    word: string;
    confidence: number;
    startFrame: number;
    endFrame: number;
    durationFrames: number;
  }>;
  tokenization: {
    totalTokens: number;
    wordsWithTokens: number;
  };
  frameStats: {
    totalFrames: number;
    blankFrames: number;
    nonBlankFrames: number;
  };
  timingMs: number;
  // Streaming simulation results
  streaming: {
    chunksProcessed: number;
    newWordsPerChunk: number[];
    finalWordIdx: number;
  };
}

// ── Single-shot FA test (full audio at once) ─────────────────────────────────

async function testForcedAlignmentSingleShot(
  id: string,
  audioPath: string,
  surah: number,
  ayah: number,
): Promise<TestResult> {
  const verseText = getVerseText(surah, ayah);
  const expectedWords = getExpectedWordCount(verseText);
  const strippedText = stripUthmaniMarks(verseText);
  const words = strippedText.trim().split(/\s+/);

  const result: TestResult = {
    id,
    surah,
    ayah,
    audioFile: audioPath,
    audioDuration: 0,
    expectedWords,
    alignedWords: 0,
    skippedWords: 0,
    allWordsAligned: false,
    completed: false,
    stalled: false,
    error: null,
    overallScore: 0,
    greedyDecode: "",
    wordDetails: [],
    tokenization: { totalTokens: 0, wordsWithTokens: 0 },
    frameStats: { totalFrames: 0, blankFrames: 0, nonBlankFrames: 0 },
    timingMs: 0,
    streaming: { chunksProcessed: 0, newWordsPerChunk: [], finalWordIdx: -1 },
  };

  const t0 = performance.now();

  try {
    // Step 1: Load audio
    const audio = loadAudio(audioPath);
    result.audioDuration = audio.length / SAMPLE_RATE;

    // Step 2: ONNX inference
    const { logprobs, timeSteps, vocabSize } = await runOnnx(audio);

    // Step 2b: CTC greedy decode (to see what model actually recognizes)
    const greedyResult = decoder.decode(logprobs, timeSteps, vocabSize);
    result.greedyDecode = greedyResult.text;

    // Step 2c: Frame statistics
    const blankId = decoder.blankId;
    let blankFrames = 0;
    for (let t = 0; t < timeSteps; t++) {
      let maxIdx = 0;
      let maxVal = logprobs[t * vocabSize];
      for (let v = 1; v < vocabSize; v++) {
        if (logprobs[t * vocabSize + v] > maxVal) {
          maxVal = logprobs[t * vocabSize + v];
          maxIdx = v;
        }
      }
      if (maxIdx === blankId) blankFrames++;
    }
    result.frameStats = {
      totalFrames: timeSteps,
      blankFrames,
      nonBlankFrames: timeSteps - blankFrames,
    };

    // Step 3: Create ForcedAligner
    const aligner = new ForcedAligner(
      strippedText,
      vocabJson,
      decoder.blankId,
      vocabSize,
    );

    result.tokenization.totalTokens = aligner.totalTokens;
    result.tokenization.wordsWithTokens = aligner.totalWords;

    // Step 4: Feed ALL logprobs at once (single-shot)
    const { newWords, currentWordIdx, allWords } = aligner.processFrames(
      logprobs,
      timeSteps,
    );

    // Step 5: Finalize
    const finalWords = aligner.finalize();
    const overallScore = aligner.getOverallScore();

    result.alignedWords = finalWords.length;
    result.skippedWords = expectedWords - finalWords.length;
    result.allWordsAligned = finalWords.length === expectedWords;
    result.completed = true;
    result.overallScore = overallScore;

    // Check for stalling: if fewer than half the words aligned, it stalled
    result.stalled = finalWords.length < expectedWords * 0.5;

    // Per-word details
    const alignedIndices = new Set(finalWords.map((w) => w.wordIndex));
    for (let i = 0; i < expectedWords; i++) {
      const aligned = finalWords.find((w) => w.wordIndex === i);
      if (aligned) {
        result.wordDetails.push({
          index: i,
          word: aligned.word,
          confidence: aligned.confidence,
          startFrame: aligned.startFrame,
          endFrame: aligned.endFrame,
          durationFrames: aligned.endFrame - aligned.startFrame,
        });
      } else {
        result.wordDetails.push({
          index: i,
          word: words[i] ?? `[word_${i}]`,
          confidence: 0,
          startFrame: -1,
          endFrame: -1,
          durationFrames: 0,
        });
      }
    }

    // Step 6: Streaming simulation (feed in ~300ms chunks)
    const chunkSamples = Math.floor(SAMPLE_RATE * 0.3);
    const aligner2 = new ForcedAligner(
      strippedText,
      vocabJson,
      decoder.blankId,
      vocabSize,
    );

    // For streaming we need logprobs frame-by-frame from separate ONNX calls
    // But we can simulate chunk-level FA by slicing logprobs
    const framesPerChunk = Math.ceil(timeSteps / Math.ceil(audio.length / chunkSamples));
    let frameOffset = 0;
    let chunkIdx = 0;

    while (frameOffset < timeSteps) {
      const chunkFrames = Math.min(framesPerChunk, timeSteps - frameOffset);
      const chunkLogprobs = logprobs.slice(
        frameOffset * vocabSize,
        (frameOffset + chunkFrames) * vocabSize,
      );

      const chunkResult = aligner2.processFrames(
        chunkLogprobs,
        chunkFrames,
      );
      result.streaming.newWordsPerChunk.push(chunkResult.newWords.length);
      result.streaming.finalWordIdx = chunkResult.currentWordIdx;
      frameOffset += chunkFrames;
      chunkIdx++;
    }
    result.streaming.chunksProcessed = chunkIdx;
  } catch (err: any) {
    result.error = err.message || String(err);
    result.completed = false;
  }

  result.timingMs = performance.now() - t0;
  return result;
}

// ── Display ──────────────────────────────────────────────────────────────────

function printResult(r: TestResult): void {
  const status = r.error
    ? "ERROR"
    : r.stalled
      ? "STALLED"
      : r.allWordsAligned
        ? "PASS"
        : "PARTIAL";

  const icon =
    status === "PASS" ? "[OK]" : status === "PARTIAL" ? "[!!]" : status === "STALLED" ? "[XX]" : "[ER]";

  console.log(`\n${"=".repeat(74)}`);
  console.log(
    `  ${icon} ${r.id} — ${r.surah}:${r.ayah}  (${status})`,
  );
  console.log(`${"=".repeat(74)}`);

  if (r.error) {
    console.log(`  ERROR: ${r.error}`);
    return;
  }

  console.log(`  Audio:        ${r.audioDuration.toFixed(2)}s`);
  console.log(`  CTC greedy:   "${r.greedyDecode}"`);
  console.log(
    `  Frames:       ${r.frameStats.totalFrames} total, ${r.frameStats.blankFrames} blank, ${r.frameStats.nonBlankFrames} non-blank`,
  );
  console.log(
    `  Tokenization: ${r.tokenization.totalTokens} BPE tokens for ${r.tokenization.wordsWithTokens} words`,
  );
  console.log(
    `  Alignment:    ${r.alignedWords}/${r.expectedWords} words aligned (${r.skippedWords} skipped)`,
  );
  console.log(`  Overall score: ${r.overallScore.toFixed(4)}`);
  console.log(`  Processing:   ${r.timingMs.toFixed(0)}ms`);
  console.log(
    `  Streaming:    ${r.streaming.chunksProcessed} chunks, words-per-chunk: [${r.streaming.newWordsPerChunk.join(", ")}]`,
  );

  // Per-word table
  console.log(`\n  Word-level results:`);
  console.log(
    `  ${"#".padStart(3)}  ${"Word".padEnd(20)}  ${"Conf".padStart(8)}  ${"Frames".padStart(10)}  ${"Status".padStart(8)}`,
  );
  console.log(`  ${"---".padStart(3)}  ${"----".padEnd(20)}  ${"----".padStart(8)}  ${"------".padStart(10)}  ${"------".padStart(8)}`);

  for (const w of r.wordDetails) {
    const confStr =
      w.confidence > 0 ? w.confidence.toFixed(4) : "   --  ";
    const framesStr =
      w.startFrame >= 0
        ? `${w.startFrame}-${w.endFrame} (${w.durationFrames})`
        : "   --   ";
    const statusStr =
      w.confidence > 0.5
        ? "GOOD"
        : w.confidence > 0.1
          ? "LOW"
          : w.confidence > 0
            ? "VLOW"
            : "SKIP";
    // Truncate Arabic word display
    const wordDisplay =
      w.word.length > 18 ? w.word.slice(0, 18) + ".." : w.word;
    console.log(
      `  ${String(w.index).padStart(3)}  ${wordDisplay.padEnd(20)}  ${confStr.padStart(8)}  ${framesStr.padStart(10)}  ${statusStr.padStart(8)}`,
    );
  }

  // Confidence histogram
  const confs = r.wordDetails.filter((w) => w.confidence > 0).map((w) => w.confidence);
  if (confs.length > 0) {
    const avg = confs.reduce((s, c) => s + c, 0) / confs.length;
    const min = Math.min(...confs);
    const max = Math.max(...confs);
    const median = confs.sort((a, b) => a - b)[Math.floor(confs.length / 2)];
    console.log(`\n  Confidence stats (aligned words only):`);
    console.log(`    avg=${avg.toFixed(4)}  min=${min.toFixed(4)}  max=${max.toFixed(4)}  median=${median.toFixed(4)}`);

    // Buckets
    const buckets = [0, 0, 0, 0, 0]; // <0.1, 0.1-0.3, 0.3-0.5, 0.5-0.8, 0.8+
    for (const c of confs) {
      if (c < 0.1) buckets[0]++;
      else if (c < 0.3) buckets[1]++;
      else if (c < 0.5) buckets[2]++;
      else if (c < 0.8) buckets[3]++;
      else buckets[4]++;
    }
    console.log(
      `    <0.1: ${buckets[0]}  0.1-0.3: ${buckets[1]}  0.3-0.5: ${buckets[2]}  0.5-0.8: ${buckets[3]}  0.8+: ${buckets[4]}`,
    );
  }
}

// ── Tokenization diagnostics ─────────────────────────────────────────────────

function printTokenizationDiag(surah: number, ayah: number): void {
  const verseText = getVerseText(surah, ayah);
  const stripped = stripUthmaniMarks(verseText);
  const tokenizer = new BPETokenizer(vocabJson);
  const { tokenIDs, tokenStrings, wordBoundaries } = tokenizer.tokenize(stripped);

  console.log(`\n  Tokenization for ${surah}:${ayah}:`);
  console.log(`    Input:  "${stripped.slice(0, 80)}${stripped.length > 80 ? "..." : ""}"`);
  console.log(`    Tokens: ${tokenIDs.length} total, ${wordBoundaries.length} words`);

  for (const wb of wordBoundaries) {
    const toks = tokenStrings.slice(wb.startTokenIdx, wb.endTokenIdx);
    const tokCount = wb.endTokenIdx - wb.startTokenIdx;
    console.log(
      `    [${wb.startTokenIdx}-${wb.endTokenIdx}) "${wb.wordText}" -> ${tokCount} tokens: ${toks.join(" | ")}`,
    );
  }
}

// ── Main ─────────────────────────────────────────────────────────────────────

interface TestCase {
  id: string;
  surah: number;
  ayah: number;
  file: string;
  description: string;
}

async function main() {
  console.log("=".repeat(74));
  console.log("  FORCED ALIGNMENT INTEGRATION TEST");
  console.log("  Testing word-level alignment on real audio files");
  console.log("=".repeat(74));
  console.log();

  await initModel();

  // Define test cases
  const tests: TestCase[] = [
    {
      id: "bismillah_1_1",
      surah: 1,
      ayah: 1,
      file: resolve(CORPUS, "001001.mp3"),
      description: "Bismillah — 4 words, clear everyayah audio",
    },
    {
      id: "ayat_kursi_2_255",
      surah: 2,
      ayah: 255,
      file: resolve(CORPUS, "002255.mp3"),
      description: "Ayat al-Kursi — long verse (~50 words)",
    },
    {
      id: "ikhlas_112_1",
      surah: 112,
      ayah: 1,
      file: resolve(CORPUS, "112001.mp3"),
      description: "Al-Ikhlas 112:1 — short verse (4 words)",
    },
    {
      id: "tarteel_user_1_1",
      surah: 1,
      ayah: 1,
      file: resolve(EXPANDED, "tarteel_000.wav"),
      description: "Tarteel user recording — noisy real-world audio",
    },
  ];

  // Validate all files exist
  for (const t of tests) {
    if (!existsSync(t.file)) {
      console.log(`  WARNING: Audio file missing: ${t.file}`);
      console.log(`           Skipping test: ${t.id}`);
    }
  }

  // Print tokenization diagnostics first
  console.log("\n--- TOKENIZATION DIAGNOSTICS ---");
  for (const t of tests) {
    try {
      printTokenizationDiag(t.surah, t.ayah);
    } catch (err: any) {
      console.log(`  Error tokenizing ${t.surah}:${t.ayah}: ${err.message}`);
    }
  }

  // Run tests
  const results: TestResult[] = [];
  console.log("\n\n--- RUNNING FORCED ALIGNMENT TESTS ---");

  for (const t of tests) {
    if (!existsSync(t.file)) continue;
    console.log(`\nRunning: ${t.description}...`);
    const r = await testForcedAlignmentSingleShot(t.id, t.file, t.surah, t.ayah);
    results.push(r);
    printResult(r);
  }

  // ── Summary ──────────────────────────────────────────────────────────────
  console.log(`\n\n${"=".repeat(74)}`);
  console.log("  SUMMARY");
  console.log(`${"=".repeat(74)}`);
  console.log(
    `\n  ${"Test".padEnd(25)} ${"Words".padStart(12)} ${"Score".padStart(8)} ${"Time".padStart(8)} ${"Status".padStart(10)}`,
  );
  console.log(
    `  ${"----".padEnd(25)} ${"-----".padStart(12)} ${"-----".padStart(8)} ${"----".padStart(8)} ${"------".padStart(10)}`,
  );

  let passCount = 0;
  for (const r of results) {
    const status = r.error
      ? "ERROR"
      : r.stalled
        ? "STALLED"
        : r.allWordsAligned
          ? "PASS"
          : "PARTIAL";
    if (status === "PASS") passCount++;
    console.log(
      `  ${r.id.padEnd(25)} ${`${r.alignedWords}/${r.expectedWords}`.padStart(12)} ${r.overallScore.toFixed(4).padStart(8)} ${`${r.timingMs.toFixed(0)}ms`.padStart(8)} ${status.padStart(10)}`,
    );
  }

  console.log(
    `\n  Result: ${passCount}/${results.length} tests fully aligned all words`,
  );

  // Key findings
  console.log("\n  Key findings:");
  for (const r of results) {
    if (r.error) {
      console.log(`    - ${r.id}: FAILED with error: ${r.error}`);
    } else if (r.stalled) {
      console.log(
        `    - ${r.id}: STALLED — only ${r.alignedWords}/${r.expectedWords} words aligned`,
      );
    } else if (!r.allWordsAligned) {
      const skippedWords = r.wordDetails
        .filter((w) => w.confidence === 0)
        .map((w) => `"${w.word}"`)
        .join(", ");
      console.log(
        `    - ${r.id}: PARTIAL — skipped words: ${skippedWords}`,
      );
    } else {
      const lowConf = r.wordDetails.filter(
        (w) => w.confidence > 0 && w.confidence < 0.3,
      );
      if (lowConf.length > 0) {
        console.log(
          `    - ${r.id}: ALL ALIGNED but ${lowConf.length} words have low confidence (<0.3)`,
        );
      } else {
        console.log(
          `    - ${r.id}: ALL ALIGNED with good confidence (score=${r.overallScore.toFixed(4)})`,
        );
      }
    }
  }

  // Verdict
  const allPassed = passCount === results.length;
  const anyStalled = results.some((r) => r.stalled);
  const anyError = results.some((r) => r.error);
  const avgScore =
    results.filter((r) => !r.error).reduce((s, r) => s + r.overallScore, 0) /
    results.filter((r) => !r.error).length;

  console.log(`\n  Overall verdict:`);
  if (allPassed && avgScore > 0.3) {
    console.log(
      `    FA engine is FUNCTIONAL — all words aligned across all test cases`,
    );
    console.log(`    Average score: ${avgScore.toFixed(4)}`);
  } else if (anyError) {
    console.log(`    FA engine has ERRORS — some tests threw exceptions`);
  } else if (anyStalled) {
    console.log(
      `    FA engine is FUNDAMENTALLY BROKEN — stalled on at least one test`,
    );
  } else {
    console.log(
      `    FA engine is PARTIALLY WORKING — some words not aligned`,
    );
    console.log(`    Average score: ${avgScore.toFixed(4)}`);
  }

  console.log(`\n${"=".repeat(74)}\n`);
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
