// ---------------------------------------------------------------------------
// FA Benchmark — Automated recitation simulation using audio-samples
// ---------------------------------------------------------------------------
// Feeds real Quran audio recordings through the full pipeline:
//   Audio → Worker (mel → ONNX → CTC logprobs → Discovery/Tracking) → Messages
// Scores: verse identification accuracy and word-level tracking coverage.
// ---------------------------------------------------------------------------

import type { WorkerOutbound } from "../src/lib/types";

// ── DOM refs ──
const $status = document.getElementById("status")!;
const $progress = document.getElementById("progress-fill")!;
const $summary = document.getElementById("summary")!;
const $log = document.getElementById("log")!;
const $btnRunAll = document.getElementById("btn-run-all") as HTMLButtonElement;
const $btnRunSelected = document.getElementById("btn-run-selected") as HTMLButtonElement;
const $btnStop = document.getElementById("btn-stop") as HTMLButtonElement;
const $scenarioSelect = document.getElementById("scenario-select") as HTMLSelectElement;

// ── Config ──
const SPEED_MULT = 8;          // 8x real-time
const CHUNK_MS = 100;           // 100ms chunks
const GAP_SAMPLES = 3200;       // 200ms silence gap between ayahs (16kHz)

// ── State ──
let worker: Worker | null = null;
let isRunning = false;
let shouldStop = false;

// ── Logging ──
function log(text: string, cls = "") {
  const span = document.createElement("span");
  span.className = cls;
  span.textContent = text + "\n";
  $log.appendChild(span);
  $log.scrollTop = $log.scrollHeight;
}

function logHeader(text: string) { log(text, "log-header"); }
function logPass(text: string) { log(text, "log-pass"); }
function logFail(text: string) { log(text, "log-fail"); }
function logWarn(text: string) { log(text, "log-warn"); }
function logInfo(text: string) { log(text, "log-info"); }
function logDim(text: string) { log(text, "log-dim"); }

// ── WAV Loader ──
async function loadWav(url: string): Promise<Float32Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  const buf = await res.arrayBuffer();

  const view = new DataView(buf);
  let offset = 12;
  while (offset < buf.byteLength - 8) {
    const chunkId = String.fromCharCode(
      view.getUint8(offset), view.getUint8(offset + 1),
      view.getUint8(offset + 2), view.getUint8(offset + 3),
    );
    const chunkSize = view.getUint32(offset + 4, true);
    if (chunkId === "data") {
      offset += 8;
      const samples = new Float32Array(chunkSize / 2);
      for (let i = 0; i < samples.length; i++) {
        const s = view.getInt16(offset + i * 2, true);
        samples[i] = s / 32768;
      }
      return samples;
    }
    offset += 8 + chunkSize;
    if (chunkSize % 2 !== 0) offset++;
  }
  throw new Error(`No data chunk found in ${url}`);
}

// ── Audio concatenation ──
function concatAudioWithGaps(clips: Float32Array[]): Float32Array {
  const gap = new Float32Array(GAP_SAMPLES); // silence
  let totalLen = 0;
  for (let i = 0; i < clips.length; i++) {
    totalLen += clips[i].length;
    if (i < clips.length - 1) totalLen += GAP_SAMPLES;
  }
  const result = new Float32Array(totalLen);
  let off = 0;
  for (let i = 0; i < clips.length; i++) {
    result.set(clips[i], off);
    off += clips[i].length;
    if (i < clips.length - 1) {
      result.set(gap, off);
      off += GAP_SAMPLES;
    }
  }
  return result;
}

// ── Worker Communication ──
let messageBuffer: WorkerOutbound[] = [];

function initWorker(): Promise<void> {
  return new Promise((resolve, reject) => {
    worker = new Worker(
      new URL("../src/worker/inference.ts", import.meta.url),
      { type: "module" },
    );

    const onReady = (e: MessageEvent<WorkerOutbound>) => {
      const msg = e.data;
      if (msg.type === "loading") {
        $status.textContent = `Loading model... ${msg.percent}%`;
        $progress.style.width = `${msg.percent}%`;
      } else if (msg.type === "loading_status") {
        $status.textContent = msg.message;
      } else if (msg.type === "ready") {
        $status.textContent = "Model ready";
        $progress.style.width = "100%";
        worker!.onmessage = (ev: MessageEvent<WorkerOutbound>) => {
          messageBuffer.push(ev.data);
        };
        resolve();
      } else if (msg.type === "error") {
        reject(new Error(msg.message));
      }
    };

    worker.onmessage = onReady;
    worker.onerror = (e) => reject(new Error(e.message));
    worker.postMessage({ type: "init" });
  });
}

/**
 * Wait until the worker is idle (no new messages for `quietMs` milliseconds).
 * This is critical — at 8x speed, audio is fed much faster than the worker can
 * process it. The worker may have dozens of inference calls queued up.
 */
async function waitForWorkerIdle(quietMs = 1500, maxWaitMs = 60000): Promise<void> {
  const start = Date.now();
  let lastMsgCount = messageBuffer.length;
  let quietSince = Date.now();

  while (Date.now() - start < maxWaitMs) {
    await new Promise((r) => setTimeout(r, 300));
    if (messageBuffer.length > lastMsgCount) {
      // New messages arrived — worker still processing
      lastMsgCount = messageBuffer.length;
      quietSince = Date.now();
    } else if (Date.now() - quietSince >= quietMs) {
      // No new messages for quietMs — worker is idle
      return;
    }
  }
}

async function resetWorker(): Promise<void> {
  // Wait for any in-flight processing to complete before resetting
  await waitForWorkerIdle(800, 30000);
  messageBuffer = [];
  worker?.postMessage({ type: "reset" });
  // Wait for reset to be processed
  await new Promise((r) => setTimeout(r, 300));
  messageBuffer = [];
}

/**
 * Feed audio to worker in fast chunks, then wait for processing.
 * Returns all messages received during feeding + wait period.
 */
async function feedAndCollect(
  samples: Float32Array,
): Promise<WorkerOutbound[]> {
  // Drain any leftover messages from previous run
  messageBuffer = [];
  await new Promise((r) => setTimeout(r, 50));
  messageBuffer = [];
  const chunkSize = Math.floor(16000 * CHUNK_MS / 1000);
  const delayMs = CHUNK_MS / SPEED_MULT;

  for (let offset = 0; offset < samples.length; offset += chunkSize) {
    if (shouldStop) break;
    const chunk = samples.slice(offset, offset + chunkSize);
    worker?.postMessage({ type: "audio", samples: chunk }, [chunk.buffer]);
    if (delayMs > 1) {
      await new Promise((r) => setTimeout(r, delayMs));
    }
  }

  // Wait for worker to actually finish all queued inference calls
  await waitForWorkerIdle();

  const collected = [...messageBuffer];
  messageBuffer = [];
  return collected;
}

// ── Test Result Types ──
interface AyahResult {
  surah: number;
  ayah: number;
  discovered: boolean;
  discoveredAs: string | null;
  discoveryConf: number;
  wordsCovered: number;
  wordsTotal: number;
}

interface ScenarioResult {
  name: string;
  results: AyahResult[];
  discoveryAccuracy: number;
  wordCoverage: number;
  avgTimeMs: number;
  totalTimeMs: number;
}

// ── Analyze messages for a set of expected ayahs ──
function analyzeMessages(
  messages: WorkerOutbound[],
  expectedAyahs: [number, number][],
): AyahResult[] {
  // Extract verse_match and word_progress messages
  const verseMatches = messages.filter((m) => m.type === "verse_match") as Extract<WorkerOutbound, { type: "verse_match" }>[];
  const wordProgresses = messages.filter((m) => m.type === "word_progress") as Extract<WorkerOutbound, { type: "word_progress" }>[];

  const results: AyahResult[] = [];

  for (const [surah, ayah] of expectedAyahs) {
    // Check if this ayah was discovered
    const match = verseMatches.find((m) => m.surah === surah && m.ayah === ayah);

    // Check word progress for this ayah
    const wps = wordProgresses.filter((m) => m.surah === surah && m.ayah === ayah);
    let maxWordIdx = -1;
    let totalWords = 0;
    for (const wp of wps) {
      totalWords = wp.total_words;
      const maxInMsg = Math.max(...wp.matched_indices);
      if (maxInMsg > maxWordIdx) maxWordIdx = maxInMsg;
    }

    results.push({
      surah,
      ayah,
      discovered: !!match,
      discoveredAs: match ? `${match.surah}:${match.ayah}` : null,
      discoveryConf: match?.confidence ?? 0,
      wordsCovered: maxWordIdx + 1,
      wordsTotal: totalWords,
    });
  }

  // Also check for false discoveries (verse_match for ayahs NOT in expected list)
  for (const vm of verseMatches) {
    const isExpected = expectedAyahs.some(([s, a]) => s === vm.surah && a === vm.ayah);
    if (!isExpected) {
      // Check if it was a wrong match for one of our expected ayahs
      // (e.g., expected 78:2 but got 78:3)
      // We don't add it as a separate result, but note it in logging
    }
  }

  return results;
}

// ── Load and concatenate audio for ayah list ──
async function loadConcatAudio(
  ayahs: [number, number][],
  reciter = "alafasy",
): Promise<{ audio: Float32Array; loadedAyahs: [number, number][] }> {
  const clips: Float32Array[] = [];
  const loaded: [number, number][] = [];

  for (const [surah, ayah] of ayahs) {
    try {
      const url = `/test/audio-samples/${surah}_${ayah}_${reciter}.wav`;
      const samples = await loadWav(url);
      clips.push(samples);
      loaded.push([surah, ayah]);
    } catch {
      logDim(`  Skipping ${surah}:${ayah} (no audio file)`);
    }
  }

  return {
    audio: clips.length > 0 ? concatAudioWithGaps(clips) : new Float32Array(0),
    loadedAyahs: loaded,
  };
}

// ── Scenario runner (concatenated mode) ──
async function runConcatScenario(
  name: string,
  ayahs: [number, number][],
): Promise<ScenarioResult> {
  logHeader(`${name} (${ayahs.length} ayahs, concatenated)`);
  const startTime = performance.now();

  await resetWorker();

  $status.textContent = `${name}: Loading audio...`;
  const { audio, loadedAyahs } = await loadConcatAudio(ayahs);

  if (audio.length === 0) {
    logFail("  No audio files found");
    return { name, results: [], discoveryAccuracy: 0, wordCoverage: 0, avgTimeMs: 0, totalTimeMs: 0 };
  }

  const durationSec = audio.length / 16000;
  logInfo(`  Audio: ${durationSec.toFixed(1)}s (${loadedAyahs.length} clips)`);

  $status.textContent = `${name}: Feeding ${durationSec.toFixed(0)}s audio at ${SPEED_MULT}x...`;
  $progress.style.width = "50%";

  const messages = await feedAndCollect(audio);
  const totalTimeMs = performance.now() - startTime;

  $progress.style.width = "100%";

  // Analyze
  const results = analyzeMessages(messages, loadedAyahs);
  const discovered = results.filter((r) => r.discovered).length;
  const discoveryAccuracy = loadedAyahs.length > 0 ? discovered / loadedAyahs.length : 0;

  const withWords = results.filter((r) => r.wordsTotal > 0);
  const wordCoverage = withWords.length > 0
    ? withWords.reduce((s, r) => s + r.wordsCovered / r.wordsTotal, 0) / withWords.length
    : 0;

  // Log individual results
  for (const r of results) {
    const icon = r.discovered ? "PASS" : "FAIL";
    const cls = r.discovered ? "log-pass" : "log-fail";
    const wordInfo = r.wordsTotal > 0 ? ` | Words: ${r.wordsCovered}/${r.wordsTotal}` : "";
    log(
      `  [${icon}] ${r.surah}:${r.ayah} → ${r.discoveredAs ?? "none"} (${(r.discoveryConf * 100).toFixed(0)}%)${wordInfo}`,
      cls,
    );
  }

  // Log messages summary
  const msgTypes: Record<string, number> = {};
  for (const m of messages) {
    msgTypes[m.type] = (msgTypes[m.type] || 0) + 1;
  }
  logDim(`  Messages: ${JSON.stringify(msgTypes)}`);

  // Log actual verse_match targets for debugging
  const allVM = messages.filter((m) => m.type === "verse_match") as Extract<WorkerOutbound, { type: "verse_match" }>[];
  const vmTargets = allVM.map((m) => `${m.surah}:${m.ayah}(${(m.confidence * 100).toFixed(0)}%)`);
  if (vmTargets.length > 0) {
    logDim(`  Matched: ${vmTargets.join(" → ")}`);
  }

  // Log raw transcripts for context
  const rawTs = messages.filter((m) => m.type === "raw_transcript") as Extract<WorkerOutbound, { type: "raw_transcript" }>[];
  if (rawTs.length > 0) {
    logDim(`  Transcripts: ${rawTs.map((r) => `"${r.text}"(${(r.confidence * 100).toFixed(0)}%)`).join(", ")}`);
  }

  // Summary line
  log("");
  const summaryLine =
    `  Summary: Discovery ${(discoveryAccuracy * 100).toFixed(1)}% (${discovered}/${loadedAyahs.length}) | ` +
    `Word Coverage ${(wordCoverage * 100).toFixed(1)}% | ` +
    `Time ${(totalTimeMs / 1000).toFixed(1)}s`;
  log(summaryLine, discoveryAccuracy >= 0.8 ? "log-pass" : "log-warn");
  log("");

  return {
    name,
    results,
    discoveryAccuracy,
    wordCoverage,
    avgTimeMs: loadedAyahs.length > 0 ? totalTimeMs / loadedAyahs.length : 0,
    totalTimeMs,
  };
}

// ── Scenario runner (individual mode — with reset between each) ──
async function runIndividualScenario(
  name: string,
  ayahs: [number, number][],
): Promise<ScenarioResult> {
  logHeader(`${name} (${ayahs.length} ayahs, individual)`);
  const startTime = performance.now();
  const results: AyahResult[] = [];
  let completed = 0;

  for (const [surah, ayah] of ayahs) {
    if (shouldStop) break;

    await resetWorker();

    $status.textContent = `${name}: ${surah}:${ayah} (${completed + 1}/${ayahs.length})`;
    $progress.style.width = `${((completed + 1) / ayahs.length * 100).toFixed(0)}%`;

    let samples: Float32Array;
    try {
      samples = await loadWav(`/test/audio-samples/${surah}_${ayah}_alafasy.wav`);
    } catch {
      results.push({
        surah, ayah,
        discovered: false, discoveredAs: null, discoveryConf: 0,
        wordsCovered: 0, wordsTotal: 0,
      });
      logDim(`  Skipping ${surah}:${ayah} (no audio)`);
      completed++;
      continue;
    }

    const messages = await feedAndCollect(samples);
    const ayahResults = analyzeMessages(messages, [[surah, ayah]]);
    const r = ayahResults[0];
    results.push(r);

    const icon = r.discovered ? "PASS" : "FAIL";
    const cls = r.discovered ? "log-pass" : "log-fail";
    const wordInfo = r.wordsTotal > 0 ? ` | Words: ${r.wordsCovered}/${r.wordsTotal}` : "";
    log(`  [${icon}] ${surah}:${ayah} → ${r.discoveredAs ?? "none"} (${(r.discoveryConf * 100).toFixed(0)}%)${wordInfo}`, cls);

    if (!r.discovered) {
      const rawTs = messages.filter((m) => m.type === "raw_transcript") as any[];
      if (rawTs.length > 0) {
        logDim(`    Heard: "${rawTs[rawTs.length - 1].text}"`);
      }
    }

    completed++;
  }

  const totalTimeMs = performance.now() - startTime;
  const discovered = results.filter((r) => r.discovered).length;
  const discoveryAccuracy = results.length > 0 ? discovered / results.length : 0;
  const withWords = results.filter((r) => r.wordsTotal > 0);
  const wordCoverage = withWords.length > 0
    ? withWords.reduce((s, r) => s + r.wordsCovered / r.wordsTotal, 0) / withWords.length
    : 0;

  log("");
  const summaryLine =
    `  Summary: Discovery ${(discoveryAccuracy * 100).toFixed(1)}% (${discovered}/${results.length}) | ` +
    `Word Coverage ${(wordCoverage * 100).toFixed(1)}% | ` +
    `Time ${(totalTimeMs / 1000).toFixed(1)}s`;
  log(summaryLine, discoveryAccuracy >= 0.8 ? "log-pass" : "log-warn");
  log("");

  return { name, results, discoveryAccuracy, wordCoverage, avgTimeMs: results.length > 0 ? totalTimeMs / results.length : 0, totalTimeMs };
}

// ── Ayah list builder ──
function makeAyahList(ranges: [number, number, number][]): [number, number][] {
  const list: [number, number][] = [];
  for (const [surah, startAyah, endAyah] of ranges) {
    for (let a = startAyah; a <= endAyah; a++) {
      list.push([surah, a]);
    }
  }
  return list;
}

// ── Scenario Definitions ──
const SCENARIOS: Record<string, () => Promise<ScenarioResult>> = {
  "fatihah": () =>
    runConcatScenario("Al-Fatihah (Sequential)", makeAyahList([[1, 1, 7]])),

  "short-surahs": () =>
    runConcatScenario("Short Surahs (112-114)", makeAyahList([
      [112, 1, 4], [113, 1, 5], [114, 1, 6],
    ])),

  "skip": () =>
    runConcatScenario("Skip Test (1:1-3, skip, 1:5-7)", [
      [1, 1], [1, 2], [1, 3], [1, 5], [1, 6], [1, 7],
    ]),

  "surah-jump": () =>
    runConcatScenario("Surah Jump (112 → 114)", [
      [112, 1], [112, 2], [114, 1], [114, 2], [114, 3],
    ]),

  "long-verse": () =>
    runConcatScenario("Long Verse (2:255 Ayat al-Kursi)", [[2, 255]]),

  "batch-50": () =>
    runIndividualScenario("Batch 50 (independent)", [
      [112, 1], [112, 2], [112, 3], [112, 4],
      [113, 1], [113, 3], [113, 5],
      [114, 1], [114, 4], [114, 6],
      [1, 1], [1, 2], [1, 4], [1, 7],
      [36, 1], [36, 2], [36, 3], [36, 4], [36, 5],
      [55, 1], [55, 2], [55, 3], [55, 13], [55, 26], [55, 27],
      [67, 1], [67, 2], [67, 3],
      [78, 1], [78, 2], [78, 3], [78, 31], [78, 40],
      [2, 1], [2, 2], [2, 255], [2, 285], [2, 286],
      [19, 1], [19, 2], [19, 3],
      [18, 1], [18, 10], [18, 109], [18, 110],
      [93, 1], [93, 2], [93, 3], [93, 11],
      [94, 1], [94, 2], [94, 3],
      [103, 1], [103, 2], [103, 3],
    ]),

  "full-juz-amma": async () => {
    const juzAmma: Record<number, number> = {
      78: 40, 79: 46, 80: 42, 81: 29, 82: 19, 83: 36, 84: 25, 85: 22,
      86: 17, 87: 19, 88: 26, 89: 30, 90: 20, 91: 15, 92: 21, 93: 11,
      94: 8, 95: 8, 96: 19, 97: 5, 98: 8, 99: 8, 100: 11, 101: 11,
      102: 8, 103: 3, 104: 9, 105: 5, 106: 4, 107: 7, 108: 3, 109: 6,
      110: 3, 111: 5, 112: 4, 113: 5, 114: 6,
    };

    const totalAyahs = Object.values(juzAmma).reduce((s, n) => s + n, 0);
    logHeader(`Full Juz Amma: ${totalAyahs} ayahs across ${Object.keys(juzAmma).length} surahs`);

    const allResults: AyahResult[] = [];
    const startTime = performance.now();

    for (const [surahNum, ayahCount] of Object.entries(juzAmma)) {
      if (shouldStop) break;
      const s = parseInt(surahNum);
      const surahAyahs = makeAyahList([[s, 1, ayahCount]]);
      const result = await runConcatScenario(`Surah ${s}`, surahAyahs);
      allResults.push(...result.results);
    }

    const totalTimeMs = performance.now() - startTime;
    const discovered = allResults.filter((r) => r.discovered).length;
    const total = allResults.length;
    const withWords = allResults.filter((r) => r.wordsTotal > 0);
    const wordCoverage = withWords.length > 0
      ? withWords.reduce((s, r) => s + r.wordsCovered / r.wordsTotal, 0) / withWords.length
      : 0;

    return {
      name: "Full Juz Amma",
      results: allResults,
      discoveryAccuracy: total > 0 ? discovered / total : 0,
      wordCoverage,
      avgTimeMs: total > 0 ? totalTimeMs / total : 0,
      totalTimeMs,
    };
  },
};

// ── Run all scenarios ──
async function runAllScenarios(): Promise<void> {
  const allResults: ScenarioResult[] = [];
  const scenarioNames = ["fatihah", "short-surahs", "skip", "surah-jump", "long-verse", "batch-50"];

  for (const name of scenarioNames) {
    if (shouldStop) break;
    const result = await SCENARIOS[name]();
    allResults.push(result);
  }

  showOverallSummary(allResults);
}

function showOverallSummary(results: ScenarioResult[]) {
  logHeader("OVERALL RESULTS");

  const totalAyahs = results.reduce((s, r) => s + r.results.length, 0);
  const totalCorrect = results.reduce((s, r) => s + r.results.filter((a) => a.discovered).length, 0);
  const overallDiscovery = totalAyahs > 0 ? totalCorrect / totalAyahs : 0;

  const allWithWords = results.flatMap((r) => r.results.filter((a) => a.wordsTotal > 0));
  const overallWordCov = allWithWords.length > 0
    ? allWithWords.reduce((s, a) => s + a.wordsCovered / a.wordsTotal, 0) / allWithWords.length
    : 0;
  const totalTimeMs = results.reduce((s, r) => s + r.totalTimeMs, 0);

  for (const r of results) {
    const icon = r.discoveryAccuracy >= 0.8 ? "PASS" : r.discoveryAccuracy >= 0.5 ? "WARN" : "FAIL";
    const cls = icon === "PASS" ? "log-pass" : icon === "WARN" ? "log-warn" : "log-fail";
    const discovered = r.results.filter((a) => a.discovered).length;
    log(
      `  [${icon}] ${r.name}: Discovery ${(r.discoveryAccuracy * 100).toFixed(1)}% (${discovered}/${r.results.length}) | Words ${(r.wordCoverage * 100).toFixed(1)}% | ${(r.totalTimeMs / 1000).toFixed(1)}s`,
      cls,
    );
  }

  log("");
  log(
    `  TOTAL: ${totalAyahs} ayahs | Discovery ${(overallDiscovery * 100).toFixed(1)}% | Words ${(overallWordCov * 100).toFixed(1)}% | Time ${(totalTimeMs / 1000).toFixed(1)}s`,
    overallDiscovery >= 0.7 ? "log-pass" : "log-warn",
  );

  $summary.style.display = "block";
  $summary.className = overallDiscovery >= 0.7 ? "" : "fail";
  $summary.innerHTML = `
    <h3>${overallDiscovery >= 0.7 ? "Benchmark Passed" : "Benchmark Needs Work"}</h3>
    <div class="summary-grid">
      <div class="stat">Ayahs tested: <span class="stat-val">${totalAyahs}</span></div>
      <div class="stat">Discovery: <span class="stat-val">${(overallDiscovery * 100).toFixed(1)}%</span></div>
      <div class="stat">Word Coverage: <span class="stat-val">${(overallWordCov * 100).toFixed(1)}%</span></div>
      <div class="stat">Total time: <span class="stat-val">${(totalTimeMs / 1000).toFixed(1)}s</span></div>
      <div class="stat">Scenarios: <span class="stat-val">${results.length}</span></div>
    </div>
  `;
}

// ── Init ──
async function warmupModel(): Promise<void> {
  // The ONNX runtime needs a few inference passes to warm up JIT compilation
  // and buffer allocation. Without this, the first real scenario produces garbage.
  logDim("  Warming up model (3 dummy inference passes)...");
  const dummySamples = new Float32Array(16000 * 3); // 3s of silence
  // Fill with very low-level noise to avoid silence skip
  for (let i = 0; i < dummySamples.length; i++) {
    dummySamples[i] = (Math.random() - 0.5) * 0.01;
  }

  // Send 3 chunks to trigger 3 inference cycles
  for (let pass = 0; pass < 3; pass++) {
    const chunk = dummySamples.slice(pass * 16000, (pass + 1) * 16000);
    worker?.postMessage({ type: "audio", samples: chunk }, [chunk.buffer]);
    await new Promise((r) => setTimeout(r, 200));
  }

  // Wait for all warmup inferences to complete
  await waitForWorkerIdle(1000, 15000);

  // Reset tracker state so warmup doesn't affect real tests
  messageBuffer = [];
  worker?.postMessage({ type: "reset" });
  await new Promise((r) => setTimeout(r, 300));
  messageBuffer = [];

  logDim("  Warmup complete.");
}

async function startup() {
  $status.textContent = "Initializing worker...";
  try {
    await initWorker();
    await warmupModel();
    $btnRunAll.disabled = false;
    $btnRunSelected.disabled = false;
    logInfo("Model loaded and warmed up. Ready to benchmark.");
  } catch (err) {
    logFail(`Failed to init: ${err}`);
    $status.textContent = `Error: ${err}`;
  }
}

$btnRunAll.addEventListener("click", async () => {
  if (isRunning) return;
  isRunning = true;
  shouldStop = false;
  $btnRunAll.disabled = true;
  $btnRunSelected.disabled = true;
  $btnRunAll.classList.add("running");
  $log.innerHTML = "";
  $summary.style.display = "none";

  try { await runAllScenarios(); } catch (err) { logFail(`Error: ${err}`); }

  isRunning = false;
  $btnRunAll.disabled = false;
  $btnRunSelected.disabled = false;
  $btnRunAll.classList.remove("running");
  $status.textContent = "Done";
});

$btnRunSelected.addEventListener("click", async () => {
  if (isRunning) return;
  const selected = $scenarioSelect.value;
  isRunning = true;
  shouldStop = false;
  $btnRunAll.disabled = true;
  $btnRunSelected.disabled = true;
  $btnRunSelected.classList.add("running");
  $log.innerHTML = "";
  $summary.style.display = "none";

  try {
    if (selected === "all") {
      await runAllScenarios();
    } else if (SCENARIOS[selected]) {
      const result = await SCENARIOS[selected]();
      showOverallSummary([result]);
    }
  } catch (err) { logFail(`Error: ${err}`); }

  isRunning = false;
  $btnRunAll.disabled = false;
  $btnRunSelected.disabled = false;
  $btnRunSelected.classList.remove("running");
  $status.textContent = "Done";
});

$btnStop.addEventListener("click", () => {
  shouldStop = true;
  $status.textContent = "Stopping...";
  logWarn("Stop requested — finishing current test...");
});

$btnRunAll.disabled = true;
$btnRunSelected.disabled = true;

startup();
