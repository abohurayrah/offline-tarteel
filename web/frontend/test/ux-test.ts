// ---------------------------------------------------------------------------
// UX Test Suite — Simulates real user reading experience
// ---------------------------------------------------------------------------
// Tests real-world scenarios:
//   - Starting from middle of an ayah
//   - Word-by-word progression monotonicity
//   - Verse oscillation detection (bouncing between verses)
//   - Sequential smooth reading
//   - Long verse tracking
//   - Verse complete detection
//   - Mid-page start behavior
//   - Continuation after pause
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
const SPEED_MULT = 8;
const CHUNK_MS = 100;
const GAP_SAMPLES = 3200; // 200ms silence

// ── State ──
let worker: Worker | null = null;
let isRunning = false;
let shouldStop = false;
let messageBuffer: WorkerOutbound[] = [];

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

// ── WAV Loader (same as benchmark) ──
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

function concatAudioWithGaps(clips: Float32Array[]): Float32Array {
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
      off += GAP_SAMPLES; // silence gap (zeros)
    }
  }
  return result;
}

// ── Worker Communication (reused from benchmark) ──
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

async function waitForWorkerIdle(quietMs = 1500, maxWaitMs = 60000): Promise<void> {
  const start = Date.now();
  let lastMsgCount = messageBuffer.length;
  let quietSince = Date.now();
  while (Date.now() - start < maxWaitMs) {
    await new Promise((r) => setTimeout(r, 300));
    if (messageBuffer.length > lastMsgCount) {
      lastMsgCount = messageBuffer.length;
      quietSince = Date.now();
    } else if (Date.now() - quietSince >= quietMs) {
      return;
    }
  }
}

async function resetWorker(): Promise<void> {
  await waitForWorkerIdle(2500, 30000);
  messageBuffer = [];
  worker?.postMessage({ type: "reset" });
  await new Promise((r) => setTimeout(r, 500));
  messageBuffer = [];
  await waitForWorkerIdle(800, 5000);
  messageBuffer = [];
}

async function feedAndCollect(samples: Float32Array): Promise<WorkerOutbound[]> {
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
  await waitForWorkerIdle();
  const collected = [...messageBuffer];
  messageBuffer = [];
  return collected;
}

// ── Analysis Functions ──

interface VerseMatchEvent {
  surah: number;
  ayah: number;
  confidence: number;
  index: number; // position in message stream
}

interface WordProgressEvent {
  surah: number;
  ayah: number;
  word_index: number;
  total_words: number;
  matched_indices: number[];
  index: number;
}

function extractVerseMatches(messages: WorkerOutbound[]): VerseMatchEvent[] {
  return messages
    .map((m, i) => m.type === "verse_match" ? { surah: m.surah, ayah: m.ayah, confidence: m.confidence, index: i } : null)
    .filter((x): x is VerseMatchEvent => x !== null);
}

function extractWordProgress(messages: WorkerOutbound[]): WordProgressEvent[] {
  return messages
    .map((m, i) => m.type === "word_progress" ? {
      surah: m.surah, ayah: m.ayah,
      word_index: m.word_index, total_words: m.total_words,
      matched_indices: m.matched_indices, index: i,
    } : null)
    .filter((x): x is WordProgressEvent => x !== null);
}

/**
 * Detect verse oscillation: A→B→A pattern (bouncing between verses).
 * Returns array of bounce events.
 */
function detectOscillation(matches: VerseMatchEvent[]): { from: string; to: string; back: string; count: number }[] {
  const bounces: { from: string; to: string; back: string; count: number }[] = [];
  if (matches.length < 3) return bounces;

  for (let i = 2; i < matches.length; i++) {
    const a = `${matches[i - 2].surah}:${matches[i - 2].ayah}`;
    const b = `${matches[i - 1].surah}:${matches[i - 1].ayah}`;
    const c = `${matches[i].surah}:${matches[i].ayah}`;

    if (a === c && a !== b) {
      // A→B→A bounce detected
      const existing = bounces.find((x) => x.from === a && x.to === b);
      if (existing) {
        existing.count++;
      } else {
        bounces.push({ from: a, to: b, back: c, count: 1 });
      }
    }
  }
  return bounces;
}

/**
 * Check if word indices advance monotonically for a given verse.
 * Returns { monotonic, regressions } where regressions are (fromIdx, toIdx) pairs.
 */
function checkWordMonotonicity(
  wordEvents: WordProgressEvent[],
  surah: number,
  ayah: number,
): { monotonic: boolean; regressions: [number, number][]; maxIdx: number; totalWords: number } {
  const filtered = wordEvents.filter((e) => e.surah === surah && e.ayah === ayah);
  if (filtered.length === 0) return { monotonic: true, regressions: [], maxIdx: -1, totalWords: 0 };

  let maxSeen = -1;
  const regressions: [number, number][] = [];
  let totalWords = 0;

  for (const e of filtered) {
    totalWords = e.total_words;
    if (e.word_index < maxSeen) {
      regressions.push([maxSeen, e.word_index]);
    }
    if (e.word_index > maxSeen) maxSeen = e.word_index;
  }

  return {
    monotonic: regressions.length === 0,
    regressions,
    maxIdx: maxSeen,
    totalWords,
  };
}

/**
 * Calculate word coverage for a specific verse from word_progress messages.
 * Uses the accumulated matched_indices (union of all events).
 */
function computeWordCoverage(
  wordEvents: WordProgressEvent[],
  surah: number,
  ayah: number,
): { covered: number; total: number; percentage: number; indices: number[] } {
  const filtered = wordEvents.filter((e) => e.surah === surah && e.ayah === ayah);
  if (filtered.length === 0) return { covered: 0, total: 0, percentage: 0, indices: [] };

  const allIndices = new Set<number>();
  let total = 0;
  for (const e of filtered) {
    total = e.total_words;
    for (const idx of e.matched_indices) {
      allIndices.add(idx);
    }
  }

  const indices = Array.from(allIndices).sort((a, b) => a - b);
  return {
    covered: allIndices.size,
    total,
    percentage: total > 0 ? allIndices.size / total : 0,
    indices,
  };
}

/**
 * Check for contiguous word progression from index 0.
 * Returns max contiguous index.
 */
function maxContiguousFromZero(indices: number[]): number {
  let max = -1;
  for (let i = 0; i < indices.length; i++) {
    if (indices.includes(i)) max = i;
    else break;
  }
  return max;
}

/**
 * Discovery latency: how many messages arrive before the correct verse_match.
 */
function discoveryLatency(messages: WorkerOutbound[], surah: number, ayah: number): number {
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    if (m.type === "verse_match" && m.surah === surah && m.ayah === ayah) {
      return i;
    }
  }
  return -1; // never discovered
}

// ── Test Result Types ──

interface Criterion {
  name: string;
  pass: boolean;
  detail: string;
}

interface UXTestResult {
  name: string;
  description: string;
  criteria: Criterion[];
  messages: WorkerOutbound[];
  passed: boolean;
  timeMs: number;
}

// ── Render timeline ──
function renderTimeline(messages: WorkerOutbound[], expectedVerse?: { surah: number; ayah: number }): HTMLElement {
  const container = document.createElement("div");
  container.className = "timeline";

  const matches = extractVerseMatches(messages);
  const words = extractWordProgress(messages);
  const raws = messages
    .map((m, i) => m.type === "raw_transcript" ? { text: m.text, confidence: m.confidence, index: i } : null)
    .filter((x): x is { text: string; confidence: number; index: number } => x !== null);

  // Verse match row
  if (matches.length > 0) {
    const row = document.createElement("div");
    row.className = "timeline-row";
    const label = document.createElement("div");
    label.className = "timeline-label";
    label.textContent = "verse_match";
    row.appendChild(label);

    const events = document.createElement("div");
    events.className = "timeline-events";
    for (const m of matches) {
      const ev = document.createElement("span");
      const isCorrect = !expectedVerse || (m.surah === expectedVerse.surah && m.ayah === expectedVerse.ayah);
      ev.className = `tl-event tl-match${isCorrect ? "" : " wrong"}`;
      ev.textContent = `${m.surah}:${m.ayah} (${(m.confidence * 100).toFixed(0)}%)`;
      events.appendChild(ev);
    }
    row.appendChild(events);
    container.appendChild(row);
  }

  // Word progress row
  if (words.length > 0) {
    const row = document.createElement("div");
    row.className = "timeline-row";
    const label = document.createElement("div");
    label.className = "timeline-label";
    label.textContent = "word_progress";
    row.appendChild(label);

    const events = document.createElement("div");
    events.className = "timeline-events";
    let lastWordIdx = -1;
    for (const w of words) {
      const ev = document.createElement("span");
      const isRegress = w.word_index < lastWordIdx;
      ev.className = `tl-event tl-word${isRegress ? " regress" : ""}`;
      ev.textContent = `${w.surah}:${w.ayah} w${w.word_index}/${w.total_words} [${w.matched_indices.join(",")}]`;
      events.appendChild(ev);
      lastWordIdx = w.word_index;
    }
    row.appendChild(events);
    container.appendChild(row);
  }

  // Oscillation detection row
  const bounces = detectOscillation(matches);
  if (bounces.length > 0) {
    const row = document.createElement("div");
    row.className = "timeline-row";
    const label = document.createElement("div");
    label.className = "timeline-label";
    label.textContent = "BOUNCES";
    row.appendChild(label);

    const events = document.createElement("div");
    events.className = "timeline-events";
    for (const b of bounces) {
      const ev = document.createElement("span");
      ev.className = "tl-event tl-bounce";
      ev.textContent = `${b.from}→${b.to}→${b.from} (x${b.count})`;
      events.appendChild(ev);
    }
    row.appendChild(events);
    container.appendChild(row);
  }

  // Raw transcript row (last 5)
  if (raws.length > 0) {
    const row = document.createElement("div");
    row.className = "timeline-row";
    const label = document.createElement("div");
    label.className = "timeline-label";
    label.textContent = "transcripts";
    row.appendChild(label);

    const events = document.createElement("div");
    events.className = "timeline-events";
    const shown = raws.slice(-5);
    for (const r of shown) {
      const ev = document.createElement("span");
      ev.className = "tl-event tl-raw";
      ev.textContent = `"${r.text.slice(0, 30)}${r.text.length > 30 ? "..." : ""}" (${(r.confidence * 100).toFixed(0)}%)`;
      events.appendChild(ev);
    }
    row.appendChild(events);
    container.appendChild(row);
  }

  return container;
}

// ── Render test result card ──
function renderResultCard(result: UXTestResult): HTMLElement {
  const card = document.createElement("div");
  const allPass = result.criteria.every((c) => c.pass);
  const anyFail = result.criteria.some((c) => !c.pass);
  card.className = `summary-card ${allPass ? "pass" : anyFail ? "fail" : "warn"}`;

  const h3 = document.createElement("h3");
  h3.textContent = `${allPass ? "PASS" : "FAIL"} — ${result.name} (${(result.timeMs / 1000).toFixed(1)}s)`;
  card.appendChild(h3);

  const desc = document.createElement("div");
  desc.className = "criteria";
  desc.style.color = "#888";
  desc.textContent = result.description;
  card.appendChild(desc);

  for (const c of result.criteria) {
    const div = document.createElement("div");
    div.className = `criteria ${c.pass ? "pass" : "fail"}`;
    div.textContent = `  ${c.pass ? "PASS" : "FAIL"} ${c.name}: ${c.detail}`;
    card.appendChild(div);
  }

  // Add timeline
  card.appendChild(renderTimeline(result.messages));

  return card;
}

// ── Load audio helper ──
async function loadAudioFile(surah: number, ayah: number, variant = ""): Promise<Float32Array> {
  const suffix = variant ? `_${variant}` : "";
  const url = `/test/audio-samples/${surah}_${ayah}_alafasy${suffix}.wav`;
  return loadWav(url);
}

// ── Test Scenarios ──

/**
 * Test 1: Mid-Ayah Start (18:13)
 * User starts reading from the middle of 18:13 — should still identify correct verse.
 */
async function testMidStart(): Promise<UXTestResult> {
  const name = "Mid-Ayah Start (18:13)";
  const description = "User starts reading from middle of an ayah — should identify correct verse without oscillation";
  const start = performance.now();

  await resetWorker();

  const audio = await loadAudioFile(18, 13, "mid50");
  const messages = await feedAndCollect(audio);

  const matches = extractVerseMatches(messages);
  const bounces = detectOscillation(matches);
  const words = extractWordProgress(messages);

  // Criteria
  const criteria: Criterion[] = [];

  // 1. Should discover 18:13 (or at least a verse in surah 18)
  const found18_13 = matches.some((m) => m.surah === 18 && m.ayah === 13);
  const foundSurah18 = matches.some((m) => m.surah === 18);
  criteria.push({
    name: "Discovers 18:13",
    pass: found18_13,
    detail: found18_13
      ? `Found 18:13 (${matches.filter((m) => m.surah === 18 && m.ayah === 13).length} times)`
      : foundSurah18
        ? `Found surah 18 but not ayah 13 — matched: ${matches.map((m) => `${m.surah}:${m.ayah}`).join(", ")}`
        : `No surah 18 match — found: ${matches.map((m) => `${m.surah}:${m.ayah}`).join(", ") || "nothing"}`,
  });

  // 2. No oscillation (A→B→A bouncing)
  criteria.push({
    name: "No verse oscillation",
    pass: bounces.length === 0,
    detail: bounces.length === 0
      ? "Clean — no bouncing detected"
      : `${bounces.length} bounce pattern(s): ${bounces.map((b) => `${b.from}↔${b.to} x${b.count}`).join(", ")}`,
  });

  // 3. Word progress should exist for identified verse
  const hasWordProgress = words.length > 0;
  criteria.push({
    name: "Has word progress",
    pass: hasWordProgress,
    detail: hasWordProgress
      ? `${words.length} word_progress events`
      : "No word progress events received",
  });

  // 4. Verse matches should stabilize (not keep re-matching same verse)
  const uniqueMatches = new Set(matches.map((m) => `${m.surah}:${m.ayah}`));
  const isStable = uniqueMatches.size <= 2; // allow 1 wrong then correct, or just correct
  criteria.push({
    name: "Verse identification stable",
    pass: isStable,
    detail: `${uniqueMatches.size} unique verses matched: ${[...uniqueMatches].join(", ")}`,
  });

  const passed = criteria.every((c) => c.pass);
  return { name, description, criteria, messages, passed, timeMs: performance.now() - start };
}

/**
 * Test 2: Word Progression (1:1-3)
 * Full ayahs — words should progress monotonically.
 */
async function testWordProgression(): Promise<UXTestResult> {
  const name = "Word Progression (1:1-3)";
  const description = "Full ayah playback — word indices should advance monotonically without regression";
  const start = performance.now();

  await resetWorker();

  const clips: Float32Array[] = [];
  const ayahs: [number, number][] = [[1, 1], [1, 2], [1, 3]];
  for (const [s, a] of ayahs) {
    try {
      clips.push(await loadAudioFile(s, a));
    } catch { logDim(`  Skipping ${s}:${a} (no audio)`); }
  }

  const audio = concatAudioWithGaps(clips);
  const messages = await feedAndCollect(audio);
  const words = extractWordProgress(messages);
  const matches = extractVerseMatches(messages);

  const criteria: Criterion[] = [];

  // 1. At least 1:1 should be discovered
  const found1_1 = matches.some((m) => m.surah === 1 && m.ayah === 1);
  criteria.push({
    name: "Discovers 1:1",
    pass: found1_1,
    detail: found1_1
      ? `Found 1:1 (conf: ${(matches.find((m) => m.surah === 1 && m.ayah === 1)?.confidence ?? 0 * 100).toFixed(0)}%)`
      : `Not found — matched: ${matches.map((m) => `${m.surah}:${m.ayah}`).join(", ") || "nothing"}`,
  });

  // 2. Word monotonicity per verse
  for (const [s, a] of ayahs) {
    const mono = checkWordMonotonicity(words, s, a);
    if (mono.totalWords > 0) {
      criteria.push({
        name: `Words monotonic ${s}:${a}`,
        pass: mono.monotonic,
        detail: mono.monotonic
          ? `Advanced to word ${mono.maxIdx}/${mono.totalWords}`
          : `${mono.regressions.length} regressions: ${mono.regressions.map(([f, t]) => `${f}→${t}`).join(", ")}`,
      });
    }
  }

  // 3. Overall word coverage for discovered verses
  let totalCov = 0;
  let covCount = 0;
  for (const [s, a] of ayahs) {
    const cov = computeWordCoverage(words, s, a);
    if (cov.total > 0) {
      totalCov += cov.percentage;
      covCount++;
    }
  }
  const avgCov = covCount > 0 ? totalCov / covCount : 0;
  criteria.push({
    name: "Word coverage ≥ 50%",
    pass: avgCov >= 0.5,
    detail: `Average word coverage: ${(avgCov * 100).toFixed(1)}% across ${covCount} tracked verse(s)`,
  });

  const passed = criteria.every((c) => c.pass);
  return { name, description, criteria, messages, passed, timeMs: performance.now() - start };
}

/**
 * Test 3: Verse Oscillation Detection (18:13 mid + 18:14)
 * The user-reported bug: reading from middle of 18:13 then 18:14,
 * tracker bounces between them.
 */
async function testVerseOscillation(): Promise<UXTestResult> {
  const name = "Verse Oscillation (18:13+14)";
  const description = "Reproduces reported bug: mid-ayah start on 18:13 then 18:14 — should not bounce between verses";
  const start = performance.now();

  await resetWorker();

  // Feed mid-50% of 18:13 followed by full 18:14
  const clip13 = await loadAudioFile(18, 13, "mid50");
  const clip14 = await loadAudioFile(18, 14);
  const audio = concatAudioWithGaps([clip13, clip14]);
  const messages = await feedAndCollect(audio);

  const matches = extractVerseMatches(messages);
  const bounces = detectOscillation(matches);

  const criteria: Criterion[] = [];

  // 1. No oscillation
  criteria.push({
    name: "No verse oscillation",
    pass: bounces.length === 0,
    detail: bounces.length === 0
      ? "No A→B→A bouncing detected"
      : `OSCILLATION: ${bounces.map((b) => `${b.from}↔${b.to} x${b.count}`).join(", ")}`,
  });

  // 2. Should eventually settle on a verse (not keep changing)
  const lastThree = matches.slice(-3);
  const lastThreeUnique = new Set(lastThree.map((m) => `${m.surah}:${m.ayah}`));
  const settles = lastThreeUnique.size <= 1 || matches.length <= 2;
  criteria.push({
    name: "Settles on a verse",
    pass: settles,
    detail: settles
      ? `Final matches: ${lastThree.map((m) => `${m.surah}:${m.ayah}`).join(" → ")}`
      : `Still unstable — last 3: ${lastThree.map((m) => `${m.surah}:${m.ayah}`).join(" → ")}`,
  });

  // 3. At least discovers a surah 18 verse
  const foundS18 = matches.some((m) => m.surah === 18);
  criteria.push({
    name: "Discovers surah 18",
    pass: foundS18,
    detail: foundS18
      ? `Matched: ${matches.map((m) => `${m.surah}:${m.ayah}`).join(", ")}`
      : `No surah 18 — found: ${matches.map((m) => `${m.surah}:${m.ayah}`).join(", ") || "nothing"}`,
  });

  // 4. Total verse_match count should be reasonable (≤6 for 2 ayahs)
  const reasonableCount = matches.length <= 6;
  criteria.push({
    name: "Reasonable match count (≤6)",
    pass: reasonableCount,
    detail: `${matches.length} verse_match events`,
  });

  const passed = criteria.every((c) => c.pass);
  return { name, description, criteria, messages, passed, timeMs: performance.now() - start };
}

/**
 * Test 4: Smooth Sequential Reading (Surah 112)
 * Full surah — verse_match should advance 112:1 → 112:2 → 112:3 → 112:4
 * without backtracking.
 */
async function testSmoothSequential(): Promise<UXTestResult> {
  const name = "Smooth Sequential (112)";
  const description = "Full surah recitation — verse progression should be monotonic without backtracking";
  const start = performance.now();

  await resetWorker();

  const clips: Float32Array[] = [];
  const ayahs: [number, number][] = [[112, 1], [112, 2], [112, 3], [112, 4]];
  for (const [s, a] of ayahs) {
    clips.push(await loadAudioFile(s, a));
  }

  const audio = concatAudioWithGaps(clips);
  const messages = await feedAndCollect(audio);

  const matches = extractVerseMatches(messages);
  const bounces = detectOscillation(matches);

  const criteria: Criterion[] = [];

  // 1. Discovers at least 3/4 ayahs
  const discoveredAyahs = new Set(
    matches.filter((m) => m.surah === 112).map((m) => m.ayah),
  );
  criteria.push({
    name: "Discovers ≥3/4 ayahs",
    pass: discoveredAyahs.size >= 3,
    detail: `Found ${discoveredAyahs.size}/4: [${[...discoveredAyahs].sort().join(", ")}]`,
  });

  // 2. No oscillation
  criteria.push({
    name: "No verse oscillation",
    pass: bounces.length === 0,
    detail: bounces.length === 0
      ? "Clean sequential flow"
      : `Bounces: ${bounces.map((b) => `${b.from}↔${b.to}`).join(", ")}`,
  });

  // 3. Verse order should be monotonically increasing
  const s112Matches = matches.filter((m) => m.surah === 112);
  let orderCorrect = true;
  let lastAyah = 0;
  for (const m of s112Matches) {
    if (m.ayah < lastAyah) {
      orderCorrect = false;
      break;
    }
    lastAyah = m.ayah;
  }
  criteria.push({
    name: "Verse order monotonic",
    pass: orderCorrect,
    detail: `Order: ${s112Matches.map((m) => m.ayah).join(" → ")}`,
  });

  const passed = criteria.every((c) => c.pass);
  return { name, description, criteria, messages, passed, timeMs: performance.now() - start };
}

/**
 * Test 5: Long Verse Tracking (2:255 Ayat al-Kursi)
 * Tests word coverage on a long verse.
 */
async function testLongVerseTracking(): Promise<UXTestResult> {
  const name = "Long Verse Tracking (2:255)";
  const description = "Long ayah — checks discovery + word-level tracking coverage";
  const start = performance.now();

  await resetWorker();

  let audio: Float32Array;
  try {
    audio = await loadAudioFile(2, 255);
  } catch {
    return {
      name, description,
      criteria: [{ name: "Audio available", pass: false, detail: "No audio file for 2:255" }],
      messages: [], passed: false, timeMs: performance.now() - start,
    };
  }

  const messages = await feedAndCollect(audio);
  const matches = extractVerseMatches(messages);
  const words = extractWordProgress(messages);

  const criteria: Criterion[] = [];

  // 1. Discovery
  const found = matches.some((m) => m.surah === 2 && m.ayah === 255);
  criteria.push({
    name: "Discovers 2:255",
    pass: found,
    detail: found
      ? `Found (conf: ${(matches.find((m) => m.surah === 2 && m.ayah === 255)?.confidence ?? 0 * 100).toFixed(0)}%)`
      : `Not found — matched: ${matches.map((m) => `${m.surah}:${m.ayah}`).join(", ") || "nothing"}`,
  });

  // 2. Word coverage ≥ 30% (long verse, partial coverage expected)
  const cov = computeWordCoverage(words, 2, 255);
  criteria.push({
    name: "Word coverage ≥ 30%",
    pass: cov.percentage >= 0.3,
    detail: `${cov.covered}/${cov.total} words (${(cov.percentage * 100).toFixed(1)}%)`,
  });

  // 3. Word monotonicity
  const mono = checkWordMonotonicity(words, 2, 255);
  criteria.push({
    name: "Word progression monotonic",
    pass: mono.monotonic,
    detail: mono.monotonic
      ? `Max word index: ${mono.maxIdx}`
      : `${mono.regressions.length} regressions`,
  });

  const passed = criteria.every((c) => c.pass);
  return { name, description, criteria, messages, passed, timeMs: performance.now() - start };
}

/**
 * Test 6: Verse Complete Detection (Surah 112)
 * When all words of a verse are matched, verse_complete or full coverage should be detected.
 */
async function testVerseComplete(): Promise<UXTestResult> {
  const name = "Verse Complete Detection (112)";
  const description = "Checks if full-coverage detection works when entire verses are recited";
  const start = performance.now();

  await resetWorker();

  // Feed full surah 112
  const clips: Float32Array[] = [];
  for (let a = 1; a <= 4; a++) {
    clips.push(await loadAudioFile(112, a));
  }
  const audio = concatAudioWithGaps(clips);
  const messages = await feedAndCollect(audio);

  const words = extractWordProgress(messages);
  const matches = extractVerseMatches(messages);

  const criteria: Criterion[] = [];

  // Check which verses reached 100% word coverage
  let fullyTracked = 0;
  for (let a = 1; a <= 4; a++) {
    const cov = computeWordCoverage(words, 112, a);
    if (cov.total > 0 && cov.covered >= cov.total) {
      fullyTracked++;
    }
  }

  criteria.push({
    name: "At least 1 verse fully tracked",
    pass: fullyTracked >= 1,
    detail: `${fullyTracked}/4 verses reached 100% word coverage`,
  });

  // Check discovery
  const discovered = new Set(matches.filter((m) => m.surah === 112).map((m) => m.ayah));
  criteria.push({
    name: "Discovers ≥3/4 ayahs",
    pass: discovered.size >= 3,
    detail: `Discovered: [${[...discovered].sort().join(", ")}]`,
  });

  // Average word coverage across discovered verses
  let totalCov = 0;
  let count = 0;
  for (let a = 1; a <= 4; a++) {
    const cov = computeWordCoverage(words, 112, a);
    if (cov.total > 0) {
      totalCov += cov.percentage;
      count++;
    }
  }
  const avgCov = count > 0 ? totalCov / count : 0;
  criteria.push({
    name: "Average word coverage ≥ 60%",
    pass: avgCov >= 0.6,
    detail: `${(avgCov * 100).toFixed(1)}% across ${count} tracked verse(s)`,
  });

  const passed = criteria.every((c) => c.pass);
  return { name, description, criteria, messages, passed, timeMs: performance.now() - start };
}

/**
 * Test 7: Mid-Page Start (18:10)
 * User starts reading from a verse that's in the middle of a mushaf page.
 */
async function testMidPageStart(): Promise<UXTestResult> {
  const name = "Mid-Page Start (18:10)";
  const description = "User starts reading from middle of a mushaf page — should identify correct verse";
  const start = performance.now();

  await resetWorker();

  const audio = await loadAudioFile(18, 10);
  const messages = await feedAndCollect(audio);

  const matches = extractVerseMatches(messages);
  const words = extractWordProgress(messages);

  const criteria: Criterion[] = [];

  // Discovery
  const found = matches.some((m) => m.surah === 18 && m.ayah === 10);
  criteria.push({
    name: "Discovers 18:10",
    pass: found,
    detail: found
      ? `Found 18:10`
      : `Not found — matched: ${matches.map((m) => `${m.surah}:${m.ayah}`).join(", ") || "nothing"}`,
  });

  // Word progress exists
  const hasWords = words.some((w) => w.surah === 18 && w.ayah === 10);
  criteria.push({
    name: "Has word progress for 18:10",
    pass: hasWords,
    detail: hasWords
      ? `${words.filter((w) => w.surah === 18 && w.ayah === 10).length} word_progress events`
      : "No word progress for 18:10",
  });

  // Discovery latency — should find it within first 10 messages
  const latency = discoveryLatency(messages, 18, 10);
  criteria.push({
    name: "Discovery latency ≤ 10 msgs",
    pass: latency >= 0 && latency <= 10,
    detail: latency >= 0 ? `Found at message index ${latency}` : "Never found",
  });

  const passed = criteria.every((c) => c.pass);
  return { name, description, criteria, messages, passed, timeMs: performance.now() - start };
}

/**
 * Test 8: Continuation After Pause
 * Feed 112:1-2, then silence, then 112:3-4.
 * Should resume tracking after the pause.
 */
async function testContinuationAfterPause(): Promise<UXTestResult> {
  const name = "Continuation After Pause";
  const description = "Recite 112:1-2, long silence, then 112:3-4 — should resume tracking";
  const start = performance.now();

  await resetWorker();

  // Part 1: 112:1-2
  const clip1 = await loadAudioFile(112, 1);
  const clip2 = await loadAudioFile(112, 2);
  const part1 = concatAudioWithGaps([clip1, clip2]);

  // Silence gap (~3 seconds)
  const silence = new Float32Array(16000 * 3);

  // Part 2: 112:3-4
  const clip3 = await loadAudioFile(112, 3);
  const clip4 = await loadAudioFile(112, 4);
  const part2 = concatAudioWithGaps([clip3, clip4]);

  // Concatenate: part1 + silence + part2
  const total = new Float32Array(part1.length + silence.length + part2.length);
  total.set(part1, 0);
  total.set(silence, part1.length);
  total.set(part2, part1.length + silence.length);

  const messages = await feedAndCollect(total);

  const matches = extractVerseMatches(messages);

  const criteria: Criterion[] = [];

  // 1. Should discover at least 112:1 from part 1
  const foundPart1 = matches.some((m) => m.surah === 112 && (m.ayah === 1 || m.ayah === 2));
  criteria.push({
    name: "Discovers 112:1-2 (before pause)",
    pass: foundPart1,
    detail: foundPart1
      ? `Found pre-pause verses`
      : `Not found — matched: ${matches.map((m) => `${m.surah}:${m.ayah}`).join(", ") || "nothing"}`,
  });

  // 2. Should resume and discover 112:3 or 112:4 after pause
  const foundPart2 = matches.some((m) => m.surah === 112 && (m.ayah === 3 || m.ayah === 4));
  criteria.push({
    name: "Discovers 112:3-4 (after pause)",
    pass: foundPart2,
    detail: foundPart2
      ? `Resumed tracking after pause`
      : `Did not resume — only found: ${matches.map((m) => `${m.surah}:${m.ayah}`).join(", ") || "nothing"}`,
  });

  // 3. Total discoveries ≥ 2 (at least 1 from each part)
  const uniqueAyahs = new Set(matches.filter((m) => m.surah === 112).map((m) => m.ayah));
  criteria.push({
    name: "Discovers ≥2 unique ayahs",
    pass: uniqueAyahs.size >= 2,
    detail: `${uniqueAyahs.size} unique ayahs: [${[...uniqueAyahs].sort().join(", ")}]`,
  });

  const passed = criteria.every((c) => c.pass);
  return { name, description, criteria, messages, passed, timeMs: performance.now() - start };
}

// ── Scenario Registry ──
const SCENARIOS: Record<string, () => Promise<UXTestResult>> = {
  "mid-start": testMidStart,
  "word-progression": testWordProgression,
  "verse-oscillation": testVerseOscillation,
  "smooth-sequential": testSmoothSequential,
  "long-verse-tracking": testLongVerseTracking,
  "verse-complete": testVerseComplete,
  "mid-page-start": testMidPageStart,
  "continuation-after-pause": testContinuationAfterPause,
};

const SCENARIO_ORDER = [
  "mid-start",
  "word-progression",
  "verse-oscillation",
  "smooth-sequential",
  "long-verse-tracking",
  "verse-complete",
  "mid-page-start",
  "continuation-after-pause",
];

// ── Run all ──
async function runAllTests(): Promise<void> {
  const results: UXTestResult[] = [];

  for (const key of SCENARIO_ORDER) {
    if (shouldStop) break;
    const idx = SCENARIO_ORDER.indexOf(key) + 1;
    $status.textContent = `Running ${idx}/${SCENARIO_ORDER.length}: ${key}...`;
    $progress.style.width = `${(idx / SCENARIO_ORDER.length * 100).toFixed(0)}%`;

    logHeader(`Test ${idx}/${SCENARIO_ORDER.length}: ${key}`);
    try {
      const result = await SCENARIOS[key]();
      results.push(result);

      // Log inline
      for (const c of result.criteria) {
        const fn = c.pass ? logPass : logFail;
        fn(`  ${c.pass ? "PASS" : "FAIL"} ${c.name}: ${c.detail}`);
      }

      // Log message summary
      const msgTypes: Record<string, number> = {};
      for (const m of result.messages) {
        msgTypes[m.type] = (msgTypes[m.type] || 0) + 1;
      }
      logDim(`  Messages: ${JSON.stringify(msgTypes)}`);
      log("");
    } catch (err) {
      logFail(`  ERROR: ${err}`);
      log("");
    }
  }

  showSummary(results);
}

function showSummary(results: UXTestResult[]) {
  $summary.style.display = "block";
  $summary.innerHTML = "";

  const totalPassed = results.filter((r) => r.passed).length;
  const totalTests = results.length;

  // Overall header
  const header = document.createElement("div");
  header.className = `summary-card ${totalPassed === totalTests ? "pass" : "fail"}`;
  const h3 = document.createElement("h3");
  h3.textContent = `UX Tests: ${totalPassed}/${totalTests} passed`;
  header.appendChild(h3);

  const totalCriteria = results.reduce((s, r) => s + r.criteria.length, 0);
  const passedCriteria = results.reduce((s, r) => s + r.criteria.filter((c) => c.pass).length, 0);
  const totalTime = results.reduce((s, r) => s + r.timeMs, 0);

  const stats = document.createElement("div");
  stats.className = "criteria";
  stats.style.color = "#888";
  stats.textContent = `${passedCriteria}/${totalCriteria} criteria passed | ${(totalTime / 1000).toFixed(1)}s total`;
  header.appendChild(stats);
  $summary.appendChild(header);

  // Individual test cards
  for (const result of results) {
    $summary.appendChild(renderResultCard(result));
  }

  // Log overall
  logHeader("OVERALL RESULTS");
  log(
    `  ${totalPassed}/${totalTests} tests passed | ${passedCriteria}/${totalCriteria} criteria | ${(totalTime / 1000).toFixed(1)}s`,
    totalPassed === totalTests ? "log-pass" : "log-warn",
  );
}

// ── Warmup ──
async function warmupModel(): Promise<void> {
  logDim("  Warming up model (3 dummy inference passes)...");
  const dummySamples = new Float32Array(16000 * 3);
  for (let i = 0; i < dummySamples.length; i++) {
    dummySamples[i] = (Math.random() - 0.5) * 0.01;
  }
  for (let pass = 0; pass < 3; pass++) {
    const chunk = dummySamples.slice(pass * 16000, (pass + 1) * 16000);
    worker?.postMessage({ type: "audio", samples: chunk }, [chunk.buffer]);
    await new Promise((r) => setTimeout(r, 200));
  }
  await waitForWorkerIdle(1000, 15000);
  messageBuffer = [];
  worker?.postMessage({ type: "reset" });
  await new Promise((r) => setTimeout(r, 300));
  messageBuffer = [];
  logDim("  Warmup complete.");
}

// ── Init ──
async function startup() {
  $status.textContent = "Initializing worker...";
  try {
    await initWorker();
    await warmupModel();
    $btnRunAll.disabled = false;
    $btnRunSelected.disabled = false;
    logInfo("Model loaded and warmed up. Ready for UX testing.");
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
  $summary.innerHTML = "";

  try { await runAllTests(); } catch (err) { logFail(`Error: ${err}`); }

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
  $summary.innerHTML = "";

  try {
    if (selected === "all") {
      await runAllTests();
    } else if (SCENARIOS[selected]) {
      logHeader(`Test: ${selected}`);
      const result = await SCENARIOS[selected]();
      for (const c of result.criteria) {
        const fn = c.pass ? logPass : logFail;
        fn(`  ${c.pass ? "PASS" : "FAIL"} ${c.name}: ${c.detail}`);
      }
      const msgTypes: Record<string, number> = {};
      for (const m of result.messages) {
        msgTypes[m.type] = (msgTypes[m.type] || 0) + 1;
      }
      logDim(`  Messages: ${JSON.stringify(msgTypes)}`);
      showSummary([result]);
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
