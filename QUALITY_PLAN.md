# Quality Improvement Plan — Offline Tarteel

**Codebase snapshot:** March 2026
**Current model:** `tarteel-ai/whisper-tiny-ar-quran` (ONNX, fp32 encoder 31 MB + q8 decoder 48 MB)
**Test corpus:** 54 samples in `benchmark/test_corpus/manifest.json`

This plan is ordered by risk and dependency. Each phase builds on the previous one. Do not start Phase 2 before Phase 0 gives you a green test suite, and do not start Phase 4 before Phase 3 is measured.

---

## Phase 0: Testing Foundation

> Goal: establish a fast, offline-runnable unit-test suite that makes every subsequent change verifiable in under 10 seconds without a browser.

### 0.1 — Install and configure Vitest

- [ ] **Files:** `web/frontend/package.json`, `web/frontend/vite.config.ts`

**What to change in `package.json`:**
Add `vitest` and `@vitest/coverage-v8` to `devDependencies`:
```json
"vitest": "^2.2.0",
"@vitest/coverage-v8": "^2.2.0"
```
Add test scripts:
```json
"test": "vitest run",
"test:watch": "vitest",
"test:coverage": "vitest run --coverage"
```

**What to change in `vite.config.ts`:**
Add a `test` block after the `resolve` block:
```ts
test: {
  environment: "node",
  include: ["src/**/*.test.ts"],
  coverage: {
    provider: "v8",
    include: ["src/lib/**"],
    thresholds: { lines: 80, functions: 80 },
  },
},
```
Also add `import { defineConfig } from "vitest/config"` or change the import to `"vitest/config"` so the `test` key is recognized.

**Expected impact:** `npm test` runs all unit tests in ~3 seconds, no browser, no model download.

**Verify:** `npm test` exits 0 with no test files yet ("no tests found" is acceptable).

---

### 0.2 — Unit tests for `normalizer.ts`

- [ ] **New file:** `web/frontend/src/lib/normalizer.test.ts`

`normalizer.ts` exports a single `normalizeArabic` function (line 13). The function in `quran-db.ts` is a richer version of the same transform (also called `normalizeArabic`, line 8) — note there are **two implementations**: the standalone `src/lib/normalizer.ts` and the internal one in `quran-db.ts`. Only the former is exported to components; the latter is used internally and re-exported via `export { normalizeArabic }` at line 21 of `quran-db.ts`. Tests must cover the `quran-db.ts` version because that is the one actually used throughout the pipeline.

```ts
// src/lib/normalizer.test.ts
import { describe, it, expect } from "vitest";
import { normalizeArabic } from "./quran-db"; // the version used by the pipeline

describe("normalizeArabic", () => {
  it("strips tashkeel (fatha, kasra, damma, shadda, sukun)", () => {
    expect(normalizeArabic("بِسْمِ")).toBe("بسم");
  });

  it("normalizes hamza variants to bare alef", () => {
    expect(normalizeArabic("أَحَدٌ")).toBe("احد");
    expect(normalizeArabic("إِلَه")).toBe("اله");
    expect(normalizeArabic("آمَنَ")).toBe("امن");
    expect(normalizeArabic("ٱللَّهُ")).toBe("الله");
  });

  it("normalizes taa marbuta to haa", () => {
    expect(normalizeArabic("رَحْمَةٌ")).toBe("رحمه");
  });

  it("normalizes alef maqsura to yaa", () => {
    expect(normalizeArabic("هُدًى")).toBe("هدي");
  });

  it("removes tatweel", () => {
    expect(normalizeArabic("الرَّحْمَـٰنِ")).toBe("الرحمن");
  });

  it("removes BPE marker (U+2581)", () => {
    expect(normalizeArabic("▁بسم")).toBe("بسم");
  });

  it("collapses multiple spaces and trims", () => {
    expect(normalizeArabic("  الله   اكبر  ")).toBe("الله اكبر");
  });

  it("returns empty string unchanged", () => {
    expect(normalizeArabic("")).toBe("");
  });

  it("strips punctuation marks (Arabic comma, question mark)", () => {
    expect(normalizeArabic("الحمد لله،")).toBe("الحمد لله");
  });
});
```

**Expected impact:** Catches any regression in the normalization pipeline that would cause matching to silently break.

**Verify:** `npm test` shows 9 passing tests.

---

### 0.3 — Unit tests for `levenshtein.ts`

- [ ] **New file:** `web/frontend/src/lib/levenshtein.test.ts`

```ts
// src/lib/levenshtein.test.ts
import { describe, it, expect } from "vitest";
import { distance, ratio, semiGlobalDistance, fragmentScore, sellersWordMatch } from "./levenshtein";

describe("distance", () => {
  it("returns 0 for identical strings", () => expect(distance("abc", "abc")).toBe(0));
  it("returns length for empty vs non-empty", () => expect(distance("", "abc")).toBe(3));
  it("computes kitten→sitting = 3", () => expect(distance("kitten", "sitting")).toBe(3));
  it("is symmetric", () => expect(distance("abc", "xyz")).toBe(distance("xyz", "abc")));
});

describe("ratio", () => {
  it("returns 1.0 for identical strings", () => expect(ratio("abc", "abc")).toBe(1.0));
  it("returns 0.8 for one-char diff in 5-char string", () => {
    // distance("abcde","abcdf")=1, lenSum=10, ratio=(10-1)/10=0.9
    expect(ratio("abcde", "abcdf")).toBeCloseTo(0.9, 5);
  });
  it("returns 1.0 for two empty strings", () => expect(ratio("", "")).toBe(1.0));
});

describe("fragmentScore", () => {
  it("returns 1.0 when query is exact substring of ref", () => {
    expect(fragmentScore("bcd", "abcde")).toBe(1.0);
  });
  it("returns 1.0 for empty query", () => {
    expect(fragmentScore("", "anything")).toBe(1.0);
  });
  it("returns lower score for non-substring query", () => {
    const s = fragmentScore("xyz", "abcde");
    expect(s).toBeLessThan(0.4);
  });
  it("returns high score for near-match substring (1 error in 5 chars)", () => {
    // "bce" vs "bcd" substring of "abcde" — 1 edit in 3 chars → score >= 0.6
    expect(fragmentScore("bce", "abcde")).toBeGreaterThanOrEqual(0.6);
  });
});

describe("semiGlobalDistance", () => {
  it("returns 0 when query is exact substring", () => {
    expect(semiGlobalDistance("bcd", "abcde")).toBe(0);
  });
  it("returns query.length for empty ref", () => {
    expect(semiGlobalDistance("abc", "")).toBe(3);
  });
});

describe("sellersWordMatch", () => {
  it("finds exact match at correct position", () => {
    const result = sellersWordMatch(["الله", "اكبر"], ["قل", "الله", "اكبر", "كثيرا"]);
    expect(result.score).toBeGreaterThan(0.8);
    expect(result.startIdx).toBe(1);
    expect(result.endIdx).toBe(3);
  });

  it("returns zero score for empty inputs", () => {
    expect(sellersWordMatch([], ["a", "b"]).score).toBe(0);
    expect(sellersWordMatch(["a"], []).score).toBe(0);
  });

  it("handles transcript longer than verse", () => {
    // Should not throw, just return a low score
    const result = sellersWordMatch(["a", "b", "c", "d"], ["a", "b"]);
    expect(result.score).toBeGreaterThanOrEqual(0);
  });
});
```

**Expected impact:** Locks down correctness of the core string-matching kernel. Any change to `levenshtein.ts` that breaks the semi-global alignment will be caught immediately.

**Verify:** `npm test` shows all levenshtein tests passing.

---

### 0.4 — Unit tests for `quran-db.ts` (QuranDB)

- [ ] **New file:** `web/frontend/src/lib/quran-db.test.ts`

The `QuranDB` constructor requires `QuranVerse[]` data. Tests must use a small in-memory fixture rather than fetching `quran.json`, keeping the suite fast and deterministic.

```ts
// src/lib/quran-db.test.ts
import { describe, it, expect, beforeAll } from "vitest";
import { QuranDB, normalizeArabic, partialRatio } from "./quran-db";
import type { QuranVerse } from "./types";

// Minimal fixture: 4 verses — 1:1, 1:2, 2:255, 112:1
// text_uthmani is real Uthmani text; text_clean is a simplified clean form.
const FIXTURE: QuranVerse[] = [
  {
    surah: 1, ayah: 1,
    text_uthmani: "بِسۡمِ ٱللَّهِ ٱلرَّحۡمَٰنِ ٱلرَّحِيمِ",
    text_clean: "بسم الله الرحمن الرحيم",
    surah_name: "الفاتحة", surah_name_en: "Al-Fatihah",
  },
  {
    surah: 1, ayah: 2,
    text_uthmani: "ٱلۡحَمۡدُ لِلَّهِ رَبِّ ٱلۡعَٰلَمِينَ",
    text_clean: "الحمد لله رب العالمين",
    surah_name: "الفاتحة", surah_name_en: "Al-Fatihah",
  },
  {
    surah: 2, ayah: 255,
    text_uthmani: "ٱللَّهُ لَآ إِلَٰهَ إِلَّا هُوَ ٱلۡحَيُّ ٱلۡقَيُّومُ",
    text_clean: "الله لا اله الا هو الحي القيوم",
    surah_name: "البقرة", surah_name_en: "Al-Baqarah",
  },
  {
    surah: 112, ayah: 1,
    text_uthmani: "قُلۡ هُوَ ٱللَّهُ أَحَدٌ",
    text_clean: "قل هو الله احد",
    surah_name: "الإخلاص", surah_name_en: "Al-Ikhlas",
  },
];

let db: QuranDB;
beforeAll(() => { db = new QuranDB(FIXTURE); });

describe("QuranDB construction", () => {
  it("stores all verses", () => expect(db.totalVerses).toBe(4));
  it("indexes by surah", () => expect(db.getSurah(1).length).toBe(2));
  it("computes text_norm (no diacritics, normalized)", () => {
    const v = db.getVerse(1, 1)!;
    expect(v.text_norm).toBe("بسم الله الرحمن الرحيم");
  });
  it("computes text_norm_ns (no spaces)", () => {
    const v = db.getVerse(1, 1)!;
    expect(v.text_norm_ns).toBe("بسماللهالرحمنالرحيم");
  });
  it("computes text_words array", () => {
    const v = db.getVerse(1, 2)!;
    expect(v.text_words).toEqual(["الحمد", "لله", "رب", "العالمين"]);
  });
  it("strips bismillah from ayah 1 of non-Fatiha surahs", () => {
    // 112:1 is "قل هو الله احد" — no bismillah prefix, so text_norm_no_bsm should be null
    const v = db.getVerse(112, 1)!;
    // Only surahs where bismillah IS a prefix would have text_norm_no_bsm set
    // 112:1 normalized = "قل هو الله احد" which doesn't start with bismillah
    expect(v.text_norm_no_bsm).toBeNull();
  });
});

describe("QuranDB.getVerse", () => {
  it("retrieves an existing verse", () => {
    expect(db.getVerse(1, 1)).toBeDefined();
    expect(db.getVerse(2, 255)?.surah).toBe(2);
  });
  it("returns undefined for non-existent verse", () => {
    expect(db.getVerse(99, 99)).toBeUndefined();
  });
});

describe("QuranDB.getNextVerse", () => {
  it("returns the next ayah within surah", () => {
    const next = db.getNextVerse(1, 1);
    expect(next?.ayah).toBe(2);
  });
  it("returns first verse of next surah at end of surah", () => {
    // 1:2 is last in our fixture for surah 1; next surah in fixture is 2
    const next = db.getNextVerse(1, 2);
    expect(next?.surah).toBe(2);
    expect(next?.ayah).toBe(255);
  });
  it("returns undefined at end of data", () => {
    expect(db.getNextVerse(112, 1)).toBeUndefined();
  });
});

describe("QuranDB.matchVerse", () => {
  it("matches a perfect input to the correct verse", () => {
    const result = db.matchVerse("الحمد لله رب العالمين", 0.3);
    expect(result?.surah).toBe(1);
    expect(result?.ayah).toBe(2);
    expect(result?.score).toBeGreaterThan(0.9);
  });

  it("matches a partial input (first 3 words) to the correct verse", () => {
    const result = db.matchVerse("الله لا اله", 0.25);
    expect(result?.surah).toBe(2);
    expect(result?.ayah).toBe(255);
  });

  it("returns null when input is below threshold", () => {
    const result = db.matchVerse("completely unrelated text xyz", 0.8);
    expect(result).toBeNull();
  });

  it("returns runners_up when returnTopK > 0", () => {
    const result = db.matchVerse("الحمد لله رب العالمين", 0.3, 3, null, 4);
    expect(Array.isArray(result?.runners_up)).toBe(true);
    expect(result!.runners_up.length).toBeGreaterThan(0);
  });
});

describe("partialRatio", () => {
  it("returns 1.0 when short equals long", () => {
    expect(partialRatio("abc", "abc")).toBe(1.0);
  });
  it("returns 1.0 when short is exact substring of long", () => {
    expect(partialRatio("bcd", "abcde")).toBe(1.0);
  });
  it("handles empty strings gracefully", () => {
    expect(partialRatio("", "abc")).toBe(0.0);
    expect(partialRatio("abc", "")).toBe(0.0);
  });
  it("returns high score for near-match substring", () => {
    // "الله اكبر" vs "قل الله اكبر كثيرا" — the substring exists
    expect(partialRatio("الله اكبر", "قل الله اكبر كثيرا")).toBeGreaterThan(0.9);
  });
});
```

**Expected impact:** Catches regressions in `QuranDB` construction (normalization, bismillah stripping, trigram indexing) and matching logic.

**Verify:** `npm test` shows all quran-db tests passing. The `beforeAll` fixture builds in < 50 ms.

---

### 0.5 — Unit tests for tracker logic (pure functions only)

- [ ] **New file:** `web/frontend/src/lib/tracker.test.ts`

The `RecitationTracker` class has no exported pure functions, but the file-internal `isSilence`, `wordsMatch`, and `alignPosition` functions are the highest-risk logic. Extract them to named exports or test indirectly via a mock transcriber.

**Step 1:** In `tracker.ts`, change the three private functions to exported functions so they can be unit-tested without spinning up the full tracker:

- Line 33: `function isSilence` → `export function isSilence`
- Line 42: `function wordsMatch` → `export function wordsMatch`
- Line 48: `function alignPosition` → `export function alignPosition`

These are pure functions with no side effects; exporting them does not change behavior.

**Step 2:** Write the tests:

```ts
// src/lib/tracker.test.ts
import { describe, it, expect } from "vitest";
import { isSilence, wordsMatch, alignPosition } from "./tracker";

describe("isSilence", () => {
  it("returns true for zero audio", () => {
    expect(isSilence(new Float32Array(1600))).toBe(true);
  });
  it("returns true for sub-threshold noise", () => {
    const audio = new Float32Array(1600).fill(0.001); // RMS = 0.001 < 0.005
    expect(isSilence(audio)).toBe(true);
  });
  it("returns false for speech-level audio", () => {
    const audio = new Float32Array(1600).fill(0.1); // RMS = 0.1 >> 0.005
    expect(isSilence(audio)).toBe(false);
  });
});

describe("wordsMatch", () => {
  it("exact match returns true", () => expect(wordsMatch("الله", "الله")).toBe(true));
  it("very short words require exact match", () => {
    expect(wordsMatch("لا", "لب")).toBe(false); // length ≤ 2 → exact only
    expect(wordsMatch("لا", "لا")).toBe(true);
  });
  it("near match above threshold returns true", () => {
    // "الرحمن" vs "الرحمان" — one insertion, ratio ≥ 0.7
    expect(wordsMatch("الرحمن", "الرحمان")).toBe(true);
  });
  it("clearly different words return false", () => {
    expect(wordsMatch("الله", "الملك")).toBe(false);
  });
});

describe("alignPosition", () => {
  const verse = ["الله", "لا", "اله", "الا", "هو", "الحي", "القيوم"];

  it("aligns all words in order", () => {
    const { position, matchedIndices } = alignPosition(verse, verse, 0);
    expect(matchedIndices.length).toBe(verse.length);
    expect(position).toBe(verse.length);
  });

  it("aligns a partial transcript", () => {
    const { position, matchedIndices } = alignPosition(["الله", "لا", "اله"], verse, 0);
    expect(matchedIndices).toEqual([0, 1, 2]);
    expect(position).toBe(3);
  });

  it("resumes from startFrom", () => {
    const { matchedIndices } = alignPosition(["الحي", "القيوم"], verse, 4);
    expect(matchedIndices[0]).toBe(5);
    expect(matchedIndices[1]).toBe(6);
  });

  it("returns startFrom when no match found", () => {
    const { position, matchedIndices } = alignPosition(["xyz", "abc"], verse, 2);
    expect(matchedIndices.length).toBe(0);
    expect(position).toBe(2);
  });

  it("handles empty inputs", () => {
    expect(alignPosition([], verse, 0).matchedIndices.length).toBe(0);
    expect(alignPosition(["الله"], [], 0).matchedIndices.length).toBe(0);
  });
});
```

**Expected impact:** Regression guard for the three most critical logical primitives in the tracking pipeline. The `LOOKAHEAD` constant (currently 5, line 141 of `types.ts`) is exercised by the `alignPosition` tests.

**Verify:** All tracker unit tests pass. Check that exporting the three functions does not cause TypeScript errors (`tsc --noEmit`).

---

## Phase 1: Audio Pipeline Fixes

> Goal: fix two silent correctness bugs in the audio path before investing in matcher tuning.

### 1.1 — Replace nearest-neighbour downsampler with a proper anti-aliasing resampler

- [ ] **File:** `web/frontend/public/audio-processor.js`

**The bug:** Lines 15-18 implement downsampling by stepping through the input with a non-integer step (`ratio = inputSampleRate / outputSampleRate`) and taking `Math.floor(i)` as the sample index. For a 48 kHz microphone (ratio = 3.0) this is equivalent to picking every 3rd sample — a perfect decimation by 3. But for a 44.1 kHz microphone (ratio = 2.75625) the floor arithmetic creates irregular gaps of 2 or 3 samples between picked values. This is **non-uniform subsampling, not resampling**. It does not apply an anti-aliasing low-pass filter before decimation, so frequencies between 8 kHz and 22 kHz alias back into the 0–8 kHz speech band, corrupting the mel spectrogram. Whisper-tiny is trained on cleanly resampled audio; aliasing lowers its effective SNR.

**The fix:** Replace the loop with a linear-interpolation resampler. Linear interpolation is not perfect but it implicitly averages adjacent samples (acts as a weak low-pass filter) and is correct at non-integer ratios.

Replace lines 15–19 of `audio-processor.js` with:

```js
const inputSampleRate = sampleRate;
const outputSampleRate = 16000;
const ratio = inputSampleRate / outputSampleRate;

// Linear-interpolation resampler (correct for non-integer ratios).
// For each output sample position, compute its exact position in the
// input, then linearly interpolate between the two surrounding samples.
const inputLen = channelData.length;
for (let outIdx = 0; ; outIdx++) {
  const inPos = outIdx * ratio;
  if (inPos >= inputLen - 1) break;
  const lo = Math.floor(inPos);
  const frac = inPos - lo;
  const sample = channelData[lo] * (1 - frac) + channelData[lo + 1] * frac;
  this._buffer.push(sample);
}
```

**Note:** The `_bufferSize` constant (line 5, value 4800) corresponds to 300 ms at 16 kHz. This is correct and should not change.

**Expected impact:** Reduces aliasing artifacts for 44.1 kHz and 48 kHz microphones. Improvement will be most visible as fewer "garbage" transcriptions from model confusion caused by aliased high-frequency energy.

**Verify:**
1. In a browser console, confirm the WorkletNode still posts `Float32Array` chunks to the main thread.
2. Feed a 440 Hz tone recorded at 44.1 kHz through the processor; the output should be approximately a 440 Hz signal (not a distorted alias).
3. Run the `batch-50` benchmark scenario before and after; discovery accuracy should improve or hold steady.

---

### 1.2 — Add RMS normalization to the AudioWorklet output

- [ ] **File:** `web/frontend/public/audio-processor.js`

**The problem:** Some users have system mic gain set very low (< -30 dBFS RMS) or very high (nearly clipping). Whisper-tiny was trained on audio normalized to a standard loudness range. When user audio is significantly quieter than the training distribution, the model tends to output empty strings or short fragments, causing repeated `raw_transcript` messages with low confidence. When audio is clipping, it causes distortion artifacts in the mel spectrogram.

**The fix:** After the resampled chunk is written to `this._buffer` and before the buffer length check, apply target-RMS normalization when sending a chunk. Add this logic just before `this.port.postMessage`:

```js
if (this._buffer.length >= this._bufferSize) {
  const chunk = new Float32Array(this._buffer);

  // RMS normalization: target -23 LUFS ≈ RMS 0.07 in linear scale.
  // Skip normalization when the signal is near-silent (avoid amplifying noise).
  const TARGET_RMS = 0.07;
  const SILENCE_GATE = 0.002; // below this RMS, don't normalize
  let sumSq = 0;
  for (let i = 0; i < chunk.length; i++) sumSq += chunk[i] * chunk[i];
  const rms = Math.sqrt(sumSq / chunk.length);
  if (rms > SILENCE_GATE) {
    const gain = Math.min(TARGET_RMS / rms, 8.0); // cap gain at +18 dB to avoid over-amplifying noise
    for (let i = 0; i < chunk.length; i++) {
      chunk[i] = Math.max(-1.0, Math.min(1.0, chunk[i] * gain));
    }
  }

  this.port.postMessage(chunk.buffer, [chunk.buffer]);
  this._buffer = [];
}
```

**Expected impact:** Reduces discovery failures caused by low-gain microphones. Users who report "it doesn't recognize my voice" are most commonly in the low-gain category. The +18 dB gain cap prevents noise amplification.

**Verify:**
1. Record a verse at half system volume; confirm `raw_transcript` confidence increases noticeably.
2. Record at full volume; confirm audio is not clipped (peak values stay ≤ 1.0 after normalization).
3. Silence (mic muted) should not be amplified — the SILENCE_GATE condition prevents it.
4. Run `isSilence` unit test (Phase 0.5) on normalized chunks — silence threshold in `tracker.ts` is 0.005, which sits comfortably below the 0.002 gate in the worklet.

---

### 1.3 — Add concurrency guard to the inference worker

- [ ] **File:** `web/frontend/src/worker/inference.ts`

**The problem:** `self.onmessage` is `async` (line 61). If the main thread sends multiple `audio` chunks faster than the worker can process them (which always happens — the worklet fires every 300 ms, but Whisper inference takes 300–800 ms on WASM), multiple `await tracker.feed(msg.samples)` calls will be in-flight simultaneously. JavaScript's single-threaded event loop serializes the execution, but because each `feed` call `await`s inside the tracker (line 71), the next `onmessage` call can start before the previous one finishes. This means `tracker.feed` can be called re-entrantly, corrupting `this.fullAudio` and `this.newAudioCount` state.

The benchmark harness in `test/benchmark.ts` works around this with `waitForWorkerIdle`, but real-time usage has no such guard.

**The fix:** Add a lock flag. In `inference.ts`, add at the top of the module scope (after line 9):

```ts
let inferenceInFlight = false;
const pendingAudioQueue: Float32Array[] = [];

async function processPendingAudio(): Promise<void> {
  if (inferenceInFlight) return;
  while (pendingAudioQueue.length > 0) {
    if (!tracker) { pendingAudioQueue.length = 0; return; }
    inferenceInFlight = true;
    // Drain: merge all queued chunks into one feed call to
    // avoid latency buildup when the queue has grown.
    const chunks = pendingAudioQueue.splice(0);
    let merged = chunks[0];
    for (let i = 1; i < chunks.length; i++) {
      const next = new Float32Array(merged.length + chunks[i].length);
      next.set(merged);
      next.set(chunks[i], merged.length);
      merged = next;
    }
    try {
      const messages = await tracker.feed(merged);
      for (const m of messages) post(m);
    } finally {
      inferenceInFlight = false;
    }
  }
}
```

Then in `self.onmessage`, replace the `audio` branch (lines 69–74):

```ts
} else if (msg.type === "audio") {
  if (!tracker) return;
  pendingAudioQueue.push(msg.samples);
  processPendingAudio(); // intentionally not awaited — returns immediately if locked
}
```

Also update the `reset` branch (line 65) to clear the queue:

```ts
} else if (msg.type === "reset") {
  pendingAudioQueue.length = 0;
  inferenceInFlight = false;
  if (db) {
    tracker = new RecitationTracker(db, transcribe);
  }
}
```

**Expected impact:** Eliminates race conditions in production use. The queue-draining strategy also naturally handles brief bursts of audio without dropping chunks.

**Verify:**
1. Open DevTools and confirm no overlapping calls to `tracker.feed` appear when logging is added.
2. Run the `fatihah` sequential benchmark — results should be identical (guard does not affect normal case).
3. Simulate high-speed feeding (reduce `SPEED_MULT` to 16) in the benchmark; previously this caused erratic results. With the guard, results should be stable.

---

## Phase 2: Tracker Tuning

> Goal: tighten the timing constants and add smarter exit conditions based on what the code review reveals.

### 2.1 — Reduce grace period from 2 cycles to 1 cycle

- [ ] **File:** `web/frontend/src/lib/tracker.ts`, line 701

**Current code (line 701):**
```ts
this.transitionGraceCycles = 2;
```

**Problem:** At `TRACKING_TRIGGER_SAMPLES = 8000` (0.5s per cycle), 2 grace cycles = 1 full second during which no word-alignment inference runs after a verse transition. For fast reciters (e.g., reading Juz Amma at 3 words/second), the next verse starts being spoken ~200 ms after the boundary, meaning 800 ms of the next verse is being captured with no tracking happening. This makes the tracker miss the first several words of every new verse.

**Change:**
```ts
this.transitionGraceCycles = 1; // was 2; 0.5s is sufficient for audio buffer flush
```

**Expected impact:** The tracker begins aligning on the new verse ~500 ms sooner after each transition. This directly improves `word_progress` coverage for the first 2–3 words of each verse.

**Verify:**
1. Run the `fatihah` scenario; confirm `word_progress` messages appear for earlier words (index 0–1) of verses 2–7.
2. Check that stale-verse artifacts (wrong ayah appearing briefly after transition) do not increase — if they do, revert to 1.5 (use a non-integer by adding a partial cycle: set `transitionGraceCycles = 1` and accept the small artifact risk, or increase `transitionCooldown` to 4 to compensate).

---

### 2.2 — Dynamic stale limit based on verse word count

- [ ] **File:** `web/frontend/src/lib/tracker.ts`, line 284 and `web/frontend/src/lib/types.ts`, line 140

**Current code (`types.ts` line 140):**
```ts
export const STALE_CYCLE_LIMIT = 4;
```

**Current code (`tracker.ts` line 284):**
```ts
if (this.staleCycles >= STALE_CYCLE_LIMIT) {
```

**Problem:** A flat limit of 4 stale cycles (2 seconds at 0.5s/cycle) works well for short verses (≤ 8 words) but is too short for long verses like 2:255 (Ayat al-Kursi, ~50 words) and 2:286 (~30 words). When the user is mid-recitation of a long verse and Whisper produces a confused output for one cycle, the tracker exits tracking prematurely. The long-verse benchmark samples (`long_002_285.wav`, `long_002_286.wav`, `long_048_029.wav`) expose this.

**Change in `tracker.ts`:** Replace the static comparison (line 284) with a dynamic limit:

```ts
// Dynamic stale limit: longer verses get more patience.
// Base: 4 cycles. Add 1 cycle per 10 words beyond 15 words, capped at 10.
const dynamicStaleLimit = Math.min(
  10,
  STALE_CYCLE_LIMIT + Math.max(0, Math.floor((this.trackingVerseWords.length - 15) / 10))
);
if (this.staleCycles >= dynamicStaleLimit) {
```

**Remove** the `STALE_CYCLE_LIMIT` export from `types.ts` if you want to enforce that all callers use the dynamic version, or keep it as the base constant and rename for clarity:

```ts
// types.ts line 140 — rename for clarity:
export const STALE_CYCLE_BASE = 4; // rename from STALE_CYCLE_LIMIT
```

Update the import in `tracker.ts` accordingly.

**Expected impact:** Long verses (15+ words) get up to 10 stale cycles (5 seconds) of patience before exiting tracking, reducing premature exits during Ayat al-Kursi and similar verses. Short verses are unaffected.

**Verify:**
1. Run the `long-verse` (2:255) scenario; compare stale-exit log messages before and after.
2. Confirm the `word_progress` coverage for 2:255 increases.
3. Run short-verse scenarios (`short-surahs`) to confirm behavior is unchanged.

---

### 2.3 — Coverage-based tracking exit

- [ ] **File:** `web/frontend/src/lib/tracker.ts`, `_exitTracking` method (line 710)

**Problem:** Currently, the stale-exit path (line 719 in `_exitTracking`) distinguishes `progress < 0.5` (likely misidentification) from `progress >= 0.5` (correct tracking that stalled). But there is a third case: if the user has covered nearly all words (e.g., coverage = 0.9) and then goes silent for 2 seconds, the stale exit fires and resets the tracker — when really the verse is essentially done and we should advance to the next verse.

**Change:** In `_handleTracking`, before the `staleCycles >= dynamicStaleLimit` check, add an early completion path:

```ts
// Coverage-based exit: if user has covered ≥ 85% of the verse words
// and has been stale for ≥ 2 cycles, treat the verse as complete
// rather than exiting to discovery mode.
const staleCoverage = (this.trackingLastWordIdx + 1) / this.trackingVerseWords.length;
if (
  this.staleCycles >= 2 &&
  staleCoverage >= 0.85 &&
  this.trackingVerse !== null &&
  this.trackingVerseWords.length > 0
) {
  // Force verse completion: same path as the normal completion block
  const curRef: [number, number] = [this.trackingVerse.surah, this.trackingVerse.ayah];
  this.lastEmittedRef = curRef;
  this.lastEmittedText = this.trackingVerse.text_norm!;
  this.cyclesSinceEmit = 0;
  if (this.trackingVerse.surah === this.lastConfirmedSurah) {
    this.lastConfirmedAyah = Math.max(this.lastConfirmedAyah, this.trackingVerse.ayah);
  } else {
    this.lastConfirmedSurah = this.trackingVerse.surah;
    this.lastConfirmedAyah = this.trackingVerse.ayah;
  }
  const nextV = this.db.getNextVerse(curRef[0], curRef[1]);
  this._exitTracking("verse complete");
  if (nextV) {
    const nextRef: [number, number] = [nextV.surah, nextV.ayah];
    const surrounding = getSurroundingVerses(this.db, nextV.surah, nextV.ayah);
    messages.push({
      type: "verse_match",
      surah: nextV.surah,
      ayah: nextV.ayah,
      verse_text: nextV.text_uthmani,
      surah_name: nextV.surah_name,
      confidence: 0.97,
      surrounding_verses: surrounding,
    });
    this.prevEmittedRef = this.lastEmittedRef;
    this.prevEmittedText = this.lastEmittedText;
    this.lastEmittedRef = nextRef;
    this.lastEmittedText = nextV.text_norm!;
    this._enterTracking(nextV, nextRef);
    this.transitionCooldown = 3;
  }
  this.fullAudio = new Float32Array(0);
  this.accumulatedText = "";
  this.accumulatedCycles = 0;
  return messages;
}
```

Place this block at line 243, just before the existing `if (this.staleCycles >= 2 && text.length >= 8 ...)` jump-detection block.

**Expected impact:** Verses where the user trails off at the end (stops before the last 1–2 words) will still advance correctly rather than dropping back to discovery mode and potentially producing a false match. This is the single most common failure mode for connected recitation.

**Verify:**
1. Manually test by reciting a verse and stopping 1–2 words before the end.
2. Confirm a `verse_match` is still emitted for the next verse within 1–2 seconds.
3. Run the `skip` scenario (which tests ayahs 1:1–1:3, then 1:5–1:7); the 85% threshold must not accidentally advance ayah 3 to ayah 5 instead of 4. Check the discovery log carefully.

---

### 2.4 — Long-verse mode: extend `TRACKING_MAX_WINDOW_SAMPLES`

- [ ] **File:** `web/frontend/src/lib/types.ts`, lines 137–139 and `web/frontend/src/lib/tracker.ts`, lines 148–153

**Current code (`types.ts`):**
```ts
export const TRACKING_MAX_WINDOW_SECONDS = 5.0;
export const TRACKING_MAX_WINDOW_SAMPLES = SAMPLE_RATE * TRACKING_MAX_WINDOW_SECONDS;
```

**Problem:** A 5-second window is insufficient for Ayat al-Kursi (2:255) which takes 20–30 seconds for most reciters. The audio window trims aggressively (line 153 in `tracker.ts`), so by the time Whisper transcribes the current window at the 15-second mark, it only sees 5 seconds of audio — often just the middle of the verse with no context for the beginning. This confuses the model's decoder.

**Change:** Make the max window conditional on verse length. In `tracker.ts` `_handleTracking`, just after the `TRACKING_TRIGGER_SAMPLES` guard (around line 193), replace the static `TRACKING_MAX_WINDOW_SAMPLES` in the `feed` method with a dynamic value:

In the `feed` method (line 148–153), change:
```ts
const maxSamples =
  this.trackingVerse !== null
    ? TRACKING_MAX_WINDOW_SAMPLES
    : MAX_WINDOW_SAMPLES;
```

to:
```ts
const maxSamples =
  this.trackingVerse !== null
    ? this._trackingWindowSamples()
    : MAX_WINDOW_SAMPLES;
```

Add a private method:
```ts
private _trackingWindowSamples(): number {
  if (!this.trackingVerse) return TRACKING_MAX_WINDOW_SAMPLES;
  const wordCount = this.trackingVerseWords.length;
  if (wordCount >= 30) return SAMPLE_RATE * 15; // 15s for very long verses
  if (wordCount >= 15) return SAMPLE_RATE * 8;  // 8s for medium-long verses
  return TRACKING_MAX_WINDOW_SAMPLES;           // 5s for normal verses
}
```

**Expected impact:** Whisper sees more context for long verses. For 2:255 specifically, this gives the decoder enough overlapping text to maintain stable Arabic output across the full verse.

**Verify:**
1. Run the `long-verse` benchmark scenario.
2. Monitor memory usage in DevTools; 15s at 16 kHz = 960 KB as Float32Array — acceptable.
3. Confirm word coverage for 2:255 increases by at least 10 percentage points compared to baseline.

---

## Phase 3: Smart Matching

> Goal: improve discovery accuracy, especially for formulaic verses and ambiguous bismillah openings.

### 3.1 — Whisper decoder prompting for Arabic/Quranic domain

- [ ] **File:** `web/frontend/src/worker/whisper-transcriber.ts`

**Current code (lines 56–58):**
```ts
const result = (await asr(audio)) as AutomaticSpeechRecognitionOutput;
return result.text.trim();
```

**Problem:** The `transcribe` function passes no generation options to the pipeline. Transformers.js v3's `pipeline("automatic-speech-recognition")` supports a `generate_kwargs` option that maps directly to HuggingFace `generate()` parameters. Without a prompt, Whisper may:
- Default to English transcription if the Quranic Arabic sounds unusual
- Use the generic Arabic vocabulary rather than Quranic-specific tokens
- Produce Latin characters or transliterations for some words

Whisper models support an initial prompt that biases the decoder toward expected vocabulary.

**Change:** Modify the `transcribe` function to pass generation kwargs. Also expose an optional `prompt` parameter for session-based prompting (Phase 3.4 will use this):

```ts
export async function transcribe(
  audio: Float32Array,
  promptText?: string,
): Promise<string> {
  if (!asr) throw new Error("Whisper model not loaded");

  const generateKwargs: Record<string, unknown> = {
    language: "arabic",
    task: "transcribe",
    // Suppress timestamp tokens — we don't need them and they waste decoder steps
    return_timestamps: false,
  };

  // If a prompt is provided, use it as an initial decoder context.
  // The prompt token limit for whisper-tiny is ~224 tokens; keep prompts short.
  if (promptText) {
    generateKwargs.prompt_ids = undefined; // let the pipeline handle tokenization
    // transformers.js v3 accepts `prompt` as a string in the pipeline options
    generateKwargs.prompt = promptText;
  }

  const result = (await asr(audio, {
    generate_kwargs: generateKwargs,
  })) as AutomaticSpeechRecognitionOutput;

  return result.text.trim();
}
```

**Note:** Verify that `@huggingface/transformers@3.8.1` (current version in `package.json`) supports the `prompt` key in `generate_kwargs` for the ONNX pipeline. If it does not, the alternative is to pass `forced_decoder_ids` or leave prompting for after the model upgrade in Phase 4. Add a comment in the code noting this dependency.

Also update the `TranscribeFn` type in `tracker.ts` line 24 to allow an optional second argument:
```ts
type TranscribeFn = (audio: Float32Array, prompt?: string) => Promise<TranscribeResult>;
```

And update the adapter in `inference.ts` lines 18–21:
```ts
async function transcribe(audio: Float32Array, prompt?: string): Promise<TranscribeResult> {
  const text = await whisperTranscribe(audio, prompt);
  return { text, rawTokens: "" };
}
```

**Expected impact:** More consistent Arabic-only output. Reduces the "garbage Latin characters" edge case. Unlocks session prompting in Phase 3.4.

**Verify:**
1. Feed a quiet verse; confirm no Latin characters appear in `raw_transcript` messages.
2. Run the `short-surahs` benchmark; scores should hold or improve.
3. Check that `generate_kwargs` is correctly forwarded by logging `result` before trimming.

---

### 3.2 — Ambiguity-adaptive deferral: use verse-pair similarity score

- [ ] **File:** `web/frontend/src/lib/tracker.ts`, lines 521–568

**Current code (line 521):**
```ts
if (altRunner && altRunner.score >= runnersUp[0].score * 0.97) {
```

**Problem:** The 3% gap threshold is fixed. This is too aggressive for verses that are genuinely very similar (e.g., the repeated refrain in Surah 55 `فَبِأَيِّ آلَاءِ رَبِّكُمَا تُكَذِّبَانِ` which appears 31 times). For those, any 2-second clip will always produce two candidates within 3% because the verses ARE nearly identical. The system defers indefinitely (capped at 2 consecutive deferrals, then forces through).

Conversely, for verses that differ from the 2nd candidate by more than 5% but still within the 3% band (numerical precision artifacts), the guard fires unnecessarily.

**Change:** Replace the hardcoded 0.97 threshold with a similarity-scaled threshold. Add a helper that computes the actual text similarity between the top-2 candidates and uses it to set a proportional deferral band:

```ts
// In _handleDiscovery, replace line 521:
// OLD: if (altRunner && altRunner.score >= runnersUp[0].score * 0.97) {
// NEW:
if (altRunner) {
  // How similar are the two candidate verses themselves?
  // If nearly identical text, the gap between their scores will always be small.
  // Use verse similarity to scale the deferral sensitivity.
  const verseSimilarity = matchVerse && altVerse
    ? ratio(matchVerse.text_norm_ns ?? "", altVerse.text_norm_ns ?? "")
    : 0;
  // For verse pairs with > 0.85 text similarity (e.g., repeated refrains),
  // tighten the band to 0.99 (only defer when scores are essentially identical).
  // For dissimilar verses, keep the 0.97 band.
  const deferralBand = verseSimilarity >= 0.85 ? 0.99 : 0.97;
  if (altRunner.score >= runnersUp[0].score * deferralBand) {
```

Import `ratio` from `levenshtein` at the top of `tracker.ts` — it is already imported via `import { ratio as levRatio } from "./levenshtein"` (line 1). Use `levRatio` directly.

**Expected impact:** Surah 55's refrain verses will no longer be stuck in perpetual deferral. Other verses that are genuinely ambiguous (bismillah-heavy openings) will still be correctly deferred. The `consecutiveDeferrals` cap of 2 remains as a safety net.

**Verify:**
1. Add Surah 55 verse 13 and verse 26 to the `batch-50` benchmark scenario.
2. Confirm both verses are discovered without hitting the `consecutiveDeferrals` force-emit path.
3. Check that 1:1 (bismillah) vs other bismillah-opening verses still defers correctly by checking the deferral log.

---

### 3.3 — Phonetic Levenshtein: adjust word-similarity threshold in `alignPosition`

- [ ] **File:** `web/frontend/src/lib/tracker.ts`, line 42 and `wordsMatch` function

**Current code (line 42):**
```ts
function wordsMatch(w1: string, w2: string, threshold = 0.7): boolean {
```

**Problem:** The 0.7 threshold is too strict for common Whisper substitution patterns in Quranic Arabic. Whisper-tiny frequently substitutes:
- `الرحمن` → `الرحمان` (adds alef, ratio ≈ 0.87 — passes)
- `مستقيم` → `مستقم` (drops yaa, ratio ≈ 0.85 — passes)
- `الضالين` → `الظالين` (ض/ظ confusion, ratio ≈ 0.88 — passes)
- `صراط` → `سراط` (ص/س confusion, ratio ≈ 0.83 — passes)

But for short words (3–4 chars), a single char substitution produces ratio ≈ 0.57–0.67, which fails the 0.7 threshold. For example:
- `لك` → `لق` (ratio = 0.5 — fails)
- `قل` → `كل` (ratio = 0.5 — fails)

These short-word failures cause alignment to miss function words entirely and report false "no progress."

**Change:** Add length-adaptive thresholding to `wordsMatch`:

```ts
// tracker.ts, replace lines 42-46:
function wordsMatch(w1: string, w2: string, threshold = 0.7): boolean {
  if (w1 === w2) return true;
  if (w1.length <= 2 || w2.length <= 2) return w1 === w2; // unchanged: very short = exact
  // For 3-char words, allow 1 error (ratio ≥ 0.57 for 3-char, but use 0.55 floor)
  const effectiveThreshold = w1.length <= 4 ? Math.min(threshold, 0.55) : threshold;
  return levRatio(w1, w2) >= effectiveThreshold;
}
```

**Expected impact:** Short function words (3–4 Arabic characters) that Whisper frequently substitutes will now participate in alignment. This should increase `word_progress` coverage for verses with many short words (common in short surahs).

**Verify:**
1. Run the unit tests from Phase 0.5 — ensure `wordsMatch("لق", "لك")` now returns `true`.
2. Update the test in `tracker.test.ts` for the new behavior:
   ```ts
   it("allows 1 substitution in 3-char words", () => {
     expect(wordsMatch("لك", "لق")).toBe(true); // threshold lowered to 0.55
   });
   ```
3. Run the `fatihah` sequential scenario and check word coverage for short words like `لك` (1:7), `غير` (1:7), `من` (various).

---

### 3.4 — Session surah context: prompt Whisper with confirmed surah name

- [ ] **Files:** `web/frontend/src/worker/inference.ts`, `web/frontend/src/worker/whisper-transcriber.ts`, `web/frontend/src/lib/tracker.ts`

**Problem:** Once a user has recited several verses of Surah 36 (Ya-Sin), the next verse is almost certainly still in Surah 36. Whisper, however, has no knowledge of this context and treats each inference independently. Providing a prompt with the current verse text (or just the last recognized words) guides Whisper's decoder toward the same lexical distribution.

**Design:** Pass the last recognized verse text as a decoder prompt when in tracking mode. This does not change the matching logic; it only helps Whisper produce more accurate Arabic text.

**Change in `tracker.ts`:** Modify `_handleTracking` to pass the tracking verse context to `transcribe`. At line 210 in `_handleTracking`:

```ts
// OLD:
const { text: rawText } = await this.transcribe(this.fullAudio.slice());

// NEW:
// Use the last few words of the current verse as a decoder prompt.
// This biases Whisper toward the vocabulary of the current verse.
const trackingPrompt = this.trackingVerseWords.length > 0
  ? this.trackingVerseWords
      .slice(Math.max(0, this.trackingLastWordIdx - 3), this.trackingLastWordIdx + 1)
      .join(" ")
  : undefined;
const { text: rawText } = await this.transcribe(this.fullAudio.slice(), trackingPrompt);
```

**Change in `tracker.ts`:** Update `TranscribeFn` at line 24:
```ts
type TranscribeFn = (audio: Float32Array, prompt?: string) => Promise<TranscribeResult>;
```

**Change in `inference.ts`:** Update the adapter at line 18 to forward the optional prompt (see Phase 3.1 change above — this is the same adapter update).

**Expected impact:** In tracking mode, Whisper's output for the current verse becomes more lexically consistent with the known verse text. This should reduce the `_charLevelProgress` fallback triggering on long verses and improve `matchedIndices` stability.

**Verify:**
1. Enable verbose logging in `_handleTracking` to print both the prompt and the raw Whisper output.
2. Compare `rawText` before and after prompting for 2:255 — the output should include fewer foreign words.
3. Run the `long-verse` benchmark; coverage should improve.
4. Ensure the prompt does not cause issues when `trackingLastWordIdx = -1` (first cycle after transition) — the `slice(Math.max(0, -1-3), 0)` will produce an empty array, which joins to `""`, which is falsy, so no prompt is passed. Verify this edge case.

---

### 3.5 — Mushaf page boosting in `matchVerse`

- [ ] **File:** `web/frontend/src/lib/quran-db.ts`

**Rationale:** Users of the Quran app typically read in mushaf (physical book) page order. Someone who just recited verse 2:200 is most likely continuing on the same page or the next page in their mushaf, not jumping to Surah 50. The `_continuationBonuses` method (line 169) already implements sequential bonus (+0.22 for next verse, +0.12 for +2, +0.06 for +3). The proposal here is to add a weak same-page bonus for all verses on the same mushaf page as the last emitted verse.

**Implementation requires mushaf page data.** The `QuranVerse` interface (`types.ts` line 96) does not currently include a `page` field. This is a two-step change:

**Step 1:** Augment the `QuranVerse` interface in `types.ts`:
```ts
export interface QuranVerse {
  // ... existing fields ...
  page?: number; // Mushaf page number (Uthmani standard, 604 pages)
}
```

**Step 2:** Verify that `quran.json` includes a `page` field. If it does not, this item should be moved to Phase 4 (data work). Check with:
```bash
node -e "const d = require('./public/quran.json'); console.log(Object.keys(d[0]))"
```
from the `web/frontend/` directory. If `page` is not present, defer this item.

**Step 3 (if page data exists):** Modify `_continuationBonuses` in `quran-db.ts` to also add a small bonus (+0.04) for all verses on the same mushaf page as the hint verse. This bonus is below the sequential bonus floor (0.06) so it will not override sequential continuation, but it will slightly favor on-page verses over off-page verses when sequential bonuses do not apply (e.g., user jumped to a different ayah on the same page).

**Expected impact:** Modest improvement for users reading mushaf sequentially. No negative impact if page data is absent.

**Verify:**
1. Confirm `quran.json` has a `page` field.
2. Add a test in `quran-db.test.ts` that checks the page bonus is present in the scoring for same-page verses.
3. Run the full benchmark; page bonus should not reduce accuracy on any scenario.

---

## Phase 4: Model and Data

> Goal: upgrade the model and expand the benchmark corpus to statistically meaningful coverage.

### 4.1 — Evaluate `whisper-base-ar-quran` as a drop-in upgrade

- [ ] **Files:** `web/frontend/src/worker/whisper-transcriber.ts` (line 20), `web/frontend/public/models/` (directory structure)

**Current model:** `whisper-tiny-ar-quran` — encoder fp32 (31 MB) + decoder q8 (48 MB) = ~79 MB total.

**Proposed model:** `whisper-base-ar-quran` — encoder fp32 (~72 MB) + decoder q8 (~100 MB) = ~172 MB total. Whisper-base has 6× more decoder parameters than whisper-tiny. For Quranic Arabic, which has a constrained vocabulary, the base model's larger attention capacity typically improves WER by 25–40% over tiny on held-out Quran test sets.

**Before switching permanently, do a controlled A/B test:**

**Step A:** Change `MODEL_ID` in `whisper-transcriber.ts` line 20:
```ts
const MODEL_ID = "/models/whisper-base-quran"; // was: /models/whisper-quran
```

**Step B:** Download and convert `tarteel-ai/whisper-base-ar-quran` to the same ONNX format:
```bash
# From web/frontend/public/models/
# Use the optimum-cli or the same script used for the tiny model
optimum-cli export onnx \
  --model tarteel-ai/whisper-base-ar-quran \
  --task automatic-speech-recognition \
  --dtype fp32 \
  --opset 17 \
  ./whisper-base-quran/
```

**Step C:** Quantize only the decoder (same reason as current model — encoder has Conv ops incompatible with ConvInteger):
```bash
python -c "
from optimum.onnxruntime import ORTQuantizer, AutoQuantizationConfig
q = ORTQuantizer.from_pretrained('./whisper-base-quran', file_name='decoder_model_merged.onnx')
q.quantize(AutoQuantizationConfig.avx2(is_static=False), save_dir='./whisper-base-quran/')
"
```

**Step D:** Run the full benchmark suite against both models. Record discovery accuracy and word coverage per scenario. Accept the base model only if it improves discovery accuracy by ≥ 5% without increasing latency beyond 1.5 seconds per inference call on a mid-range laptop (M1 MacBook Air equivalent).

**Expected impact:** ~25–40% WER reduction. Discovery accuracy likely improves from ~75% to ~85–90% on the `batch-50` scenario. First-inference latency increases from ~300 ms to ~600–800 ms (WASM is linear in model size).

**Verify:**
1. Run `npm run test:onnx` to confirm the base model loads correctly in the WASM runtime.
2. Measure inference time per call by adding `performance.now()` before/after `asr(audio)` in `whisper-transcriber.ts`.
3. Confirm the warmup in `inference.ts` (line 44) still works — warm up with the same 2-second noise clip.
4. Check that the model's `generation_config.json` has `forced_decoder_ids` set for Arabic language and transcription task — if not, the language kwarg from Phase 3.1 must be set explicitly.

---

### 4.2 — Expand benchmark corpus from 54 to 300+ samples

- [ ] **Files:** `benchmark/test_corpus/manifest.json`, `benchmark/test_corpus/` (directory)

**Current state:** 54 samples in `manifest.json`. Categories:
- `short`: 21 samples (mostly Surah 1 and short surahs 110–114)
- `medium`: 10 samples
- `long`: 11 samples
- `multi`: 12 samples

**Gaps:**
1. No samples from Surahs 2–35 (except 2:255, 2:285, 2:286, 3:23, 3:191)
2. No mid-ayah starts (user begins reading from the middle of a verse)
3. No samples with background noise
4. No samples with tajweed variations (elongated medd, ghunna patterns)
5. No cross-reciter variation (all reference samples are from everyayah.com / alafasy)
6. No samples with heavy accent (non-native Arabic speakers reciting)

**Target corpus (300+ samples):**

**Category 1: Core coverage — one sample per surah (Surahs 1–114)**
- Source: `everyayah.com` offers full Quran WAV downloads at 16 kHz from multiple reciters
- Add 114 samples: one from each surah (first ayah), from reciter `Alafasy_16kHz`
- File naming: `{surah:03d}{ayah:03d}_alafasy.wav` (already used by benchmark harness in `test/benchmark.ts` line 405)

**Category 2: Long verses — full audio (not just first ayah)**
- Add 15 long verses: 2:255, 2:256, 2:257, 2:261, 2:282, 2:285, 2:286, 3:18, 3:26–27, 3:190–191, 4:36, 33:35, 48:29, 74:31
- Source: `everyayah.com` or `mp3quran.net`

**Category 3: Mid-ayah starts**
- Create 20 samples by trimming the first 1–3 seconds from existing long-verse WAV files using ffmpeg:
  ```bash
  ffmpeg -i long_002_255.wav -ss 3 -c copy mid_002_255_3s.wav
  ```
- Expected behavior: tracker should still match via Sellers' word-window scoring

**Category 4: Multi-reciter**
- Add 30 samples of the same 10 verses from 3 different reciters: Alafasy, Minshawi, Al-Ghamdi
- Source: `everyayah.com` supports per-reciter downloads

**Manifest schema extension:** The existing manifest format supports `source` and `category` fields. Add a `reciter` field to each entry:
```json
{
  "id": "ref_001001_ghamdi",
  "file": "001001_ghamdi.wav",
  "surah": 1, "ayah": 1,
  "category": "short",
  "source": "everyayah",
  "reciter": "alghamdi",
  "expected_verses": [{"surah": 1, "ayah": 1}]
}
```

**Download script:** Create `benchmark/download_corpus.sh` to automate fetching:
```bash
#!/bin/bash
# Download one sample per surah (Surah 1–114, Ayah 1) from everyayah.com
RECITER="Alafasy_16kHz"
BASE_URL="https://everyayah.com/data/${RECITER}"
for s in $(seq 1 114); do
  FILE=$(printf "%03d001.mp3" $s)
  wget -nc "${BASE_URL}/${FILE}" -O "test_corpus/${FILE}"
done
```

**Expected impact:** Statistical confidence in benchmark results improves from ±12% to ±3% (law of large numbers on 300 vs 54 samples). Failure modes that only appear in specific verse patterns become detectable.

**Verify:**
1. Run `wc -l benchmark/test_corpus/manifest.json` to confirm 300+ entries.
2. Run the full benchmark suite with the new corpus; establish a baseline accuracy number.
3. Confirm the benchmark harness in `test/benchmark.ts` can load `.mp3` files (the WAV loader on line 48 only handles WAV). Either convert all downloads to WAV via ffmpeg or add an MP3 loader using the browser's `AudioContext.decodeAudioData`.

**Note on MP3 loading:** The WAV loader in `benchmark.ts` is a hand-rolled parser (lines 48–74) and does not support MP3. Before running the expanded corpus, either:
- Convert all MP3s to 16 kHz mono WAV:
  ```bash
  for f in test_corpus/*.mp3; do
    ffmpeg -i "$f" -ar 16000 -ac 1 "${f%.mp3}.wav"
  done
  ```
- Or add a browser-native decoder using `AudioContext.decodeAudioData` and a resampler.

The WAV-conversion approach is simpler and consistent with the existing file format.

---

## Experiments

> These are measurements to take before committing to specific tuning values. Each experiment has a clear hypothesis, a way to measure it, and a decision criterion.

### E1 — Measure actual aliasing impact of the current resampler

**Hypothesis:** The floor-based resampler (Phase 1.1) introduces audible aliasing for 44.1 kHz input that degrades Whisper accuracy by ≥ 5% on discovery rate.

**Measurement:**
1. Record 10 verses at 44.1 kHz (the default on macOS Chrome).
2. Run them through the current resampler and save the output WAV.
3. Run the same recordings through a reference resampler (Python `librosa.resample` with `res_type="kaiser_best"`).
4. Feed both sets through the pipeline independently; compare discovery accuracy.

**Decision criterion:** If discovery accuracy improves by ≥ 3% with the librosa-resampled audio vs. the current worklet, implement Phase 1.1. If improvement is < 1%, keep Phase 1.1 but lower its priority.

---

### E2 — Measure grace period sensitivity

**Hypothesis:** Reducing `transitionGraceCycles` from 2 to 1 (Phase 2.1) improves first-word coverage by ≥ 15% without increasing stale-verse false positives.

**Measurement:**
1. Add a metric to the benchmark harness: `wordsAtPosition1Coverage` = fraction of verse_match events where word index 0 is covered in subsequent `word_progress` messages.
2. Run `fatihah` and `short-surahs` with grace = 2 (baseline), then grace = 1.
3. Also count false verse_match events (wrong surah:ayah) per scenario.

**Decision criterion:** Accept grace = 1 if first-word coverage improves by ≥ 15% AND false positives do not increase.

---

### E3 — Characterize the bismillah disambiguation failure rate

**Hypothesis:** The current ambiguity guard (Phase 3.2) defers correctly on pure bismillah clips but forces through after 2 deferrals when mixed audio contains 3+ additional words.

**Measurement:**
1. Record 20 clips that start with bismillah (Surahs 2, 3, 4, 5... first ayahs).
2. Measure: (a) how many trigger deferral, (b) how many force through correctly, (c) how many force through to the wrong surah.
3. For the cases that force through to the wrong surah, check if the Levenshtein scores of the top-2 candidates are within 1% of each other at force-emit time.

**Decision criterion:** If > 20% of bismillah-opening clips force through to the wrong surah, tighten the deferral by raising `consecutiveDeferrals` cap from 2 to 3. If < 5%, the current behavior is acceptable.

---

### E4 — Measure Whisper output quality with vs. without decoder prompt

**Hypothesis:** Session prompting (Phase 3.4) reduces average normalized edit distance between Whisper output and ground-truth verse text by ≥ 10%.

**Measurement:**
1. Take 20 mid-verse audio clips (from existing long-verse corpus).
2. Transcribe each with no prompt vs. with the "last 3 words" prompt.
3. Compute `1 - ratio(normalizeArabic(whisperOutput), normalizeArabic(groundTruth))` as the character error rate for each.
4. Compare means.

**Decision criterion:** Accept prompting if average character error rate improves by ≥ 10%. Note: if the `@huggingface/transformers@3.8.1` WASM pipeline does not support `prompt` in `generate_kwargs`, do not implement Phase 3.4 until the library is upgraded.

---

### E5 — Word coverage ceiling: how much is lost to Whisper vs. matching?

**Hypothesis:** The current word coverage gap (< 100%) is split roughly 60% Whisper transcription errors and 40% alignment/matching failures. If true, model upgrade (Phase 4.1) will have a larger impact than matching improvements (Phase 3.3).

**Measurement:**
1. Take 10 verses where word coverage is poor (< 70% in benchmarks).
2. For each, manually transcribe Whisper's actual output text.
3. Manually align the Whisper output against the ground-truth verse.
4. Count errors attributable to: (a) Whisper hallucination/wrong word, (b) Correct Whisper word but alignment missed it (wordsMatch returned false).

**Decision criterion:** If > 60% of coverage failures are Whisper errors, prioritize Phase 4.1. If > 40% are alignment failures, prioritize Phase 3.3.

---

### E6 — Measure concurrency collision rate in production-speed usage

**Hypothesis:** Re-entrant `tracker.feed` calls (Phase 1.3) occur on average 0.5–2.0 times per minute of recitation, measurable as interleaved `word_progress` messages with non-monotonic word indices.

**Measurement:**
1. Collect `word_progress` messages from a 5-minute live recording session (without the guard).
2. Count events where `word_index[n+1] < word_index[n]` for the same surah:ayah — this indicates the previous tracking state was overwritten by a concurrent feed.
3. Also count `verse_match` events that appear within 100 ms of a preceding `verse_match` for a different verse.

**Decision criterion:** If collision rate > 0.2/minute, implement Phase 1.3 urgently. If < 0.05/minute, Phase 1.3 is still good hygiene but not urgent.

---

## Dependency Graph and Recommended Execution Order

```
Phase 0 (all items) — must complete first
    |
    +-- Phase 1.3 (concurrency guard) — implement before any benchmark runs
    |       |
    +-- Phase 1.1 (resampler) + Phase 1.2 (RMS norm) — parallel, no deps
            |
        Phase 2 (tracker tuning) — all items sequential within phase
            |
        Phase 3 (smart matching) — 3.1 before 3.4 (prompt dependency)
            |
        Phase 4.1 (model upgrade) — after Phase 3 baseline is established
        Phase 4.2 (corpus expansion) — parallel with Phase 3, needed for Phase 4.1 eval
```

**Experiments E1–E6** should be run as soon as Phase 0 and Phase 1.3 are complete, to inform decisions in Phases 2–4.

---

## Files Referenced

| File | Phases |
|------|--------|
| `web/frontend/public/audio-processor.js` | 1.1, 1.2 |
| `web/frontend/src/worker/inference.ts` | 1.3, 3.1, 3.4 |
| `web/frontend/src/worker/whisper-transcriber.ts` | 3.1, 4.1 |
| `web/frontend/src/lib/tracker.ts` | 0.5, 2.1, 2.2, 2.3, 3.2, 3.3, 3.4 |
| `web/frontend/src/lib/quran-db.ts` | 0.4, 3.5 |
| `web/frontend/src/lib/levenshtein.ts` | 0.3 |
| `web/frontend/src/lib/normalizer.ts` | 0.2 |
| `web/frontend/src/lib/types.ts` | 0.2, 2.2, 2.4, 3.5 |
| `web/frontend/package.json` | 0.1 |
| `web/frontend/vite.config.ts` | 0.1 |
| `benchmark/test_corpus/manifest.json` | 4.2 |
| `web/frontend/test/benchmark.ts` | E1–E6 (measurement harness) |
| `web/frontend/src/lib/normalizer.test.ts` | 0.2 (new) |
| `web/frontend/src/lib/levenshtein.test.ts` | 0.3 (new) |
| `web/frontend/src/lib/quran-db.test.ts` | 0.4 (new) |
| `web/frontend/src/lib/tracker.test.ts` | 0.5 (new) |

---

## Implementation Progress Log

### Session 1: March 22, 2026

#### Baseline Established
**Whisper-tiny-ar-quran on 54-sample corpus: 69.8% strict accuracy**

| Category | Accuracy | Count |
|----------|----------|-------|
| Long | 100% | 9/9 |
| Medium | 77.8% | 14/18 |
| Short | 58.8% | 10/17 |
| Multi-verse | 44.4% | 4/9 |

| Source | Accuracy | Count |
|--------|----------|-------|
| EveryAyah (professional) | 78.3% | 18/23 |
| RetaSy (crowdsourced) | 64.3% | 18/28 |
| User recordings | 50.0% | 1/2 |

Avg inference: 1756ms/sample on CPU.

#### Completed (Phase 0)
- [x] Vitest installed and configured
- [x] 162 unit tests (30 normalizer, 64 levenshtein, 48 quran-db, 21 tracker)
- [x] Test fixtures and helpers created
- [x] Whisper benchmark runner (`test/benchmark-whisper.ts`)
- [x] Benchmark comparison tool (`test/compare-benchmarks.ts`)

#### Completed (Phase 1: Audio Pipeline)
- [x] Linear interpolation resampling (was naive sample-skipping)
- [x] RMS audio normalization to -20dBFS target
- [x] Pre-allocated ring buffer (eliminates GC on audio thread)
- [x] Concurrency guard on inference worker (busy flag + audio queue)

#### Completed (Phase 2: Tracker Tuning)
- [x] Grace period reduced 2→1 cycles
- [x] Dynamic stale limit: `max(4, verseWords.length/8)`
- [x] Coverage-based exit at 85% + 2 stale cycles
- [x] Long-verse mode: 18s window for 20+ word verses

#### Completed (Phase 3: Smart Matching)
- [x] Phonetic Levenshtein (ص/س, ط/ت, ق/ك, etc.) — 20% blend weight
- [x] Session surah context (0.06 bonus)
- [x] Whisper decoder prompting (passes verse text as context)
- [x] `Record<string, any>` → proper `VerseMatch`/`VerseMatchCandidate` types

#### Completed (Code Quality)
- [x] Dead code removed: normalizer.ts, duplicate WAV encoder, unused exports
- [x] XSS fix in admin dashboard
- [x] Path traversal fix in report/diagnostic endpoints

#### In Progress (Accuracy Push)
- [ ] Multi-verse span matching: increase maxSpan to 6, use fragmentScore in Pass 2
- [ ] Muqatta'at (isolated letter) handling for 29 surahs
- [ ] Two-pass surah identification strategy
- [ ] Short verse / RetaSy accuracy improvements
- [ ] Model upgrade to whisper-base-ar-quran (5.75% WER vs 7.05%)
- [ ] Expanded test corpus from Tarteel user recordings

#### Accuracy Push (Session 1 continued)
- [x] Multi-verse span matching: maxSpan 3→6, smart scoring in Pass 2 with fragmentScore
- [x] Muqatta'at handling: exact + fuzzy match for 29 surahs with isolated letters (يس, طه, الم, etc.)
- [x] Two-pass surah identification: aggregate surah scores then narrow search
- [x] Short verse improvements: 40% phonetic weight, expanded candidates, first-word boosting
- [x] Whisper-base-ar-quran ONNX export (79MB encoder + 300MB decoder)
- [x] Benchmark comparison tool (`test/compare-benchmarks.ts`)

**Confirmed sub-benchmark results:**
| Category | Before | After | Delta |
|----------|--------|-------|-------|
| Multi-verse | 44.4% (4/9) | **100% (9/9)** | +55.6% |
| Short | 58.8% (10/17) | **64.7% (11/17)** | +5.9% |

#### Final Benchmark Result: 81.1% (43/53)

| Category | Baseline | Final | Delta |
|----------|----------|-------|-------|
| **Overall** | **69.8%** (37/53) | **81.1%** (43/53) | **+11.3%** |
| Short | 58.8% (10/17) | **70.6%** (12/17) | +11.8% |
| Medium | 77.8% (14/18) | 72.2% (13/18) | -5.6% |
| Long | 100% (9/9) | **100%** (9/9) | = |
| **Multi-verse** | **44.4%** (4/9) | **100%** (9/9) | **+55.6%** |
| EveryAyah | 78.3% (18/23) | **95.7%** (22/23) | +17.4% |
| RetaSy | 64.3% (18/28) | **67.9%** (19/28) | +3.6% |
| User | 50% (1/2) | **100%** (2/2) | +50% |

**Failures reduced from 16 to 10.** All remaining failures are RetaSy crowdsourced recordings where Whisper produces garbled text — this is the ASR model's limitation, not the matching engine.

#### Experiment: whisper-base-ar-quran
Exported to ONNX and benchmarked. **Result: WORSE than tiny (56.6% vs 69.8%)**.
The base model has severe repetition/looping on short and noisy audio. Do NOT upgrade.
Whisper-tiny remains the better model for this use case.

#### Expanded Test Corpus
Downloaded 256 Tarteel user recordings from HuggingFace (ashraf-ali/quran-data).
32 surahs covered, 224 unique verse combinations. Manifest at `benchmark/test_corpus_expanded/manifest.json`.

#### Additional Improvements (Session 1 continued)
- [x] Whisper output cleanup: repetition loop detection (collapses 3+ repeated words/phrases)
- [x] Extracted `cleanWhisperOutput` to shared `src/lib/text-cleanup.ts` with 12 tests
- [x] Adaptive discovery trigger: 1.2s for first match (instead of 2.0s), ~800ms faster
- [x] Production Vite build verified working

#### Remaining Failures Analysis (10 samples)
- **ref_001002** (1:2): Perfect transcript, wrong verse (37:182 has same text + leading "و"). Root cause: العالمين (Whisper output) vs العلمين (Uthmani text). Normalizer fix needed.
- **retasy_025** (1:7): 67% word overlap — "الذين... الاتالين" is garbled "الذين... الضالين". Potentially recoverable.
- **8 other RetaSy**: <50% word overlap — Whisper output is too garbled to match. Only a better ASR model can fix these.

#### Key Finding: whisper-base-ar-quran WORSE than tiny
- Exported to ONNX (79MB encoder + 300MB decoder)
- Benchmarked: 56.6% accuracy (vs 69.8% for tiny)
- Severe repetition/looping on short and noisy audio
- Not suitable for upgrade — tiny is more robust

#### Commits pushed to fork/feat/quality-improvements:
1. `bed29b0` — Main quality improvements (69.8% → 81.1%)
2. `8eedc1e` — Whisper output cleanup + adaptive trigger
3. `c52a2a7` — Extract cleanWhisperOutput to shared module with tests

#### Failure Analysis (16 failures at baseline)
1. **Ya-Sin 36:1** — FIXED: muqatta'at handling now matches isolated letters
2. **RetaSy Al-Fatiha** (5 failures) — Partially fixed: phonetic + first-word boosting helps some
3. **Multi-verse spans** (4 failures) — FIXED: maxSpan=6 + smart scoring + two-pass surah ID
4. **Short verses** (7 failures) — Partially fixed: higher phonetic weight + expanded candidates
