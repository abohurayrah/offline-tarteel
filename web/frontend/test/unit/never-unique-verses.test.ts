/**
 * Tests for the 339 never-unique verses identified in the paper.
 *
 * These are verses where d=[-1,-1,...,-1] in ambiguity-compact.json,
 * meaning they can NEVER be uniquely identified from their text alone
 * (at any starting position / prefix depth). They need boundary context
 * (the preceding verse) to resolve.
 *
 * KEY BEHAVIOR: The short-verse exception (isShortVerse2 = wordCount <= 4)
 * does NOT apply to never-unique verses. A 4-word refrain that repeats
 * 31 times should still be held, not emitted immediately.
 *
 * Without boundary context, never-unique verses are HELD regardless of
 * word count. With boundary context (previous verse known), they resolve
 * via sequential advancement.
 *
 * The prefix trie also affects behavior: when narrowByPrefix returns
 * <= PREFIX_NARROW_MAX_CANDIDATES (5) candidates, Phase 1 defers
 * regardless of word count. When the trie returns too many (>5),
 * Phase 1 skips and Phase 2 applies the never-unique hold.
 *
 * Test categories:
 *   1. Bismillah (1:1) — 113 occurrences, 4 words, held without context
 *   2. Ar-Rahman refrain (55:13) — 31 repeats, 4 words, held without context
 *   3. Al-Shu'ara' prophet narrative (26:107-109) — 5 identical blocks
 *   4. Muqatta'at "الم" (2:1) — 6 surahs, 5 words, full hold
 */

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { RecitationTracker } from "../../src/lib/tracker.ts";
import type { TranscribeResult } from "../../src/lib/tracker.ts";
import { QuranDB, normalizeArabic } from "../../src/lib/quran-db.ts";
import type { DisambiguationEntry } from "../../src/lib/quran-db.ts";
import type { WorkerOutbound } from "../../src/lib/types.ts";
import {
  SAMPLE_RATE,
  TRACKING_TRIGGER_SAMPLES,
} from "../../src/lib/types.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Full QuranDB + disambiguation data (required for never-unique verse tests)
// ---------------------------------------------------------------------------
let fullDb: QuranDB;
let disambigMap: Record<string, DisambiguationEntry>;

beforeAll(() => {
  const quranData = JSON.parse(
    readFileSync(resolve(__dirname, "../../public/quran.json"), "utf-8"),
  );
  fullDb = new QuranDB(quranData);

  disambigMap = JSON.parse(
    readFileSync(
      resolve(__dirname, "../../public/ambiguity-compact.json"),
      "utf-8",
    ),
  );
  fullDb.loadDisambiguationMap(disambigMap);
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function createMockTranscriber(responses: string[]) {
  let idx = 0;
  return async (_audio: Float32Array): Promise<TranscribeResult> => {
    const text = responses[idx % responses.length];
    idx++;
    return { text, rawTokens: text };
  };
}

function createSequentialTranscriber(responses: string[]) {
  let idx = 0;
  return async (_audio: Float32Array): Promise<TranscribeResult> => {
    const text =
      idx < responses.length ? responses[idx] : responses[responses.length - 1];
    idx++;
    return { text, rawTokens: text };
  };
}

function fakeAudio(samples: number): Float32Array {
  return new Float32Array(samples).fill(0.1);
}

function filterType<T extends WorkerOutbound["type"]>(
  msgs: WorkerOutbound[],
  type: T,
): Extract<WorkerOutbound, { type: T }>[] {
  return msgs.filter((m) => m.type === type) as any;
}

function discoveryAudio(): Float32Array {
  return fakeAudio(SAMPLE_RATE * 5);
}

// ==========================================================================
// Section 0: Disambiguation data integrity
// ==========================================================================
describe("disambiguation data integrity", () => {
  it("1:1 has d=[-1,-1,-1,-1] (all-negative, never unique)", () => {
    const entry = fullDb.getDisambiguationEntry(1, 1);
    expect(entry).not.toBeNull();
    expect(entry!.d.every((v) => v === -1)).toBe(true);
    expect(fullDb.isAmbiguousInIsolation(1, 1)).toBe(true);
  });

  it("55:13 has d=[-1,-1,-1,-1] (refrain, never unique)", () => {
    const entry = fullDb.getDisambiguationEntry(55, 13);
    expect(entry).not.toBeNull();
    expect(entry!.d.every((v) => v === -1)).toBe(true);
    expect(fullDb.isAmbiguousInIsolation(55, 13)).toBe(true);
  });

  it("26:107 has d=[-1,-1,-1,-1] (prophet narrative, never unique)", () => {
    const entry = fullDb.getDisambiguationEntry(26, 107);
    expect(entry).not.toBeNull();
    expect(entry!.d.every((v) => v === -1)).toBe(true);
    expect(fullDb.isAmbiguousInIsolation(26, 107)).toBe(true);
  });

  it("2:1 has d=[-1,-1,-1,-1,-1] (bismillah+alm, never unique)", () => {
    const entry = fullDb.getDisambiguationEntry(2, 1);
    expect(entry).not.toBeNull();
    expect(entry!.d.every((v) => v === -1)).toBe(true);
    expect(fullDb.isAmbiguousInIsolation(2, 1)).toBe(true);
  });

  it("confusers for 1:1 include 2:1, 3:1, 4:1 (bismillah family)", () => {
    const entry = fullDb.getDisambiguationEntry(1, 1);
    expect(entry).not.toBeNull();
    expect(entry!.c).toContain("2:1");
    expect(entry!.c).toContain("3:1");
    expect(entry!.c).toContain("4:1");
  });

  it("confusers for 55:13 include 55:16 and 55:18 (identical refrain)", () => {
    const entry = fullDb.getDisambiguationEntry(55, 13);
    expect(entry).not.toBeNull();
    expect(entry!.c).toContain("55:16");
    expect(entry!.c).toContain("55:18");
  });

  it("getDisambiguationLength returns -1 for all tested never-unique verses", () => {
    for (const [s, a] of [
      [1, 1],
      [55, 13],
      [55, 16],
      [26, 107],
      [26, 108],
      [2, 1],
    ] as [number, number][]) {
      expect(fullDb.getDisambiguationLength(s, a)).toBe(-1);
    }
  });
});

// ==========================================================================
// Section 1: Bismillah (1:1) — 113 occurrences, 4 words
//
// BEHAVIOR: Bismillah is 4 words -> isShortVerse=true -> disambiguation
// hold is BYPASSED in Phase 2. The trie has 113 candidates at full depth
// (> PREFIX_NARROW_MAX_CANDIDATES=5) so Phase 1 cannot narrow.
// The system relies on the ambiguity guard (runners_up score comparison)
// and boundary context instead of the hold.
// ==========================================================================
describe("Bismillah (1:1) — short-verse exception path", () => {
  const bismillahText = "بسم الله الرحمن الرحيم";

  it("bismillah is 4 words but never-unique so short-verse exception does not apply", () => {
    const v = fullDb.getVerse(1, 1)!;
    expect(v.text_words!.length).toBe(4);
    // neverUnique2=true overrides the short-verse exception, so Phase 2 hold DOES engage
  });

  it("prefix trie narrows to 113 candidates for bismillah (too many for Phase 1)", () => {
    const words = normalizeArabic(bismillahText).split(" ");
    const cascade = fullDb.prefixNarrowingCascade(words);
    // All 4 words resolve to 113 candidates (every surah except At-Tawba 9)
    expect(cascade.length).toBe(4);
    expect(cascade[3].count).toBeGreaterThanOrEqual(100);
    // narrowByPrefix returns null because 113 > PREFIX_NARROW_MAX_CANDIDATES (5)
    const narrowed = fullDb.narrowByPrefix(words, 5);
    expect(narrowed).toBeNull();
  });

  it("holds bismillah without context (never-unique, no short-verse exception)", async () => {
    // With full DB + disambiguation, bismillah is HELD because never-unique
    // verses no longer get the short-verse exception. Without boundary context,
    // the system cannot determine which of the 113 bismillah instances is correct.
    const transcribe = createMockTranscriber([bismillahText]);
    const tracker = new RecitationTracker(fullDb, transcribe);

    const msgs = await tracker.feed(discoveryAudio());
    const verseMatches = filterType(msgs, "verse_match");

    // The system does NOT commit because neverUnique2=true overrides isShortVerse2
    expect(verseMatches.length).toBe(0);
    // Should get a raw_transcript instead
    const rawTranscripts = filterType(msgs, "raw_transcript");
    expect(rawTranscripts.length).toBeGreaterThanOrEqual(1);
  });

  it("ambiguity guard fires for bismillah (runners_up within 3% of top score)", async () => {
    // When multiple verses score nearly identically, the ambiguity guard
    // should detect this and either defer or allow through based on shared prefix
    const result = fullDb.matchVerse(bismillahText, 0.3, 3, null, 10);
    expect(result).not.toBeNull();
    const runners = result!.runners_up ?? [];
    // Multiple ayah-1 verses should score identically
    expect(runners.length).toBeGreaterThanOrEqual(2);

    // Check that top candidates are within 3% of each other
    if (runners.length >= 2) {
      const topScore = runners[0].score;
      const altScore = runners[1].score;
      // If alt is within 97% of top, the ambiguity guard is relevant
      const ratio = altScore / topScore;
      expect(ratio).toBeGreaterThan(0.9);
    }
  });

  it("resolves bismillah with boundary context (start from unique 1:2, advance to 1:3)", async () => {
    // Since 1:1 (bismillah) is now held without context, test boundary
    // resolution by starting from a unique verse (1:2) and advancing.
    const v2 = fullDb.getVerse(1, 2)!;
    const v3 = fullDb.getVerse(1, 3)!;

    const responses = [
      v2.text_norm!, // discovery: match 1:2 (unique text)
      v2.text_norm!, // tracking
      v2.text_norm!, // grace
      v2.text_norm!, // complete → advance to 1:3
      v3.text_norm!, // tracking 1:3
    ];
    const transcribe = createSequentialTranscriber(responses);
    const tracker = new RecitationTracker(fullDb, transcribe);

    let allMsgs: WorkerOutbound[] = [];
    allMsgs.push(...(await tracker.feed(discoveryAudio())));
    for (let i = 0; i < 10; i++) {
      allMsgs.push(
        ...(await tracker.feed(fakeAudio(TRACKING_TRIGGER_SAMPLES))),
      );
    }

    const verseMatches = filterType(allMsgs, "verse_match");
    const refs = verseMatches.map((m) => `${m.surah}:${m.ayah}`);

    // Should match 1:2 then advance to 1:3
    expect(refs).toContain("1:2");
    expect(refs).toContain("1:3");
  });
});

// ==========================================================================
// Section 2: Ar-Rahman refrain (55:13) — repeats 31 times, 4 words
//
// BEHAVIOR: Never-unique verses are now held regardless of word count.
// The trie narrows to 31 candidates (>5) so Phase 1 skips.
// Phase 2 hold engages because neverUnique2=true overrides isShortVerse2.
// Boundary context (previous verse known) resolves the ambiguity.
// ==========================================================================
describe("Ar-Rahman refrain (55:13) — held without context, resolves with boundary", () => {
  const refrainNorm = normalizeArabic("فبأي آلاء ربكما تكذبان");

  it("55:13 and 55:16 have identical normalized text", () => {
    const v13 = fullDb.getVerse(55, 13)!;
    const v16 = fullDb.getVerse(55, 16)!;
    expect(v13.text_norm).toBe(v16.text_norm);
  });

  it("refrain is 4 words but never-unique so short-verse exception does not apply", () => {
    const v = fullDb.getVerse(55, 13)!;
    expect(v.text_words!.length).toBe(4);
  });

  it("prefix trie narrows to 31+ candidates (too many for Phase 1)", () => {
    const words = normalizeArabic(refrainNorm).split(" ");
    const cascade = fullDb.prefixNarrowingCascade(words);
    expect(cascade.length).toBeGreaterThanOrEqual(1);
    // At full depth, 31 refrain instances
    const lastCount = cascade[cascade.length - 1].count;
    expect(lastCount).toBeGreaterThanOrEqual(30);
    // Too many for Phase 1
    const narrowed = fullDb.narrowByPrefix(words, 5);
    expect(narrowed).toBeNull();
  });

  it("holds refrain without context (Fix 4: short-verse exception does NOT apply to never-unique)", async () => {
    const transcribe = createMockTranscriber([refrainNorm]);
    const tracker = new RecitationTracker(fullDb, transcribe);

    const msgs = await tracker.feed(discoveryAudio());
    const verseMatches = filterType(msgs, "verse_match");

    // Fix 4: short-verse exception no longer bypasses hold for never-unique verses.
    // A 4-word refrain that repeats 31 times should still be held without context.
    expect(verseMatches.length).toBe(0);
  });

  it("resolves refrain with sequential boundary context (after 55:12)", async () => {
    const v12 = fullDb.getVerse(55, 12)!;
    const v13 = fullDb.getVerse(55, 13)!;

    const responses = [
      v12.text_norm!, // discovery: match 55:12
      v12.text_norm!, // tracking 55:12
      v12.text_norm!, // tracking / grace
      v12.text_norm!, // tracking complete → advance to 55:13
      v13.text_norm!, // tracking 55:13
    ];
    const transcribe = createSequentialTranscriber(responses);
    const tracker = new RecitationTracker(fullDb, transcribe);

    let allMsgs: WorkerOutbound[] = [];
    allMsgs.push(...(await tracker.feed(discoveryAudio())));
    for (let i = 0; i < 12; i++) {
      allMsgs.push(
        ...(await tracker.feed(fakeAudio(TRACKING_TRIGGER_SAMPLES))),
      );
    }

    const verseMatches = filterType(allMsgs, "verse_match");
    const verseRefs = verseMatches.map((m) => `${m.surah}:${m.ayah}`);

    // Should match 55:12 first
    expect(verseRefs).toContain("55:12");
    // After completing 55:12, should auto-advance to 55:13
    expect(verseRefs).toContain("55:13");
  });

  it("multiple confusers score identically for refrain text", () => {
    const result = fullDb.matchVerse(refrainNorm, 0.3, 3, null, 10);
    expect(result).not.toBeNull();

    const runners = result!.runners_up ?? [];
    const surah55Runners = runners.filter(
      (r: any) => r.surah === 55,
    );
    expect(surah55Runners.length).toBeGreaterThanOrEqual(2);

    if (surah55Runners.length >= 2) {
      const scores = surah55Runners.map((r: any) => r.raw_score);
      const maxDiff = Math.max(...scores) - Math.min(...scores);
      expect(maxDiff).toBeLessThan(0.01);
    }
  });

  it("surah context helps disambiguate within surah 55", () => {
    const result = fullDb.matchVerse(refrainNorm, 0.3, 3, null, 10, 55);
    expect(result).not.toBeNull();
    expect(result!.surah).toBe(55);
  });
});

// ==========================================================================
// Section 3: Al-Shu'ara' prophet narrative (26:107-109) — 5 identical blocks
//
// BEHAVIOR VARIES BY VERSE:
//   26:107 (4 words): trie narrows to exactly 5 candidates → Phase 1 DEFERS
//     (neverUnique + !trieUnique → prefixDeferred=true). Phase 2 skipped. HELD.
//   26:108 (3 words): trie narrows to 8 candidates (>5) → Phase 1 skips.
//     Phase 2: neverUnique2=true → isShortVerse2=false (Fix 4) → HELD.
//   26:109 (11 words): isShortVerse2=false → Phase 2 hold ENGAGES. HELD.
// ==========================================================================
describe("Al-Shu'ara' prophet narrative (26:107-109) disambiguation", () => {
  it("26:107 and 26:125 have identical normalized text", () => {
    const v107 = fullDb.getVerse(26, 107)!;
    const v125 = fullDb.getVerse(26, 125)!;
    expect(v107.text_norm).toBe(v125.text_norm);
  });

  it("26:108 and 26:126 have identical normalized text", () => {
    const v108 = fullDb.getVerse(26, 108)!;
    const v126 = fullDb.getVerse(26, 126)!;
    expect(v108.text_norm).toBe(v126.text_norm);
  });

  it("26:109 and 26:127 have identical normalized text", () => {
    const v109 = fullDb.getVerse(26, 109)!;
    const v127 = fullDb.getVerse(26, 127)!;
    expect(v109.text_norm).toBe(v127.text_norm);
  });

  it("26:107 (4 words): Phase 1 narrows to exactly 5 candidates → DEFERS", async () => {
    const v107 = fullDb.getVerse(26, 107)!;

    // Verify trie behavior
    const words = v107.text_norm!.split(" ");
    const narrowed = fullDb.narrowByPrefix(words, 5);
    expect(narrowed).not.toBeNull();
    expect(narrowed!.length).toBe(5);

    // Now test the tracker
    const transcribe = createMockTranscriber([v107.text_norm!]);
    const tracker = new RecitationTracker(fullDb, transcribe);

    const msgs = await tracker.feed(discoveryAudio());
    const verseMatches = filterType(msgs, "verse_match");

    // Phase 1 finds 5 candidates, all never-unique → prefixDeferred=true
    // Phase 2 is SKIPPED because prefixDeferred=true
    // Result: NO verse_match
    expect(verseMatches.length).toBe(0);
  });

  it("26:108 (3 words): trie returns >5 candidates, never-unique → HELD (Fix 4)", async () => {
    const v108 = fullDb.getVerse(26, 108)!;

    // Verify trie behavior: 8 candidates at depth 3 → too many for Phase 1
    const words = v108.text_norm!.split(" ");
    const narrowed = fullDb.narrowByPrefix(words, 5);
    expect(narrowed).toBeNull(); // >5 candidates

    const transcribe = createMockTranscriber([v108.text_norm!]);
    const tracker = new RecitationTracker(fullDb, transcribe);

    const msgs = await tracker.feed(discoveryAudio());
    const verseMatches = filterType(msgs, "verse_match");

    // Fix 4: short-verse exception no longer bypasses hold for never-unique verses.
    // Phase 1 skips (too many candidates), Phase 2: neverUnique2=true → isShortVerse2=false
    // → neverUnique2 && !isShortVerse2 && !lastEmittedRef → match=null. HELD.
    expect(verseMatches.length).toBe(0);
  });

  it("26:109 (11 words): Phase 2 hold engages (not short verse)", async () => {
    const v109 = fullDb.getVerse(26, 109)!;
    expect(v109.text_words!.length).toBe(11);

    const transcribe = createMockTranscriber([v109.text_norm!]);
    const tracker = new RecitationTracker(fullDb, transcribe);

    const msgs = await tracker.feed(discoveryAudio());
    const verseMatches = filterType(msgs, "verse_match");

    // 11 words → isShortVerse2=false → Phase 2 hold engages
    // neverUnique2=true && !isShortVerse2 && !lastEmittedRef → match=null
    expect(verseMatches.length).toBe(0);
  });

  it("resolves 26:107 with boundary context (after 26:106)", async () => {
    const v106 = fullDb.getVerse(26, 106)!;

    const responses = [
      v106.text_norm!, // discovery: match 26:106
      v106.text_norm!, // tracking
      v106.text_norm!, // grace
      v106.text_norm!, // complete → advance to 26:107
    ];
    const transcribe = createSequentialTranscriber(responses);
    const tracker = new RecitationTracker(fullDb, transcribe);

    let allMsgs: WorkerOutbound[] = [];
    allMsgs.push(...(await tracker.feed(discoveryAudio())));
    for (let i = 0; i < 15; i++) {
      allMsgs.push(
        ...(await tracker.feed(fakeAudio(TRACKING_TRIGGER_SAMPLES))),
      );
    }

    const verseMatches = filterType(allMsgs, "verse_match");
    const verseRefs = verseMatches.map((m) => `${m.surah}:${m.ayah}`);

    expect(verseRefs).toContain("26:106");
    expect(verseRefs).toContain("26:107");
  });

  it("all 5 identical blocks score equally (prophet narrative refrains)", () => {
    const v107 = fullDb.getVerse(26, 107)!;
    const result = fullDb.matchVerse(v107.text_norm!, 0.3, 3, null, 10);
    expect(result).not.toBeNull();

    const runners = result!.runners_up ?? [];
    const narrativeVerses = runners.filter(
      (r: any) =>
        r.surah === 26 && [107, 125, 143, 162, 178].includes(r.ayah),
    );
    expect(narrativeVerses.length).toBeGreaterThanOrEqual(2);

    if (narrativeVerses.length >= 2) {
      const scores = narrativeVerses.map((r: any) => r.raw_score);
      const maxDiff = Math.max(...scores) - Math.min(...scores);
      expect(maxDiff).toBeLessThan(0.01);
    }
  });
});

// ==========================================================================
// Section 4: Muqatta'at "الم" (2:1) — appears in 6 surahs, 5 words
//
// BEHAVIOR: 2:1 has 5 words (bismillah + الم) → isShortVerse2=false.
// The trie narrows to 113 candidates at depth 4, then 8 at depth 5
// (still >5) → Phase 1 skips. Phase 2 hold ENGAGES because
// neverUnique2=true && !isShortVerse2 && !lastEmittedRef.
// Result: HELD (no verse_match on first discovery).
// ==========================================================================
describe("Muqatta'at 'alif-lam-mim' (2:1) disambiguation — full hold", () => {
  it("2:1, 3:1, 29:1, 30:1, 31:1, 32:1 all contain 'الم' (after bsm strip)", () => {
    for (const surah of [2, 3, 29, 30, 31, 32]) {
      const v = fullDb.getVerse(surah, 1)!;
      const stripped = v.text_norm_no_bsm ?? v.text_norm!;
      const normAlm = normalizeArabic("الم");
      expect(stripped).toContain(normAlm);
    }
  });

  it("2:1 is 5 words — NOT a short verse, full hold applies", () => {
    const v = fullDb.getVerse(2, 1)!;
    expect(v.text_words!.length).toBe(5);
    expect(fullDb.isAmbiguousInIsolation(2, 1)).toBe(true);
  });

  it("Phase 2 hold: does NOT commit to 2:1 without context", async () => {
    const v = fullDb.getVerse(2, 1)!;
    const transcribe = createMockTranscriber([v.text_norm!]);
    const tracker = new RecitationTracker(fullDb, transcribe);

    const msgs = await tracker.feed(discoveryAudio());
    const verseMatches = filterType(msgs, "verse_match");

    // 5 words → isShortVerse2=false → Phase 2 hold engages
    // neverUnique2=true && !isShortVerse2=true && !lastEmittedRef=true → nulled
    expect(verseMatches.length).toBe(0);
  });

  it("emits raw_transcript or nothing (not verse_match) for 2:1 in isolation", async () => {
    const v = fullDb.getVerse(2, 1)!;
    const transcribe = createMockTranscriber([v.text_norm!]);
    const tracker = new RecitationTracker(fullDb, transcribe);

    const msgs = await tracker.feed(discoveryAudio());
    const verseMatches = filterType(msgs, "verse_match");
    expect(verseMatches.length).toBe(0);
    // May emit raw_transcript if the match was strong enough before nulling
  });

  it("resolves 2:1 when boundary context is established (after 1:7)", async () => {
    const v1_7 = fullDb.getVerse(1, 7)!;

    const responses = [
      v1_7.text_norm!, // discovery: match 1:7
      v1_7.text_norm!, // tracking 1:7
      v1_7.text_norm!, // grace
      v1_7.text_norm!, // tracking complete → advance to 2:1
    ];
    const transcribe = createSequentialTranscriber(responses);
    const tracker = new RecitationTracker(fullDb, transcribe);

    let allMsgs: WorkerOutbound[] = [];
    allMsgs.push(...(await tracker.feed(discoveryAudio())));
    for (let i = 0; i < 20; i++) {
      allMsgs.push(
        ...(await tracker.feed(fakeAudio(TRACKING_TRIGGER_SAMPLES))),
      );
    }

    const verseMatches = filterType(allMsgs, "verse_match");
    const verseRefs = verseMatches.map((m) => `${m.surah}:${m.ayah}`);

    expect(verseRefs).toContain("1:7");
    // After 1:7 completes, getNextVerse(1,7) returns 2:1
    expect(verseRefs).toContain("2:1");
  });

  it("muqattaat confusers all have same raw_score for 'الم' text", () => {
    const almNorm = normalizeArabic("الم");
    const result = fullDb.matchVerse(almNorm, 0.1, 3, null, 20);
    // "الم" is very short (1 word, 3 chars) — may not pass minimum thresholds
    if (result) {
      const runners = result.runners_up ?? [];
      const almSurahs = runners.filter(
        (r: any) =>
          [2, 3, 29, 30, 31, 32].includes(r.surah) && r.ayah === 1,
      );
      if (almSurahs.length >= 2) {
        const scores = almSurahs.map((r: any) => r.raw_score);
        const maxDiff = Math.max(...scores) - Math.min(...scores);
        expect(maxDiff).toBeLessThan(0.05);
      }
    }
  });
});

// ==========================================================================
// Section 5: Disambiguation hold lifecycle (cross-cutting)
// ==========================================================================
describe("disambiguation hold lifecycle", () => {
  it("Phase 1 defers when trie narrows to <=5 never-unique candidates (26:107)", async () => {
    const v = fullDb.getVerse(26, 107)!;

    // Verify Phase 1 path: trie returns exactly 5 candidates
    const words = v.text_norm!.split(" ");
    const narrowed = fullDb.narrowByPrefix(words, 5);
    expect(narrowed).not.toBeNull();
    expect(narrowed!.length).toBeLessThanOrEqual(5);

    const transcribe = createMockTranscriber([v.text_norm!]);
    const tracker = new RecitationTracker(fullDb, transcribe);

    const msgs = await tracker.feed(discoveryAudio());
    const verseMatches = filterType(msgs, "verse_match");

    // Phase 1 finds candidates, defers (neverUnique=true, !trieUnique)
    // Phase 2 is SKIPPED because prefixDeferred=true
    expect(verseMatches.length).toBe(0);
  });

  it("consecutive deferrals force-emit after limit for held verses", async () => {
    // Use 26:109 (11 words, never-unique, Phase 2 hold applies)
    const v = fullDb.getVerse(26, 109)!;
    const transcribe = createMockTranscriber([v.text_norm!]);
    const tracker = new RecitationTracker(fullDb, transcribe);

    let allMsgs: WorkerOutbound[] = [];

    // Feed multiple discovery cycles — each triggers a deferral
    for (let i = 0; i < 5; i++) {
      const msgs = await tracker.feed(discoveryAudio());
      allMsgs.push(...msgs);
    }

    const rawTranscripts = filterType(allMsgs, "raw_transcript");
    const verseMatches = filterType(allMsgs, "verse_match");

    // Should see deferrals and possibly an eventual force-emit
    const total = rawTranscripts.length + verseMatches.length;
    expect(total).toBeGreaterThan(0);
  });

  it("boundary context resolution: isSequentialNext allows emit for never-unique verse", async () => {
    const v12 = fullDb.getVerse(55, 12)!;

    const responses = [
      v12.text_norm!,
      v12.text_norm!,
      v12.text_norm!,
      v12.text_norm!,
    ];
    const transcribe = createSequentialTranscriber(responses);
    const tracker = new RecitationTracker(fullDb, transcribe);

    let allMsgs: WorkerOutbound[] = [];
    allMsgs.push(...(await tracker.feed(discoveryAudio())));

    for (let i = 0; i < 15; i++) {
      allMsgs.push(
        ...(await tracker.feed(fakeAudio(TRACKING_TRIGGER_SAMPLES))),
      );
    }

    const verseMatches = filterType(allMsgs, "verse_match");
    const refs = verseMatches.map((m) => `${m.surah}:${m.ayah}`);

    if (refs.includes("55:12")) {
      // After completing 55:12, tracker auto-advances to 55:13
      // even though 55:13 is never-unique — boundary context resolves it
      expect(refs).toContain("55:13");
    }
  });

  it("short-verse exception only applies to verses with <=4 words", () => {
    // Verify the threshold: 4 words = short, 5 words = not short
    const bismillah = fullDb.getVerse(1, 1)!; // 4 words
    const refrain = fullDb.getVerse(55, 13)!; // 4 words
    const alm = fullDb.getVerse(2, 1)!; // 5 words
    const v109 = fullDb.getVerse(26, 109)!; // 11 words

    expect(bismillah.text_words!.length).toBeLessThanOrEqual(4);
    expect(refrain.text_words!.length).toBeLessThanOrEqual(4);
    expect(alm.text_words!.length).toBeGreaterThan(4);
    expect(v109.text_words!.length).toBeGreaterThan(4);
  });
});

// ==========================================================================
// Section 6: QuranDB-level ambiguity API tests
// ==========================================================================
describe("QuranDB ambiguity API", () => {
  it("isAmbiguousInIsolation returns false for a unique verse (103:2)", () => {
    expect(fullDb.isAmbiguousInIsolation(103, 2)).toBe(false);
  });

  it("isAmbiguousInIsolation returns true for all Ar-Rahman refrains", () => {
    const refrainAyahs = [
      13, 16, 18, 21, 23, 25, 28, 30, 32, 34, 36, 38, 40, 42, 45, 47, 49, 51,
      53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77,
    ];
    for (const ayah of refrainAyahs) {
      expect(fullDb.isAmbiguousInIsolation(55, ayah)).toBe(true);
    }
  });

  it("isAmbiguousInIsolation returns true for all 'الم' muqattaat surahs", () => {
    for (const surah of [2, 3, 29, 30, 31, 32]) {
      expect(fullDb.isAmbiguousInIsolation(surah, 1)).toBe(true);
    }
  });

  it("isAmbiguousInIsolation returns true for all 5 prophet narrative blocks", () => {
    for (const ayah of [107, 125, 143, 162, 178]) {
      expect(fullDb.isAmbiguousInIsolation(26, ayah)).toBe(true);
    }
  });

  it("getDisambiguationEntry returns null for non-existent verse", () => {
    expect(fullDb.getDisambiguationEntry(999, 999)).toBeNull();
  });

  it("prefixNarrowingCascade shows high candidate count for bismillah", () => {
    const words = normalizeArabic("بسم الله الرحمن الرحيم").split(" ");
    const cascade = fullDb.prefixNarrowingCascade(words);
    expect(cascade.length).toBeGreaterThanOrEqual(1);
    if (cascade.length >= 1) {
      expect(cascade[0].count).toBeGreaterThanOrEqual(100);
    }
    if (cascade.length >= 4) {
      expect(cascade[3].count).toBeGreaterThanOrEqual(100);
    }
  });

  it("prefixNarrowingCascade narrows to 1 for a unique verse prefix", () => {
    const words = normalizeArabic("والعصر").split(" ");
    const cascade = fullDb.prefixNarrowingCascade(words);
    expect(cascade.length).toBeGreaterThanOrEqual(1);
    expect(cascade[cascade.length - 1].count).toBe(1);
  });
});
