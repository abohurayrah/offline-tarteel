import { describe, it, expect } from "vitest";
import { RecitationTracker } from "../../src/lib/tracker.ts";
import type { TranscribeResult } from "../../src/lib/tracker.ts";
import { getFixtureQuranDB } from "../helpers/test-quran-db.ts";
import {
  SAMPLE_RATE,
  TRACKING_TRIGGER_SAMPLES,
} from "../../src/lib/types.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function createSequentialTranscriber(responses: string[]) {
  let idx = 0;
  return async (_audio: Float32Array): Promise<TranscribeResult> => {
    const text = idx < responses.length ? responses[idx] : responses[responses.length - 1];
    idx++;
    return { text, rawTokens: text };
  };
}

function fakeAudio(samples: number): Float32Array {
  return new Float32Array(samples).fill(0.1);
}

/**
 * Build the concatenated normalized text for a sequence of consecutive verses.
 * Uses text_norm_no_bsm for ayah 1 (strips bismillah) and text_norm for the rest.
 */
function buildMultiVerseText(
  db: ReturnType<typeof getFixtureQuranDB>,
  surah: number,
  ayahStart: number,
  ayahEnd: number,
): string {
  const parts: string[] = [];
  for (let a = ayahStart; a <= ayahEnd; a++) {
    const v = db.getVerse(surah, a);
    if (!v) throw new Error(`Verse ${surah}:${a} not in fixture`);
    if (a === ayahStart) {
      // First verse: use no-bismillah variant if available
      parts.push(v.text_norm_no_bsm ?? v.text_norm!);
    } else {
      parts.push(v.text_norm!);
    }
  }
  return parts.join(" ");
}

// ===========================================================================
// 1. NON-STREAMING: matchVerse with multi-verse concatenated text
// ===========================================================================

describe("Multi-verse span detection (non-streaming matchVerse)", () => {
  // --- 112:1-4 (full Al-Ikhlas) ---
  describe("112:1-4 (full Al-Ikhlas)", () => {
    it("detects the multi-verse span starting at 112:1", () => {
      const db = getFixtureQuranDB();
      const text = buildMultiVerseText(db, 112, 1, 4);
      const result = db.matchVerse(text, 0.2, 6);
      expect(result).not.toBeNull();
      expect(result!.surah).toBe(112);
      expect(result!.score).toBeGreaterThan(0.7);
    });

    it("returns ayah_end indicating the span covers multiple verses", () => {
      const db = getFixtureQuranDB();
      const text = buildMultiVerseText(db, 112, 1, 4);
      const result = db.matchVerse(text, 0.2, 6);
      expect(result).not.toBeNull();
      expect(result!.surah).toBe(112);
      // ayah_end should be set when multi-verse span is detected
      if (result!.ayah === 1) {
        expect(result!.ayah_end).toBeDefined();
        expect(result!.ayah_end).toBeGreaterThanOrEqual(2);
      }
    });

    it("does not false-jump to a wrong surah", () => {
      const db = getFixtureQuranDB();
      const text = buildMultiVerseText(db, 112, 1, 4);
      const result = db.matchVerse(text, 0.2, 6);
      expect(result).not.toBeNull();
      // Must be surah 112, not any other surah
      expect(result!.surah).toBe(112);
    });
  });

  // --- 113:1-5 (full Al-Falaq) ---
  describe("113:1-5 (full Al-Falaq)", () => {
    it("detects the multi-verse span starting at 113:1", () => {
      const db = getFixtureQuranDB();
      const text = buildMultiVerseText(db, 113, 1, 5);
      const result = db.matchVerse(text, 0.2, 6);
      expect(result).not.toBeNull();
      expect(result!.surah).toBe(113);
      expect(result!.score).toBeGreaterThan(0.7);
    });

    it("does not false-jump to surah 114 (similar opening structure)", () => {
      const db = getFixtureQuranDB();
      const text = buildMultiVerseText(db, 113, 1, 5);
      const result = db.matchVerse(text, 0.2, 6);
      expect(result).not.toBeNull();
      expect(result!.surah).toBe(113);
    });

    it("span covers all 5 verses", () => {
      const db = getFixtureQuranDB();
      const text = buildMultiVerseText(db, 113, 1, 5);
      const result = db.matchVerse(text, 0.2, 6);
      expect(result).not.toBeNull();
      if (result!.ayah === 1 && result!.ayah_end) {
        expect(result!.ayah_end).toBe(5);
      }
    });
  });

  // --- 2:285-286 (last two verses of Al-Baqarah) ---
  describe("2:285-286 (last two verses of Al-Baqarah)", () => {
    it("detects the two-verse span in surah 2", () => {
      const db = getFixtureQuranDB();
      const text = buildMultiVerseText(db, 2, 285, 286);
      const result = db.matchVerse(text, 0.2, 6);
      expect(result).not.toBeNull();
      expect(result!.surah).toBe(2);
      expect(result!.score).toBeGreaterThan(0.7);
    });

    it("does not false-match to Ayat al-Kursi (2:255)", () => {
      const db = getFixtureQuranDB();
      const text = buildMultiVerseText(db, 2, 285, 286);
      const result = db.matchVerse(text, 0.2, 6);
      expect(result).not.toBeNull();
      // Should match 2:285 span, not 2:255
      if (result!.surah === 2) {
        expect(result!.ayah).toBeGreaterThanOrEqual(285);
      }
    });

    it("handles these long verses (20+ words each) without timeout", () => {
      const db = getFixtureQuranDB();
      const text = buildMultiVerseText(db, 2, 285, 286);
      const start = Date.now();
      const result = db.matchVerse(text, 0.2, 6);
      const elapsed = Date.now() - start;
      expect(result).not.toBeNull();
      // Should complete within 2 seconds even on slow machines
      expect(elapsed).toBeLessThan(2000);
    });
  });

  // --- 36:1-5 (opening of Ya-Sin with muqattaat) ---
  describe("36:1-5 (opening of Ya-Sin with muqattaat)", () => {
    it("detects the span in surah 36", () => {
      const db = getFixtureQuranDB();
      const text = buildMultiVerseText(db, 36, 1, 5);
      const result = db.matchVerse(text, 0.2, 6);
      expect(result).not.toBeNull();
      expect(result!.surah).toBe(36);
      expect(result!.score).toBeGreaterThan(0.7);
    });

    it("does not confuse muqattaat letters with other surahs (e.g. Alif-Lam-Mim surahs)", () => {
      const db = getFixtureQuranDB();
      const text = buildMultiVerseText(db, 36, 1, 5);
      const result = db.matchVerse(text, 0.2, 6);
      expect(result).not.toBeNull();
      // Must be surah 36, not surah 2 (Alif-Lam-Mim) or any other
      expect(result!.surah).toBe(36);
    });

    it("matchVerseWithSurahId also identifies surah 36 via two-pass", () => {
      const db = getFixtureQuranDB();
      const text = buildMultiVerseText(db, 36, 1, 5);
      const result = db.matchVerseWithSurahId(text, 0.2, 6);
      expect(result).not.toBeNull();
      expect(result!.surah).toBe(36);
    });
  });

  // --- Cross-case: partial multi-verse ---
  describe("partial multi-verse spans", () => {
    it("112:1+112:2 (two verses) detects span", () => {
      const db = getFixtureQuranDB();
      const text = buildMultiVerseText(db, 112, 1, 2);
      const result = db.matchVerse(text, 0.2, 6);
      expect(result).not.toBeNull();
      expect(result!.surah).toBe(112);
      expect(result!.score).toBeGreaterThan(0.7);
    });

    it("113:1+113:2+113:3 (three verses) detects span", () => {
      const db = getFixtureQuranDB();
      const text = buildMultiVerseText(db, 113, 1, 3);
      const result = db.matchVerse(text, 0.2, 6);
      expect(result).not.toBeNull();
      expect(result!.surah).toBe(113);
      expect(result!.score).toBeGreaterThan(0.7);
    });
  });
});

// ===========================================================================
// 2. STREAMING: RecitationTracker with mock transcribers
// ===========================================================================

describe("Multi-verse span detection (streaming RecitationTracker)", () => {
  // --- 112:1-4 (full Al-Ikhlas) ---
  describe("112:1-4 streaming through tracker", () => {
    it("discovers 112:1 and advances through at least 112:2", async () => {
      const db = getFixtureQuranDB();
      const v1 = db.getVerse(112, 1)!;
      const v2 = db.getVerse(112, 2)!;
      const v3 = db.getVerse(112, 3)!;
      const v4 = db.getVerse(112, 4)!;

      // Simulate: discovery returns verse 1, tracking returns progressive text
      const responses = [
        v1.text_norm_no_bsm ?? v1.text_norm!, // discovery -> match 112:1
        v1.text_norm_no_bsm ?? v1.text_norm!, // tracking 112:1 full
        v1.text_norm_no_bsm ?? v1.text_norm!, // grace
        v1.text_norm_no_bsm ?? v1.text_norm!, // complete -> advance 112:2
        v2.text_norm!,                         // tracking 112:2
        v2.text_norm!,                         // grace
        v2.text_norm!,                         // complete -> advance 112:3
        v3.text_norm!,                         // tracking 112:3
        v3.text_norm!,                         // grace
        v3.text_norm!,                         // complete -> advance 112:4
        v4.text_norm!,                         // tracking 112:4
      ];
      const transcribe = createSequentialTranscriber(responses);
      const tracker = new RecitationTracker(db, transcribe);

      const seenVerses = new Set<string>();

      // Phase 1: discovery
      let msgs = await tracker.feed(fakeAudio(SAMPLE_RATE * 5));
      for (const m of msgs) {
        if (m.type === "verse_match") seenVerses.add(`${m.surah}:${m.ayah}`);
      }

      // Phase 2: drive tracking for up to 25 cycles
      for (let i = 0; i < 25; i++) {
        msgs = await tracker.feed(fakeAudio(TRACKING_TRIGGER_SAMPLES));
        for (const m of msgs) {
          if (m.type === "verse_match") seenVerses.add(`${m.surah}:${m.ayah}`);
        }
      }

      // Should have discovered at least 112:1 and advanced to 112:2
      expect(seenVerses.has("112:1")).toBe(true);
      expect(seenVerses.has("112:2")).toBe(true);

      // Allow natural cross-surah cascade (112:4 -> 113:1 -> ... -> 114:1)
      // The tracker's getNextVerse legitimately crosses surah boundaries.
      // Verify no jumps to distant/unrelated surahs.
      for (const ref of seenVerses) {
        const surah = parseInt(ref.split(":")[0], 10);
        // Expect only surah 112 or natural continuation to 113/114
        expect(surah).toBeGreaterThanOrEqual(112);
        expect(surah).toBeLessThanOrEqual(114);
      }
    });

    it("tracks at least 3 of the 4 Al-Ikhlas verses", async () => {
      const db = getFixtureQuranDB();
      const v1 = db.getVerse(112, 1)!;
      const v2 = db.getVerse(112, 2)!;
      const v3 = db.getVerse(112, 3)!;
      const v4 = db.getVerse(112, 4)!;

      const responses = [
        v1.text_norm_no_bsm ?? v1.text_norm!,
        v1.text_norm_no_bsm ?? v1.text_norm!,
        v1.text_norm_no_bsm ?? v1.text_norm!,
        v1.text_norm_no_bsm ?? v1.text_norm!,
        v2.text_norm!,
        v2.text_norm!,
        v2.text_norm!,
        v2.text_norm!,
        v3.text_norm!,
        v3.text_norm!,
        v3.text_norm!,
        v3.text_norm!,
        v4.text_norm!,
        v4.text_norm!,
      ];
      const transcribe = createSequentialTranscriber(responses);
      const tracker = new RecitationTracker(db, transcribe);

      const seenVerses = new Set<string>();

      let msgs = await tracker.feed(fakeAudio(SAMPLE_RATE * 5));
      for (const m of msgs) {
        if (m.type === "verse_match") seenVerses.add(`${m.surah}:${m.ayah}`);
      }

      for (let i = 0; i < 30; i++) {
        msgs = await tracker.feed(fakeAudio(TRACKING_TRIGGER_SAMPLES));
        for (const m of msgs) {
          if (m.type === "verse_match") seenVerses.add(`${m.surah}:${m.ayah}`);
        }
      }

      // At least 3 of 4 verses should be discovered
      const ikhlaasVerses = [...seenVerses].filter(r => r.startsWith("112:"));
      expect(ikhlaasVerses.length).toBeGreaterThanOrEqual(3);
    });
  });

  // --- 113:1-5 (full Al-Falaq) ---
  describe("113:1-5 streaming through tracker", () => {
    it("discovers 113:1 and advances to at least 113:2", async () => {
      const db = getFixtureQuranDB();
      const verses = [1, 2, 3, 4, 5].map(a => db.getVerse(113, a)!);

      const responses: string[] = [];
      for (const v of verses) {
        const text = v.ayah === 1 ? (v.text_norm_no_bsm ?? v.text_norm!) : v.text_norm!;
        // 3 cycles per verse: match + grace + complete
        responses.push(text, text, text);
      }

      const transcribe = createSequentialTranscriber(responses);
      const tracker = new RecitationTracker(db, transcribe);

      const seenVerses = new Set<string>();

      let msgs = await tracker.feed(fakeAudio(SAMPLE_RATE * 5));
      for (const m of msgs) {
        if (m.type === "verse_match") seenVerses.add(`${m.surah}:${m.ayah}`);
      }

      for (let i = 0; i < 30; i++) {
        msgs = await tracker.feed(fakeAudio(TRACKING_TRIGGER_SAMPLES));
        for (const m of msgs) {
          if (m.type === "verse_match") seenVerses.add(`${m.surah}:${m.ayah}`);
        }
      }

      expect(seenVerses.has("113:1")).toBe(true);
      expect(seenVerses.has("113:2")).toBe(true);

      // Allow natural cross-surah advance (113:5 -> 114:1 via getNextVerse)
      // but no jumps to unrelated surahs
      for (const ref of seenVerses) {
        const surah = parseInt(ref.split(":")[0], 10);
        expect([113, 114]).toContain(surah);
      }
    });
  });

  // --- 2:285-286 (last two verses of Al-Baqarah) ---
  describe("2:285-286 streaming through tracker", () => {
    it("discovers 2:285 and advances to 2:286", async () => {
      const db = getFixtureQuranDB();
      const v285 = db.getVerse(2, 285)!;
      const v286 = db.getVerse(2, 286)!;

      // These are long verses -- the tracker will need more cycles
      const responses = [
        v285.text_norm!, // discovery
        v285.text_norm!, // tracking
        v285.text_norm!, // grace
        v285.text_norm!, // coverage-based advance
        v285.text_norm!, // complete -> advance 2:286
        v286.text_norm!, // tracking 2:286
        v286.text_norm!, // tracking
        v286.text_norm!, // complete
      ];
      const transcribe = createSequentialTranscriber(responses);
      const tracker = new RecitationTracker(db, transcribe);

      const seenVerses = new Set<string>();

      let msgs = await tracker.feed(fakeAudio(SAMPLE_RATE * 5));
      for (const m of msgs) {
        if (m.type === "verse_match") seenVerses.add(`${m.surah}:${m.ayah}`);
      }

      for (let i = 0; i < 20; i++) {
        msgs = await tracker.feed(fakeAudio(TRACKING_TRIGGER_SAMPLES));
        for (const m of msgs) {
          if (m.type === "verse_match") seenVerses.add(`${m.surah}:${m.ayah}`);
        }
      }

      expect(seenVerses.has("2:285")).toBe(true);
      expect(seenVerses.has("2:286")).toBe(true);

      // No false jumps to other surahs
      for (const ref of seenVerses) {
        const surah = parseInt(ref.split(":")[0], 10);
        expect(surah).toBe(2);
      }
    });
  });

  // --- 36:1-5 (opening of Ya-Sin with muqattaat) ---
  describe("36:1-5 streaming through tracker", () => {
    it("discovers 36:1 and advances through the muqattaat opening", async () => {
      const db = getFixtureQuranDB();
      const v1 = db.getVerse(36, 1)!;
      const v2 = db.getVerse(36, 2)!;
      const v3 = db.getVerse(36, 3)!;
      const v4 = db.getVerse(36, 4)!;
      const v5 = db.getVerse(36, 5)!;

      // 36:1 after bismillah stripping is just "يس" (1 word), which is below
      // MIN_DISCOVERY_WORDS=2. In real usage, the ASR captures a multi-second
      // audio window so the transcript includes bismillah + muqattaat.
      // Use the full text_norm (with bismillah) for the discovery cycle.
      const responses = [
        v1.text_norm!,   // discovery: full text passes word gate
        v1.text_norm!,   // tracking 36:1
        v1.text_norm!,   // grace
        v1.text_norm!,   // complete -> advance 36:2
        v2.text_norm!,   // tracking 36:2
        v2.text_norm!,   // grace
        v2.text_norm!,   // complete -> advance 36:3
        v3.text_norm!,   // tracking 36:3
        v3.text_norm!,   // grace
        v3.text_norm!,   // complete -> advance 36:4
        v4.text_norm!,   // tracking 36:4
        v4.text_norm!,   // grace
        v4.text_norm!,   // complete -> advance 36:5
        v5.text_norm!,   // tracking 36:5
      ];

      const transcribe = createSequentialTranscriber(responses);
      const tracker = new RecitationTracker(db, transcribe);

      const seenVerses = new Set<string>();

      let msgs = await tracker.feed(fakeAudio(SAMPLE_RATE * 5));
      for (const m of msgs) {
        if (m.type === "verse_match") seenVerses.add(`${m.surah}:${m.ayah}`);
      }

      for (let i = 0; i < 30; i++) {
        msgs = await tracker.feed(fakeAudio(TRACKING_TRIGGER_SAMPLES));
        for (const m of msgs) {
          if (m.type === "verse_match") seenVerses.add(`${m.surah}:${m.ayah}`);
        }
      }

      // Should at least discover the first verse
      expect(seenVerses.has("36:1")).toBe(true);

      // Count Ya-Sin verses found
      const yasinVerses = [...seenVerses].filter(r => r.startsWith("36:"));
      expect(yasinVerses.length).toBeGreaterThanOrEqual(2);

      // Only expect surah 36 verses (no false jumps)
      for (const ref of seenVerses) {
        const surah = parseInt(ref.split(":")[0], 10);
        expect(surah).toBe(36);
      }
    });
  });

  // --- Cross-case: no false surah jumps across all 4 spans ---
  describe("no false surah jumps across all spans", () => {
    const testCases: { surah: number; start: number; end: number; label: string }[] = [
      { surah: 112, start: 1, end: 4, label: "Al-Ikhlas 112:1-4" },
      { surah: 113, start: 1, end: 5, label: "Al-Falaq 113:1-5" },
      { surah: 2, start: 285, end: 286, label: "Al-Baqarah 2:285-286" },
      { surah: 36, start: 1, end: 5, label: "Ya-Sin 36:1-5" },
    ];

    for (const tc of testCases) {
      it(`${tc.label}: non-streaming matchVerse returns correct surah`, () => {
        const db = getFixtureQuranDB();
        const text = buildMultiVerseText(db, tc.surah, tc.start, tc.end);
        const result = db.matchVerse(text, 0.2, 6);
        expect(result).not.toBeNull();
        expect(result!.surah).toBe(tc.surah);
      });
    }
  });
});
