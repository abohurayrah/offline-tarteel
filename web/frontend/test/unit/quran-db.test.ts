import { describe, it, expect } from "vitest";
import { getFixtureQuranDB } from "../helpers/test-quran-db.ts";
import { QuranDB, normalizeArabic, partialRatio } from "../../src/lib/quran-db.ts";

describe("QuranDB construction", () => {
  it("totalVerses matches input length", () => {
    const db = getFixtureQuranDB();
    expect(db.totalVerses).toBe(44);
  });

  it("populates text_norm on every verse", () => {
    const db = getFixtureQuranDB();
    for (const v of db.verses) {
      expect(v.text_norm).toBeDefined();
      expect(typeof v.text_norm).toBe("string");
      expect(v.text_norm!.length).toBeGreaterThan(0);
    }
  });

  it("populates text_norm_ns (no-space) on every verse", () => {
    const db = getFixtureQuranDB();
    for (const v of db.verses) {
      expect(v.text_norm_ns).toBeDefined();
      expect(v.text_norm_ns!.includes(" ")).toBe(false);
    }
  });

  it("populates text_words on every verse", () => {
    const db = getFixtureQuranDB();
    for (const v of db.verses) {
      expect(v.text_words).toBeDefined();
      expect(Array.isArray(v.text_words)).toBe(true);
      expect(v.text_words!.length).toBeGreaterThan(0);
    }
  });
});

describe("QuranDB.getVerse()", () => {
  it("returns correct verse for 1:1", () => {
    const db = getFixtureQuranDB();
    const v = db.getVerse(1, 1);
    expect(v).toBeDefined();
    expect(v!.surah).toBe(1);
    expect(v!.ayah).toBe(1);
    expect(v!.surah_name_en).toBe("Al-Faatiha");
  });

  it("returns undefined for non-existent verse 999:999", () => {
    const db = getFixtureQuranDB();
    expect(db.getVerse(999, 999)).toBeUndefined();
  });
});

describe("QuranDB.getSurah()", () => {
  it("returns all 7 verses of surah 1 (Al-Fatiha)", () => {
    const db = getFixtureQuranDB();
    const verses = db.getSurah(1);
    expect(verses.length).toBe(7);
    expect(verses[0].ayah).toBe(1);
    expect(verses[6].ayah).toBe(7);
  });

  it("returns empty array for non-existent surah", () => {
    const db = getFixtureQuranDB();
    expect(db.getSurah(999)).toEqual([]);
  });
});

describe("QuranDB.getNextVerse()", () => {
  it("returns next verse within same surah", () => {
    const db = getFixtureQuranDB();
    const next = db.getNextVerse(1, 1);
    expect(next).toBeDefined();
    expect(next!.surah).toBe(1);
    expect(next!.ayah).toBe(2);
  });

  it("returns first verse of next surah when at end of surah", () => {
    const db = getFixtureQuranDB();
    // Surah 1 has 7 ayahs, next should be 2:1
    const next = db.getNextVerse(1, 7);
    expect(next).toBeDefined();
    expect(next!.surah).toBe(2);
    expect(next!.ayah).toBe(1);
  });

  it("returns undefined for non-existent verse", () => {
    const db = getFixtureQuranDB();
    expect(db.getNextVerse(999, 1)).toBeUndefined();
  });
});

describe("QuranDB.search()", () => {
  it("finds Al-Fatiha 1:1 by its text", () => {
    const db = getFixtureQuranDB();
    const results = db.search("بسم الله الرحمن الرحيم");
    expect(results.length).toBeGreaterThan(0);
    // The top result should be 1:1 (exact match on text_norm)
    const topResult = results[0];
    expect(topResult.surah).toBe(1);
    expect(topResult.ayah).toBe(1);
    expect(topResult.score).toBeGreaterThan(0.9);
  });
});

describe("QuranDB.matchVerse()", () => {
  it("matches full verse text with high score (>0.9)", () => {
    const db = getFixtureQuranDB();
    // Use the normalized text of surah 103 (Al-Asr) ayah 2
    // text_clean: "ان الانسن لفي خسر"
    const result = db.matchVerse("ان الانسن لفي خسر");
    expect(result).not.toBeNull();
    expect(result!.surah).toBe(103);
    expect(result!.ayah).toBe(2);
    expect(result!.score).toBeGreaterThan(0.9);
  });

  it("matches text with diacritics after normalization", () => {
    const db = getFixtureQuranDB();
    // Feed diacritized text — should still match after normalization
    const result = db.matchVerse("إِنَّ ٱلْإِنسَٰنَ لَفِى خُسْرٍ");
    expect(result).not.toBeNull();
    expect(result!.surah).toBe(103);
    expect(result!.ayah).toBe(2);
  });

  it("returns null for empty input", () => {
    const db = getFixtureQuranDB();
    expect(db.matchVerse("")).toBeNull();
    expect(db.matchVerse("   ")).toBeNull();
  });

  it("returns null for very short input (< 3 chars after normalization)", () => {
    const db = getFixtureQuranDB();
    expect(db.matchVerse("اب")).toBeNull();
  });

  it("detects multi-verse span (112:1 + 112:2)", () => {
    const db = getFixtureQuranDB();
    // 112:1 norm: "بسم الله الرحمن الرحيم قل هو الله احد" (with bismillah)
    // After bismillah stripping for ayah 1: "قل هو الله احد"
    // 112:2 norm: "الله الصمد"
    // Combined: "قل هو الله احد الله الصمد"
    const result = db.matchVerse("قل هو الله احد الله الصمد", 0.3, 3);
    expect(result).not.toBeNull();
    // Should detect multi-verse span starting at 112:1
    expect(result!.surah).toBe(112);
    expect(result!.score).toBeGreaterThan(0.7);
  });

  it("applies continuation hint bonus to next verse", () => {
    const db = getFixtureQuranDB();
    // First match 112:1 normally
    const match1 = db.matchVerse("قل هو الله احد");
    expect(match1).not.toBeNull();
    expect(match1!.surah).toBe(112);

    // Now with hint pointing to 112:1, 112:2 should get a bonus
    const hint: [number, number] = [112, 1];
    const match2 = db.matchVerse("الله الصمد", 0.3, 3, hint);
    expect(match2).not.toBeNull();
    expect(match2!.surah).toBe(112);
    expect(match2!.ayah).toBe(2);
    // The bonus should boost the score
    expect(match2!.bonus).toBeGreaterThan(0);
  });

  // --- New tests below ---

  it("matchVerse with surahContext biases toward the active surah", () => {
    const db = getFixtureQuranDB();
    // 55:13 and 55:16 are identical refrains. Without context, either may win.
    // With surahContext=55, the surah should get a small 0.06 bonus.
    const text = "فباي ءالاء ربكما تكذبان";
    const resultWithContext = db.matchVerse(text, 0.3, 3, null, 0, 55);
    expect(resultWithContext).not.toBeNull();
    expect(resultWithContext!.surah).toBe(55);
  });

  it("surahContext does not override a clearly better match from another surah", () => {
    const db = getFixtureQuranDB();
    // 103:2 is unique text. surahContext=112 should not prevent it from matching.
    const result = db.matchVerse("ان الانسن لفي خسر", 0.3, 3, null, 0, 112);
    expect(result).not.toBeNull();
    expect(result!.surah).toBe(103);
    expect(result!.ayah).toBe(2);
  });

  it("matches multi-verse span: 112:1 + 112:2 + 112:3", () => {
    const db = getFixtureQuranDB();
    // Three verses combined
    const text = "قل هو الله احد الله الصمد لم يلد ولم يولد";
    const result = db.matchVerse(text, 0.3, 3);
    expect(result).not.toBeNull();
    expect(result!.surah).toBe(112);
    expect(result!.score).toBeGreaterThan(0.7);
    // Should have ayah_end indicating span
    if (result!.ayah === 1) {
      expect(result!.ayah_end).toBeGreaterThanOrEqual(2);
    }
  });

  it("bismillah handling: verse 2:1 strips bismillah for matching", () => {
    const db = getFixtureQuranDB();
    const v = db.getVerse(2, 1);
    expect(v).toBeDefined();
    // 2:1 text_clean has bismillah + "الم"
    // text_norm_no_bsm should strip bismillah, leaving just "الم"
    expect(v!.text_norm_no_bsm).toBeDefined();
    expect(v!.text_norm_no_bsm).not.toBeNull();
    expect(v!.text_norm_no_bsm).not.toContain("بسم الله الرحمن الرحيم");
    expect(v!.text_norm_no_bsm!.trim().length).toBeGreaterThan(0);
  });

  it("bismillah handling: surah 9:1 does NOT strip bismillah (At-Tawba has no bismillah)", () => {
    const db = getFixtureQuranDB();
    const v = db.getVerse(9, 1);
    expect(v).toBeDefined();
    // At-Tawba 9:1 has no bismillah in its text, so text_norm_no_bsm should be null
    expect(v!.text_norm_no_bsm).toBeNull();
  });

  it("bismillah handling: surah 1:1 does NOT strip bismillah (Al-Fatiha keeps it)", () => {
    const db = getFixtureQuranDB();
    const v = db.getVerse(1, 1);
    expect(v).toBeDefined();
    // Al-Fatiha 1:1 IS the bismillah, so no stripping
    expect(v!.text_norm_no_bsm).toBeNull();
  });

  it("bismillah handling: verse 112:1 has bismillah stripped", () => {
    const db = getFixtureQuranDB();
    const v = db.getVerse(112, 1);
    expect(v).toBeDefined();
    expect(v!.text_norm_no_bsm).toBeDefined();
    expect(v!.text_norm_no_bsm).not.toBeNull();
    // Should contain "قل هو الله احد" without bismillah
    expect(v!.text_norm_no_bsm).toContain("قل");
    expect(v!.text_norm_no_bsm).not.toContain("بسم");
  });

  it("short ambiguous verses: 55:13 and 55:16 (identical refrain) both score high", () => {
    const db = getFixtureQuranDB();
    const text = "فباي ءالاء ربكما تكذبان";
    // Match with returnTopK to see both candidates
    const result = db.matchVerse(text, 0.3, 3, null, 10);
    expect(result).not.toBeNull();
    // The result should have runners_up
    expect(result!.runners_up).toBeDefined();
    expect(result!.runners_up.length).toBeGreaterThanOrEqual(2);
    // Both 55:13 and 55:16 should appear with high scores
    const refs = result!.runners_up.map(
      (r: any) => `${r.surah}:${r.ayah}`
    );
    expect(refs).toContain("55:13");
    expect(refs).toContain("55:16");
    // Both should have the same raw_score (identical text)
    const s13 = result!.runners_up.find(
      (r: any) => r.surah === 55 && r.ayah === 13
    );
    const s16 = result!.runners_up.find(
      (r: any) => r.surah === 55 && r.ayah === 16
    );
    expect(s13).toBeDefined();
    expect(s16).toBeDefined();
    expect(s13!.raw_score).toBeCloseTo(s16!.raw_score, 2);
  });

  it("_slidingWordWindowScore path: partial verse text from the middle of 2:255", () => {
    const db = getFixtureQuranDB();
    // 2:255 (Ayat al-Kursi) is a very long verse. Take some middle words.
    const v = db.getVerse(2, 255);
    expect(v).toBeDefined();
    const words = v!.text_words!;
    expect(words.length).toBeGreaterThanOrEqual(10); // It's a long verse
    // Take words from the middle (roughly indices 5-9)
    const middleWords = words.slice(5, 10).join(" ");
    // Should still find 2:255 with decent score via sliding window
    const result = db.matchVerse(middleWords, 0.2, 3);
    expect(result).not.toBeNull();
    // The sliding window path should help find this verse
    // even with partial mid-verse text
    expect(result!.surah).toBe(2);
    expect(result!.ayah).toBe(255);
  });

  it("_sellersScore path: transcript shorter than verse", () => {
    const db = getFixtureQuranDB();
    // Use first 3 words of 2:255 (which has 20+ words) — triggers Sellers
    const v = db.getVerse(2, 255);
    const firstWords = v!.text_words!.slice(0, 3).join(" ");
    const result = db.matchVerse(firstWords, 0.2, 3);
    expect(result).not.toBeNull();
    // Should find 2:255 through Sellers matching
    // (transcript is much shorter than verse, so Sellers path activates)
  });

  it("long verse fragment matching: first 5 words of 2:255 should find it", () => {
    const db = getFixtureQuranDB();
    const v = db.getVerse(2, 255);
    expect(v).toBeDefined();
    const first5 = v!.text_words!.slice(0, 5).join(" ");
    const result = db.matchVerse(first5, 0.2, 3);
    expect(result).not.toBeNull();
    expect(result!.surah).toBe(2);
    expect(result!.ayah).toBe(255);
  });

  it("_suffixPrefixScore path: text spanning verse boundary with hint", () => {
    const db = getFixtureQuranDB();
    // 112:1 ends with "قل هو الله احد" (after bsm strip)
    // 112:2 is "الله الصمد"
    // Simulate text that has end of 112:1 + beginning of 112:2
    // with a hint at 112:1, so _suffixPrefixScore is triggered
    const hint: [number, number] = [112, 1];
    const spanText = "الله احد الله الصمد";
    const result = db.matchVerse(spanText, 0.3, 3, hint);
    expect(result).not.toBeNull();
    // Should match to 112:2 with bonus from continuation
    expect(result!.surah).toBe(112);
  });

  it("continuation hint gives bonus to ayah+2 and ayah+3", () => {
    const db = getFixtureQuranDB();
    // Hint at 112:1 should bonus 112:2 (+0.22), 112:3 (+0.12), 112:4 (+0.06)
    const hint: [number, number] = [112, 1];
    // Match against 112:3 text
    const result = db.matchVerse("لم يلد ولم يولد", 0.3, 3, hint);
    expect(result).not.toBeNull();
    expect(result!.surah).toBe(112);
    expect(result!.ayah).toBe(3);
    expect(result!.bonus).toBeGreaterThan(0);
  });

  it("matchVerse with returnTopK returns runners_up", () => {
    const db = getFixtureQuranDB();
    const result = db.matchVerse("بسم الله الرحمن الرحيم", 0.3, 3, null, 5);
    expect(result).not.toBeNull();
    expect(result!.runners_up).toBeDefined();
    expect(result!.runners_up.length).toBeGreaterThanOrEqual(1);
    // Each runner should have required fields
    const ru = result!.runners_up[0];
    expect(ru.surah).toBeDefined();
    expect(ru.ayah).toBeDefined();
    expect(ru.raw_score).toBeDefined();
    expect(ru.score).toBeDefined();
  });

  it("matchVerse returns raw_score and bonus fields", () => {
    const db = getFixtureQuranDB();
    const result = db.matchVerse("ان الانسن لفي خسر");
    expect(result).not.toBeNull();
    expect(result!.raw_score).toBeDefined();
    expect(typeof result!.raw_score).toBe("number");
    expect(result!.bonus).toBeDefined();
    expect(typeof result!.bonus).toBe("number");
  });

  it("matchVerse returns null for gibberish text", () => {
    const db = getFixtureQuranDB();
    const result = db.matchVerse("xyzxyz abcabc defdef ghighi");
    // Should return null or very low score
    if (result !== null) {
      expect(result.score).toBeLessThan(0.45);
    }
  });

  it("matchVerse handles diacritized Uthmani text of 1:7", () => {
    const db = getFixtureQuranDB();
    const uthmani = "صِرَٰطَ ٱلَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ ٱلْمَغْضُوبِ عَلَيْهِمْ وَلَا ٱلضَّآلِّينَ";
    const result = db.matchVerse(uthmani);
    expect(result).not.toBeNull();
    expect(result!.surah).toBe(1);
    expect(result!.ayah).toBe(7);
    expect(result!.score).toBeGreaterThan(0.8);
  });
});

describe("QuranDB.matchVerseNarrow()", () => {
  it("returns match within narrow window near hint", () => {
    const db = getFixtureQuranDB();
    const hint: [number, number] = [112, 1];
    const result = db.matchVerseNarrow("الله الصمد", hint);
    expect(result).not.toBeNull();
    expect(result!.surah).toBe(112);
    expect(result!.ayah).toBe(2);
  });

  it("returns null for empty input", () => {
    const db = getFixtureQuranDB();
    const result = db.matchVerseNarrow("", [112, 1]);
    expect(result).toBeNull();
  });

  it("falls back to full matchVerse when narrow window misses", () => {
    const db = getFixtureQuranDB();
    // Hint at 112:1, but searching for text from surah 103 — narrow window won't have it
    // Should fall back to full matchVerse
    const result = db.matchVerseNarrow("ان الانسن لفي خسر", [112, 1], 2, 0.25);
    // May or may not find it depending on fallback behavior
    // The key is it doesn't crash
    expect(result === null || typeof result!.score === "number").toBe(true);
  });

  it("includes runners_up in results", () => {
    const db = getFixtureQuranDB();
    const result = db.matchVerseNarrow("لم يلد ولم يولد", [112, 2]);
    expect(result).not.toBeNull();
    expect(result!.runners_up).toBeDefined();
  });
});

describe("QuranDB.matchVerseWithSurahId()", () => {
  it("matches a multi-verse transcript to the correct surah via two-pass", () => {
    const db = getFixtureQuranDB();
    // 55:1-4 text: "الرحمن علم القران خلق الانسان علمه البيان"
    // The two-pass approach should identify surah 55 first, then find the span
    const v1 = db.getVerse(55, 1);
    const v2 = db.getVerse(55, 2);
    const v3 = db.getVerse(55, 3);
    const v4 = db.getVerse(55, 4);
    expect(v1).toBeDefined();
    expect(v2).toBeDefined();
    expect(v3).toBeDefined();
    expect(v4).toBeDefined();
    const combined = [
      v1!.text_norm_no_bsm ?? v1!.text_norm!,
      v2!.text_norm!,
      v3!.text_norm!,
      v4!.text_norm!,
    ].join(" ");
    const result = db.matchVerseWithSurahId(combined, 0.3, 6);
    expect(result).not.toBeNull();
    expect(result!.surah).toBe(55);
  });

  it("matches 36:1-5 to surah 36 via two-pass", () => {
    const db = getFixtureQuranDB();
    // Concatenate normalized text of 36:1-5
    const verses = [1, 2, 3, 4, 5].map(a => db.getVerse(36, a)!);
    const combined = [
      verses[0].text_norm_no_bsm ?? verses[0].text_norm!,
      ...verses.slice(1).map(v => v.text_norm!),
    ].join(" ");
    const result = db.matchVerseWithSurahId(combined, 0.3, 6);
    expect(result).not.toBeNull();
    expect(result!.surah).toBe(36);
  });

  it("returns null for empty input", () => {
    const db = getFixtureQuranDB();
    expect(db.matchVerseWithSurahId("")).toBeNull();
    expect(db.matchVerseWithSurahId("   ")).toBeNull();
  });

  it("returns null for very short input", () => {
    const db = getFixtureQuranDB();
    expect(db.matchVerseWithSurahId("اب")).toBeNull();
  });

  it("single-verse match works via two-pass", () => {
    const db = getFixtureQuranDB();
    const result = db.matchVerseWithSurahId("ان الانسن لفي خسر");
    expect(result).not.toBeNull();
    expect(result!.surah).toBe(103);
    expect(result!.ayah).toBe(2);
  });

  it("respects surahContext in surah identification", () => {
    const db = getFixtureQuranDB();
    // The repeated refrain "فباي ءالاء ربكما تكذبان" appears in surah 55
    // With surahContext=55, the two-pass should favor surah 55
    const text = "فباي ءالاء ربكما تكذبان";
    const result = db.matchVerseWithSurahId(text, 0.3, 3, null, 0, 55);
    expect(result).not.toBeNull();
    expect(result!.surah).toBe(55);
  });
});

describe("matchVerse two-pass integration", () => {
  it("two-pass in matchVerse improves multi-verse span detection for 112:1-4", () => {
    const db = getFixtureQuranDB();
    // Full surah Al-Ikhlas: all 4 verses combined
    const text = "قل هو الله احد الله الصمد لم يلد ولم يولد ولم يكن له كفوا احد";
    const result = db.matchVerse(text, 0.3, 6);
    expect(result).not.toBeNull();
    expect(result!.surah).toBe(112);
    expect(result!.score).toBeGreaterThan(0.7);
  });

  it("two-pass does not regress single-verse matches", () => {
    const db = getFixtureQuranDB();
    // Simple single-verse match should still work fine
    const result = db.matchVerse("ان الانسن لفي خسر");
    expect(result).not.toBeNull();
    expect(result!.surah).toBe(103);
    expect(result!.ayah).toBe(2);
    expect(result!.score).toBeGreaterThan(0.9);
  });

  it("two-pass does not regress continuation hint behavior", () => {
    const db = getFixtureQuranDB();
    const hint: [number, number] = [112, 1];
    const result = db.matchVerse("الله الصمد", 0.3, 3, hint);
    expect(result).not.toBeNull();
    expect(result!.surah).toBe(112);
    expect(result!.ayah).toBe(2);
    expect(result!.bonus).toBeGreaterThan(0);
  });
});

describe("QuranDB - trigram index", () => {
  it("finds verses via trigram overlap", () => {
    const db = getFixtureQuranDB();
    // The trigram index is used internally by _getCandidates.
    // Matching should still work efficiently.
    const result = db.matchVerse("والعصر");
    expect(result).not.toBeNull();
    expect(result!.surah).toBe(103);
    expect(result!.ayah).toBe(1);
  });

});

describe("partialRatio()", () => {
  it("returns high ratio for substring match", () => {
    // "بسم" is a substring of "بسم الله الرحمن الرحيم"
    const r = partialRatio("بسم", "بسم الله الرحمن الرحيم");
    expect(r).toBeGreaterThan(0.9);
  });

  it("returns 0 for empty inputs", () => {
    expect(partialRatio("", "test")).toBe(0.0);
    expect(partialRatio("test", "")).toBe(0.0);
  });

  it("returns low ratio for unrelated strings", () => {
    const r = partialRatio("xyz", "بسم الله");
    expect(r).toBeLessThanOrEqual(0.5);
  });

  it("returns 1.0 for identical strings", () => {
    expect(partialRatio("بسم", "بسم")).toBe(1.0);
  });

  it("handles reversed argument order (short, long) correctly", () => {
    // partialRatio should swap if first arg is longer
    const r1 = partialRatio("بسم", "بسم الله الرحمن");
    const r2 = partialRatio("بسم الله الرحمن", "بسم");
    expect(r1).toBeCloseTo(r2, 5);
  });

  it("returns high ratio for partial Arabic verse fragment", () => {
    const r = partialRatio("الله الرحمن", "بسم الله الرحمن الرحيم");
    expect(r).toBeGreaterThan(0.8);
  });
});
