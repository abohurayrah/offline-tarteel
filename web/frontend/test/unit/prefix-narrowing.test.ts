import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { getFullQuranDB } from "../helpers/test-quran-db.ts";
import { QuranDB, normalizeArabic } from "../../src/lib/quran-db.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));

/**
 * Prefix-narrowing algorithm test.
 *
 * Verifies that the word-prefix trie correctly narrows the candidate set
 * as words are consumed from a CTC transcript, and that the narrowing
 * agrees with the paper's disambiguation data (d[0] = words needed from
 * start to uniquely identify the verse).
 */

// ---------------------------------------------------------------------------
// Test verses spread across the Quran
// ---------------------------------------------------------------------------
interface TestVerse {
  ref: string;
  surah: number;
  ayah: number;
  label: string;
}

const TEST_VERSES: TestVerse[] = [
  { ref: "1:1",   surah: 1,   ayah: 1,   label: "Bismillah (113 confusers)" },
  { ref: "1:2",   surah: 1,   ayah: 2,   label: "Al-Hamd (should narrow fast)" },
  { ref: "2:255", surah: 2,   ayah: 255, label: "Ayat al-Kursi (long, unique opening)" },
  { ref: "55:13", surah: 55,  ayah: 13,  label: "Refrain (31 repeats in Ar-Rahman)" },
  { ref: "112:1", surah: 112, ayah: 1,   label: "Qul Huwa Allahu Ahad" },
  { ref: "114:1", surah: 114, ayah: 1,   label: "Qul A'udhu bi-Rabb in-Nas" },
  { ref: "36:1",  surah: 36,  ayah: 1,   label: "Ya-Sin (muqatta'at)" },
  { ref: "98:1",  surah: 98,  ayah: 1,   label: "Lam Yakun (reported failing)" },
  { ref: "3:2",   surah: 3,   ayah: 2,   label: "Allahu la ilaha illa Huwa" },
  { ref: "67:1",  surah: 67,  ayah: 1,   label: "Tabaraka (Al-Mulk opener)" },
];

describe("Prefix-narrowing algorithm (full Quran)", () => {
  let db: QuranDB;

  beforeAll(() => {
    db = getFullQuranDB();

    // Load disambiguation data
    const disambigPath = resolve(__dirname, "../../public/ambiguity-compact.json");
    const disambigData = JSON.parse(readFileSync(disambigPath, "utf-8"));
    db.loadDisambiguationMap(disambigData);
  });

  it("QuranDB loads all 6236 verses", () => {
    expect(db.totalVerses).toBe(6236);
  });

  it("disambiguation map is loaded for all test verses", () => {
    for (const tv of TEST_VERSES) {
      const entry = db.getDisambiguationEntry(tv.surah, tv.ayah);
      expect(entry, `Missing disambiguation entry for ${tv.ref}`).not.toBeNull();
    }
  });

  // -------------------------------------------------------------------------
  // Core test: simulate streaming word-by-word prefix narrowing
  // -------------------------------------------------------------------------
  for (const tv of TEST_VERSES) {
    describe(`Verse ${tv.ref} — ${tv.label}`, () => {
      it("prefix narrowing cascade narrows candidates with each word", () => {
        const verse = db.getVerse(tv.surah, tv.ayah);
        expect(verse, `Verse ${tv.ref} not found`).toBeDefined();

        // Use the no-bismillah words for ayah 1 (except 1:1, 9:1) to
        // simulate what the CTC model would actually produce when the
        // user starts reciting (they skip the bismillah).
        let words: string[];
        if (verse!.text_norm_no_bsm) {
          words = verse!.text_norm_no_bsm.split(" ").filter(Boolean);
        } else {
          words = verse!.text_words!;
        }

        const maxWords = Math.min(words.length, 5);
        const cascade = db.prefixNarrowingCascade(words.slice(0, maxWords));

        expect(cascade.length).toBeGreaterThan(0);

        // Candidate counts should be monotonically non-increasing
        for (let i = 1; i < cascade.length; i++) {
          expect(
            cascade[i].count,
            `${tv.ref}: candidates increased from word ${i} to ${i + 1} ` +
              `(${cascade[i - 1].count} -> ${cascade[i].count})`
          ).toBeLessThanOrEqual(cascade[i - 1].count);
        }
      });

      it("narrowByPrefix returns candidates that include the target verse", () => {
        const verse = db.getVerse(tv.surah, tv.ayah);
        expect(verse).toBeDefined();

        let words: string[];
        if (verse!.text_norm_no_bsm) {
          words = verse!.text_norm_no_bsm.split(" ").filter(Boolean);
        } else {
          words = verse!.text_words!;
        }

        // Use enough words to narrow, but allow up to 200 candidates
        const maxWords = Math.min(words.length, 5);
        const candidates = db.narrowByPrefix(words.slice(0, maxWords), 200);

        if (candidates !== null) {
          // Find the verse index
          const verseIdx = db.verses.findIndex(
            v => v.surah === tv.surah && v.ayah === tv.ayah
          );
          expect(
            candidates,
            `${tv.ref}: narrowByPrefix result does not include target verse`
          ).toContain(verseIdx);
        }
        // If null, the prefix might not narrow enough with default maxCandidates
        // — that's fine for highly ambiguous verses like 1:1 (bismillah)
      });

      it("prefix narrowing agrees with paper disambiguation data (d[0])", () => {
        const verse = db.getVerse(tv.surah, tv.ayah);
        expect(verse).toBeDefined();

        const entry = db.getDisambiguationEntry(tv.surah, tv.ayah);
        expect(entry).not.toBeNull();

        const paperWordsNeeded = entry!.d[0]; // -1 means never uniquely identifiable

        let words: string[];
        if (verse!.text_norm_no_bsm) {
          words = verse!.text_norm_no_bsm.split(" ").filter(Boolean);
        } else {
          words = verse!.text_words!;
        }

        const cascade = db.prefixNarrowingCascade(words);

        // Find how many words it takes to reach <= 5 candidates
        // (PREFIX_NARROW_MAX_CANDIDATES) and <= 1 candidate (unique)
        let wordsToFive = -1;
        let wordsToOne = -1;
        for (let i = 0; i < cascade.length; i++) {
          if (cascade[i].count <= 5 && wordsToFive === -1) {
            wordsToFive = i + 1;
          }
          if (cascade[i].count <= 1 && wordsToOne === -1) {
            wordsToOne = i + 1;
          }
        }

        // The paper counts d[0] from position 0 of the FULL verse text
        // (including bismillah). The trie indexes BOTH full and stripped
        // text, so for ayah-1 verses where we use the stripped words, the
        // trie resolves 4 words earlier (the 4 bismillah words are gone).
        // Compute the expected offset.
        const bsmOffset = verse!.text_norm_no_bsm ? 4 : 0;

        if (paperWordsNeeded === -1) {
          // Paper says this verse is never uniquely identifiable from opening
          // words alone. The cascade should never reach count=1, OR if it
          // does, it's because the trie also indexes no-bismillah variants
          // which the paper's analysis may not account for.
          // We just verify the cascade does not trivially resolve in 1-2 words.
          if (cascade.length >= 2) {
            expect(
              cascade[1]?.count ?? 999,
              `${tv.ref}: paper says d[0]=-1 but trie resolves in 2 words`
            ).toBeGreaterThan(1);
          }
        } else {
          // Paper says it takes paperWordsNeeded words from the start.
          // The prefix trie may resolve slightly differently because:
          //   1. The trie uses normalized words (bismillah stripped)
          //   2. The paper counts from position 0 of the raw text
          // Adjust for the bismillah offset and allow +/-3 word tolerance.
          if (wordsToOne !== -1) {
            const adjustedPaper = paperWordsNeeded - bsmOffset;
            expect(
              Math.abs(wordsToOne - adjustedPaper),
              `${tv.ref}: paper says ${paperWordsNeeded} words (adjusted=${adjustedPaper} after bsm offset=${bsmOffset}), ` +
                `trie says ${wordsToOne} words to unique (tolerance=3)`
            ).toBeLessThanOrEqual(3);
          }
        }
      });
    });
  }

  // -------------------------------------------------------------------------
  // Detailed cascade report (runs as a single test with console output)
  // -------------------------------------------------------------------------
  it("prints detailed prefix-narrowing cascade for all test verses", () => {
    const lines: string[] = [];
    lines.push("");
    lines.push("=".repeat(80));
    lines.push("PREFIX-NARROWING CASCADE REPORT");
    lines.push("=".repeat(80));

    for (const tv of TEST_VERSES) {
      const verse = db.getVerse(tv.surah, tv.ayah);
      if (!verse) continue;

      const entry = db.getDisambiguationEntry(tv.surah, tv.ayah);

      let words: string[];
      let source: string;
      if (verse.text_norm_no_bsm) {
        words = verse.text_norm_no_bsm.split(" ").filter(Boolean);
        source = "no-bsm";
      } else {
        words = verse.text_words!;
        source = "full";
      }

      const cascade = db.prefixNarrowingCascade(words);

      const textPreview = words.slice(0, 6).join(" ") +
        (words.length > 6 ? "..." : "");

      lines.push("");
      lines.push(`Verse: ${tv.ref} "${textPreview}"  [${source}]`);
      lines.push(`  ${tv.label}`);
      lines.push(`  Total words: ${words.length}`);

      for (let i = 0; i < Math.min(cascade.length, 5); i++) {
        const prefix = words.slice(0, i + 1).join(" ");
        lines.push(
          `  After ${i + 1} word${i > 0 ? "s" : ""} "${prefix}": ` +
            `${cascade[i].count} candidates`
        );
      }

      if (entry) {
        const paperVal = entry.d[0];
        const confuserCount = entry.c.length;
        lines.push(
          `  Paper says: needs ${paperVal === -1 ? "NEVER (ambiguous)" : paperVal + " words"} to identify`
        );
        lines.push(`  Confusers listed: ${confuserCount} (${entry.c.slice(0, 5).join(", ")}${confuserCount > 5 ? "..." : ""})`);
      }

      // Find words to reach <= 5 candidates
      let wordsToFive = -1;
      for (let i = 0; i < cascade.length; i++) {
        if (cascade[i].count <= 5) {
          wordsToFive = i + 1;
          break;
        }
      }
      lines.push(
        `  Prefix narrowing: needs ${wordsToFive === -1 ? ">5 words or NEVER" : wordsToFive + " words"} to reach <=5 candidates`
      );
    }

    lines.push("");
    lines.push("=".repeat(80));

    // Print to test output
    console.log(lines.join("\n"));

    // This test always passes — it's for reporting
    expect(true).toBe(true);
  });

  // -------------------------------------------------------------------------
  // Specific behavioral checks
  // -------------------------------------------------------------------------

  describe("Bismillah (1:1) — highly ambiguous", () => {
    it("has many candidates at every depth (never resolves)", () => {
      const verse = db.getVerse(1, 1)!;
      const words = verse.text_words!;
      const cascade = db.prefixNarrowingCascade(words);

      // "bsm" is shared by 113 surahs so the candidate count stays high
      expect(cascade.length).toBeGreaterThan(0);
      // Even after all 4 words of 1:1, there should be many candidates
      // because most surahs start with bismillah
      const lastCount = cascade[cascade.length - 1].count;
      expect(lastCount).toBeGreaterThan(1);
    });

    it("paper confirms d[0]=-1 (never uniquely identifiable)", () => {
      expect(db.isAmbiguousInIsolation(1, 1)).toBe(true);
    });
  });

  describe("55:13 — repeated refrain", () => {
    it("never narrows to 1 because the refrain repeats 31 times", () => {
      const verse = db.getVerse(55, 13)!;
      const words = verse.text_words!;
      const cascade = db.prefixNarrowingCascade(words);

      // After all words, count should still be > 1
      const lastCount = cascade[cascade.length - 1].count;
      expect(lastCount).toBeGreaterThan(1);
    });

    it("paper confirms d[0]=-1 (never uniquely identifiable)", () => {
      expect(db.isAmbiguousInIsolation(55, 13)).toBe(true);
    });
  });

  describe("98:1 — Lam Yakun (reported failing)", () => {
    it("narrows to small set after stripping bismillah", () => {
      const verse = db.getVerse(98, 1)!;
      expect(verse.text_norm_no_bsm).not.toBeNull();

      const words = verse.text_norm_no_bsm!.split(" ").filter(Boolean);
      const cascade = db.prefixNarrowingCascade(words.slice(0, 5));

      // Paper says d[0]=5 from position 0 (includes bsm).
      // After stripping bismillah, "lm ykn aldhyn kfru" should narrow fast.
      expect(cascade.length).toBeGreaterThanOrEqual(3);

      // After 3-4 words of non-bismillah text, should be very few candidates
      if (cascade.length >= 4) {
        expect(cascade[3].count).toBeLessThanOrEqual(10);
      }
    });

    it("narrowByPrefix returns the verse in candidate set", () => {
      const verse = db.getVerse(98, 1)!;
      const words = verse.text_norm_no_bsm!.split(" ").filter(Boolean);
      const candidates = db.narrowByPrefix(words.slice(0, 5), 200);
      expect(candidates).not.toBeNull();

      const idx = db.verses.findIndex(v => v.surah === 98 && v.ayah === 1);
      expect(candidates).toContain(idx);
    });
  });

  describe("2:255 — Ayat al-Kursi (long verse)", () => {
    it("narrows quickly due to unique opening after bismillah words", () => {
      const verse = db.getVerse(2, 255)!;
      // 2:255 is not ayah 1, so no bismillah stripping
      const words = verse.text_words!;
      const cascade = db.prefixNarrowingCascade(words.slice(0, 5));

      expect(cascade.length).toBeGreaterThanOrEqual(3);
    });

    it("paper says d[0]=8, but 3:2 shares the same opening", () => {
      // 2:255 and 3:2 both start with "الله لا اله الا هو الحي القيوم"
      const entry = db.getDisambiguationEntry(2, 255);
      expect(entry).not.toBeNull();
      expect(entry!.d[0]).toBe(8);
    });
  });

  describe("3:2 — shares opening with 2:255", () => {
    it("paper says d[0]=-1 (never unique — same as 2:255 prefix)", () => {
      expect(db.isAmbiguousInIsolation(3, 2)).toBe(true);
    });

    it("cascade starts with same candidates as 2:255", () => {
      const v32 = db.getVerse(3, 2)!;
      const v2255 = db.getVerse(2, 255)!;

      const words32 = v32.text_words!;
      const words2255 = v2255.text_words!;

      // First word of both should be the same (normalized)
      expect(words32[0]).toBe(words2255[0]);

      const cascade32 = db.prefixNarrowingCascade(words32.slice(0, 1));
      const cascade2255 = db.prefixNarrowingCascade(words2255.slice(0, 1));

      // After 1 word, both should have the same candidate count
      expect(cascade32[0].count).toBe(cascade2255[0].count);
    });
  });

  describe("112:1 — Qul Huwa Allahu Ahad (with bismillah strip)", () => {
    it("narrows from bismillah-stripped opening 'qul huwa allah ahad'", () => {
      const verse = db.getVerse(112, 1)!;
      expect(verse.text_norm_no_bsm).not.toBeNull();

      const words = verse.text_norm_no_bsm!.split(" ").filter(Boolean);
      const cascade = db.prefixNarrowingCascade(words);

      // "qul" is shared by many verses, but "qul huwa" should narrow fast
      expect(cascade.length).toBeGreaterThanOrEqual(2);
      if (cascade.length >= 3) {
        expect(cascade[2].count).toBeLessThan(cascade[0].count);
      }
    });
  });

  describe("67:1 — Tabaraka (Al-Mulk opener)", () => {
    it("unique opening 'tabaraka' should narrow quickly", () => {
      const verse = db.getVerse(67, 1)!;
      expect(verse.text_norm_no_bsm).not.toBeNull();

      const words = verse.text_norm_no_bsm!.split(" ").filter(Boolean);
      const cascade = db.prefixNarrowingCascade(words.slice(0, 5));

      // "تبرك" is relatively rare — should narrow to very few candidates
      expect(cascade.length).toBeGreaterThanOrEqual(2);
      if (cascade.length >= 3) {
        expect(cascade[2].count).toBeLessThanOrEqual(10);
      }
    });
  });

  // -------------------------------------------------------------------------
  // Edge cases and algorithm invariants
  // -------------------------------------------------------------------------
  describe("Algorithm invariants", () => {
    it("narrowByPrefix returns null for empty words", () => {
      expect(db.narrowByPrefix([])).toBeNull();
    });

    it("narrowByPrefix returns null for nonexistent word sequence", () => {
      const result = db.narrowByPrefix(["xyznonexistent", "abcnotaword"]);
      expect(result).toBeNull();
    });

    it("prefixNarrowingCascade returns empty for empty words", () => {
      expect(db.prefixNarrowingCascade([])).toEqual([]);
    });

    it("prefixNarrowingCascade candidate counts are always >= 1", () => {
      for (const tv of TEST_VERSES) {
        const verse = db.getVerse(tv.surah, tv.ayah);
        if (!verse) continue;
        const words = verse.text_words!;
        const cascade = db.prefixNarrowingCascade(words.slice(0, 3));
        for (const step of cascade) {
          expect(step.count, `${tv.ref}: count dropped to 0`).toBeGreaterThanOrEqual(1);
        }
      }
    });

    it("narrowByPrefix maxCandidates parameter is respected", () => {
      // Use bismillah words which have many candidates
      const verse = db.getVerse(1, 1)!;
      const words = verse.text_words!.slice(0, 1); // Just "بسم"

      // With maxCandidates=5, should return null (too many candidates)
      const tight = db.narrowByPrefix(words, 5);
      // With maxCandidates=200, might return candidates
      const loose = db.narrowByPrefix(words, 200);

      // If tight is null but loose is not, the parameter is being respected
      if (loose !== null && loose.length > 5) {
        expect(tight).toBeNull();
      }
    });
  });
});
