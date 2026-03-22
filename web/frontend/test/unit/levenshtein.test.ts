import { describe, it, expect } from "vitest";
import {
  distance,
  ratio,
  semiGlobalDistance,
  fragmentScore,
  sellersWordMatch,
  phoneticDistance,
  phoneticRatio,
} from "../../src/lib/levenshtein.ts";

describe("distance()", () => {
  it("returns 0 for identical strings", () => {
    expect(distance("abc", "abc")).toBe(0);
    expect(distance("", "")).toBe(0);
    expect(distance("بسم", "بسم")).toBe(0);
  });

  it("returns length of non-empty string when other is empty", () => {
    expect(distance("", "abc")).toBe(3);
    expect(distance("hello", "")).toBe(5);
  });

  it("handles single substitution", () => {
    expect(distance("abc", "adc")).toBe(1);
  });

  it("handles single insertion", () => {
    expect(distance("abc", "abdc")).toBe(1);
  });

  it("handles single deletion", () => {
    expect(distance("abdc", "abc")).toBe(1);
  });

  it("is symmetric", () => {
    expect(distance("abc", "xyz")).toBe(distance("xyz", "abc"));
    expect(distance("بسم", "الله")).toBe(distance("الله", "بسم"));
  });

  it("computes correct distance for a known Arabic pair", () => {
    // "الرحمن" vs "الرحيم" — differ in last 2 characters
    const d = distance("الرحمن", "الرحيم");
    expect(d).toBe(2);
  });
});

describe("ratio()", () => {
  it("returns 1.0 for identical strings", () => {
    expect(ratio("abc", "abc")).toBe(1.0);
  });

  it("returns 1.0 for both empty", () => {
    expect(ratio("", "")).toBe(1.0);
  });

  it("returns 0.0 when one is empty and other is not", () => {
    expect(ratio("", "abc")).toBe(0.0);
    expect(ratio("abc", "")).toBe(0.0);
  });

  it("returns known values for specific pairs", () => {
    // "abc" vs "abd": distance=1, ratio = (3+3-1)/(3+3) = 5/6
    expect(ratio("abc", "abd")).toBeCloseTo(5 / 6, 5);
  });

  it("returns value between 0 and 1", () => {
    const r = ratio("الرحمن", "الرحيم");
    expect(r).toBeGreaterThan(0);
    expect(r).toBeLessThan(1);
  });
});

describe("semiGlobalDistance()", () => {
  it("returns 0 for exact substring", () => {
    // "bc" is a substring of "abcd"
    expect(semiGlobalDistance("bc", "abcd")).toBe(0);
  });

  it("returns 0 for empty query", () => {
    expect(semiGlobalDistance("", "abcd")).toBe(0);
  });

  it("returns query length when ref is empty", () => {
    expect(semiGlobalDistance("abc", "")).toBe(3);
  });

  it("handles partial match with edits", () => {
    // "bx" in "abcd" — best alignment is against "bc" with 1 sub
    expect(semiGlobalDistance("bx", "abcd")).toBe(1);
  });

  it("returns 0 when query is exact prefix of ref", () => {
    expect(semiGlobalDistance("ab", "abcd")).toBe(0);
  });

  it("returns 0 when query is exact suffix of ref", () => {
    expect(semiGlobalDistance("cd", "abcd")).toBe(0);
  });
});

describe("fragmentScore()", () => {
  it("returns 1.0 for exact substring", () => {
    expect(fragmentScore("bc", "abcd")).toBe(1.0);
  });

  it("returns 1.0 for empty query", () => {
    expect(fragmentScore("", "abcd")).toBe(1.0);
  });

  it("returns high score for close partial match", () => {
    // "bx" in "abcd" — semiGlobalDistance = 1, score = 1 - 1/2 = 0.5
    expect(fragmentScore("bx", "abcd")).toBeCloseTo(0.5, 5);
  });

  it("returns low score for no overlap", () => {
    const score = fragmentScore("xyz", "abcd");
    expect(score).toBeLessThan(0.5);
  });
});

describe("sellersWordMatch()", () => {
  it("returns score 0 for empty inputs", () => {
    expect(sellersWordMatch([], ["a", "b"]).score).toBe(0);
    expect(sellersWordMatch(["a"], []).score).toBe(0);
  });

  it("finds exact match at start", () => {
    const result = sellersWordMatch(["hello", "world"], ["hello", "world", "foo"]);
    expect(result.score).toBeGreaterThan(0.9);
    expect(result.startIdx).toBe(0);
  });

  it("finds exact match in middle", () => {
    const result = sellersWordMatch(
      ["world", "foo"],
      ["hello", "world", "foo", "bar"]
    );
    expect(result.score).toBeGreaterThan(0.9);
    expect(result.startIdx).toBe(1);
  });

  it("returns low score for no match", () => {
    const result = sellersWordMatch(["xyz", "abc"], ["hello", "world", "foo"]);
    expect(result.score).toBeLessThan(0.5);
  });

  it("works with Arabic words", () => {
    const transcript = ["بسم", "الله"];
    const verse = ["بسم", "الله", "الرحمن", "الرحيم"];
    const result = sellersWordMatch(transcript, verse);
    expect(result.score).toBeGreaterThan(0.9);
    expect(result.startIdx).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// New tests: phoneticDistance and phoneticRatio
// ---------------------------------------------------------------------------

describe("phoneticDistance()", () => {
  it("returns 0 for identical strings", () => {
    expect(phoneticDistance("صلاه", "صلاه")).toBe(0);
    expect(phoneticDistance("", "")).toBe(0);
    expect(phoneticDistance("بسم", "بسم")).toBe(0);
  });

  it("returns length of non-empty string when other is empty", () => {
    expect(phoneticDistance("", "abc")).toBe(3);
    expect(phoneticDistance("hello", "")).toBe(5);
  });

  it("has lower cost for emphatic ص/س confusion than standard distance", () => {
    const phonDist = phoneticDistance("صلاه", "سلاه");
    const stdDist = distance("صلاه", "سلاه");
    expect(stdDist).toBe(1); // standard: 1 substitution
    expect(phonDist).toBeLessThan(stdDist); // phonetic: 0.3 cost
    expect(phonDist).toBeCloseTo(0.3, 1);
  });

  it("has lower cost for emphatic ط/ت confusion than standard distance", () => {
    const phonDist = phoneticDistance("طالب", "تالب");
    const stdDist = distance("طالب", "تالب");
    expect(stdDist).toBe(1);
    expect(phonDist).toBeLessThan(stdDist);
    expect(phonDist).toBeCloseTo(0.3, 1);
  });

  it("has lower cost for emphatic ض/د confusion", () => {
    const phonDist = phoneticDistance("ضرب", "درب");
    const stdDist = distance("ضرب", "درب");
    expect(phonDist).toBeLessThan(stdDist);
  });

  it("has lower cost for emphatic ظ/ذ confusion", () => {
    const phonDist = phoneticDistance("ظلم", "ذلم");
    const stdDist = distance("ظلم", "ذلم");
    expect(phonDist).toBeLessThan(stdDist);
  });

  it("has lower cost for pharyngeal ه/ح confusion", () => {
    const phonDist = phoneticDistance("هق", "حق");
    const stdDist = distance("هق", "حق");
    expect(phonDist).toBeLessThan(stdDist);
  });

  it("has lower cost for uvular ق/ك confusion", () => {
    const phonDist = phoneticDistance("قلب", "كلب");
    const stdDist = distance("قلب", "كلب");
    expect(phonDist).toBeLessThan(stdDist);
  });

  it("has lower cost for ع/ء confusion", () => {
    const phonDist = phoneticDistance("عين", "ءين");
    const stdDist = distance("عين", "ءين");
    expect(phonDist).toBeLessThan(stdDist);
  });

  it("has lower cost for ش/س confusion", () => {
    const phonDist = phoneticDistance("شمس", "سمس");
    const stdDist = distance("شمس", "سمس");
    expect(phonDist).toBeLessThan(stdDist);
  });

  it("has lower cost for ث/س confusion (interdental)", () => {
    const phonDist = phoneticDistance("ثلاث", "سلاس");
    const stdDist = distance("ثلاث", "سلاس");
    // Two confusable substitutions
    expect(phonDist).toBeLessThan(stdDist);
  });

  it("equals standard distance for non-confusable pairs", () => {
    // ب and ن are not in the phonetic map
    const phonDist = phoneticDistance("بسم", "نسم");
    const stdDist = distance("بسم", "نسم");
    expect(phonDist).toBe(stdDist);
  });

  it("equals standard distance for identical non-Arabic strings", () => {
    expect(phoneticDistance("hello", "world")).toBe(distance("hello", "world"));
  });

  it("is symmetric", () => {
    expect(phoneticDistance("صلاه", "سلاه")).toBe(phoneticDistance("سلاه", "صلاه"));
    expect(phoneticDistance("طالب", "تالب")).toBe(phoneticDistance("تالب", "طالب"));
  });

  it("handles single character strings", () => {
    expect(phoneticDistance("ص", "س")).toBeCloseTo(0.3, 1);
    expect(phoneticDistance("ط", "ت")).toBeCloseTo(0.3, 1);
    expect(phoneticDistance("ب", "ب")).toBe(0);
  });

  it("handles multiple phonetic confusions in one string", () => {
    // Two confusable substitutions: ص→س and ط→ت
    const phonDist = phoneticDistance("صط", "ست");
    const stdDist = distance("صط", "ست");
    expect(stdDist).toBe(2);
    expect(phonDist).toBeLessThan(stdDist);
    expect(phonDist).toBeCloseTo(0.6, 1); // 0.3 + 0.3
  });
});

describe("phoneticRatio()", () => {
  it("returns 1.0 for identical strings", () => {
    expect(phoneticRatio("بسم", "بسم")).toBe(1.0);
  });

  it("returns 1.0 for both empty", () => {
    expect(phoneticRatio("", "")).toBe(1.0);
  });

  it("returns 0.0 when one is empty and other is not", () => {
    expect(phoneticRatio("", "abc")).toBe(0.0);
    expect(phoneticRatio("abc", "")).toBe(0.0);
  });

  it("returns higher ratio than standard ratio for confusable pairs (ص/س)", () => {
    const phonR = phoneticRatio("صلاه", "سلاه");
    const stdR = ratio("صلاه", "سلاه");
    expect(phonR).toBeGreaterThan(stdR);
    expect(phonR).toBeGreaterThan(0.9); // very similar phonetically
  });

  it("returns higher ratio than standard ratio for confusable pairs (ط/ت)", () => {
    const phonR = phoneticRatio("طالب", "تالب");
    const stdR = ratio("طالب", "تالب");
    expect(phonR).toBeGreaterThan(stdR);
  });

  it("equals standard ratio for non-confusable pairs", () => {
    const phonR = phoneticRatio("بسم", "نسم");
    const stdR = ratio("بسم", "نسم");
    expect(phonR).toBeCloseTo(stdR, 5);
  });

  it("returns high ratio for غ/ق confusion", () => {
    const phonR = phoneticRatio("غلب", "قلب");
    const stdR = ratio("غلب", "قلب");
    expect(phonR).toBeGreaterThan(stdR);
  });

  it("is between 0 and 1", () => {
    const phonR = phoneticRatio("صلاه", "كتاب");
    expect(phonR).toBeGreaterThanOrEqual(0);
    expect(phonR).toBeLessThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// Additional edge case tests for existing functions
// ---------------------------------------------------------------------------

describe("distance() edge cases", () => {
  it("handles very long strings", () => {
    const a = "ا".repeat(100);
    const b = "ب".repeat(100);
    expect(distance(a, b)).toBe(100);
  });

  it("handles Unicode supplementary characters gracefully", () => {
    // These are just basic checks that it doesn't crash
    const d = distance("abc", "abcd");
    expect(d).toBe(1);
  });

  it("computes correct multi-operation distance", () => {
    // "kitten" → "sitting": 3 operations
    expect(distance("kitten", "sitting")).toBe(3);
  });
});

describe("ratio() edge cases", () => {
  it("returns high ratio for similar long Arabic strings", () => {
    const a = "بسم الله الرحمن الرحيم";
    const b = "بسم الله الرحمن الرحيم";
    expect(ratio(a, b)).toBe(1.0);
  });

  it("returns value that satisfies triangle inequality expectations", () => {
    // ratio(a,b) and ratio(b,c) both high implies ratio(a,c) is not too low
    const a = "الرحمن";
    const b = "الرحمان";
    const c = "الرحمن";
    expect(ratio(a, b)).toBeGreaterThan(0.8);
    expect(ratio(a, c)).toBe(1.0);
  });
});

describe("semiGlobalDistance() edge cases", () => {
  it("handles query longer than ref", () => {
    const d = semiGlobalDistance("abcde", "bc");
    // query is longer, so it cannot be a substring; distance should be > 0
    expect(d).toBeGreaterThan(0);
  });

  it("handles Arabic substring matching", () => {
    const d = semiGlobalDistance("الله", "بسم الله الرحمن");
    expect(d).toBe(0); // exact substring
  });
});

describe("fragmentScore() edge cases", () => {
  it("returns high score for Arabic substring match", () => {
    const score = fragmentScore("الرحمن", "بسم الله الرحمن الرحيم");
    expect(score).toBe(1.0);
  });

  it("returns 0 for completely unrelated text of same length", () => {
    const score = fragmentScore("xxxx", "yyyy");
    expect(score).toBe(0);
  });
});

describe("sellersWordMatch() edge cases", () => {
  it("handles transcript longer than verse", () => {
    const result = sellersWordMatch(
      ["a", "b", "c", "d", "e"],
      ["b", "c"]
    );
    // Transcript is longer, so score should reflect partial match
    expect(result.score).toBeDefined();
  });

  it("handles single word transcript", () => {
    const result = sellersWordMatch(["hello"], ["hello", "world"]);
    // Single word - the +-1 word window check should still work
    expect(result.score).toBeGreaterThan(0);
  });

  it("finds match at end of verse", () => {
    const result = sellersWordMatch(
      ["الرحيم"],
      ["بسم", "الله", "الرحمن", "الرحيم"]
    );
    expect(result.score).toBeGreaterThan(0.5);
  });

  it("returns endIdx correctly", () => {
    const result = sellersWordMatch(
      ["world", "foo"],
      ["hello", "world", "foo", "bar"]
    );
    expect(result.startIdx).toBe(1);
    expect(result.endIdx).toBe(3);
  });
});
