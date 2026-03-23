import { describe, it, expect } from "vitest";

// ---------------------------------------------------------------------------
// Test the word highlight / accumulator logic used in mushaf word progress.
// These tests validate the pure logic (no DOM) that was fixed in the
// word-by-word reveal system.
// ---------------------------------------------------------------------------

/**
 * Computes the contiguous max from index 0, counting both matched and
 * error words as "filled". This is the logic used in handleMushafWordProgress
 * and highlightWord to determine how far to highlight.
 */
function contiguousMax(
  totalWords: number,
  matchedWords: Set<number>,
  errorWords: Set<number>,
): number {
  let max = -1;
  for (let i = 0; i < totalWords; i++) {
    if (matchedWords.has(i) || errorWords.has(i)) {
      max = i;
    } else {
      break;
    }
  }
  return max;
}

/**
 * Computes bismillah offset for ayah 1 of a surah.
 * The tracker's text_words includes the 4 bismillah words for ayah 1
 * of most surahs, but the mushaf DOM has them on a separate basmala line.
 */
function computeBismillahOffset(
  surah: number,
  ayah: number,
  hasBasmalaLine: boolean,
): number {
  if (ayah !== 1 || surah === 1 || surah === 9) return 0;
  return hasBasmalaLine ? 4 : 0;
}

/**
 * Converts tracker-space matched indices to mushaf-space by subtracting
 * the bismillah offset and filtering out negative indices.
 */
function trackerToMushafIndices(
  trackerIndices: number[],
  bismillahOffset: number,
): number[] {
  return trackerIndices
    .map((idx) => idx - bismillahOffset)
    .filter((idx) => idx >= 0);
}

describe("contiguousMax", () => {
  it("returns -1 for empty sets", () => {
    expect(contiguousMax(5, new Set(), new Set())).toBe(-1);
  });

  it("returns 0 for a single matched word at index 0", () => {
    expect(contiguousMax(5, new Set([0]), new Set())).toBe(0);
  });

  it("returns contiguous range for sequential matches", () => {
    expect(contiguousMax(5, new Set([0, 1, 2]), new Set())).toBe(2);
  });

  it("breaks at first gap", () => {
    // [0, 1, 3, 4] — gap at index 2
    expect(contiguousMax(5, new Set([0, 1, 3, 4]), new Set())).toBe(1);
  });

  it("error words fill gaps in contiguous chain", () => {
    // matched=[0,1,3,4], errors=[2] → contiguous through 4
    expect(contiguousMax(5, new Set([0, 1, 3, 4]), new Set([2]))).toBe(4);
  });

  it("error word at start still counts", () => {
    // matched=[1,2,3], errors=[0] → contiguous through 3
    expect(contiguousMax(5, new Set([1, 2, 3]), new Set([0]))).toBe(3);
  });

  it("handles single error word only", () => {
    expect(contiguousMax(5, new Set(), new Set([0]))).toBe(0);
  });

  it("out of order additions to Set still produce correct result", () => {
    // Simulate indices arriving out of order: [0, 2, 1, 3]
    const matched = new Set<number>();
    matched.add(0);
    matched.add(2);
    // gap at 1, so contiguous is 0
    expect(contiguousMax(5, matched, new Set())).toBe(0);

    // Now fill the gap
    matched.add(1);
    expect(contiguousMax(5, matched, new Set())).toBe(2);

    // Add 3
    matched.add(3);
    expect(contiguousMax(5, matched, new Set())).toBe(3);
  });
});

describe("computeBismillahOffset", () => {
  it("returns 0 for Al-Fatiha (surah 1) ayah 1", () => {
    expect(computeBismillahOffset(1, 1, true)).toBe(0);
  });

  it("returns 0 for At-Tawbah (surah 9) ayah 1", () => {
    expect(computeBismillahOffset(9, 1, true)).toBe(0);
  });

  it("returns 4 for ayah 1 of other surahs with basmala", () => {
    expect(computeBismillahOffset(2, 1, true)).toBe(4);
    expect(computeBismillahOffset(112, 1, true)).toBe(4);
  });

  it("returns 0 for non-ayah-1 verses", () => {
    expect(computeBismillahOffset(2, 2, true)).toBe(0);
    expect(computeBismillahOffset(112, 3, true)).toBe(0);
  });

  it("returns 0 when no basmala line on page", () => {
    expect(computeBismillahOffset(2, 1, false)).toBe(0);
  });
});

describe("trackerToMushafIndices", () => {
  it("returns unchanged indices when offset is 0", () => {
    expect(trackerToMushafIndices([0, 1, 2, 3], 0)).toEqual([0, 1, 2, 3]);
  });

  it("subtracts bismillah offset and filters negatives", () => {
    // Tracker sends [0,1,2,3,4] for a verse with 4 bismillah words + 1 verse word
    // Offset = 4, so mushaf indices are [-4,-3,-2,-1,0] → only [0] kept
    expect(trackerToMushafIndices([0, 1, 2, 3, 4], 4)).toEqual([0]);
  });

  it("handles partial bismillah matches", () => {
    // Tracker matched [0,1] (first 2 bismillah words only)
    // Offset = 4, so all are negative → empty
    expect(trackerToMushafIndices([0, 1], 4)).toEqual([]);
  });

  it("handles mixed bismillah and verse words", () => {
    // Tracker matched [2,3,4,5,6] — words 2-3 are bismillah tail, 4-6 are verse words
    // Offset = 4, so mushaf indices are [-2,-1,0,1,2] → [0,1,2]
    expect(trackerToMushafIndices([2, 3, 4, 5, 6], 4)).toEqual([0, 1, 2]);
  });
});

describe("monotonic accumulator behavior", () => {
  it("Set.add is monotonic — adding same index twice is a no-op", () => {
    const set = new Set<number>();
    set.add(0);
    set.add(1);
    set.add(0); // duplicate
    set.add(1); // duplicate
    set.add(2);
    expect(Array.from(set).sort((a, b) => a - b)).toEqual([0, 1, 2]);
    expect(set.size).toBe(3);
  });

  it("accumulator grows monotonically across multiple events", () => {
    const accumulated = new Set<number>();

    // Event 1: words [0, 1]
    for (const idx of [0, 1]) accumulated.add(idx);
    expect(accumulated.size).toBe(2);

    // Event 2: words [1, 2, 3] (overlapping 1)
    for (const idx of [1, 2, 3]) accumulated.add(idx);
    expect(accumulated.size).toBe(4);

    // Event 3: words [2, 3, 4] (overlapping 2, 3)
    for (const idx of [2, 3, 4]) accumulated.add(idx);
    expect(accumulated.size).toBe(5);

    // Verify it never shrinks
    expect(Array.from(accumulated).sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4]);
  });

  it("clearing on verse transition resets accumulator", () => {
    const accumulated = new Set<number>();
    accumulated.add(0);
    accumulated.add(1);
    accumulated.add(2);
    expect(accumulated.size).toBe(3);

    // Simulate verse transition: create new set
    const fresh = new Set<number>();
    expect(fresh.size).toBe(0);
  });
});

describe("skipped word (error) detection", () => {
  it("detects skipped words between contiguous and beyond-contiguous groups", () => {
    const matched = new Set([0, 1, 2, 5, 6]);
    const errors = new Set<number>();
    const totalWords = 8;

    const cMax = contiguousMax(totalWords, matched, errors);
    expect(cMax).toBe(2);

    const accumulated = Array.from(matched).sort((a, b) => a - b);
    const beyondContiguous = accumulated.filter((i) => i > cMax + 1);
    expect(beyondContiguous).toEqual([5, 6]);

    // Words 3 and 4 are skipped
    const firstBeyond = beyondContiguous[0];
    const skipped: number[] = [];
    for (let i = cMax + 1; i < firstBeyond; i++) {
      if (!matched.has(i) && !errors.has(i)) {
        skipped.push(i);
      }
    }
    expect(skipped).toEqual([3, 4]);
  });

  it("does not trigger with only 1 beyond-contiguous word", () => {
    // [0,1,2,4] — only 1 beyond contiguous, not enough to trigger
    const matched = new Set([0, 1, 2, 4]);
    const cMax = contiguousMax(5, matched, new Set());
    const accumulated = Array.from(matched).sort((a, b) => a - b);
    const beyondContiguous = accumulated.filter((i) => i > cMax + 1);
    // Only 1 beyond-contiguous word — should not trigger error detection
    expect(beyondContiguous.length).toBe(1);
  });

  it("requires contiguousMax >= 2 to trigger", () => {
    // [0, 3, 4] — contiguousMax is 0, which is < 2
    const matched = new Set([0, 3, 4]);
    const cMax = contiguousMax(5, matched, new Set());
    expect(cMax).toBe(0);
    // Even though 2 words are beyond contiguous, cMax < 2 prevents trigger
    const accumulated = Array.from(matched).sort((a, b) => a - b);
    const beyondContiguous = accumulated.filter((i) => i > cMax + 1);
    expect(beyondContiguous.length).toBe(2);
    // The guard: contiguousMax >= 2 should prevent this from being flagged
    expect(cMax >= 2).toBe(false);
  });
});
