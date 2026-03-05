/**
 * Maps phoneme-level alignment errors back to word-level corrections.
 *
 * Given predicted phonemes (raw, with | boundaries) and reference phonemes,
 * aligns them and identifies which words contain errors.
 */

import { alignPhonemes } from "./phoneme-aligner";
import type { WordCorrection } from "./types";

/**
 * Compute word-level corrections from phoneme alignment.
 *
 * @param predictedRaw - Raw phoneme string from CTC: "b i s m i | a l l a h i"
 * @param referenceRaw - Reference phoneme string: "b i s m i | a l l a h i"
 * @param maxWordIndex - Only report errors up to this word index (exclusive).
 *                       Use to avoid noise from unrecited portion.
 * @returns Array of word corrections with error details
 */
export function computeCorrection(
  predictedRaw: string,
  referenceRaw: string,
  maxWordIndex?: number,
): WordCorrection[] {
  const predTokens = predictedRaw.trim().split(/\s+/).filter(Boolean);
  const refTokens = referenceRaw.trim().split(/\s+/).filter(Boolean);

  if (!predTokens.length || !refTokens.length) return [];

  // Build word boundary maps for reference: refTokenIndex -> wordIndex
  const refTokenToWord: number[] = [];
  let wordIdx = 0;
  for (const tok of refTokens) {
    if (tok === "|") {
      wordIdx++;
      refTokenToWord.push(-1); // boundary marker, not a real phoneme
    } else {
      refTokenToWord.push(wordIdx);
    }
  }

  // Strip | from both sequences for alignment (align pure phonemes)
  const predClean = predTokens.filter((t) => t !== "|");
  const refClean = refTokens.filter((t) => t !== "|");
  // Map from clean index to word index
  const refCleanToWord: number[] = [];
  let wi = 0;
  for (const tok of refTokens) {
    if (tok === "|") {
      wi++;
    } else {
      refCleanToWord.push(wi);
    }
  }

  const result = alignPhonemes(predClean, refClean);

  if (result.errors.length === 0) return [];

  // Group errors by word index
  const wordErrors = new Map<number, { expected: string[]; got: string[]; type: string }>();

  for (const err of result.errors) {
    const wIdx = err.position < refCleanToWord.length
      ? refCleanToWord[err.position]
      : refCleanToWord[refCleanToWord.length - 1];

    if (maxWordIndex !== undefined && wIdx >= maxWordIndex) continue;

    if (!wordErrors.has(wIdx)) {
      wordErrors.set(wIdx, { expected: [], got: [], type: err.type });
    }
    const entry = wordErrors.get(wIdx)!;
    if (err.expected) entry.expected.push(err.expected);
    if (err.got) entry.got.push(err.got);
    // Use the most severe error type for the word
    if (err.type === "substitution") entry.type = "substitution";
  }

  const corrections: WordCorrection[] = [];
  for (const [wIdx, info] of wordErrors) {
    corrections.push({
      word_index: wIdx,
      expected: info.expected.join(""),
      got: info.got.join(""),
      error_type: info.type as WordCorrection["error_type"],
    });
  }

  return corrections;
}
