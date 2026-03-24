// ---------------------------------------------------------------------------
// CTC Viterbi Direct Verse Scoring
// ---------------------------------------------------------------------------
//
// Instead of decoding CTC logprobs to text and then fuzzy-matching against
// verse candidates, this module scores verse templates DIRECTLY against the
// raw CTC logprob matrix using Viterbi dynamic programming.
//
// Each candidate verse is tokenized into BPE token IDs, and the Viterbi DP
// finds the optimal CTC alignment path through the logprob matrix for that
// token sequence. The resulting path log-probability, normalized by token
// count, gives a direct acoustic score for how well the audio matches the
// verse — no text decoding or Levenshtein distance needed.
//
// Performance: scoring 5-10 candidates x ~500 frames is ~30ms (trivial).
// Memory: pre-tokenized 6,236 verses adds ~1MB — acceptable.
// ---------------------------------------------------------------------------

import { ViterbiDP, BPETokenizer } from "./forced-alignment";

export interface VerseScoringCandidate {
  /** Index into QuranDB.verses or any external identifier */
  index: number;
  /** Pre-tokenized BPE token IDs for this verse */
  tokenIds: number[];
}

export interface VerseScoringResult {
  /** Index matching the input candidate */
  index: number;
  /** Normalized Viterbi score: exp(logProb / tokenCount), range ~(0, 1] */
  score: number;
  /** Raw log-probability from Viterbi best path */
  logProb: number;
}

export class CTCVerseScorer {
  private tokenizer: BPETokenizer;
  private blankId: number;

  constructor(tokenizer: BPETokenizer, blankId: number) {
    this.tokenizer = tokenizer;
    this.blankId = blankId;
  }

  /**
   * Tokenize a verse text into BPE token IDs suitable for Viterbi scoring.
   * Handles Uthmani mark stripping and BPE segmentation.
   */
  tokenizeVerse(text: string): number[] {
    const { tokenIDs } = this.tokenizer.tokenize(text);
    return tokenIDs;
  }

  /**
   * Score multiple verse candidates against a CTC logprob matrix using
   * Viterbi forced alignment.
   *
   * For each candidate, runs a full Viterbi DP forward pass and extracts
   * the best-path log-probability. The score is normalized by token count
   * so that shorter and longer verses are comparable.
   *
   * @param logprobs  Raw CTC logprob matrix as flat Float32Array [timeSteps x vocabSize]
   * @param timeSteps Number of time frames
   * @param vocabSize Vocabulary size (columns in logprob matrix)
   * @param candidates Array of {index, tokenIds} for each candidate verse
   * @returns Scored candidates sorted by score descending (best first)
   */
  scoreVerses(
    logprobs: Float32Array,
    timeSteps: number,
    vocabSize: number,
    candidates: VerseScoringCandidate[],
  ): VerseScoringResult[] {
    const results: VerseScoringResult[] = [];

    for (const cand of candidates) {
      if (!cand.tokenIds.length) continue;

      // Create a fresh ViterbiDP for each candidate
      // maxFrames = timeSteps + 10 for a small safety margin
      const dp = new ViterbiDP(cand.tokenIds, this.blankId, vocabSize, timeSteps + 10);

      // Feed all frames through the DP
      for (let t = 0; t < timeSteps; t++) {
        const frame = logprobs.subarray(t * vocabSize, (t + 1) * vocabSize);
        if (t === 0) {
          dp.initFrame(frame);
        } else {
          dp.extendFrame(frame);
        }
      }

      const logProb = dp.getBestPathProb();

      // Normalize by token count so shorter/longer verses are comparable.
      // exp(logProb / N) gives the geometric mean probability per token,
      // which is a natural normalization for CTC alignment scores.
      const score = Math.exp(logProb / cand.tokenIds.length);

      results.push({ index: cand.index, score, logProb });
    }

    // Sort by score descending (best match first)
    results.sort((a, b) => b.score - a.score);
    return results;
  }
}
