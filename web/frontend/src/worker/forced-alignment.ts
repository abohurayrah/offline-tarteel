// ---------------------------------------------------------------------------
// CTC Forced Alignment — Viterbi DP for aligning audio to known Quran text
// ---------------------------------------------------------------------------
//
// Given CTC logprobs [T x V] and a known target text (verse), finds the
// optimal frame-to-token alignment using dynamic programming. Supports
// streaming: extend alignment incrementally as new audio frames arrive.
// ---------------------------------------------------------------------------

// ── Types ──

export interface WordAlignment {
  wordIndex: number;      // 0-based word index in the verse
  word: string;           // original Arabic word text
  startFrame: number;
  endFrame: number;       // exclusive
  confidence: number;     // 0-1, geometric mean of token confidences
}

interface WordInfo {
  wordText: string;
  startTokenIdx: number;  // first BPE token index (inclusive)
  endTokenIdx: number;    // last BPE token index (exclusive)
}

interface TokenBoundary {
  tokenIdx: number;
  startFrame: number;
  endFrame: number;       // exclusive
  confidence: number;
}

// ── BPE Tokenizer ──

/**
 * Greedy longest-match BPE tokenizer using the existing vocab.json.
 * Handles Arabic ▁ word boundaries.
 */
export class BPETokenizer {
  private tokenToId: Map<string, number>;
  private idToToken: Map<number, string>;
  private maxTokenLen: number;

  constructor(vocabJson: Record<string, string>) {
    this.tokenToId = new Map();
    this.idToToken = new Map();
    this.maxTokenLen = 0;

    for (const [id, token] of Object.entries(vocabJson)) {
      const numId = parseInt(id);
      this.tokenToId.set(token, numId);
      this.idToToken.set(numId, token);
      // Track max token length (excluding ▁ prefix for matching purposes)
      const raw = token.startsWith("▁") ? token.slice(1) : token;
      if (raw.length > this.maxTokenLen) this.maxTokenLen = raw.length;
    }
  }

  /**
   * Tokenize Arabic text into BPE token IDs.
   * Returns { tokenIDs, tokenStrings, wordBoundaries }.
   */
  tokenize(text: string): {
    tokenIDs: number[];
    tokenStrings: string[];
    wordBoundaries: WordInfo[];
  } {
    // Normalize: collapse spaces, trim
    text = text.trim().replace(/\s+/g, " ");
    const arabicWords = text.split(" ");

    const tokenIDs: number[] = [];
    const tokenStrings: string[] = [];
    const wordBoundaries: WordInfo[] = [];

    for (let wIdx = 0; wIdx < arabicWords.length; wIdx++) {
      const word = arabicWords[wIdx];
      const startTokenIdx = tokenIDs.length;

      let i = 0;
      let isFirst = true; // first token of word gets ▁ prefix

      while (i < word.length) {
        let matched = false;
        const remaining = word.length - i;
        const maxLen = Math.min(remaining, this.maxTokenLen);

        for (let len = maxLen; len >= 1; len--) {
          const candidate = word.substring(i, i + len);

          if (isFirst) {
            // Try with ▁ prefix first (word boundary marker)
            const withBoundary = "▁" + candidate;
            if (this.tokenToId.has(withBoundary)) {
              const id = this.tokenToId.get(withBoundary)!;
              tokenIDs.push(id);
              tokenStrings.push(withBoundary);
              i += len;
              isFirst = false;
              matched = true;
              break;
            }
          }

          // Try without prefix
          if (this.tokenToId.has(candidate)) {
            const id = this.tokenToId.get(candidate)!;
            tokenIDs.push(id);
            tokenStrings.push(candidate);
            i += len;
            isFirst = false;
            matched = true;
            break;
          }
        }

        if (!matched) {
          // Unknown character — skip it
          console.warn(
            `[FA] BPE: no token for char "${word[i]}" (U+${word.charCodeAt(i).toString(16)}) in word "${word}"`,
          );
          i++;
          isFirst = false;
        }
      }

      wordBoundaries.push({
        wordText: word,
        startTokenIdx,
        endTokenIdx: tokenIDs.length,
      });
    }

    return { tokenIDs, tokenStrings, wordBoundaries };
  }
}

// ── Viterbi DP ──

/**
 * CTC Forced Alignment via Viterbi dynamic programming.
 *
 * State space: [blank_0, token_0, blank_1, token_1, ..., token_{L-1}, blank_L]
 * Total states: 2L + 1 where L = target token sequence length
 *
 * Supports incremental frame extension for streaming.
 */
class ViterbiDP {
  private targetTokenIDs: number[];
  private blankId: number;
  private numStates: number;

  // DP tables — preallocated, extended per frame
  private alpha: Float32Array;   // [maxFrames x numStates]
  private bp: Int16Array;        // backpointers [maxFrames x numStates]
  private maxFrames: number;
  private currentFrame: number = 0;

  // Store per-frame logprobs for confidence computation
  private frameLogprobs: Float32Array[]; // sparse: only target token probs

  constructor(
    targetTokenIDs: number[],
    blankId: number,
    _vocabSize: number,
    maxFrames: number = 3000,
  ) {
    this.targetTokenIDs = targetTokenIDs;
    this.blankId = blankId;
    this.numStates = 2 * targetTokenIDs.length + 1;
    this.maxFrames = maxFrames;

    // Allocate DP tables
    const tableSize = maxFrames * this.numStates;
    this.alpha = new Float32Array(tableSize).fill(-Infinity);
    this.bp = new Int16Array(tableSize).fill(-1);
    this.frameLogprobs = [];

    // Initialize frame 0: can only be in blank_0 (state 0) or token_0 (state 1)
    // We'll initialize properly when first frame arrives
  }

  get framesProcessed(): number {
    return this.currentFrame;
  }

  /**
   * Process initial frame (t=0).
   */
  initFrame(logprobs: Float32Array): void {
    const blankLogP = logprobs[this.blankId];

    // State 0: blank_0
    this.alpha[0] = blankLogP;
    this.bp[0] = 0;

    // State 1: token_0 (first target token)
    if (this.targetTokenIDs.length > 0) {
      const tokenLogP = logprobs[this.targetTokenIDs[0]];
      this.alpha[1] = tokenLogP;
      this.bp[1] = 1;
    }

    // All other states stay at -Infinity (unreachable at t=0)
    this.currentFrame = 1;

    // Store frame logprobs for confidence
    this._storeFrameLogprobs(logprobs);
  }

  /**
   * Extend DP by one frame. Call for each new audio frame.
   * logprobs: Float32Array of size vocabSize (log-probabilities for this frame).
   */
  extendFrame(logprobs: Float32Array): void {
    const t = this.currentFrame;
    if (t >= this.maxFrames) {
      // Grow tables if needed
      this._grow();
    }

    const S = this.numStates;
    const blankLogP = logprobs[this.blankId];
    const prevRow = (t - 1) * S;
    const curRow = t * S;

    for (let s = 0; s < S; s++) {
      const isBlank = s % 2 === 0;
      const tokenIndex = isBlank ? -1 : (s - 1) >> 1; // which target token

      let bestProb = -Infinity;
      let bestPrev = s; // default self-loop

      if (isBlank) {
        // blank_i state (even index)
        const emitLogP = blankLogP;

        // Transition 1: self-loop (stay in blank_i)
        const selfProb = this.alpha[prevRow + s];
        if (selfProb > bestProb) {
          bestProb = selfProb;
          bestPrev = s;
        }

        // Transition 2: from token_{i-1} (state s-1), if s > 0
        if (s > 0) {
          const fromToken = this.alpha[prevRow + s - 1];
          if (fromToken > bestProb) {
            bestProb = fromToken;
            bestPrev = s - 1;
          }
        }

        this.alpha[curRow + s] = bestProb + emitLogP;
        this.bp[curRow + s] = bestPrev;
      } else {
        // token_i state (odd index)
        const emitLogP = logprobs[this.targetTokenIDs[tokenIndex]];

        // Transition 1: self-loop (stay in token_i — CTC collapse)
        const selfProb = this.alpha[prevRow + s];
        if (selfProb > bestProb) {
          bestProb = selfProb;
          bestPrev = s;
        }

        // Transition 2: from blank_i (state s-1)
        const fromBlank = this.alpha[prevRow + s - 1];
        if (fromBlank > bestProb) {
          bestProb = fromBlank;
          bestPrev = s - 1;
        }

        // Transition 3: from token_{i-1} (state s-2), skip blank
        // Only if i > 0 and current token != previous token
        if (tokenIndex > 0 && s >= 2) {
          const prevTokenIdx = tokenIndex - 1;
          if (this.targetTokenIDs[tokenIndex] !== this.targetTokenIDs[prevTokenIdx]) {
            const fromPrevToken = this.alpha[prevRow + s - 2];
            if (fromPrevToken > bestProb) {
              bestProb = fromPrevToken;
              bestPrev = s - 2;
            }
          }
        }

        this.alpha[curRow + s] = bestProb + emitLogP;
        this.bp[curRow + s] = bestPrev;
      }
    }

    this.currentFrame = t + 1;
    this._storeFrameLogprobs(logprobs);
  }

  /**
   * Store relevant logprobs for confidence scoring.
   * Only stores the blank + target token probs (not entire vocab).
   */
  private _storeFrameLogprobs(logprobs: Float32Array): void {
    // Store probabilities of each target token + blank for confidence calc
    const L = this.targetTokenIDs.length;
    const stored = new Float32Array(L + 1); // [blank, token0, token1, ...]
    stored[0] = logprobs[this.blankId];
    for (let i = 0; i < L; i++) {
      stored[i + 1] = logprobs[this.targetTokenIDs[i]];
    }
    this.frameLogprobs.push(stored);
  }

  /**
   * Backtrack to find optimal state path.
   * Returns array of state indices, one per frame.
   */
  backtrack(): number[] {
    const T = this.currentFrame;
    if (T === 0) return [];

    const S = this.numStates;
    const path = new Array<number>(T);

    // Find best final state (last frame)
    const lastRow = (T - 1) * S;
    let bestState = 0;
    let bestProb = this.alpha[lastRow];
    for (let s = 1; s < S; s++) {
      if (this.alpha[lastRow + s] > bestProb) {
        bestProb = this.alpha[lastRow + s];
        bestState = s;
      }
    }
    path[T - 1] = bestState;

    // Follow backpointers
    for (let t = T - 2; t >= 0; t--) {
      bestState = this.bp[(t + 1) * S + bestState];
      path[t] = bestState;
    }

    return path;
  }

  /**
   * Convert state path to token boundaries with confidence scores.
   */
  pathToTokenBoundaries(path: number[]): TokenBoundary[] {
    if (path.length === 0) return [];

    const boundaries: TokenBoundary[] = [];
    let currentTokenIdx = -1;
    let startFrame = -1;
    let sumLogProb = 0;
    let frameCount = 0;

    for (let t = 0; t < path.length; t++) {
      const state = path[t];
      const isBlank = state % 2 === 0;

      if (isBlank) {
        // Blank state — finalize current token if any
        if (currentTokenIdx >= 0) {
          boundaries.push({
            tokenIdx: currentTokenIdx,
            startFrame,
            endFrame: t,
            confidence: this._computeConfidence(sumLogProb, frameCount),
          });
          currentTokenIdx = -1;
        }
      } else {
        const tokenIdx = (state - 1) >> 1;

        if (tokenIdx !== currentTokenIdx) {
          // New token starting
          if (currentTokenIdx >= 0) {
            boundaries.push({
              tokenIdx: currentTokenIdx,
              startFrame,
              endFrame: t,
              confidence: this._computeConfidence(sumLogProb, frameCount),
            });
          }
          currentTokenIdx = tokenIdx;
          startFrame = t;
          sumLogProb = 0;
          frameCount = 0;
        }

        // Accumulate logprob for confidence
        if (t < this.frameLogprobs.length) {
          sumLogProb += this.frameLogprobs[t][tokenIdx + 1]; // +1 because [0]=blank
          frameCount++;
        }
      }
    }

    // Finalize last token
    if (currentTokenIdx >= 0) {
      boundaries.push({
        tokenIdx: currentTokenIdx,
        startFrame,
        endFrame: path.length,
        confidence: this._computeConfidence(sumLogProb, frameCount),
      });
    }

    return boundaries;
  }

  /**
   * Get the furthest token reached by the current best path (without full backtrack).
   * Useful for quick streaming checks.
   */
  getFurthestToken(): number {
    const T = this.currentFrame;
    if (T === 0) return -1;

    const S = this.numStates;
    const lastRow = (T - 1) * S;

    // Find the highest token state that has a non-negligible probability
    let furthest = -1;
    for (let s = S - 1; s >= 0; s--) {
      if (this.alpha[lastRow + s] > -1e10) {
        const isBlank = s % 2 === 0;
        if (isBlank) {
          const blankIdx = s >> 1;
          furthest = blankIdx - 1; // blank_i is after token_{i-1}
        } else {
          furthest = (s - 1) >> 1;
        }
        break;
      }
    }
    return furthest;
  }

  /**
   * Get the best path probability at the current frame.
   */
  getBestPathProb(): number {
    const T = this.currentFrame;
    if (T === 0) return -Infinity;

    const S = this.numStates;
    const lastRow = (T - 1) * S;
    let best = -Infinity;
    for (let s = 0; s < S; s++) {
      if (this.alpha[lastRow + s] > best) {
        best = this.alpha[lastRow + s];
      }
    }
    return best;
  }

  private _computeConfidence(sumLogProb: number, frameCount: number): number {
    if (frameCount === 0) return 0;
    // Average log prob → probability (geometric mean)
    const avgLogProb = sumLogProb / frameCount;
    // Clamp to prevent underflow
    return Math.exp(Math.max(avgLogProb, -20));
  }

  private _grow(): void {
    const newMax = this.maxFrames * 2;
    const S = this.numStates;

    const newAlpha = new Float32Array(newMax * S).fill(-Infinity);
    newAlpha.set(this.alpha);
    this.alpha = newAlpha;

    const newBp = new Int16Array(newMax * S).fill(-1);
    newBp.set(this.bp);
    this.bp = newBp;

    this.maxFrames = newMax;
  }
}

// ── ForcedAligner (main class) ──

export class ForcedAligner {
  private tokenizer: BPETokenizer;
  private viterbi: ViterbiDP;
  private tokenIDs: number[];
  private wordBoundaries: WordInfo[];
  private blankId: number;
  private vocabSize: number;
  private initialized = false;

  // Streaming state
  private lastEmittedWordIdx = -1;
  private stableLag = 8; // frames of lag before considering alignment stable

  constructor(
    targetText: string,
    vocabJson: Record<string, string>,
    blankId: number,
    vocabSize: number,
  ) {
    this.blankId = blankId;
    this.vocabSize = vocabSize;

    // Tokenize
    this.tokenizer = new BPETokenizer(vocabJson);
    const { tokenIDs, tokenStrings, wordBoundaries } = this.tokenizer.tokenize(targetText);
    this.tokenIDs = tokenIDs;
    this.wordBoundaries = wordBoundaries;

    // Initialize Viterbi
    this.viterbi = new ViterbiDP(tokenIDs, blankId, vocabSize);

    console.log(
      `[FA] Initialized: "${targetText.slice(0, 40)}..." → ${tokenIDs.length} BPE tokens, ${wordBoundaries.length} words`,
    );
    console.log(
      `[FA] Tokens: ${tokenStrings.join(" | ")}`,
    );
  }

  get totalWords(): number {
    return this.wordBoundaries.length;
  }

  get totalTokens(): number {
    return this.tokenIDs.length;
  }

  /**
   * Feed raw CTC logprobs for multiple frames.
   * logprobs: Float32Array of size [numFrames x vocabSize]
   *
   * Returns newly completed word alignments (incremental).
   */
  processFrames(
    logprobs: Float32Array,
    numFrames: number,
  ): { newWords: WordAlignment[]; currentWordIdx: number; allWords: WordAlignment[] } {
    // Process each frame
    for (let f = 0; f < numFrames; f++) {
      const frameOffset = f * this.vocabSize;
      const frameLogprobs = logprobs.subarray(frameOffset, frameOffset + this.vocabSize);

      if (!this.initialized) {
        this.viterbi.initFrame(frameLogprobs);
        this.initialized = true;
      } else {
        this.viterbi.extendFrame(frameLogprobs);
      }
    }

    // Get current full alignment
    const allWords = this._getCurrentWordAlignment();

    // Determine which words are newly stable
    const totalFrames = this.viterbi.framesProcessed;
    const newWords: WordAlignment[] = [];

    for (let i = this.lastEmittedWordIdx + 1; i < allWords.length; i++) {
      const w = allWords[i];
      // Word is "stable" if its end frame is at least `stableLag` frames
      // before the current frame, OR if it's the last processable word
      const isStable = w.endFrame <= totalFrames - this.stableLag;
      const isComplete = w.endFrame < totalFrames && w.confidence > 0;

      if (isStable || isComplete) {
        newWords.push(w);
        this.lastEmittedWordIdx = i;
      } else {
        break; // Stop at first unstable word
      }
    }

    // Current word being spoken (furthest token → word mapping)
    const furthestToken = this.viterbi.getFurthestToken();
    let currentWordIdx = -1;
    for (let i = 0; i < this.wordBoundaries.length; i++) {
      if (furthestToken >= this.wordBoundaries[i].startTokenIdx &&
          furthestToken < this.wordBoundaries[i].endTokenIdx) {
        currentWordIdx = i;
        break;
      }
      if (furthestToken >= this.wordBoundaries[i].endTokenIdx) {
        currentWordIdx = i; // passed this word
      }
    }

    return { newWords, currentWordIdx, allWords };
  }

  /**
   * Get complete word alignment from current DP state.
   */
  private _getCurrentWordAlignment(): WordAlignment[] {
    const path = this.viterbi.backtrack();
    const tokenBounds = this.viterbi.pathToTokenBoundaries(path);

    // Map token boundaries to word boundaries
    const wordAlignments: WordAlignment[] = [];

    for (let wIdx = 0; wIdx < this.wordBoundaries.length; wIdx++) {
      const wb = this.wordBoundaries[wIdx];

      // Find token boundaries that belong to this word
      const tokensInWord = tokenBounds.filter(
        (tb) => tb.tokenIdx >= wb.startTokenIdx && tb.tokenIdx < wb.endTokenIdx,
      );

      if (tokensInWord.length === 0) {
        // Word not yet reached by alignment
        continue;
      }

      // Aggregate
      const startFrame = tokensInWord[0].startFrame;
      const endFrame = tokensInWord[tokensInWord.length - 1].endFrame;

      // Geometric mean of token confidences
      let logSum = 0;
      for (const tb of tokensInWord) {
        logSum += Math.log(Math.max(tb.confidence, 1e-10));
      }
      const confidence = Math.exp(logSum / tokensInWord.length);

      wordAlignments.push({
        wordIndex: wIdx,
        word: wb.wordText,
        startFrame,
        endFrame,
        confidence,
      });
    }

    return wordAlignments;
  }

  /**
   * Finalize alignment (call when verse is done).
   * Returns complete word alignment with final confidence scores.
   */
  finalize(): WordAlignment[] {
    return this._getCurrentWordAlignment();
  }

  /**
   * Get overall alignment quality (0-1).
   * Based on fraction of words aligned and average confidence.
   */
  getOverallScore(): number {
    const words = this._getCurrentWordAlignment();
    if (words.length === 0) return 0;

    const coverage = words.length / this.wordBoundaries.length;
    const avgConf =
      words.reduce((s, w) => s + w.confidence, 0) / words.length;

    return coverage * avgConf;
  }

  /**
   * Reset for reuse with same target text (e.g., user retries).
   */
  reset(): void {
    this.viterbi = new ViterbiDP(this.tokenIDs, this.blankId, this.vocabSize);
    this.initialized = false;
    this.lastEmittedWordIdx = -1;
  }
}
