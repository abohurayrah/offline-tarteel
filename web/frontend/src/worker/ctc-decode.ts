export interface CTCResult {
  /** Decoded Arabic text with spaces (from BPE ▁ word boundaries) */
  text: string;
  /** Raw BPE tokens space-separated */
  rawTokens: string;
}

export interface Hypothesis {
  /** Decoded Arabic text */
  text: string;
  /** Raw BPE tokens space-separated */
  rawTokens: string;
  /** Log probability (normalized by length) */
  score: number;
}

export interface BeamSearchOptions {
  beamWidth?: number;    // default 10
  topK?: number;         // top tokens per frame, default 20
}

// Numerically stable log-space addition: log(exp(a) + exp(b))
function logAdd(a: number, b: number): number {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const max = Math.max(a, b);
  return max + Math.log1p(Math.exp(Math.min(a, b) - max));
}

interface BeamState {
  /** Log prob of paths ending in blank */
  probBlank: number;
  /** Log prob of paths ending in non-blank */
  probNonBlank: number;
  /** Last non-blank token ID emitted */
  lastToken: number;
  /** Sequence of non-blank token IDs */
  tokens: number[];
}

function totalProb(s: BeamState): number {
  return logAdd(s.probBlank, s.probNonBlank);
}

export class CTCDecoder {
  private vocab: Map<number, string>;
  private blankId: number;

  constructor(vocabJson: Record<string, string>) {
    this.vocab = new Map();
    this.blankId = -1;
    for (const [id, token] of Object.entries(vocabJson)) {
      const numId = parseInt(id);
      this.vocab.set(numId, token);
      if (token === "<blank>") {
        this.blankId = numId;
      }
    }
    // Fallback: blank is last token if not found by value
    if (this.blankId === -1) {
      let maxId = 0;
      for (const id of this.vocab.keys()) {
        if (id > maxId) maxId = id;
      }
      this.blankId = maxId;
    }
  }

  decode(logprobs: Float32Array, timeSteps: number, vocabSize: number): CTCResult {
    // argmax per timestep
    const ids: number[] = [];
    for (let t = 0; t < timeSteps; t++) {
      let maxIdx = 0;
      let maxVal = logprobs[t * vocabSize];
      for (let v = 1; v < vocabSize; v++) {
        const val = logprobs[t * vocabSize + v];
        if (val > maxVal) {
          maxVal = val;
          maxIdx = v;
        }
      }
      ids.push(maxIdx);
    }

    // Collapse consecutive duplicates, remove blanks
    const tokens: string[] = [];
    let prev = -1;
    for (const id of ids) {
      if (id !== prev && id !== this.blankId) {
        const token = this.vocab.get(id) ?? "";
        tokens.push(token);
      }
      prev = id;
    }

    // Raw tokens: all tokens space-separated
    const rawTokens = tokens.join(" ");

    // BPE tokens use ▁ (U+2581) as word boundary marker.
    // Concatenate all tokens, then replace ▁ with space.
    const joined = tokens.join("");
    // Replace ▁ with space, then trim
    const text = joined.replace(/▁/g, " ").trim();

    return { text, rawTokens };
  }

  /**
   * CTC prefix beam search.
   * Returns top-K hypotheses sorted by probability (best first).
   *
   * Uses the standard CTC prefix beam search algorithm:
   * - Beams keyed by text prefix for automatic merging of equivalent paths
   * - Separate blank/nonblank probability tracking per prefix
   * - TopK token pruning per frame for efficiency
   */
  beamSearch(
    logprobs: Float32Array,
    timeSteps: number,
    vocabSize: number,
    opts: BeamSearchOptions = {},
  ): Hypothesis[] {
    const beamWidth = opts.beamWidth ?? 10;
    const topK = opts.topK ?? 20;
    const blankId = this.blankId;

    // Initialize with empty prefix
    // Key = joined token IDs (e.g. "5,12,3"), value = BeamState
    let beams = new Map<string, BeamState>();
    beams.set("", {
      probBlank: 0,       // log(1) = 0 — empty prefix starts with blank
      probNonBlank: -Infinity,
      lastToken: -1,
      tokens: [],
    });

    for (let t = 0; t < timeSteps; t++) {
      const frameOffset = t * vocabSize;

      // Get top-K token indices for this frame (by log prob)
      const topTokens = this._getTopKTokens(logprobs, frameOffset, vocabSize, topK, blankId);

      // Get blank log prob for this frame
      const blankLogProb = logprobs[frameOffset + blankId];

      const newBeams = new Map<string, BeamState>();

      for (const [key, beam] of beams) {
        const beamTotal = totalProb(beam);
        // Prune beams with negligible probability early
        if (beamTotal < -100) continue;

        // --- Case 1: Emit blank ---
        // Blank extends the same prefix, adds to probBlank
        const blankProb = beamTotal + blankLogProb;
        if (newBeams.has(key)) {
          const existing = newBeams.get(key)!;
          existing.probBlank = logAdd(existing.probBlank, blankProb);
        } else {
          newBeams.set(key, {
            probBlank: blankProb,
            probNonBlank: -Infinity,
            lastToken: beam.lastToken,
            tokens: beam.tokens,
          });
        }

        // --- Case 2: Emit each non-blank token ---
        for (const [tokenId, tokenLogProb] of topTokens) {
          if (tokenId === beam.lastToken) {
            // Repeat of last token:
            // - From blank path: extends prefix (emits new copy of same token)
            // - From nonblank path: stays on same prefix (CTC collapse)

            // Collapse case: same prefix, only from nonblank
            const collapseProb = beam.probNonBlank + tokenLogProb;
            if (newBeams.has(key)) {
              const existing = newBeams.get(key)!;
              existing.probNonBlank = logAdd(existing.probNonBlank, collapseProb);
            } else {
              newBeams.set(key, {
                probBlank: -Infinity,
                probNonBlank: collapseProb,
                lastToken: beam.lastToken,
                tokens: beam.tokens,
              });
            }

            // Extend case: new prefix with repeated token, only from blank
            const extendTokens = [...beam.tokens, tokenId];
            const extendKey = extendTokens.join(",");
            const extendProb = beam.probBlank + tokenLogProb;
            if (newBeams.has(extendKey)) {
              const existing = newBeams.get(extendKey)!;
              existing.probNonBlank = logAdd(existing.probNonBlank, extendProb);
            } else {
              newBeams.set(extendKey, {
                probBlank: -Infinity,
                probNonBlank: extendProb,
                lastToken: tokenId,
                tokens: extendTokens,
              });
            }
          } else {
            // New token (different from last): extends prefix from either path
            const extendTokens = [...beam.tokens, tokenId];
            const extendKey = extendTokens.join(",");
            const extendProb = beamTotal + tokenLogProb;
            if (newBeams.has(extendKey)) {
              const existing = newBeams.get(extendKey)!;
              existing.probNonBlank = logAdd(existing.probNonBlank, extendProb);
            } else {
              newBeams.set(extendKey, {
                probBlank: -Infinity,
                probNonBlank: extendProb,
                lastToken: tokenId,
                tokens: extendTokens,
              });
            }
          }
        }
      }

      // Prune to top beamWidth by total probability
      const sorted = [...newBeams.entries()].sort(
        (a, b) => totalProb(b[1]) - totalProb(a[1]),
      );
      beams = new Map(sorted.slice(0, beamWidth));
    }

    // Convert final beams to Hypothesis results
    const results: Hypothesis[] = [];
    for (const [, beam] of beams) {
      const tokenStrs = beam.tokens.map(id => this.vocab.get(id) ?? "");
      const rawTokens = tokenStrs.join(" ");
      const joined = tokenStrs.join("");
      const text = joined.replace(/▁/g, " ").trim();

      // Normalize score by sequence length to avoid length bias
      const seqLen = Math.max(beam.tokens.length, 1);
      const score = totalProb(beam) / seqLen;

      if (text.length > 0) {
        results.push({ text, rawTokens, score });
      }
    }

    // Sort by score descending
    results.sort((a, b) => b.score - a.score);
    return results;
  }

  /** Extract top-K tokens by log probability for a single frame.
   *  Always includes blankId in the result. */
  private _getTopKTokens(
    logprobs: Float32Array,
    frameOffset: number,
    vocabSize: number,
    k: number,
    blankId: number,
  ): [number, number][] {
    // Collect all (tokenId, logProb) pairs excluding blank
    // (blank is handled separately in the beam loop)
    const entries: [number, number][] = [];
    for (let v = 0; v < vocabSize; v++) {
      if (v === blankId) continue;
      entries.push([v, logprobs[frameOffset + v]]);
    }
    // Partial sort: only need top-K
    entries.sort((a, b) => b[1] - a[1]);
    return entries.slice(0, k);
  }
}
