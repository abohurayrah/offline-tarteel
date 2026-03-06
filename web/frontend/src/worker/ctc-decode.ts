export interface CTCResult {
  /** Decoded Arabic text with spaces (from BPE ▁ word boundaries) */
  text: string;
  /** Raw BPE tokens space-separated */
  rawTokens: string;
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
}
