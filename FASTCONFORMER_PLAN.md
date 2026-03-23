# FastConformer CTC Migration Plan — Offline Tarteel

**Date:** March 2026
**Current state:** Whisper-tiny-ar-quran via @huggingface/transformers, 81.6% accuracy on 256-sample corpus
**Target:** FastConformer-CTC (131MB q8 ONNX) + constrained BPE-trie decoding
**Baseline to beat:** 81.6% strict accuracy (209/256), 74.3% on short verses

---

## Context and Key Findings

Before reading this plan, understand these critical facts discovered from the codebase:

**The entire FastConformer pipeline was previously implemented and then deleted.** Commit `d17d1e9` ("feat: replace FastConformer CTC with Whisper ASR via transformers.js") removed five complete files on March 7 2026:
- `mel.ts` — NeMo-compatible 80-bin mel spectrogram with per-feature normalization (112 lines)
- `ctc-decode.ts` — Greedy decoder + full CTC prefix beam search (273 lines at deletion, 421 lines in the trie commit `cc1b6f0`)
- `session.ts` — ONNX Runtime Web session wrapper (47 lines)
- `model-cache.ts` — IndexedDB-based model caching with progress download (70 lines)
- `forced-alignment.ts` — Complete Viterbi CTC forced aligner with BPE tokenizer and streaming support (668 lines)

Additionally, commit `cc1b6f0` ("exp: Quran prefix trie with constrained CTC beam search") implemented:
- `quran-trie.ts` — BPE token-level prefix trie for all 6,236 verses (105 lines)
- `constrainedBeamSearch()` added to `ctc-decode.ts`

**All of this code is recoverable verbatim from git history.** The plan below is therefore primarily a restoration + integration + tuning exercise, not a green-field build.

**Why was FastConformer replaced?** The commit message says "UX test results: 5/8 passed, 20/27 criteria (on par with CTC baseline)." Whisper was adopted because it produced readable Arabic text directly (useful for the prompt-based tracking strategy), while FastConformer's BPE output required more post-processing. However, Whisper-tiny has a hard ceiling on accuracy due to ASR transcription quality — the 8 remaining failures in the 54-sample corpus are all "Whisper output is too garbled to match." FastConformer with constrained decoding bypasses this ceiling entirely.

**vocab.json** already exists at `/Users/omarjarad/Desktop/personal/quran/offline-tarteel/data/vocab.json`. It is a BPE vocabulary with subword tokens including ▁ word-boundary markers (IDs 0–1024, with `<blank>` as the last token ID). The CTCDecoder auto-detects the blank token.

**Current dependencies already present:** `onnxruntime-web@1.24.2` is in `package.json` — the ONNX runtime is already installed and was used by the original pipeline.

---

## Phase 1: Restore the FastConformer Browser Pipeline (Week 1)

**Goal:** Get FastConformer running in the browser, producing Arabic text, benchmarked against Whisper.

### Step 1.1 — Restore `mel.ts`

**File to create:** `web/frontend/src/worker/mel.ts`

Recover from git with:
```
git show d17d1e9^:web/frontend/src/worker/mel.ts > web/frontend/src/worker/mel.ts
```

The file is complete and correct. It implements:
- Pre-emphasis (coeff 0.97), dither (1e-5)
- STFT via `@huggingface/transformers` `spectrogram()` function (already a dependency)
- 80-bin mel filterbank, HTK mel scale, 0–8000 Hz, N_FFT=512, hop=160 (10ms), win=400 (25ms)
- Log with guard (1e-5), per-feature mean/std normalization
- Returns `{ features: Float32Array, timeFrames: number }` in [n_mels, time] layout

**No changes needed.** The `spectrogram`, `mel_filter_bank`, and `window_function` imports from `@huggingface/transformers` are available in v3.8.1.

**Verify:** Write a unit test at `web/frontend/test/unit/mel.test.ts`:
- Feed a 1-second sine wave (440 Hz, 16 kHz sample rate) and assert `timeFrames` equals `Math.floor((16000 - 400) / 160) + 1` = 99 frames
- Assert `features.length === 80 * timeFrames`
- Assert no NaN or Infinity values in features
- Assert all values are in approximately [-5, 5] range (per-feature normalization)

---

### Step 1.2 — Restore `session.ts`

**File to create:** `web/frontend/src/worker/session.ts`

Recover from git with:
```
git show d17d1e9^:web/frontend/src/worker/session.ts > web/frontend/src/worker/session.ts
```

The file is 47 lines. It uses `onnxruntime-web/wasm` (already installed), creates a single ONNX session with WASM execution provider, single-threaded, SIMD enabled. Input names are read from `session.inputNames` dynamically (the FastConformer model has two inputs: mel features `[1, 80, T]` and length `int64 [1]`).

**One update required:** The original code imports from `"onnxruntime-web/wasm"`. In `onnxruntime-web@1.24.2`, the import path may need to be `"onnxruntime-web"` with `ort.env.wasm.wasmPaths` configured. Check by running `node -e "require('onnxruntime-web/wasm')"` — if it fails, change the import to `import * as ort from "onnxruntime-web"` and set `ort.env.wasm.wasmPaths = "/"` to use the WASM files served from the frontend's public directory.

**Verify:** The existing `npm run test:onnx` script tests ONNX loading. After restoring session.ts, add a smoke test that creates a dummy session from a small ONNX model.

---

### Step 1.3 — Restore `model-cache.ts`

**File to create:** `web/frontend/src/worker/model-cache.ts`

Recover from git with:
```
git show d17d1e9^:web/frontend/src/worker/model-cache.ts > web/frontend/src/worker/model-cache.ts
```

The file implements IndexedDB caching with a streaming download that reports progress. The model key is hardcoded as `"fastconformer-ar-ctc-v1"`. This matches the planned model file naming.

**One update required:** The model URL is passed as a parameter. In the new `inference.ts`, pass `"/fastconformer_ar_ctc_q8.onnx"` as the URL — this assumes the model is served from the public directory. If the model is instead served from GitHub Releases (as stated in the prompt), the `loadModel` function already handles remote URLs with progress tracking. The model should be placed in `web/frontend/public/fastconformer_ar_ctc_q8.onnx` for local serving, or configured with the GitHub releases URL.

**Model file placement:** The 131 MB q8 ONNX file should be placed at `web/frontend/public/fastconformer_ar_ctc_q8.onnx`. Add this path to `.gitignore` (it is already there: the existing `.gitignore` excludes `public/models/whisper-base-quran/` — add `public/fastconformer_ar_ctc_q8.onnx` to the same block).

**Verify:** Unit test the cache functions against a real IndexedDB mock, or test manually by verifying the model is cached after first load and not re-downloaded on second load.

---

### Step 1.4 — Restore `ctc-decode.ts` (greedy decoder only)

**File to create:** `web/frontend/src/worker/ctc-decode.ts`

Recover from the trie commit (which has both the original decoder plus the constrained beam search additions):
```
git show cc1b6f0:web/frontend/src/worker/ctc-decode.ts > web/frontend/src/worker/ctc-decode.ts
```

This gives you the complete file (421 lines) including:
- `CTCDecoder` class with greedy `decode()` method
- `beamSearch()` — unconstrained CTC prefix beam search (beamWidth=10, topK=20)
- `constrainedBeamSearch()` — trie-constrained version (Phase 2)
- Exported types: `CTCResult`, `Hypothesis`, `BeamSearchOptions`

The `CTCDecoder` constructor takes `Record<string, string>` (the vocab.json format: `{ "0": "<unk>", "1": "ة", ... }`). It auto-detects blank as the token with value `"<blank>"`, falling back to the max ID if not found.

**Verify:** Unit test at `web/frontend/test/unit/ctc-decode.test.ts`:
- Construct a `CTCDecoder` from a 5-token vocab `{ "0": "<blank>", "1": "ا", "2": "ل", "3": "ل", "4": "ه" }` (this will deduplicate keys — use a proper test with distinct chars)
- Build a synthetic logprob array [3 timesteps × 5 vocab] where timestep 0 favors token 2 (ل), timestep 1 favors token 2 again (duplicate), timestep 2 favors token 4 (ه)
- Assert greedy decode produces "لله" or similar after CTC collapse
- Assert `beamSearch` returns the same top hypothesis as greedy for this simple case

---

### Step 1.5 — Rewrite `inference.ts` to use FastConformer

**File to modify:** `web/frontend/src/worker/inference.ts`

The current `inference.ts` (116 lines) uses `loadWhisper`/`whisperTranscribe` from `whisper-transcriber.ts`. Replace it with the FastConformer pipeline.

Do **not** delete `whisper-transcriber.ts` yet — keep it for A/B comparison via feature flag.

**New `inference.ts` structure:**

```typescript
// Feature flag: set to "fastconformer" or "whisper" to switch backends
const BACKEND: "fastconformer" | "whisper" = "fastconformer";

// FastConformer imports
import { loadModel } from "./model-cache";
import { computeMelSpectrogram } from "./mel";
import { CTCDecoder } from "./ctc-decode";
import { createSession, runInference } from "./session";
// Whisper imports (kept for A/B)
import { loadWhisper, transcribe as whisperTranscribe } from "./whisper-transcriber";

import { QuranDB } from "../lib/quran-db";
import { RecitationTracker } from "../lib/tracker";
import type { TranscribeResult } from "../lib/tracker";
import type { WorkerInbound, WorkerOutbound } from "../lib/types";
import { SAMPLE_RATE } from "../lib/types";

const MODEL_URL = "/fastconformer_ar_ctc_q8.onnx";

let decoder: CTCDecoder | null = null;
let tracker: RecitationTracker | null = null;
let db: QuranDB | null = null;
```

The `transcribe` function for FastConformer:
```typescript
async function transcribe(audio: Float32Array): Promise<TranscribeResult> {
  const { features, timeFrames } = await computeMelSpectrogram(audio);
  const { logprobs, timeSteps, vocabSize } = await runInference(features, 80, timeFrames);
  return decoder!.decode(logprobs, timeSteps, vocabSize);
}
```

The `init` function must:
1. Load `vocab.json` via fetch from `/vocab.json` (copy file to `public/vocab.json`)
2. Construct `CTCDecoder` from vocab
3. Load model via `loadModel(MODEL_URL, progressCallback)`
4. Call `createSession(modelBuffer)`
5. Load quran.json and construct `QuranDB`
6. Warmup: run a 2-second silence chunk through mel + session
7. Construct `RecitationTracker`

**vocab.json placement:** Copy `data/vocab.json` to `web/frontend/public/vocab.json` so it is served statically. Add a build step or document this as a manual step.

**The `reset` message handler** stays identical to the current implementation — it reconstructs a new `RecitationTracker` with the same `transcribe` function.

**The `audio` message handler** stays identical — it calls `processAudio(msg.samples)` which calls `tracker.feed()`.

**Important:** The FastConformer `CTCDecoder.decode()` returns `{ text, rawTokens }` which matches the `TranscribeResult` interface exactly. No adapter change is needed in `tracker.ts`.

**Critical difference from Whisper:** FastConformer cannot accept a text prompt. The `tracker.ts` passes an optional `prompt` parameter to `transcribe()` in tracking mode (`trackingPrompt = this.trackingVerse?.text_norm?.slice(-80)`). The FastConformer `transcribe` function signature must accept and silently ignore this parameter to maintain interface compatibility. Update the `TranscribeFn` type in `tracker.ts` — it already accepts `prompt?: string`, and the FastConformer version will just ignore it.

**Verify:** After wiring, run `npm run dev`, open the browser, speak Al-Fatiha. Confirm the worker loads and produces a `verse_match` message. Check the browser console for any ONNX Runtime errors.

---

### Step 1.6 — Add FastConformer Benchmark Runner

**File to create:** `web/frontend/test/benchmark-fastconformer.ts`

Copy `test/benchmark-whisper.ts` as a starting point. Replace the `@huggingface/transformers` pipeline with the Node.js ONNX pipeline:

```typescript
import * as ort from "onnxruntime-node"; // devDependency, already installed
```

The benchmark uses `onnxruntime-node` (already in devDependencies as `^1.21.0`) to run inference server-side, matching the browser's ONNX model. The mel spectrogram computation uses `computeMelSpectrogram` from `mel.ts` — this works in Node.js since it only uses `@huggingface/transformers` math utilities, not browser APIs.

**Add script to `package.json`:**
```json
"benchmark:fastconformer": "tsx test/benchmark-fastconformer.ts",
"benchmark:fastconformer:greedy": "tsx test/benchmark-fastconformer.ts --decoder=greedy",
"benchmark:fastconformer:beam": "tsx test/benchmark-fastconformer.ts --decoder=beam"
```

The benchmark should measure:
- Transcription accuracy (Levenshtein ratio of output vs normalized verse text)
- Verse identification accuracy (same eval categories as benchmark-whisper.ts: correct/equiv/adjacent/wrong/no_match)
- Inference latency (mel computation time + ONNX session time, separately)
- Save results to `test/benchmark-results/fastconformer-{timestamp}.json`

**Verify:** Run `npm run benchmark:fastconformer -- --sample=5` on the first 5 samples. Confirm it runs without errors. Expected initial accuracy: unknown, but likely 50–70% without constrained decoding (the BPE output will have subword errors that the QuranDB fuzzy matching may or may not recover).

---

## Phase 2: Constrained CTC Decoding (Week 2)

**Goal:** Build the BPE-trie from all 6,236 verses and wire it into beam search. This eliminates hallucinated text and constrains outputs to valid Quran prefixes.

### Step 2.1 — Restore `quran-trie.ts`

**File to create:** `web/frontend/src/worker/quran-trie.ts`

Recover from git:
```
git show cc1b6f0:web/frontend/src/worker/quran-trie.ts > web/frontend/src/worker/quran-trie.ts
```

The `QuranTrie` class (105 lines) imports `BPETokenizer` from `"./forced-alignment"`. Since `forced-alignment.ts` is being restored in Phase 3, you have two options for Phase 2:
- Option A: Restore `forced-alignment.ts` first (just the `BPETokenizer` class). Recommended.
- Option B: Extract `BPETokenizer` into its own file `bpe-tokenizer.ts`. This is cleaner long-term since both the trie and the forced aligner need it.

**Recommended: create `web/frontend/src/worker/bpe-tokenizer.ts`** as a standalone file containing just the `BPETokenizer` class (lines 43–130 of the original `forced-alignment.ts`). Then update both `quran-trie.ts` and `forced-alignment.ts` to import from `"./bpe-tokenizer"`.

`BPETokenizer` implements:
- Greedy longest-match BPE tokenization using `vocab.json` entries
- Word-boundary marker `▁` (U+2581) handling for inter-word boundaries
- Returns `{ tokenIDs: number[], tokenStrings: string[], wordBoundaries: WordInfo[] }`

**`QuranTrie` interface summary:**
- `buildFromVerses(verses)` — tokenizes all 6,236 normalized verse texts and inserts into trie
- `getNode(tokenIDs)` — navigates trie, returns node with `children: Map<number, TrieNode>` and `isVerseEnd: boolean`
- `isValidPrefix(tokenIDs)` — boolean check
- `getValidNextTokens(tokenIDs)` — returns valid next token IDs at a given trie position

**Trie size estimation:** With ~170,000 tokens across 6,236 verses and significant prefix sharing (Quran has heavy repetition — bismillah, common phrases), expected node count is approximately 50,000–80,000 nodes. Each node is a Map entry + two booleans + an array. Estimated memory: ~15–25 MB in JS heap. This is acceptable for a browser worker.

**Build time:** Tokenizing all 6,236 verses at init time. Each verse averages ~27 BPE tokens. Total: ~168,000 tokenizations with longest-match scan. Expected: <500ms on a modern browser.

**Verify:** Unit test at `web/frontend/test/unit/quran-trie.test.ts`:
- Build a mini-trie from 3 known verses (Al-Fatiha 1:1, 1:2, 2:1)
- Assert that the BPE tokenization of "بسم الله الرحمن الرحيم" is a valid prefix in the trie
- Assert that a random Arabic string "كلب في الشارع" is NOT a valid prefix
- Assert `getValidNextTokens([])` returns a non-empty array (root node always has children)
- Assert `isVerseEnd` is true for the complete token sequence of a known verse

---

### Step 2.2 — Wire Constrained Beam Search into `inference.ts`

**File to modify:** `web/frontend/src/worker/inference.ts`

Add trie construction to `init()`:
```typescript
let trie: QuranTrie | null = null;

// In init(), after QuranDB is constructed:
post({ type: "loading_status", message: "Building Quran trie..." });
trie = new QuranTrie(vocabJsonCache);
trie.buildFromVerses(db.verses.filter(v => v.text_norm).map(v => ({
  text_norm: v.text_norm!,
  surah: v.surah,
  ayah: v.ayah,
})));
```

Modify the `transcribe` function to accept a `useConstrainedDecoding` flag or to always use constrained beam search:

```typescript
async function transcribe(audio: Float32Array, _prompt?: string): Promise<TranscribeResult> {
  const { features, timeFrames } = await computeMelSpectrogram(audio);
  const { logprobs, timeSteps, vocabSize } = await runInference(features, 80, timeFrames);

  if (trie) {
    // Constrained beam search: only outputs valid Quran prefixes
    const hypotheses = decoder!.constrainedBeamSearch(
      logprobs, timeSteps, vocabSize, trie,
      { beamWidth: 10, topK: 20 }
    );
    if (hypotheses.length > 0) {
      return { text: hypotheses[0].text, rawTokens: hypotheses[0].rawTokens };
    }
  }
  // Fallback: unconstrained greedy
  return decoder!.decode(logprobs, timeSteps, vocabSize);
}
```

**Performance concern:** Constrained beam search with beamWidth=10 on a 300ms audio chunk (~30 frames after mel) should run in <50ms. For longer utterances (2 seconds = ~197 frames), it could be 200–400ms. The concurrency guard in `inference.ts` (the `busy` flag + `pendingChunks` queue) already handles this correctly — audio chunks arriving during inference are queued.

**The trie constraint changes the QuranDB matching strategy:** When constrained decoding is active, the decoded text is guaranteed to be a valid Quran prefix. This means `QuranDB.matchVerse()` can potentially use exact prefix matching instead of fuzzy Levenshtein, significantly improving accuracy for clean audio. However, for noisy audio where the CTC output deviates slightly from the true verse, fuzzy matching is still needed. Keep both paths.

**Verify:** Run `npm run benchmark:fastconformer -- --decoder=beam --sample=10`. Compare results against `--decoder=greedy`. Expected: constrained beam search should eliminate cases where greedy decoding produces impossible Arabic text (hallucinated words), improving accuracy by 5–15% on noisy/short samples.

---

### Step 2.3 — Trie-Based Direct Verse Identification

**File to modify:** `web/frontend/src/worker/quran-trie.ts`

Add a new method `getMatchingVerses(tokenIDs: number[]): Array<{ surah: number, ayah: number }>`:

```typescript
getMatchingVerses(tokenIDs: number[]): Array<{ surah: number, ayah: number }> {
  // Navigate to the deepest node reachable with the given token sequence
  let node = this.root;
  let deepestVerseRefs: Array<{ surah: number, ayah: number }> = [];

  for (const tokenID of tokenIDs) {
    if (!node.children.has(tokenID)) break;
    node = node.children.get(tokenID)!;
    if (node.isVerseEnd) {
      deepestVerseRefs = [...node.verseRefs];
    }
  }

  return deepestVerseRefs;
}
```

This allows direct lookup: if the constrained beam search produces a complete verse sequence, we can skip QuranDB fuzzy matching entirely and do O(1) trie lookup. For partial sequences, fall back to QuranDB.

**File to modify:** `web/frontend/src/worker/inference.ts`

In the `transcribe` function, after constrained beam search:
```typescript
// Try direct trie-based verse identification first
const topHypothesis = hypotheses[0];
const directMatch = trie.getMatchingVerses(/* topHypothesis token IDs */);
if (directMatch.length === 1) {
  // Exact verse identified — return immediately with high confidence
  return {
    text: topHypothesis.text,
    rawTokens: topHypothesis.rawTokens,
    directMatch: directMatch[0]  // Pass to tracker for immediate verse_match
  };
}
```

**Note:** This requires extending `TranscribeResult` (defined in `tracker.ts`) with an optional `directMatch` field. The `RecitationTracker` would then use this to bypass fuzzy matching in discovery mode when a direct match is available. This is a significant accuracy win for clean audio.

**Assess carefully:** Only implement the `directMatch` shortcut if the Phase 1.6 benchmark shows that the constrained decoder is reliably producing complete verse sequences (not just prefixes) for 2-second audio chunks. If most outputs are partial sequences, the direct lookup path will rarely trigger and is not worth the added complexity yet.

**Verify:** Unit test with a complete 2-second audio clip of a known short verse. Assert that `getMatchingVerses` returns exactly one result matching the expected verse.

---

### Step 2.4 — Phase 2 Benchmark

Run the full 256-sample benchmark with constrained decoding:
```
npm run benchmark:fastconformer -- --decoder=beam
```

Also run the 54-sample curated benchmark:
```
npm run benchmark:fastconformer -- --decoder=beam --corpus=54
```

Use `npm run benchmark:compare` to compare against the saved Whisper baseline (`whisper-2026-03-22T03-05-24.json`).

**Target metrics:**
- Strict accuracy: target ≥83% on 256-sample corpus (vs current 81.6%)
- Short verse accuracy: target ≥78% (vs current 74.3% — this is the hardest category)
- Inference time: target ≤200ms per 2-second chunk on a MacBook M-series (vs ~436ms for Whisper)

---

## Phase 3: CTC Forced Alignment Re-scoring (Week 2-3)

**Goal:** For the top-5 beam search candidates, compute P(verse | audio) using the CTC forward algorithm and re-rank. This replaces Levenshtein fuzzy matching for verse identification with acoustic probability scoring.

### Step 3.1 — Restore `forced-alignment.ts`

**File to create:** `web/frontend/src/worker/forced-alignment.ts`

Recover from git:
```
git show d17d1e9^:web/frontend/src/worker/forced-alignment.ts > web/frontend/src/worker/forced-alignment.ts
```

This is the largest file at 668 lines. It contains:
- `BPETokenizer` class — move to `bpe-tokenizer.ts` per Step 2.1, then update this file's import
- `WordInfo` and `TokenBoundary` interfaces
- `ViterbiDP` class — streaming CTC Viterbi algorithm with dynamic array growth
  - `initFrame()`, `extendFrame()` — O(S) per frame where S = 2N+1 CTC sequence length
  - `backtrack()` — O(T×S) backtracking to recover alignment path
  - `pathToTokenBoundaries()` — maps path to per-token frame ranges
  - `getFurthestToken()` — returns current alignment frontier
- `ForcedAligner` class — high-level wrapper
  - `processFrames(logprobs, numFrames)` — feeds frames to ViterbiDP, returns newly stable word alignments
  - `finalize()` — returns complete word alignment
  - `getOverallScore()` — coverage × average confidence (0–1)

**Important architectural decision for Phase 3:** The original `ForcedAligner` was designed for real-time streaming word-by-word progress display during recitation (the `WordAlignedMessage` and `VerseCompleteMessage` types already exist in `types.ts`). For Phase 3's re-scoring use case, we need a **different usage pattern**: run forced alignment over a complete 2-second chunk, score the top-5 hypotheses from beam search, and return the best one. This is a batch re-scoring operation, not streaming.

Create a wrapper function `scoreCandidates` in `forced-alignment.ts`:

```typescript
/**
 * Score multiple verse candidates against CTC logprobs.
 * Returns candidates sorted by CTC log-likelihood (best first).
 *
 * @param logprobs CTC log probabilities [T x V]
 * @param timeSteps Number of time steps
 * @param vocabSize Vocabulary size
 * @param candidates Array of {text_norm, surah, ayah} to score
 * @param vocabJson The vocab.json for BPE tokenization
 * @param blankId CTC blank token ID
 */
export function scoreCandidates(
  logprobs: Float32Array,
  timeSteps: number,
  vocabSize: number,
  candidates: Array<{ text_norm: string; surah: number; ayah: number }>,
  vocabJson: Record<string, string>,
  blankId: number,
): Array<{ surah: number; ayah: number; ctcScore: number; overallScore: number }> {
  return candidates
    .map(candidate => {
      const aligner = new ForcedAligner(
        candidate.text_norm, vocabJson, blankId, vocabSize
      );
      aligner.processFrames(logprobs, timeSteps);
      const overallScore = aligner.getOverallScore();

      // Extract CTC log-likelihood from ViterbiDP
      // (add a getLogLikelihood() method to ForcedAligner)
      const ctcScore = aligner.getLogLikelihood();

      return { surah: candidate.surah, ayah: candidate.ayah, ctcScore, overallScore };
    })
    .sort((a, b) => b.ctcScore - a.ctcScore);
}
```

**Add `getLogLikelihood()` to `ForcedAligner`** — this returns the Viterbi path log-probability normalized by sequence length:
```typescript
getLogLikelihood(): number {
  const path = this.viterbi.backtrack();
  if (path.length === 0) return -Infinity;
  return this.viterbi.getPathLogProb() / Math.max(this.tokenIDs.length, 1);
}
```

Also add `getPathLogProb()` to `ViterbiDP` — it sums the log probs along the backtracked path.

**Verify:** Unit test that `scoreCandidates` correctly ranks two candidates where one is the true verse and one is a nearby verse, given synthetic logprobs that strongly favor the true verse's token sequence.

---

### Step 3.2 — Wire Re-scoring into `inference.ts`

**File to modify:** `web/frontend/src/worker/inference.ts`

Add a new function `transcribeWithRescoring`:

```typescript
async function transcribeWithRescoring(
  audio: Float32Array,
  _prompt?: string
): Promise<TranscribeResult> {
  const { features, timeFrames } = await computeMelSpectrogram(audio);
  const { logprobs, timeSteps, vocabSize } = await runInference(features, 80, timeFrames);

  // Step 1: Get top-5 hypotheses from constrained beam search
  const hypotheses = decoder!.constrainedBeamSearch(
    logprobs, timeSteps, vocabSize, trie!,
    { beamWidth: 5, topK: 20 }
  );

  if (hypotheses.length === 0) {
    return decoder!.decode(logprobs, timeSteps, vocabSize);
  }

  // Step 2: Match hypotheses to verse candidates via QuranDB
  const uniqueCandidates = new Map<string, { text_norm: string; surah: number; ayah: number }>();
  for (const hyp of hypotheses) {
    const match = db!.matchVerse(hyp.text, 0.1, 1, null, 1, null);
    if (match) {
      const key = `${match.surah}:${match.ayah}`;
      if (!uniqueCandidates.has(key)) {
        const verse = db!.getVerse(match.surah, match.ayah);
        if (verse?.text_norm) {
          uniqueCandidates.set(key, {
            text_norm: verse.text_norm,
            surah: match.surah,
            ayah: match.ayah
          });
        }
      }
    }
  }

  // Step 3: Re-score candidates with CTC forced alignment
  if (uniqueCandidates.size > 1) {
    const blankId = (decoder as any).blankId;
    const scored = scoreCandidates(
      logprobs, timeSteps, vocabSize,
      [...uniqueCandidates.values()],
      vocabJsonCache!, blankId
    );

    if (scored.length > 0) {
      const best = scored[0];
      const bestVerse = db!.getVerse(best.surah, best.ayah);
      if (bestVerse?.text_norm) {
        return { text: bestVerse.text_norm, rawTokens: hypotheses[0].rawTokens };
      }
    }
  }

  // Fallback: use top beam search hypothesis directly
  return { text: hypotheses[0].text, rawTokens: hypotheses[0].rawTokens };
}
```

**Performance budget:** Forced alignment for each candidate verse requires one `ForcedAligner` construction + `processFrames()` call. For a 2-second chunk (~197 frames) and a verse with ~35 BPE tokens (CTC sequence length S = 71), the Viterbi DP is O(T×S) = O(197×71) ≈ 14,000 operations per candidate. For 5 candidates: ~70,000 operations. This runs in <5ms in JavaScript. Re-scoring is cheap and worth doing.

**Verify:** Benchmark shows that re-scored results outperform non-re-scored results on the 54-sample corpus. If not, the re-scoring step may introduce noise — keep it feature-flagged.

---

### Step 3.3 — Phase 3 Benchmark

```
npm run benchmark:fastconformer -- --decoder=beam+rescore
```

Add a `--decoder` flag to the benchmark runner that selects:
- `greedy` — `CTCDecoder.decode()`
- `beam` — `CTCDecoder.beamSearch()`
- `constrained` — `CTCDecoder.constrainedBeamSearch()` with trie
- `beam+rescore` — constrained beam search + CTC forced alignment re-scoring

Save results as `test/benchmark-results/fastconformer-rescore-{timestamp}.json`.

**Target:** ≥85% strict accuracy on 256-sample corpus (vs 81.6% Whisper baseline).

---

## Phase 4: Full Integration and Streaming Optimization (Week 3-4)

**Goal:** Wire everything into the live recitation tracker, test end-to-end streaming, handle edge cases.

### Step 4.1 — Restore `forced-alignment.ts` Streaming Mode for Word Progress

The original `ForcedAligner` already supports streaming (designed for real-time word-by-word highlighting). Wire this back into `inference.ts` for tracking mode:

**File to modify:** `web/frontend/src/worker/inference.ts`

The original `inference.ts` before the Whisper switch had a complete forced alignment subsystem (`enterFA`, `exitFA`, `processFAChunk` functions, plus FA state variables `faAligner`, `faVerse`, `faAudioBuffer`, etc.). This is recoverable from git at `d17d1e9^:web/frontend/src/worker/inference.ts`.

For tracking mode, the current system uses `RecitationTracker._handleTracking()` which calls `transcribe()` for word alignment. This works but is wasteful — it runs full mel+ONNX inference just to get word positions. The forced aligner can do this more efficiently and accurately using the CTC logprobs directly.

**Implementation approach:** Keep the `RecitationTracker` as-is for discovery mode. For tracking mode, add an alternative path that:
1. Runs mel + ONNX → logprobs
2. Feeds logprobs to `ForcedAligner.processFrames()` for the known tracking verse
3. Emits `word_aligned` messages (already defined in `types.ts`) as new words are confirmed
4. Emits `verse_complete` (already defined in `types.ts`) when alignment reaches the end

This requires adding a new `"forced_align"` mode to the worker alongside the existing tracker. The tracker handles discovery; forced alignment handles tracking.

**Worker state machine:**
```
DISCOVERY mode (RecitationTracker._handleDiscovery)
  -> verse_match found
  -> TRACKING mode (ForcedAligner for the matched verse)
    -> verse_complete or stale
    -> DISCOVERY mode (next verse search)
```

**Critical:** The `WorkerInbound` and `WorkerOutbound` types in `types.ts` already include `WordAlignedMessage` and `VerseCompleteMessage` — these were designed for exactly this purpose and have never been used. Wire them in now.

**Verify:** End-to-end UX test: speak Al-Fatiha 1:1 continuously. Confirm:
- `verse_match` fires within 1.2s of first syllable
- `word_aligned` messages fire for each word as it is spoken
- `verse_complete` fires at the end with per-word confidence scores
- `verse_match` fires for 1:2 automatically

---

### Step 4.2 — Streaming Performance Optimization for 300ms Chunks

The current architecture processes audio in 2-second chunks (TRIGGER_SAMPLES = SAMPLE_RATE * 2.0). For FastConformer, mel computation is fast enough that 300ms chunks are viable for real-time word highlighting.

**File to modify:** `web/frontend/src/lib/types.ts`

For forced alignment tracking mode, reduce the trigger:
```typescript
// For forced alignment tracking (word-by-word)
export const FA_TRIGGER_SAMPLES = SAMPLE_RATE * 0.3; // 300ms
```

This constant already exists in the original `types.ts` before Whisper (recoverable from git). Check:
```
git show d17d1e9^:web/frontend/src/lib/types.ts | grep FA_
```

**Key constraint:** 300ms audio → ~30 mel frames. The FastConformer model runs quickly on WASM for 30 frames. But for 2-second discovery chunks, it still produces ~197 frames — the mel computation is O(T) and the ONNX inference is O(T²) in the worst case. Profile separately.

**Verify:** In the browser, measure `performance.now()` around mel computation and ONNX `session.run()` for:
- 300ms chunk (30 frames): expected <20ms mel + <30ms ONNX = <50ms total
- 2000ms chunk (197 frames): expected <80ms mel + <150ms ONNX = <230ms total

If 2-second chunks exceed 300ms on a typical device, reduce `MAX_WINDOW_SAMPLES` from 10s to 6s and adjust discovery trigger accordingly.

---

### Step 4.3 — Handle the Prompt-less Tracking Regression

**Problem:** The current `RecitationTracker._handleTracking()` passes the known verse text as a `prompt` to the Whisper transcriber, which biases the decoder toward the correct vocabulary. FastConformer cannot accept text prompts. The constrained trie-beam search provides an equivalent benefit for discovery mode (it constrains to valid Quran text), but tracking mode currently relies on prompting to prevent the model from hallucinating.

**Solution:** In tracking mode, build a **verse-specific sub-trie** from the current tracking verse and the next 3 verses. Pass this sub-trie to `constrainedBeamSearch()` instead of the full 6,236-verse trie. This achieves the same bias effect as prompting.

**Implementation:**
```typescript
function buildSubTrie(verses: QuranVerse[], vocabJson: Record<string, string>): QuranTrie {
  const subTrie = new QuranTrie(vocabJson);
  subTrie.buildFromVerses(verses.map(v => ({
    text_norm: v.text_norm!,
    surah: v.surah,
    ayah: v.ayah,
  })));
  return subTrie;
}
```

In `inference.ts`, maintain a `trackingTrie: QuranTrie | null` that is rebuilt whenever the tracking verse changes. Use `trackingTrie` in `transcribeWithRescoring()` when a tracking verse is known.

This is a significant accuracy improvement over unconstrained decoding and should partially compensate for the loss of Whisper prompting.

**Verify:** Repeat the UX test from Step 4.1. Count false-positive `word_progress` jumps (word index going backward or skipping). Should be <5% of transitions.

---

### Step 4.4 — Final Benchmark

Run all four benchmarks and compare:

```bash
# Current Whisper baseline (already saved)
# → test/benchmark-results/whisper-2026-03-22T03-05-24.json (54-sample)
# → test/benchmark-results/expanded-2026-03-22T03-59-53.json (256-sample)

# FastConformer greedy (Phase 1 baseline)
npm run benchmark:fastconformer -- --decoder=greedy
# FastConformer constrained beam (Phase 2)
npm run benchmark:fastconformer -- --decoder=constrained
# FastConformer constrained beam + CTC rescore (Phase 3)
npm run benchmark:fastconformer -- --decoder=beam+rescore

# Compare all
npm run benchmark:compare
```

**Expected accuracy progression on 256-sample corpus:**
| Configuration | Expected Accuracy | Basis for Estimate |
|--------------|-------------------|---------------------|
| Whisper baseline | 81.6% (209/256) | Measured |
| FC greedy | ~65-72% | Original CTC baseline before Whisper switch was "on par" at 5/8 UX tests |
| FC constrained beam | ~78-84% | Eliminates hallucinations, still subject to BPE decoding artifacts |
| FC + CTC rescore | ~83-88% | Re-scoring with acoustic likelihood should push past Whisper on clean audio |

Note: "on par with CTC baseline" in the Whisper switch commit referred to UX criteria, not raw accuracy on the 256-sample corpus (which did not exist at that time). The actual accuracy delta could be higher or lower.

---

## File Summary

### Files to create (all via git restore + minor edits):

| File | Source | Est. Lines | Phase |
|------|--------|------------|-------|
| `web/frontend/src/worker/mel.ts` | `git show d17d1e9^:..../mel.ts` | 112 | 1.1 |
| `web/frontend/src/worker/session.ts` | `git show d17d1e9^:..../session.ts` | 47 | 1.2 |
| `web/frontend/src/worker/model-cache.ts` | `git show d17d1e9^:..../model-cache.ts` | 70 | 1.3 |
| `web/frontend/src/worker/ctc-decode.ts` | `git show cc1b6f0:..../ctc-decode.ts` | 421 | 1.4 |
| `web/frontend/src/worker/bpe-tokenizer.ts` | extracted from `forced-alignment.ts` | ~90 | 2.1 |
| `web/frontend/src/worker/quran-trie.ts` | `git show cc1b6f0:..../quran-trie.ts` | 105 | 2.1 |
| `web/frontend/src/worker/forced-alignment.ts` | `git show d17d1e9^:..../forced-alignment.ts` | 668 | 3.1 |
| `web/frontend/test/benchmark-fastconformer.ts` | copied from `benchmark-whisper.ts` | ~250 | 1.6 |
| `web/frontend/test/unit/mel.test.ts` | new | ~40 | 1.1 |
| `web/frontend/test/unit/ctc-decode.test.ts` | new | ~60 | 1.4 |
| `web/frontend/test/unit/quran-trie.test.ts` | new | ~50 | 2.1 |
| `web/frontend/public/vocab.json` | copy from `data/vocab.json` | 1 | 1.5 |
| `web/frontend/public/fastconformer_ar_ctc_q8.onnx` | download from GitHub releases | 131MB | 1.3 |

### Files to modify:

| File | What Changes | Phase |
|------|-------------|-------|
| `web/frontend/src/worker/inference.ts` | Replace Whisper pipeline with FastConformer; add trie init; add rescore path | 1.5, 2.2, 3.2 |
| `web/frontend/src/lib/types.ts` | Add `FA_TRIGGER_SAMPLES` constant (if not already present); potentially extend `TranscribeResult` | 4.2 |
| `web/frontend/package.json` | Add benchmark:fastconformer scripts | 1.6 |
| `web/frontend/.gitignore` | Add `public/fastconformer_ar_ctc_q8.onnx` | 1.3 |

### Files to keep unchanged:
- `web/frontend/src/lib/tracker.ts` — `RecitationTracker` is fully compatible; it calls `transcribe(audio, prompt?)` and the FastConformer version accepts and ignores the prompt parameter
- `web/frontend/src/lib/quran-db.ts` — no changes needed; the matching engine works on normalized Arabic text regardless of ASR source
- `web/frontend/src/lib/types.ts` — mostly unchanged; `WordAlignedMessage` and `VerseCompleteMessage` are already defined and waiting
- `web/frontend/src/worker/whisper-transcriber.ts` — keep for A/B testing; delete only after FastConformer exceeds Whisper on all benchmarks

---

## Risk Register

### Risk 1: FastConformer ONNX model input/output shape has changed

The model was last used with the original session.ts. The `onnxruntime-web` version has changed from whatever was used originally to `1.24.2`. If the model's input/output tensor names differ from what session.ts expects, inference will fail silently.

**Mitigation:** Before writing any new code, load the model in Node.js with `onnxruntime-node` and print `session.inputNames` and `session.outputNames`:
```typescript
import * as ort from "onnxruntime-node";
const session = await ort.InferenceSession.create("public/fastconformer_ar_ctc_q8.onnx");
console.log("inputs:", session.inputNames);
console.log("outputs:", session.outputNames);
const dummy = new ort.Tensor("float32", new Float32Array(80 * 50), [1, 80, 50]);
const len = new ort.Tensor("int64", BigInt64Array.from([50n]), [1]);
const result = await session.run({ [session.inputNames[0]]: dummy, [session.inputNames[1]]: len });
console.log("output shape:", result[session.outputNames[0]].dims);
```

The output should be `[1, T, 1025]` where 1025 is the vocab size. If the output is `[1, T, V]` with V ≠ 1025, update the `CTCDecoder` construction accordingly.

### Risk 2: BPE tokenizer produces different token sequences than the model was trained with

The model was trained by NVIDIA using NeMo's BPE tokenizer. The `BPETokenizer` in `forced-alignment.ts` is a greedy longest-match implementation that approximates NeMo's behavior. If there are subtle differences (e.g., ▁ boundary handling on sentence-initial words), the constrained trie will reject valid prefixes and the decoder will fall back to greedy.

**Detection:** Run the `BPETokenizer` on all 6,236 verses and check the trie coverage. If `trie.isValidPrefix(tokenize(verse.text_norm).tokenIDs)` returns false for any complete verse, there is a tokenization mismatch.

**Mitigation:** Add a post-build validation step in `QuranTrie.buildFromVerses()` that asserts all inserted verses are self-consistent (i.e., the tokenized form is retrievable from the trie).

### Risk 3: WASM ONNX performance is worse than expected for 131MB model

The FastConformer q8 model is 131MB. Loading time from IndexedDB is fast (<100ms). But inference on WASM for a conformer block may be significantly slower than the mel computation suggests. If inference takes >500ms per 2-second chunk, the streaming pipeline will introduce unacceptable lag.

**Mitigation:**
- Benchmark inference time in the browser before committing to the architecture (Step 1.5 verification)
- If WASM is too slow, consider: (a) reducing audio window to 1.5s for discovery, (b) using WebGPU backend via `ort.env.webgpu = true` if available, (c) falling back to Whisper for slow devices
- The existing Whisper pipeline can serve as an automatic fallback: if FastConformer init fails (OOM, WASM timeout), fall back to `whisper-transcriber.ts`

### Risk 4: Constrained beam search produces empty results for short audio

For very short audio clips (<500ms), the CTC output may have very few frames (~50). If no trie path reaches an `isVerseEnd` node within these frames, the constrained beam search returns an empty hypothesis list. The fallback to greedy decode will then produce unconstrained BPE output which may not match any verse.

**Mitigation:** The fallback to `decoder!.decode()` in the `transcribe` function handles this. Additionally, consider a "relaxed constraint" mode for short clips: if constrained search returns <3 hypotheses, retry with a reduced `topK=50` to allow more exploration. Short clips are exactly the hardest category (74.3% current accuracy) — this is where FastConformer needs the most help.

### Risk 5: Memory pressure from simultaneous trie + model + QuranDB in worker

The worker already holds QuranDB (all verse data, ~3MB after JSON parse). Adding:
- FastConformer model: 131MB ArrayBuffer in IndexedDB, but ONNX session uses separate heap
- Quran trie: ~15–25MB JS heap
- BPE tokenizer vocab maps: ~2MB
- ForcedAligner state per verse: ~1MB during alignment

Total expected worker memory: ~50–80MB heap + 131MB ONNX session. On iOS Safari, workers are limited to 256MB heap. This is tight but workable.

**Mitigation:** After model loading, do not keep the raw `modelBuffer` ArrayBuffer in the worker scope — release it after `createSession()` completes, as the ONNX session holds its own internal copy. This frees 131MB from the JS heap immediately after session creation.

---

## Git Commands for File Recovery

Run these from `/Users/omarjarad/Desktop/personal/quran/offline-tarteel`:

```bash
# Recover deleted FastConformer pipeline files
git show d17d1e9^:web/frontend/src/worker/mel.ts \
  > web/frontend/src/worker/mel.ts

git show d17d1e9^:web/frontend/src/worker/session.ts \
  > web/frontend/src/worker/session.ts

git show d17d1e9^:web/frontend/src/worker/model-cache.ts \
  > web/frontend/src/worker/model-cache.ts

git show d17d1e9^:web/frontend/src/worker/forced-alignment.ts \
  > web/frontend/src/worker/forced-alignment.ts

# Recover ctc-decode.ts from the trie commit (has all versions + constrained search)
git show cc1b6f0:web/frontend/src/worker/ctc-decode.ts \
  > web/frontend/src/worker/ctc-decode.ts

# Recover quran-trie.ts from the trie commit
git show cc1b6f0:web/frontend/src/worker/quran-trie.ts \
  > web/frontend/src/worker/quran-trie.ts

# Check FA_TRIGGER_SAMPLES and other FA constants from old types.ts
git show d17d1e9^:web/frontend/src/lib/types.ts | grep -E "FA_|FORCED"

# Copy vocab.json to public/
cp data/vocab.json web/frontend/public/vocab.json
```

---

## Success Criteria

Phase 1 is complete when:
- `npm run dev` loads the FastConformer model in the browser without errors
- `npm run benchmark:fastconformer -- --sample=10` runs 10 samples in <30 seconds
- Greedy decode produces recognizable Arabic text (not garbage) for all 10 samples

Phase 2 is complete when:
- Constrained beam search produces only valid Quran prefixes (verified by unit test)
- `npm run benchmark:fastconformer -- --decoder=constrained` shows ≥1% accuracy improvement over greedy
- Trie builds in <500ms during worker init

Phase 3 is complete when:
- `scoreCandidates` unit test passes
- `npm run benchmark:fastconformer -- --decoder=beam+rescore` shows ≥2% improvement over constrained-only
- No measurable latency increase (re-scoring adds <10ms per inference call)

Phase 4 is complete when:
- Live recitation of Al-Fatiha produces correct `verse_match` for all 7 verses in sequence
- `word_aligned` messages fire with accurate word indices
- Total end-to-end latency (audio chunk arrival → verse_match) is ≤800ms on MacBook M-series
- `npm run benchmark:fastconformer -- --decoder=beam+rescore` shows ≥83% strict accuracy on 256-sample corpus
