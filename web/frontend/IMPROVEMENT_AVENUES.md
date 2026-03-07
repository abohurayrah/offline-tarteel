# Improvement Avenues for Offline Quran Recitation Tracker

## Current State (March 2026)

**UX Test Results: 5/8 passed, 21/28 criteria**

| Test | Status | Notes |
|------|--------|-------|
| Mid-Ayah Start (18:13) | FAIL | CTC model transcribes mid-verse audio as surah 96 |
| Word Progression (1:1-3) | PASS | 100% word coverage across 3 verses |
| Verse Oscillation (18:13+14) | FAIL | No bouncing (fixed), but wrong surah from model |
| Smooth Sequential (112) | PASS | Clean 1->2->3->4 progression |
| Long Verse (2:255) | FAIL | CTC model transcribes as surah 3 content |
| Verse Complete (112) | PASS | 67.5% avg word coverage, 2/4 fully tracked |
| Mid-Page Start (18:10) | PASS | Found at message index 1 |
| Continuation After Pause | PASS | All 4 ayahs discovered |

### Root Cause of Remaining Failures

All 3 failures share one root cause: **the CTC model produces incorrect transcriptions, and no matching algorithm can fix wrong input**. The general Arabic FastConformer was not trained on Quranic recitation and misidentifies audio for certain verses entirely.

### What We Tried (Worktree Experiments)

| Approach | Result | Verdict |
|----------|--------|---------|
| Sellers' substring matching | 5/8 (21/28) | Best word coverage, cherry-picked into main |
| N-gram BM25 retrieval | 5/8 (21/28) | Marginal improvement |
| EMA hysteresis gating | 1/8 (12/25) | Too aggressive, but **only approach to find 2:255** |
| CTC blank density trimming | 5/8 (21/28) | No regression, smarter buffer trimming |
| Combined (Sellers+bigram+EMA) | 3/8 (16/25) | Regressed - too many changes at once |

---

## Tier 1: Highest Impact

### 1. CTC Forced Alignment (Score Verses Directly from Logits)

**Impact: Very High | Effort: ~1 week | Dependencies: None**

The single biggest architectural unlock. Instead of the current pipeline:

```
audio -> CTC model -> greedy decode -> text -> Levenshtein match against verses
```

Compute verse probability directly:

```
audio -> CTC model -> logits -> P(verse_text | logits) via CTC forward algorithm
```

**Why this works:** The CTC model outputs a 1025-dimensional probability vector per audio frame. Greedy decoding collapses this rich distribution into a single text string, losing information. The model might greedily decode "surah 96 text" but the actual probability `P("18:13 text" | audio)` could still be highest among all candidates.

**Technical approach:**
- Extract per-frame log-probabilities from the existing ONNX CTC output
- For each candidate verse (top ~50 from trigram index), compute the CTC forward probability using the standard forward algorithm
- The forward algorithm runs in O(T * L) where T = frames (~500 for 10s) and L = verse length in characters (~50-100)
- Total runtime for 50 candidates: ~50 * 500 * 100 = 2.5M operations = <10ms

**This is how music score-following works** — the closest academic analog to real-time Quran tracking. Systems like NVIDIA Riva use CTC alignment internally for timestamp generation.

**Implementation:** ~200 lines in a new `ctc-align.ts` module. Uses the same model output, zero new dependencies, zero additional model download.

**References:**
- Graves et al., "Connectionist Temporal Classification" (2006) — Section 4, Forward-Backward Algorithm
- NVIDIA NeMo CTC alignment utilities
- sherpa-onnx forced alignment implementation

---

### 2. Quran-Specific ASR Model

**Impact: Very High | Effort: ~2-3 weeks | Dependencies: New model**

The current FastConformer is a general Arabic CTC model. Models specifically trained on Quranic recitation exist and would directly fix "wrong surah" transcriptions:

| Model | Size | WER on Quran | Format | Browser Ready |
|-------|------|-------------|--------|---------------|
| `tarteel-ai/whisper-base-ar-quran` | ~290MB | ~6% | PyTorch/ONNX | Needs conversion |
| `KheemP/whisper-base-quran-lora` | ~300MB | ~5.98% | LoRA adapter | Needs conversion |
| Moonshine Arabic (quantized) | ~103MB | Matches Whisper Large | ONNX | Via sherpa-onnx |
| NVIDIA `stt_ar_fastconformer_hybrid_large_pcd_v1.0` | ~500MB | SOTA on Arabic | NeMo/ONNX | Needs quantization |

**Key finding from offline-tarteel experiments:** Moonshine Tiny Arabic (103MB) matched Whisper Large-v3-Turbo (3.1GB) on Quran benchmarks. This means a browser-deployable model can achieve near-SOTA accuracy.

**Training data available:**
- EveryAyah dataset: 829 hours from 36 professional reciters, MIT license
- Tarteel.io user recordings: ~50,000+ unrated recordings
- Common Voice Arabic: ~300 hours

**Recommended path:**
1. Export Moonshine Arabic to ONNX int8 (~50MB)
2. Benchmark against current FastConformer on UX test suite
3. If insufficient, fine-tune on EveryAyah dataset

---

### 3. N-Best Beam Search Decoding

**Impact: High | Effort: ~3 days | Dependencies: None**

Current decoder uses greedy decoding (take highest-probability token per frame). This is lossy — the correct token sequence might have the 2nd or 3rd highest probability at each frame.

**CTC beam search** with width=5 produces 5 alternative transcriptions. Match ALL of them against the Quran. If even one hypothesis contains the correct verse text, we find it.

**Technical approach:**
- Implement prefix beam search over CTC logits (well-documented algorithm)
- For each beam, maintain running prefix and log-probability
- At each frame, extend top-K beams with top-K tokens
- Prune to beam width after each frame
- Return N-best final hypotheses

**Expected improvement:** For cases where the correct transcription has the 2nd or 3rd highest probability per frame, beam search will find it. Estimated 20-40% of current misidentifications are recoverable.

**Implementation:** ~100 lines in the CTC decoder. No new model needed.

**Existing implementations:**
- `ctc-decoder` npm package (but may not support WASM)
- Sean Naren's beam search in DeepSpeech
- flashlight-text CTC decoder (C++, compilable to WASM)

---

## Tier 2: Medium Impact

### 4. Constrained CTC Decoding (Quran Prefix Tree)

**Impact: Medium-High | Effort: ~1 week | Dependencies: None**

We know the user is reading the Quran — only ~14,870 unique words exist. Build a **prefix tree (trie)** of valid Quran word sequences and constrain beam search to only follow valid paths.

**Why this is powerful:**
- Eliminates impossible outputs entirely
- A general Arabic model might decode "كتب" but the Quran text says "كُتِبَ" — the trie forces the correct form
- Combined with beam search, this creates a Quran-specific language model at zero training cost

**Technical approach:**
- Build character-level trie from all 6,236 verse texts (normalized)
- During beam search, only extend beams along valid trie paths
- When a word boundary is reached, only allow transitions to valid next-words in the Quran
- Memory: ~2MB for the trie structure

**Limitation:** Only works for sequential recitation. If user skips verses or makes errors, the trie constraints could block correct recognition. Need a fallback to unconstrained decoding.

---

### 5. Audio Feature Matching (Fingerprinting)

**Impact: Medium | Effort: ~2 weeks | Dependencies: Reference audio database**

For known reciters like Alafasy (whose audio we already have for testing), bypass text entirely and compare audio features directly.

**Technical approach:**
- Pre-compute mel spectrogram features for each verse from reference audio
- At runtime, compute mel features for incoming audio
- Use Dynamic Time Warping (DTW) to align and score against reference features
- DTW naturally handles tempo variations and mid-verse starts

**Advantages:**
- Completely bypasses ASR errors
- Naturally handles mid-ayah starts (DTW finds best alignment anywhere)
- Works even for verses the ASR model struggles with

**Limitations:**
- Only works for reciters whose reference audio we have
- Different reciters have different melodic patterns (maqam)
- Requires pre-computed feature database (~50MB for all verses)

**Hybrid approach:** Use audio fingerprinting as a secondary signal alongside ASR. If ASR confidence is low but fingerprint confidence is high, prefer the fingerprint match.

---

### 6. Long Verse Mode (Adaptive Patience)

**Impact: Medium | Effort: ~2 days | Dependencies: None**

The EMA experiment (WT3) was the **only approach that found 2:255**, because it was more patient — it accumulated more audio before committing to a match.

**Insight:** For long verses (>20 words), the 10-second audio window only captures a fraction. The system needs to be more patient and accumulate more evidence before matching.

**Technical approach:**
- After 2-3 discovery cycles with the same top candidate surah but low confidence, enter "long verse mode"
- In this mode: increase `MAX_WINDOW_SECONDS` from 10 to 15-20 seconds
- Raise the match threshold to require higher confidence
- Accumulate transcripts across cycles more aggressively
- Only trigger for candidate verses with >20 words

**Implementation:** ~50 lines in tracker.ts. The key parameters:
- Trigger: 3 consecutive cycles where top candidates are sequential ayahs in same surah
- Extended window: 20 seconds (vs 10 default)
- Elevated threshold: 0.55 (vs 0.45 default)

---

## Tier 3: Lower Priority / Long-Term

### 7. Streaming CTC Model

**Impact: Very High | Effort: 4-16 weeks | Dependencies: Model training**

The definitive long-term fix. With streaming CTC:
- Each audio frame is processed exactly once (no re-transcription)
- No buffer length limit (handles any verse length)
- No boundary artifacts (no arbitrary window boundaries)

**Path:** sherpa-onnx WASM with streaming Zipformer CTC, fine-tuned on EveryAyah dataset.

**Risk:** No pre-trained streaming Arabic Quran model exists. Would need to train one.

### 8. Embedding-Based Semantic Matching

**Impact: Low-Medium | Effort: ~1 week | Dependencies: Embedding model (~45MB)**

Use Arabic sentence transformers (`paraphrase-multilingual-MiniLM-L12-v2`, ~45MB quantized) to compute semantic similarity between transcript and verse texts.

**Limitation:** Embeddings capture meaning rather than surface form. Similar-themed verses score 0.6-0.8, making discrimination difficult. Best used as a 15% weight in a hybrid scoring function.

### 9. User Context / Mushaf Page Hints

**Impact: Medium | Effort: ~3 days | Dependencies: UI integration**

When the user is viewing a specific mushaf page, use the visible verses as strong priors for matching. If the user is looking at page 293 (which contains 18:10-18:15), massively boost scores for those verses.

**This is what Tarteel.ai does** — the user selects their starting position. We could infer it from the mushaf page being displayed.

---

## Recommended Implementation Roadmap

### Phase 1 (This Week): CTC Forced Alignment
- Implement CTC forward algorithm for direct verse scoring
- Integrate as primary scoring signal alongside Levenshtein
- Expected: fixes 18:13 mid-ayah and potentially 2:255
- **This is the single highest-ROI change**

### Phase 2 (Next Week): N-Best + Constrained Decoding
- Add beam search with width=5 to CTC decoder
- Build Quran word prefix tree
- Constrain beam search to valid Quran sequences
- Expected: catches cases where correct text is 2nd/3rd hypothesis

### Phase 3 (Week 3-4): Model Upgrade
- Benchmark Moonshine Arabic against current FastConformer
- If better: swap model, re-run all tests
- If insufficient: fine-tune on EveryAyah dataset

### Phase 4 (Month 2+): Streaming + Polish
- Evaluate streaming CTC feasibility
- Add mushaf page context hints
- Audio fingerprinting for reference reciters
