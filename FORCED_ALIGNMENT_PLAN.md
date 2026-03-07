# CTC Forced Alignment for Real-Time Quran Recitation Tracking

## Problem

The current pipeline uses **open-vocabulary ASR** (Nvidia FastConformer CTC, general Arabic) to decode audio into arbitrary Arabic text, then fuzzy-matches against 6,236 Quran verses. This causes:

- **Poor accuracy on Quranic Arabic** — model trained on conversational/news Arabic, doesn't understand tajweed
- **Verse jumping** — model rapidly switches between wrong verses
- **No error detection** — word_correction module disabled (designed for phonemes, incompatible with BPE)
- **Cascade reveals** — false verse matches trigger unread verses to be revealed

## Key Insight

The Quran is a **fixed, known text**. Once we identify which verse the user is reading, we don't need open-vocabulary decoding anymore. We know exactly what text to expect. This transforms the problem from ASR (hard) to **forced alignment** (much easier).

## Architecture

### Two-Phase Approach

```
Phase 1: IDENTIFICATION (first few seconds)
┌──────────────────────────────────────────────────────┐
│ Audio → CTC Model → Open Decode → Match 6,236 verses │
│                                                        │
│ Same as current pipeline. Gets us the starting verse.  │
└───────────────────────────┬──────────────────────────┘
                            │ verse identified
                            ▼
Phase 2: FORCED ALIGNMENT (ongoing)
┌──────────────────────────────────────────────────────┐
│ Audio frames → CTC logprobs → Viterbi alignment      │
│ against known verse text → word boundaries +          │
│ pronunciation quality scores                          │
│                                                        │
│ When verse completes → auto-advance to next verse     │
│ (known: surah:ayah+1) → continue forced alignment    │
└──────────────────────────────────────────────────────┘
```

### Why This Works

1. **Accuracy near-perfect** — not guessing what they said, measuring how well audio matches known text
2. **Built-in error detection** — low CTC probability at a character position = mispronunciation
3. **No verse jumping** — constrained to expected verse, can't cascade to wrong verses
4. **Real-time streaming** — Viterbi extends incrementally as audio frames arrive
5. **Works with current model** — forced alignment compensates for model weakness on Quranic Arabic

## Implementation Plan

### Step 1: CTC Forced Alignment Algorithm

**File:** `src/worker/forced-alignment.ts`

Implement CTC forced alignment using the Viterbi algorithm:

```
Input:  CTC log-probabilities [T × V] (T=time frames, V=vocab size)
        Target text: "بسم الله الرحمن الرحيم" (tokenized to BPE/char IDs)
Output: Frame-to-token alignment, per-token confidence scores
```

**Algorithm:**
- Convert target verse text → sequence of BPE token IDs (using existing vocab.json)
- For each audio frame, compute probability of each target token
- Viterbi dynamic programming: find optimal alignment of frames → tokens
- Allow CTC blank transitions between tokens
- Track word boundaries (tokens → words mapping)

**Key properties:**
- Streaming: extend alignment as new frames arrive (no need to recompute from scratch)
- Confidence: per-token alignment probability → word-level pronunciation score
- Word boundaries: exact frame where each word starts/ends

### Step 2: Integration with Worker Pipeline

**File:** `src/worker/inference.ts`

After Phase 1 identifies the starting verse:

1. Worker enters "forced alignment mode" with the identified verse text
2. Each audio chunk → mel → CTC logprobs → extend forced alignment
3. Emit new message types:
   - `word_aligned`: word N confirmed at frame T with confidence C
   - `pronunciation_score`: per-word quality score (0-1)
   - `verse_complete`: all words aligned, auto-load next verse
4. When verse completes, automatically set next verse as target

### Step 3: Tracker Updates

**File:** `src/lib/tracker.ts`

Add forced alignment mode to RecitationTracker:

- `startForcedAlignment(surah, ayah, verseText)` — enter FA mode
- `feedFA(ctcLogprobs)` — process frame in FA mode
- Keep open-vocabulary mode for initial identification
- Switch to FA mode once verse is identified
- Handle verse transitions (current verse done → next verse)

### Step 4: New Message Types

**File:** `src/lib/types.ts`

```typescript
interface WordAlignedMessage {
  type: "word_aligned";
  surah: number;
  ayah: number;
  word_index: number;
  total_words: number;
  confidence: number;        // 0-1 pronunciation quality
  frame_start: number;
  frame_end: number;
}

interface PronunciationScoreMessage {
  type: "pronunciation_score";
  surah: number;
  ayah: number;
  word_scores: { word_index: number; score: number; }[];
  verse_score: number;       // overall verse quality
}

interface VerseCompleteMessage {
  type: "verse_complete";
  surah: number;
  ayah: number;
  overall_score: number;
  next_surah: number;
  next_ayah: number;
}
```

### Step 5: UI Updates

**File:** `src/main.ts`

- Handle `word_aligned` → reveal word with confidence-based coloring
  - High confidence (>0.8): gold highlight (correct)
  - Medium (0.5-0.8): orange (acceptable)
  - Low (<0.5): red (mispronounced)
- Handle `verse_complete` → auto-advance to next verse
- Remove dependency on `verse_match` for word tracking (only used for initial identification)

### Step 6: Mushaf Renderer Updates

**File:** `src/lib/mushaf-renderer.ts`

- Add `mp-word--good`, `mp-word--warning`, `mp-word--error` CSS classes
- Color-coded pronunciation quality on each word
- Tooltip or overlay showing pronunciation score

## Future Enhancements

### Model Improvements

1. **Fine-tune CTC model on Quran data** — Use existing 8.3GB of EveryAyah audio (3 reciters, full Quran) to fine-tune the FastConformer model. Even with forced alignment, a Quran-trained model gives better frame-level probabilities.

2. **Distill muaalem model** — Use obadx/muaalem-model-v3_2 (0.6B params, 0.16% PER, 850hrs Quran data) as teacher to train a small student model (~100M params) for browser deployment.

3. **Quran Phonetic Script (QPS)** — Adopt the phoneme encoding from the muaalem paper for character-level tajweed assessment (madd, ghunnah, idgham, etc.).

### Available Resources

- **Training data (local):** 8.3GB, 6,236 ayahs × 3 reciters (Alafasy full, Husary/Minshawy partial)
- **Training data (online):** obadx/recitation-segmentation — 850+ hours, ~300K annotated utterances
- **Quran-specific CTC models:** HamzaSidhu786/wav2vec2-base-word-by-word-quran-asr (7.9% WER)
- **Best Quran model:** obadx/muaalem-model-v3_2 (0.16% PER, 0.6B params — too large for browser, good for distillation)

### Pronunciation Feedback Features

- Per-word tajweed quality score
- Character-level error highlighting (which letter was mispronounced)
- Common tajweed rule detection (madd, qalqalah, etc.)
- Progress tracking across sessions

## Technical Notes

### CTC Forced Alignment vs Open Decoding

| Aspect | Open Decoding (current) | Forced Alignment (proposed) |
|--------|------------------------|---------------------------|
| Output | Any Arabic text | Alignment to known text |
| Accuracy | Limited by model quality | Near-perfect (constrained) |
| Speed | Beam search (expensive) | Viterbi DP (fast) |
| Error detection | None | Built-in (low confidence) |
| Streaming | Yes (CTC is streaming) | Yes (extend Viterbi) |
| First verse | Required | Falls back to open decode |

### Browser Constraints

- ONNX Runtime WASM: single-threaded inference (~100ms per chunk)
- Model size: ≤150MB for reasonable download
- Memory: ≤512MB working set
- Forced alignment DP table: O(T × L) where T=frames, L=target length — negligible memory
