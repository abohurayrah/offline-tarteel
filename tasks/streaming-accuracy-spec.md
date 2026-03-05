# Spec: Improve Streaming Accuracy from 41% to 70%+

## Problem

The Quran recitation recognition app uses a streaming pipeline: audio arrives in 300ms chunks, gets transcribed via ONNX model, and matched against 6236 Quran verses. The streaming pipeline scores **22/53 (41.5%)** on the test corpus, while non-streaming (full-file transcription) scores **37/53 (69.8%)**. The gap is caused by the tracker's windowing, thresholds, and state transitions — not the model or mel spectrogram.

## Architecture Overview

```
Audio chunks (300ms) → RecitationTracker.feed()
  ├── Discovery mode: accumulate 2s → transcribe → matchVerse()
  │   └── Match found → emit verse_match → enter tracking mode
  └── Tracking mode: accumulate 0.5s → transcribe → alignPosition()
      ├── Progress → emit word_progress
      ├── Verse complete → emit next verse_match → track next verse
      └── Stale (4 cycles) → exit tracking → back to discovery
```

Key files (all in `web/frontend/`):
- `src/lib/tracker.ts` — `RecitationTracker` class (discovery + tracking logic)
- `src/lib/quran-db.ts` — `QuranDB.matchVerse()` (verse matching with Levenshtein)
- `src/lib/types.ts` — all tunable constants
- `src/worker/mel.ts` — mel spectrogram (don't modify)
- `src/worker/ctc-decode.ts` — CTC decoder (don't modify)
- `test/validate-streaming.ts` — test harness

## Root Causes (ranked by impact)

### 1. FIRST_MATCH_THRESHOLD = 0.75 is too high for partial audio (15 empty-result failures)

The first-ever verse match requires 0.75 confidence. But in streaming, the first discovery fires after only 2s of audio. A 2s window of a 6-second verse produces a partial transcript that scores ~0.4-0.6 against the full verse text via Levenshtein ratio. The non-streaming baseline uses the effective threshold of 0.45 on the complete audio.

**Affected samples:** all 15 "empty result" failures — the tracker fires discovery 2-5 times but never hits 0.75.

### 2. MAX_WINDOW_SAMPLES = 10s caps destroy long-verse matching (10+ failures)

Any verse longer than ~15s can never be matched from a 10s rolling window. Levenshtein ratio for a 10s excerpt vs a 60s verse: `2*partial / (partial + full) ≈ 0.29`. This is below even the 0.45 subsequent-match threshold.

**Affected samples:** ref_002255 (52s), ref_024035 (80s), ref_002285 (33s), ref_002286 (53s), ref_003191 (35s), ref_048029 (77s), ref_074031 (65s), ref_033056 (18s), ref_059023 (23s), ref_059024 (23s)

### 3. Post-verse window reset drops audio for multi-verse transitions (7 failures)

When a verse completes, the tracker trims audio to last 2s (`TRIGGER_SAMPLES`) and resets `newAudioCount = 0`. It then needs to accumulate 2s of NEW audio before discovery fires again. During that ~2s dead zone, the next verse is already being recited but not evaluated. Short verses (2-4 words) can pass entirely within this gap.

**Affected samples:** multi_036_001_005, multi_113_001_005, multi_114_001_006, multi_067_001_004, user_ikhlas_2_3, multi_059_022_024, multi_002_285_286

### 4. Residual check blocks re-discovery after wrong-verse emit (contributes to 9 wrong-verse failures)

After emitting a wrong verse, `partialRatio(newText, lastEmittedText) > 0.7` blocks subsequent discovery of the correct verse — because similar Quranic verses share phoneme subsequences. The wrong initial emit cascades into a permanent block.

**Affected samples:** ref_002255 (got 3 wrong verses), ref_059023, ref_059024, ref_048029, ref_033056, ref_074031

## Constraints

1. **Do NOT modify** `mel.ts`, `ctc-decode.ts`, `levenshtein.ts`, `phoneme-aligner.ts`, or `correction.ts` — these are correct
2. **Do NOT change** `validate-streaming.ts` test logic (chunk size, silence tail, pass/fail criteria) — you need a stable test to measure improvements
3. **You CAN modify**: `tracker.ts`, `quran-db.ts`, `types.ts`
4. **You CAN add** new utility files if needed (e.g., a partial matching function)
5. Changes must not break the non-streaming path or the browser-side usage (the tracker is also used in the web worker via `inference.ts`)
6. The `WorkerOutbound` message types in `types.ts` are the public API — don't remove or rename existing message types

## Suggested Approaches (pick what works, not all)

### A. Lower/adapt first-match threshold
The 0.75 threshold was meant to prevent false cold-starts, but it's too strict for streaming. Consider:
- Lower `FIRST_MATCH_THRESHOLD` to ~0.55-0.60
- Or use a dynamic threshold that tightens as the audio window grows (e.g., start at 0.5 with 2s, rise to 0.75 with 8s+)
- Or require 2 consecutive discovery cycles to agree on the same verse before emitting

### B. Use partial/prefix matching instead of full-verse Levenshtein
`matchVerse()` compares the transcript against the full verse text using `ratio()`. For a 10-word transcript vs a 40-word verse, this penalizes heavily. Consider:
- Compare the transcript against only the first N words of each verse (where N ~ transcript word count)
- Or use `partialRatio()` (already exists) instead of `ratio()` for discovery
- Or implement a prefix-matching score: align transcript words against the verse prefix and compute coverage

### C. Grow the window for long verses instead of hard-capping
Instead of a fixed 10s cap, allow the window to grow when discovery keeps failing:
- Start with 10s window
- If X discovery cycles fire without a match, expand to 15s, then 20s
- Cap at some reasonable max (30s?) to avoid OOM

### D. Fix multi-verse transitions
After verse completion, instead of trimming to 2s and waiting for 2s of new audio:
- Keep the tracking-mode trigger interval (0.5s) for the first discovery attempt after exiting tracking
- Or don't reset `newAudioCount` on verse completion — let discovery fire immediately with accumulated audio
- Or after verse complete, pre-seed the next discovery with the retained audio so it fires on the next chunk

### E. Improve residual check
The current `partialRatio > 0.7` check is too broad — it blocks legitimate new verses that happen to share phonemes with the last emitted verse. Consider:
- Only apply the residual check for the first 1-2 discovery cycles after a verse emit (not permanently)
- Or compare against a shorter suffix of `lastEmittedText` (last few words, not the full verse)
- Or remove the check entirely and rely on the dedup check (`lastEmittedRef` same-verse skip) instead

## How to Test

Run from `web/frontend/`:

```bash
# Streaming mode (the one you're improving)
npm run test:streaming

# Non-streaming baseline (should stay at 37/53 or improve — never regress)
npm run test:streaming -- --no-streaming

# Single sample for quick iteration
npm run test:streaming -- retasy_017
npm run test:streaming -- ref_002255
npm run test:streaming -- multi_036_001_005
```

The test takes ~15-20 minutes for all 53 samples. For faster iteration, test against specific failure buckets:

```bash
# Short verses that return empty (threshold issue)
npm run test:streaming -- retasy_003
npm run test:streaming -- retasy_012
npm run test:streaming -- ref_036001

# Long verses (window cap issue)
npm run test:streaming -- ref_002255
npm run test:streaming -- ref_024035

# Multi-verse transitions
npm run test:streaming -- multi_036_001_005
npm run test:streaming -- multi_103_001_003
```

## Success Criteria

- Streaming accuracy: **35/53 (66%)** or higher (currently 22/53 = 41.5%)
- Non-streaming must not regress below 37/53
- No new message types or breaking changes to `WorkerOutbound`
- The browser app (`npm run dev`) must still work — tracker is used identically

## Current Constants Reference (`types.ts`)

```
SAMPLE_RATE = 16000
TRIGGER_SECONDS = 2.0          → TRIGGER_SAMPLES = 32000
MAX_WINDOW_SECONDS = 10.0      → MAX_WINDOW_SAMPLES = 160000
SILENCE_RMS_THRESHOLD = 0.005
VERSE_MATCH_THRESHOLD = 0.45
FIRST_MATCH_THRESHOLD = 0.75
RAW_TRANSCRIPT_THRESHOLD = 0.25
SURROUNDING_CONTEXT = 2
TRACKING_TRIGGER_SECONDS = 0.5 → TRACKING_TRIGGER_SAMPLES = 8000
TRACKING_SILENCE_TIMEOUT = 4.0 → TRACKING_SILENCE_SAMPLES = 64000
TRACKING_MAX_WINDOW_SECONDS = 5.0 → TRACKING_MAX_WINDOW_SAMPLES = 80000
STALE_CYCLE_LIMIT = 4
LOOKAHEAD = 5
```
