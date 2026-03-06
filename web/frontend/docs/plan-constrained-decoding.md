# Plan: Constrained CTC Beam Search for Quran Verse Recognition

## Problem Statement

Our current pipeline does two separate steps:
1. **Transcribe**: CTC greedy decode → raw Arabic text (lossy — throws away all non-argmax probabilities)
2. **Match**: Levenshtein compare transcript against 6,236 verses

This loses information. When the greedy path produces garbage like "تفسيرينب الرحمن الرحيم",
there may have been a perfectly valid Quran path through the logits that scored nearly as well.

## Core Idea

Replace greedy decode with **CTC beam search constrained to valid Quran text**. Instead of
picking argmax at every frame and hoping the result matches a verse, explore multiple hypotheses
simultaneously and only keep those that spell out text found in the Quran.

## How CTC Beam Search Works

### Current: Greedy Decode
```
Frame 0: [0.01, 0.8, 0.1, ...] → pick token 1 ("ة")
Frame 1: [0.7, 0.02, 0.1, ...] → pick token 0 ("<unk>")
Frame 2: ...
Result: single hypothesis, often garbled
```

### Proposed: Beam Search (width K=10)
```
Frame 0: Keep top-K partial hypotheses:
  beam[0]: "ة"    (prob=0.8)
  beam[1]: "▁في"  (prob=0.1)
  beam[2]: ...

Frame 1: Extend each beam with each token, keep top-K overall:
  beam[0]: "ة" + blank  (prob=0.8 * 0.7 = 0.56)
  beam[1]: "▁في" + "ها" (prob=0.1 * 0.6 = 0.06)
  ...

Result: K hypotheses ranked by probability
```

### With Quran Constraint
Same as above, but at each step, **boost beams that follow valid Quran text** and
**penalize beams that deviate**. A character-level trie of all Quran verses guides the search.

## Architecture

### Component 1: Quran Character Trie

Build a trie from all 6,236 verses (normalized Arabic, no diacritics):

```
Root
├── "ب" → "س" → "م" → " " → "ا" → "ل" → ...  (bismillah prefix, 113 verses)
├── "ا" → "ل" → "ح" → "م" → "د" → " " → ...  (al-Fatiha 1:2)
│         → "م" → (muqatta'at: 2:1, 3:1, 29:1, ...)
│         → "ر" → (muqatta'at: 10:1, 11:1, ...)
├── "ق" → "ل" → " " → ...  (many "qul" verses)
...
```

**Key design decisions:**
- **Normalize before trie insertion**: strip diacritics, normalize hamza/taa/yaa (same as normalizeArabic)
- **Store verse IDs at leaf nodes**: when a beam reaches a leaf, we know exactly which verse(s) it matches
- **Support partial entry**: a beam can enter the trie at any point (user may start mid-verse)
- **Memory**: ~6,236 verses × ~50 chars avg = ~300K characters. Trie with sharing ≈ 1-2MB. Fine for browser.

### Component 2: BPE-to-Character Bridge

The model outputs BPE tokens, but the trie is character-level. We need a bridge:

```
BPE token "▁الله" → characters: " ", "ا", "ل", "ل", "ه"
BPE token "َ"     → diacritic (skip in trie, don't advance position)
BPE token "▁في"   → characters: " ", "ف", "ي"
```

**Algorithm:**
1. When a beam emits a BPE token, decode it to characters
2. Strip diacritics and `▁` markers → get normalized characters
3. Try to advance in the trie one character at a time
4. If all characters advance successfully → beam is "on-trie" (boosted)
5. If any character fails → beam is "off-trie" (penalized but not killed)

### Component 3: Scoring Function

For each beam at each frame:
```
total_score = acoustic_score + α * trie_bonus
```

Where:
- `acoustic_score` = log probability from CTC logits (standard)
- `trie_bonus` = reward for being on a valid Quran path
- `α` = tunable weight (start with 0.5, optimize on test set)

**Trie bonus values:**
- On-trie, advancing: `+1.0` per character advanced
- On-trie, at verse end: `+5.0` (we completed a valid verse!)
- Off-trie: `0.0`
- Was on-trie but fell off: `-2.0` (actively penalize deviation)

### Component 4: Modified CTC Decode Loop

```typescript
interface Beam {
  tokens: number[];           // BPE token IDs emitted
  text: string;               // Decoded text so far
  logProb: number;            // Cumulative acoustic log probability
  trieNode: TrieNode | null;  // Current position in Quran trie (null = off-trie)
  trieBonus: number;          // Cumulative trie bonus
  verseMatches: number[];     // Verse indices matched so far
  lastTokenId: number;        // For CTC repeat collapsing
}

function beamSearchDecode(
  logits: Float32Array,    // T × V
  timeSteps: number,
  vocabSize: number,
  trie: QuranTrie,
  beamWidth: number = 10,
  alpha: number = 0.5,
): BeamResult[] {
  let beams: Beam[] = [initialBeam];

  for (let t = 0; t < timeSteps; t++) {
    const candidates: Beam[] = [];

    for (const beam of beams) {
      // Get top-K tokens for this frame (don't need to try all 1025)
      const topK = getTopKTokens(logits, t, vocabSize, 20);

      for (const [tokenId, logProb] of topK) {
        // CTC rules: handle blank and repeat
        if (tokenId === blankId) {
          // Blank: keep beam as-is, add probability
          candidates.push({...beam, logProb: beam.logProb + logProb});
          continue;
        }
        if (tokenId === beam.lastTokenId) {
          // Repeat: collapse (CTC rule)
          candidates.push({...beam, logProb: beam.logProb + logProb});
          continue;
        }

        // New token: decode to characters, advance trie
        const chars = bpeToChars(tokenId);
        const {newNode, bonus, matches} = advanceTrie(beam.trieNode, chars, trie);

        candidates.push({
          tokens: [...beam.tokens, tokenId],
          text: beam.text + vocab[tokenId],
          logProb: beam.logProb + logProb,
          trieNode: newNode,
          trieBonus: beam.trieBonus + bonus,
          verseMatches: [...beam.verseMatches, ...matches],
          lastTokenId: tokenId,
        });
      }
    }

    // Prune: sort by (logProb + alpha * trieBonus), keep top beamWidth
    candidates.sort((a, b) =>
      (b.logProb + alpha * b.trieBonus) - (a.logProb + alpha * a.trieBonus)
    );
    beams = candidates.slice(0, beamWidth);
  }

  return beams;
}
```

## Handling Partial Recitation

Users may start mid-verse. The trie only has verse beginnings. Solutions:

### Option A: Suffix Trie (memory-heavy)
Build a trie of ALL suffixes of all verses. Memory: ~300K chars × avg 25 suffixes = 7.5M nodes.
Too large for browser.

### Option B: Delayed Trie Entry (recommended)
1. Start with unconstrained beam search for the first N frames
2. Once beams have accumulated 5+ characters of text, try to match each beam into the trie
3. Use the trigram index (already built) to find candidate verses
4. For each candidate, check if the beam's text appears as a substring
5. If found, "attach" the beam to the trie at the matching position

This is more complex but memory-efficient.

### Option C: Two-Pass (simplest)
1. Pass 1: Standard beam search (unconstrained) → top-K hypotheses
2. Pass 2: For each hypothesis, run Levenshtein matching against Quran (existing matchVerse)
3. Score = acoustic_score + match_score
4. Pick best

This is the easiest to implement and gets most of the benefit. The key insight: beam search
gives us K hypotheses instead of 1. Even without trie constraints, having 10 candidates means
the correct transcript is much more likely to be among them.

## Implementation Plan

### Phase 1: Unconstrained Beam Search (1-2 days)
- Implement CTC beam search in `ctc-decode.ts`
- Return top-K hypotheses with probabilities
- **Test**: Compare greedy top-1 vs beam top-1 accuracy. Expect small improvement.
- **Key test**: Check if correct transcript is in top-K for currently-failing cases

### Phase 2: Multi-Hypothesis Matching (1 day)
- Run matchVerse on ALL K hypotheses
- Pick the hypothesis+verse pair with best combined score
- Combined score = normalize(acoustic_prob) + normalize(levenshtein_ratio)
- **Test**: Expect significant improvement — many garbled greedy outputs may have
  a clean alternative in position 2-5

### Phase 3: Quran Trie Construction (1 day)
- Build character-level trie from quran.json
- BPE-to-character mapping with diacritic handling
- Unit tests for trie traversal

### Phase 4: Trie-Constrained Beam Search (2-3 days)
- Integrate trie with beam search loop
- Implement soft constraints (bonus/penalty)
- Tune α parameter on test set
- Handle partial recitation (Option B or C)
- **Test**: Expect major improvement on garbled cases

### Phase 5: Performance Optimization (1-2 days)
- Profile in browser (Web Worker)
- Optimize hot paths (trie traversal, beam pruning)
- Target: <500ms latency for 5-second audio clip
- Consider WASM for trie operations if JS is too slow

## Performance Considerations

- **Beam width K=10**: 10× more work per frame than greedy. For 200 frames × 20 top tokens × 10 beams = 40K operations per frame. Should be <50ms total.
- **Trie traversal**: O(token_length) per beam per frame. Negligible.
- **Memory**: Trie ≈ 1-2MB. Beams ≈ negligible. Total overhead ≈ 2MB.
- **Latency target**: <500ms for a 5-second clip (currently ~200ms with greedy). 2.5× budget is comfortable.

## Expected Impact

| Approach | Expected Equiv-Aware Accuracy | Effort |
|----------|-------------------------------|--------|
| Current (greedy + match) | 97.5% | — |
| Phase 2 (beam K=10 + multi-match) | 98.5-99% | 2 days |
| Phase 4 (trie-constrained beam) | 99-99.5% | 5-7 days |

## Risks

1. **BPE-to-character alignment**: Multi-character BPE tokens may partially advance in trie, creating edge cases
2. **Performance in browser**: Beam search is more compute-intensive; may need WASM acceleration
3. **Partial recitation**: Delayed trie entry adds complexity
4. **Over-constraining**: If trie bonus is too strong, may force wrong verse when audio is ambiguous

## Files to Modify

- `src/worker/ctc-decode.ts` — Add beam search decoder
- `src/worker/quran-trie.ts` — New file: Quran character trie
- `src/worker/recognition-worker.ts` — Use beam search instead of greedy
- `test/test-pipeline-full.ts` — Test beam search accuracy
- `public/quran-trie.json` — Pre-built trie (or build at startup from quran.json)
