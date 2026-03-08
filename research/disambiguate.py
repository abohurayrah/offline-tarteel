"""
Quran Verse Disambiguation — Exact Word-Sequence Analysis

For every ayah (6236 total), at every starting word position, compute
the minimum number of consecutive words needed to uniquely identify
that ayah against the entire Quran corpus.

Two analysis modes:
  1. ISOLATED: each ayah analyzed independently (original)
  2. CONTINUOUS: the Quran as one word stream — windows can span
     ayah boundaries (e.g., last 2 words of ayah N + first word of N+1)

Method: Pure exact word-sequence matching via n-gram inverted index.
Zero parameters. Deterministic. Reproducible.

Output:
  - ambiguity-map.json      (full per-ayah results)
  - ambiguity-compact.json  (compact version for frontend)
  - boundary-analysis.json  (cross-ayah-boundary disambiguation)
"""

import json
import os
import re
import statistics
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Arabic normalization (mirrors quran-db.ts normalizeArabic)
# ---------------------------------------------------------------------------
_DIACRITIC_RE = re.compile(
    "[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for comparison — mirrors the TypeScript version."""
    text = text.replace("\u2581", " ")    # BPE marker
    text = text.replace("\uFEFF", "")     # BOM
    text = _DIACRITIC_RE.sub("", text)    # strip tashkeel
    for ch in "أإآٱ":
        text = text.replace(ch, "ا")
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = text.replace("ـ", "")          # tatweel
    text = re.sub(r"[،؟.!:]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_quran(path: Path) -> list[dict]:
    """Load and normalize quran.json."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    verses = []
    for v in raw:
        norm = normalize_arabic(v.get("text_clean") or v["text_uthmani"])
        words = norm.split()
        verses.append({
            "surah": v["surah"],
            "ayah": v["ayah"],
            "text_norm": norm,
            "words": words,
            "surah_name": v.get("surah_name", ""),
            "surah_name_en": v.get("surah_name_en", ""),
        })
    return verses


# ---------------------------------------------------------------------------
# N-gram inverted index
# ---------------------------------------------------------------------------
def build_ngram_index(verses: list[dict], max_n: int = 15) -> dict[tuple, set[int]]:
    """
    Inverted index: (word_1, word_2, ..., word_n) -> {ayah_idx_1, ayah_idx_2, ...}

    For any contiguous word sequence of length <= max_n, this gives the exact
    set of ayahs that contain it. No approximation, no threshold.
    """
    index: dict[tuple, set[int]] = {}
    for idx, v in enumerate(verses):
        words = v["words"]
        for n in range(1, min(max_n + 1, len(words) + 1)):
            for start in range(len(words) - n + 1):
                gram = tuple(words[start:start + n])
                if gram not in index:
                    index[gram] = set()
                index[gram].add(idx)
    return index


def sequence_in_words(seq: tuple[str, ...], words: list[str]) -> bool:
    """Check if an exact word sequence appears contiguously in a word list."""
    seq_len = len(seq)
    for i in range(len(words) - seq_len + 1):
        if tuple(words[i:i + seq_len]) == seq:
            return True
    return False


# ---------------------------------------------------------------------------
# Core: analyze a single ayah
# ---------------------------------------------------------------------------
def analyze_ayah(
    ayah_idx: int,
    verses: list[dict],
    ngram_index: dict[tuple, set[int]],
    max_n: int = 15,
) -> dict:
    """
    For a single ayah, at every starting word position, extend a sliding
    window one word at a time. At each length, look up the exact word
    sequence in the n-gram index to count how many other ayahs share it.
    Stop when the count reaches 0 (unique) or the ayah ends.
    """
    verse = verses[ayah_idx]
    words = verse["words"]
    total_words = len(words)

    if total_words == 0:
        return {
            "surah": verse["surah"],
            "ayah": verse["ayah"],
            "total_words": 0,
            "windows": [],
        }

    windows = []

    for start in range(total_words):
        max_len = total_words - start
        disambiguation_len = -1  # -1 = never unique from this position
        narrowing = []
        prev_candidates: set[int] | None = None

        for length in range(1, max_len + 1):
            window = tuple(words[start:start + length])

            if length <= max_n:
                # Direct index lookup — O(1)
                matches = ngram_index.get(window, set())
            else:
                # Window exceeds index depth: narrow from previous candidates
                if prev_candidates is not None:
                    matches = {
                        c for c in prev_candidates
                        if sequence_in_words(window, verses[c]["words"])
                    }
                else:
                    # Fallback: use longest indexed prefix, then verify
                    prefix = tuple(words[start:start + max_n])
                    candidates = ngram_index.get(prefix, set())
                    matches = {
                        c for c in candidates
                        if sequence_in_words(window, verses[c]["words"])
                    }

            confusers = matches - {ayah_idx}
            num_confusers = len(confusers)

            # Record narrowing: how many confusers at this window length
            # Include confuser refs (up to 5) for the full research output
            confuser_refs = sorted(
                [{"s": verses[c]["surah"], "a": verses[c]["ayah"]} for c in confusers],
                key=lambda x: (x["s"], x["a"]),
            )[:5]

            narrowing.append({
                "len": length,
                "n": num_confusers,
                "refs": confuser_refs,
            })

            if num_confusers == 0:
                disambiguation_len = length
                break

            # Carry forward candidate set for next iteration
            prev_candidates = matches

        window_result: dict = {
            "start": start,
            "d": disambiguation_len,
            "narrowing": narrowing,
        }

        # If never unique, record final confuser count
        if disambiguation_len == -1 and narrowing:
            window_result["remaining"] = narrowing[-1]["n"]

        windows.append(window_result)

    # Summary stats
    unique_lens = [w["d"] for w in windows if w["d"] > 0]

    return {
        "surah": verse["surah"],
        "ayah": verse["ayah"],
        "total_words": total_words,
        "text_norm": verse["text_norm"],
        "surah_name_en": verse["surah_name_en"],
        "windows": windows,
        # Per-ayah summary
        "from_start": windows[0]["d"] if windows else -1,
        "best": min(unique_lens) if unique_lens else -1,
        "worst": max(unique_lens) if unique_lens else -1,
        "never_unique_positions": sum(1 for w in windows if w["d"] == -1),
    }


# ---------------------------------------------------------------------------
# Cross-boundary: continuous stream analysis
# ---------------------------------------------------------------------------
def build_continuous_stream(verses: list[dict]) -> tuple[list[str], list[dict], list[int]]:
    """
    Build the entire Quran as one continuous word sequence.
    Returns:
      stream: flat list of all words in recitation order
      pos_meta: for each stream position, {surah, ayah, word_idx, verse_idx}
      boundaries: stream positions where a new ayah starts
    """
    stream: list[str] = []
    pos_meta: list[dict] = []
    boundaries: list[int] = []

    for vi, v in enumerate(verses):
        boundaries.append(len(stream))
        for wi, w in enumerate(v["words"]):
            stream.append(w)
            pos_meta.append({
                "surah": v["surah"],
                "ayah": v["ayah"],
                "word_idx": wi,
                "verse_idx": vi,
                "surah_name_en": v["surah_name_en"],
            })

    return stream, pos_meta, boundaries


def build_continuous_ngram_index(
    stream: list[str], max_n: int = 15
) -> dict[tuple, set[int]]:
    """
    N-gram index on the continuous stream.
    Maps word tuples -> set of starting positions in the stream.
    """
    index: dict[tuple, set[int]] = {}
    stream_len = len(stream)
    for start in range(stream_len):
        for n in range(1, min(max_n + 1, stream_len - start + 1)):
            gram = tuple(stream[start:start + n])
            if gram not in index:
                index[gram] = set()
            index[gram].add(start)
    return index


def analyze_boundaries(
    verses: list[dict],
    stream: list[str],
    pos_meta: list[dict],
    boundaries: list[int],
    cont_index: dict[tuple, set[int]],
    max_n: int = 15,
    context_words: int = 10,
) -> list[dict]:
    """
    For each ayah boundary (transition from ayah N to ayah N+1),
    analyze windows that span across the boundary.

    For each boundary, we test starting positions from up to `context_words`
    before the boundary, extending into the next ayah. For each start,
    find the minimum window length that is unique in the continuous stream.

    Returns one entry per boundary with disambiguation data.
    """
    stream_len = len(stream)
    boundary_results = []

    for bi in range(len(boundaries) - 1):
        bpos = boundaries[bi + 1]  # first word of the next ayah
        prev_verse = verses[bi]
        next_verse = verses[bi + 1]

        # Skip surah transitions where the boundary is between different surahs
        # and the next surah starts with basmalah — still analyze, it's useful data
        is_surah_transition = prev_verse["surah"] != next_verse["surah"]

        # Test windows starting from up to context_words before boundary
        # that MUST cross the boundary (i.e., extend into the next ayah)
        crossing_windows = []

        for tail_len in range(1, min(context_words + 1, len(prev_verse["words"]) + 1)):
            # Start position in stream: tail_len words before boundary
            start_pos = bpos - tail_len

            # Extend into next ayah, 1 word at a time
            for head_len in range(1, min(context_words + 1, len(next_verse["words"]) + 1)):
                total_len = tail_len + head_len
                end_pos = bpos + head_len
                if end_pos > stream_len:
                    break

                window = tuple(stream[start_pos:end_pos])

                if total_len <= max_n:
                    matches = cont_index.get(window, set())
                else:
                    # Use prefix lookup + verify
                    prefix = tuple(stream[start_pos:start_pos + max_n])
                    candidates = cont_index.get(prefix, set())
                    matches = set()
                    for c in candidates:
                        if c + total_len <= stream_len:
                            if tuple(stream[c:c + total_len]) == window:
                                matches.add(c)

                num_matches = len(matches)  # includes the position itself
                is_unique = num_matches == 1  # only this position

                if is_unique:
                    crossing_windows.append({
                        "tail": tail_len,  # words from previous ayah
                        "head": head_len,  # words from next ayah
                        "total": total_len,
                        "unique": True,
                    })
                    break  # found minimum head_len for this tail_len

            else:
                # Never became unique for this tail_len
                crossing_windows.append({
                    "tail": tail_len,
                    "head": context_words,
                    "total": tail_len + context_words,
                    "unique": False,
                })

        # Find the best crossing window (minimum total to achieve uniqueness)
        unique_crossings = [w for w in crossing_windows if w["unique"]]
        best_crossing = min(unique_crossings, key=lambda w: w["total"]) if unique_crossings else None

        # Also compute: starting from last word of prev ayah, how many words
        # into the next ayah until unique?
        from_last_word = None
        last_word_pos = bpos - 1
        for length in range(1, min(context_words * 2 + 1, stream_len - last_word_pos + 1)):
            window = tuple(stream[last_word_pos:last_word_pos + length])
            if length <= max_n:
                matches = cont_index.get(window, set())
            else:
                prefix = tuple(stream[last_word_pos:last_word_pos + max_n])
                candidates = cont_index.get(prefix, set())
                matches = {c for c in candidates
                          if c + length <= stream_len and
                          tuple(stream[c:c + length]) == window}
            if len(matches) == 1:
                from_last_word = length
                break

        boundary_results.append({
            "boundary_idx": bi,
            "prev_ayah": f"{prev_verse['surah']}:{prev_verse['ayah']}",
            "next_ayah": f"{next_verse['surah']}:{next_verse['ayah']}",
            "prev_name": prev_verse["surah_name_en"],
            "next_name": next_verse["surah_name_en"],
            "is_surah_transition": is_surah_transition,
            "prev_word_count": len(prev_verse["words"]),
            "next_word_count": len(next_verse["words"]),
            "best_crossing": best_crossing,
            "from_last_word": from_last_word,
            "crossings_tested": len(crossing_windows),
            "crossings_unique": len(unique_crossings),
        })

    return boundary_results


# ---------------------------------------------------------------------------
# Statistics and reporting
# ---------------------------------------------------------------------------
def print_stats(results: list[dict], total: int) -> None:
    """Print corpus-level statistics."""
    from_start = [r["from_start"] for r in results if r["from_start"] > 0]
    never_from_start = sum(1 for r in results if r["from_start"] == -1)
    best_all = [r["best"] for r in results if r["best"] > 0]
    worst_all = [r["worst"] for r in results if r["worst"] > 0]

    print("\n" + "=" * 60)
    print("GLOBAL STATISTICS")
    print("=" * 60)

    if from_start:
        print(f"\nDisambiguation from start (words needed):")
        print(f"  Mean:   {statistics.mean(from_start):.2f}")
        print(f"  Median: {statistics.median(from_start):.1f}")
        print(f"  Min:    {min(from_start)}")
        print(f"  Max:    {max(from_start)}")
        print(f"  Stdev:  {statistics.stdev(from_start):.2f}")
    print(f"  Never unique from start: {never_from_start}/{total}")

    if best_all:
        print(f"\nBest disambiguation (any start position):")
        print(f"  Mean:   {statistics.mean(best_all):.2f}")
        print(f"  Median: {statistics.median(best_all):.1f}")

    if worst_all:
        print(f"\nWorst disambiguation (hardest start position):")
        print(f"  Mean:   {statistics.mean(worst_all):.2f}")
        print(f"  Median: {statistics.median(worst_all):.1f}")
        print(f"  Max:    {max(worst_all)}")

    # Distribution histogram
    print("\nDisambiguation length distribution (from start):")
    hist: dict[int, int] = {}
    for v in from_start:
        bucket = min(v, 10)  # group 10+ together
        hist[bucket] = hist.get(bucket, 0) + 1
    for k in sorted(hist.keys()):
        label = f"{k}+" if k == 10 else str(k)
        bar = "#" * (hist[k] // 20)
        print(f"  {label:>3} words: {hist[k]:>4} ayahs {bar}")
    if never_from_start > 0:
        print(f"  N/A:     {never_from_start:>4} ayahs (never unique from start)")

    # Top never-unique ayahs
    never_unique = [r for r in results if r["from_start"] == -1]
    if never_unique:
        print(f"\nAyahs never uniquely identifiable from start (showing up to 20):")
        for r in never_unique[:20]:
            remaining = r["windows"][0].get("remaining", "?") if r["windows"] else "?"
            print(f"  {r['surah']}:{r['ayah']} ({r['surah_name_en']}) "
                  f"— {r['total_words']}w, {remaining} confusers at max length")

    # Hardest ayahs (highest disambiguation length from start)
    hardest = sorted(
        [r for r in results if r["from_start"] > 0],
        key=lambda x: -x["from_start"],
    )[:20]
    if hardest:
        print(f"\nHardest ayahs to disambiguate from start (top 20):")
        for r in hardest:
            print(f"  {r['surah']}:{r['ayah']} ({r['surah_name_en']}) "
                  f"— needs {r['from_start']} words")

    # Surah-level summary
    surah_stats: dict[int, list[int]] = {}
    surah_names: dict[int, str] = {}
    for r in results:
        s = r["surah"]
        surah_names[s] = r["surah_name_en"]
        if s not in surah_stats:
            surah_stats[s] = []
        if r["from_start"] > 0:
            surah_stats[s].append(r["from_start"])

    print(f"\nSurah-level average disambiguation (top 20 hardest):")
    surah_avgs = [
        (s, statistics.mean(vals), len(vals))
        for s, vals in surah_stats.items()
        if vals
    ]
    surah_avgs.sort(key=lambda x: -x[1])
    for s, avg, count in surah_avgs[:20]:
        print(f"  Surah {s:>3} ({surah_names[s]:>20}) — avg {avg:.1f} words ({count} ayahs)")

    # Formulaic phrases (word sequences in 5+ ayahs)
    print(f"\nFormulaic phrases (exact sequences in 5+ ayahs):")
    # We'll compute this from the index in main() and pass it here
    # For now just note the count


def extract_formulaic_phrases(
    ngram_index: dict[tuple, set[int]],
    min_count: int = 5,
    min_words: int = 2,
) -> list[tuple[tuple[str, ...], int]]:
    """Find word sequences appearing in min_count+ distinct ayahs."""
    phrases = []
    for gram, ayah_set in ngram_index.items():
        if len(gram) >= min_words and len(ayah_set) >= min_count:
            phrases.append((gram, len(ayah_set)))

    # Sort by count descending, then by length descending
    phrases.sort(key=lambda x: (-x[1], -len(x[0])))

    # Remove subsequences: if "A B C" is in the list and "A B" has the
    # same count, drop "A B" since it's less informative
    filtered = []
    for gram, count in phrases:
        # Check if any longer phrase in filtered has the same count and contains this gram
        is_subsumed = False
        for existing_gram, existing_count in filtered:
            if existing_count == count and len(existing_gram) > len(gram):
                if sequence_in_words(gram, list(existing_gram)):
                    is_subsumed = True
                    break
        if not is_subsumed:
            filtered.append((gram, count))

    return filtered[:200]  # cap at 200


# ---------------------------------------------------------------------------
# Confuser pair extraction
# ---------------------------------------------------------------------------
def extract_confuser_pairs(results: list[dict]) -> list[dict]:
    """
    Extract all (ayah_A, ayah_B) confuser pairs from the narrowing data.
    Returns pairs sorted by the maximum shared window length.
    """
    pairs: dict[tuple[str, str], int] = {}  # (key_A, key_B) -> max shared length

    for r in results:
        key_a = f"{r['surah']}:{r['ayah']}"
        for w in r["windows"]:
            for entry in w["narrowing"]:
                for ref in entry["refs"]:
                    key_b = f"{ref['s']}:{ref['a']}"
                    # Canonical ordering
                    pair = tuple(sorted([key_a, key_b]))
                    current_len = entry["len"]
                    if pair not in pairs or current_len > pairs[pair]:
                        pairs[pair] = current_len

    # Sort by shared length descending
    pair_list = [
        {"a": p[0], "b": p[1], "shared_len": l}
        for p, l in sorted(pairs.items(), key=lambda x: -x[1])
    ]
    return pair_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    script_dir = Path(__file__).parent
    quran_path = script_dir.parent / "web" / "frontend" / "public" / "quran.json"

    print("=" * 60)
    print("Quran Verse Disambiguation — Exact Word-Sequence Matching")
    print("Zero parameters. Deterministic. Reproducible.")
    print("=" * 60)

    # Load
    t0 = time.time()
    verses = load_quran(quran_path)
    print(f"\nLoaded {len(verses)} ayahs in {time.time() - t0:.2f}s")

    word_counts = [len(v["words"]) for v in verses]
    print(f"  Word counts: min={min(word_counts)}, max={max(word_counts)}, "
          f"mean={statistics.mean(word_counts):.1f}, median={statistics.median(word_counts):.0f}")

    # Build n-gram index
    MAX_N = 15
    t0 = time.time()
    ngram_index = build_ngram_index(verses, max_n=MAX_N)
    print(f"\nBuilt n-gram index (max_n={MAX_N}, {len(ngram_index):,} unique n-grams) "
          f"in {time.time() - t0:.2f}s")

    # ===== Phase 1: Per-ayah isolated analysis =====
    print("\n--- Per-ayah isolated analysis ---")
    t0 = time.time()
    results = []
    for i in range(len(verses)):
        result = analyze_ayah(i, verses, ngram_index, max_n=MAX_N)
        results.append(result)
        if (i + 1) % 500 == 0 or i == len(verses) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            ref = f"{result['surah']}:{result['ayah']}"
            print(f"  [{i + 1:>5}/{len(verses)}] {ref:<10} "
                  f"— {elapsed:.1f}s elapsed ({rate:.0f} ayahs/s)")

    phase1_time = time.time() - t0
    print(f"\nPer-ayah analysis: {len(verses)} ayahs in {phase1_time:.1f}s")

    # ===== Phase 2: Cross-boundary continuous analysis =====
    print("\n--- Cross-boundary analysis (continuous stream) ---")
    t0 = time.time()
    stream, pos_meta, boundary_positions = build_continuous_stream(verses)
    print(f"  Continuous stream: {len(stream):,} words, "
          f"{len(boundary_positions):,} ayah boundaries")

    t1 = time.time()
    cont_index = build_continuous_ngram_index(stream, max_n=MAX_N)
    print(f"  Built continuous index ({len(cont_index):,} n-grams) in {time.time() - t1:.1f}s")

    t1 = time.time()
    boundary_results = analyze_boundaries(
        verses, stream, pos_meta, boundary_positions,
        cont_index, max_n=MAX_N, context_words=10,
    )
    phase2_time = time.time() - t0
    print(f"  Analyzed {len(boundary_results)} boundaries in {time.time() - t1:.1f}s")

    # Print boundary stats
    unique_boundaries = [b for b in boundary_results if b["best_crossing"] is not None]
    never_unique_boundaries = [b for b in boundary_results if b["best_crossing"] is None]
    from_last = [b["from_last_word"] for b in boundary_results if b["from_last_word"] is not None]

    print(f"\n  BOUNDARY STATS:")
    print(f"    Total boundaries: {len(boundary_results)}")
    print(f"    Uniquely identifiable crossings: {len(unique_boundaries)} "
          f"({100*len(unique_boundaries)/len(boundary_results):.1f}%)")
    print(f"    Never-unique crossings: {len(never_unique_boundaries)}")
    if unique_boundaries:
        best_totals = [b["best_crossing"]["total"] for b in unique_boundaries]
        print(f"    Best crossing length: mean={statistics.mean(best_totals):.2f}, "
              f"median={statistics.median(best_totals):.1f}, max={max(best_totals)}")
    if from_last:
        print(f"    From last word: mean={statistics.mean(from_last):.2f}, "
              f"median={statistics.median(from_last):.1f}")

    # Show hardest boundaries
    hardest_boundaries = sorted(
        unique_boundaries,
        key=lambda b: -b["best_crossing"]["total"],
    )[:20]
    print(f"\n  Hardest boundaries (most words needed to cross):")
    for b in hardest_boundaries:
        bc = b["best_crossing"]
        print(f"    {b['prev_ayah']:>7} -> {b['next_ayah']:<7} "
              f"({b['prev_name']}/{b['next_name']}) "
              f"— {bc['tail']} tail + {bc['head']} head = {bc['total']} words"
              f"{' [SURAH]' if b['is_surah_transition'] else ''}")

    total_time = phase1_time + phase2_time
    print(f"\nTotal analysis time: {total_time:.1f}s")

    # Print statistics
    print_stats(results, len(verses))

    # Extract formulaic phrases
    phrases = extract_formulaic_phrases(ngram_index, min_count=5, min_words=2)
    print(f"\nFormulaic phrases (sequences in 5+ ayahs): {len(phrases)} found")
    for gram, count in phrases[:30]:
        print(f"  [{count:>3} ayahs] {' '.join(gram)}")

    # Extract confuser pairs
    pairs = extract_confuser_pairs(results)
    print(f"\nConfuser pairs: {len(pairs)} total")
    print("Top 30 by shared substring length:")
    for p in pairs[:30]:
        print(f"  {p['a']} <-> {p['b']}  (shared {p['shared_len']} words)")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------

    # 1. Full research output
    output_path = script_dir / "ambiguity-map.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 2. Compact frontend output
    compact: dict[str, dict] = {}
    for r in results:
        key = f"{r['surah']}:{r['ayah']}"
        # Collect unique confuser refs from start=0 narrowing
        confuser_keys: list[str] = []
        if r["windows"]:
            for entry in r["windows"][0]["narrowing"][:3]:
                for ref in entry.get("refs", [])[:3]:
                    ckey = f"{ref['s']}:{ref['a']}"
                    if ckey not in confuser_keys:
                        confuser_keys.append(ckey)

        compact[key] = {
            "w": r["total_words"],
            "d": [w["d"] for w in r["windows"]],
            "c": confuser_keys[:5],
        }

    compact_path = script_dir / "ambiguity-compact.json"
    with open(compact_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False)

    # 3. Formulaic phrases
    phrases_output = [
        {"phrase": " ".join(gram), "count": count, "words": len(gram)}
        for gram, count in phrases
    ]
    phrases_path = script_dir / "formulaic-phrases.json"
    with open(phrases_path, "w", encoding="utf-8") as f:
        json.dump(phrases_output, f, ensure_ascii=False, indent=2)

    # 4. Confuser pairs (top 500)
    pairs_path = script_dir / "confuser-pairs.json"
    with open(pairs_path, "w", encoding="utf-8") as f:
        json.dump(pairs[:500], f, ensure_ascii=False, indent=2)

    # 5. Boundary analysis
    boundary_path = script_dir / "boundary-analysis.json"
    with open(boundary_path, "w", encoding="utf-8") as f:
        json.dump(boundary_results, f, ensure_ascii=False, indent=2)

    # Print file sizes
    print(f"\n{'=' * 60}")
    print("OUTPUT FILES")
    print(f"{'=' * 60}")
    for p in [output_path, compact_path, phrases_path, pairs_path, boundary_path]:
        size = os.path.getsize(p)
        if size > 1024 * 1024:
            print(f"  {p.name}: {size / 1024 / 1024:.1f} MB")
        else:
            print(f"  {p.name}: {size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
