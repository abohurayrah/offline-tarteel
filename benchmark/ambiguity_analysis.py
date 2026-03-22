"""
Quran Ambiguity Analysis — Modal Parallelized

For every ayah in the Quran (6236 total), for every possible starting word
position within that ayah, compute:
  1. How many words must the user read before the ayah is uniquely identifiable?
  2. Which other ayahs are confusers (share the same substring)?
  3. At each window length, how many candidates remain?

Users can start reading from the BEGINNING, MIDDLE, or END of any ayah.
So we analyze every possible starting position.

Output: ambiguity-map.json — used by the tracker to decide patience level.

Architecture:
  - 100 Modal containers, each handles ~62 ayahs
  - Each ayah × each start position → fuzzy substring search against all 6236 ayahs
  - Uses rapidfuzz for fast Levenshtein-based matching
  - Precomputed n-gram index for candidate filtering (avoid O(n²) full scan)
"""

import json
import math
import os
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("quran-ambiguity-analysis")

quran_json_path = Path(__file__).parent.parent / "web" / "frontend" / "public" / "quran.json"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("rapidfuzz>=3.0")
    .add_local_file(str(quran_json_path), remote_path="/data/quran.json")
)

# ---------------------------------------------------------------------------
# Arabic normalization (mirrors quran-db.ts normalizeArabic)
# ---------------------------------------------------------------------------
import re

_DIACRITIC_RE = re.compile(
    "[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)

def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for comparison — mirrors the TypeScript version."""
    text = text.replace("\u2581", " ")   # BPE marker
    text = text.replace("\uFEFF", "")    # BOM
    text = _DIACRITIC_RE.sub("", text)   # strip tashkeel
    for ch in "أإآٱ":
        text = text.replace(ch, "ا")
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = text.replace("ـ", "")         # tatweel
    text = re.sub(r"[،؟.!:]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_quran():
    """Load and normalize quran.json."""
    with open("/data/quran.json", "r", encoding="utf-8") as f:
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


def build_ngram_index(verses, max_n=6):
    """
    Build inverted index: ngram_string → set of verse indices.
    An n-gram is a contiguous sequence of n normalized words from a verse.
    Used to quickly find candidate verses that share a substring.
    """
    index = {}  # tuple of words → set of verse indices
    for idx, v in enumerate(verses):
        words = v["words"]
        for n in range(1, min(max_n + 1, len(words) + 1)):
            for start in range(len(words) - n + 1):
                gram = tuple(words[start:start + n])
                if gram not in index:
                    index[gram] = set()
                index[gram].add(idx)
    return index


# ---------------------------------------------------------------------------
# Core analysis function (runs per ayah)
# ---------------------------------------------------------------------------
def analyze_single_ayah(ayah_idx, verses, ngram_index, max_ngram_n=6):
    """
    For a single ayah, analyze every possible starting position.
    Returns the disambiguation data for this ayah.
    """
    from rapidfuzz import fuzz

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

    FUZZY_THRESHOLD = 70  # rapidfuzz uses 0-100 scale

    windows = []

    for start in range(total_words):
        max_len = total_words - start
        disambiguation_len = -1  # -1 means never unique from this position
        confusers_at_each_len = []

        for length in range(1, max_len + 1):
            window_words = words[start:start + length]
            window_str = " ".join(window_words)

            # Phase 1: Use n-gram index for exact candidate filtering
            # Find all verses that contain any sub-ngram of our window
            if length <= max_ngram_n:
                gram = tuple(window_words)
                candidates = ngram_index.get(gram, set()).copy()
            else:
                # For longer windows, intersect shorter n-gram lookups
                # Start with candidates from the first max_n words
                gram = tuple(window_words[:max_ngram_n])
                candidates = ngram_index.get(gram, set()).copy()
                # Also check the last max_n words to catch mid-verse matches
                gram_end = tuple(window_words[-max_ngram_n:])
                candidates = candidates.union(ngram_index.get(gram_end, set()))

            # Remove self
            candidates.discard(ayah_idx)

            # Phase 2: Fuzzy matching against candidates
            # For each candidate, check if our window appears as a substring
            fuzzy_matches = []
            for cand_idx in candidates:
                cand_text = verses[cand_idx]["text_norm"]
                # partial_ratio finds the best substring match
                score = fuzz.partial_ratio(window_str, cand_text)
                if score >= FUZZY_THRESHOLD:
                    fuzzy_matches.append({
                        "idx": cand_idx,
                        "surah": verses[cand_idx]["surah"],
                        "ayah": verses[cand_idx]["ayah"],
                        "score": score,
                    })

            # Phase 3: Also check non-indexed candidates for short windows
            # Short windows (1-2 words) may match anywhere, n-gram index
            # might miss some due to word-boundary differences
            if length <= 2:
                for cand_idx, cand_v in enumerate(verses):
                    if cand_idx == ayah_idx or cand_idx in candidates:
                        continue
                    score = fuzz.partial_ratio(window_str, cand_v["text_norm"])
                    if score >= FUZZY_THRESHOLD:
                        fuzzy_matches.append({
                            "idx": cand_idx,
                            "surah": cand_v["surah"],
                            "ayah": cand_v["ayah"],
                            "score": score,
                        })

            num_confusers = len(fuzzy_matches)
            # Store top-3 confusers for this length
            top_confusers = sorted(fuzzy_matches, key=lambda x: -x["score"])[:3]
            confusers_at_each_len.append({
                "length": length,
                "num_confusers": num_confusers,
                "top": [
                    {"surah": c["surah"], "ayah": c["ayah"], "score": c["score"]}
                    for c in top_confusers
                ],
            })

            if num_confusers == 0:
                disambiguation_len = length
                break

        # Build window result
        window_result = {
            "start": start,
            "min_unique_len": disambiguation_len,
            "narrowing": confusers_at_each_len,  # how candidates narrow with each word
        }

        # If never unique, record how many confusers remain at max length
        if disambiguation_len == -1 and confusers_at_each_len:
            last = confusers_at_each_len[-1]
            window_result["remaining_confusers"] = last["num_confusers"]

        windows.append(window_result)

    return {
        "surah": verse["surah"],
        "ayah": verse["ayah"],
        "total_words": total_words,
        "text_norm": verse["text_norm"],
        "surah_name_en": verse["surah_name_en"],
        "windows": windows,
        # Summary stats
        "min_disambiguation_from_start": windows[0]["min_unique_len"] if windows else -1,
        "best_disambiguation": min(
            (w["min_unique_len"] for w in windows if w["min_unique_len"] > 0),
            default=-1,
        ),
        "worst_disambiguation": max(
            (w["min_unique_len"] for w in windows if w["min_unique_len"] > 0),
            default=-1,
        ),
        "num_never_unique_positions": sum(
            1 for w in windows if w["min_unique_len"] == -1
        ),
    }


# ---------------------------------------------------------------------------
# Modal function — processes a batch of ayahs
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    timeout=600,
    memory=2048,
)
def analyze_batch(batch_indices: list[int]) -> list[dict]:
    """Analyze a batch of ayahs. Each container gets ~62 ayahs."""
    verses = load_quran()
    ngram_index = build_ngram_index(verses)

    results = []
    for i, idx in enumerate(batch_indices):
        t0 = time.time()
        result = analyze_single_ayah(idx, verses, ngram_index)
        elapsed = time.time() - t0
        ref = f"{result['surah']}:{result['ayah']}"
        print(f"  [{i+1}/{len(batch_indices)}] {ref} ({result['total_words']}w) — {elapsed:.1f}s")
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Orchestrator — distributes work across containers
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    print("=" * 60)
    print("Quran Ambiguity Analysis")
    print("=" * 60)

    # Load locally to get count
    with open(str(quran_json_path), "r", encoding="utf-8") as f:
        raw = json.load(f)
    total_ayahs = len(raw)
    print(f"Total ayahs: {total_ayahs}")

    # Split into batches for 100 containers
    NUM_CONTAINERS = 100
    batch_size = math.ceil(total_ayahs / NUM_CONTAINERS)
    batches = []
    for i in range(0, total_ayahs, batch_size):
        batches.append(list(range(i, min(i + batch_size, total_ayahs))))

    print(f"Distributing across {len(batches)} containers, ~{batch_size} ayahs each")
    print()

    # Launch all batches in parallel
    t0 = time.time()
    all_results = []

    for batch_result in analyze_batch.map(batches):
        all_results.extend(batch_result)
        print(f"  Progress: {len(all_results)}/{total_ayahs} ayahs complete")

    elapsed = time.time() - t0
    print(f"\nAll {total_ayahs} ayahs analyzed in {elapsed:.1f}s")

    # Sort by surah:ayah
    all_results.sort(key=lambda x: (x["surah"], x["ayah"]))

    # Compute global statistics
    disambig_from_start = [
        r["min_disambiguation_from_start"]
        for r in all_results
        if r["min_disambiguation_from_start"] > 0
    ]
    never_unique_count = sum(
        1 for r in all_results if r["min_disambiguation_from_start"] == -1
    )
    best_disambig = [
        r["best_disambiguation"]
        for r in all_results
        if r["best_disambiguation"] > 0
    ]

    print("\n" + "=" * 60)
    print("GLOBAL STATISTICS")
    print("=" * 60)
    if disambig_from_start:
        print(f"Disambiguation from start (words needed):")
        print(f"  Mean:   {sum(disambig_from_start)/len(disambig_from_start):.1f}")
        print(f"  Median: {sorted(disambig_from_start)[len(disambig_from_start)//2]}")
        print(f"  Min:    {min(disambig_from_start)}")
        print(f"  Max:    {max(disambig_from_start)}")
    print(f"  Never unique from start: {never_unique_count}/{total_ayahs}")

    if best_disambig:
        print(f"\nBest disambiguation (any start position):")
        print(f"  Mean:   {sum(best_disambig)/len(best_disambig):.1f}")
        print(f"  Median: {sorted(best_disambig)[len(best_disambig)//2]}")

    # Distribution histogram
    print("\nDisambiguation length distribution (from start):")
    hist = {}
    for v in disambig_from_start:
        bucket = min(v, 10)  # group 10+ together
        hist[bucket] = hist.get(bucket, 0) + 1
    for k in sorted(hist.keys()):
        label = f"{k}+" if k == 10 else str(k)
        bar = "█" * (hist[k] // 20)
        print(f"  {label:>3} words: {hist[k]:>4} ayahs {bar}")
    if never_unique_count > 0:
        print(f"  N/A:     {never_unique_count:>4} ayahs (never unique from start)")

    # Save full results
    output_path = Path(__file__).parent / "ambiguity-map.json"
    with open(str(output_path), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")

    # Also save a compact version for the frontend
    compact = {}
    for r in all_results:
        key = f"{r['surah']}:{r['ayah']}"
        compact[key] = {
            "w": r["total_words"],
            # For each start position: min words needed (-1 = never unique)
            "d": [w["min_unique_len"] for w in r["windows"]],
            # Top confusers for start=0 at length 1-3
            "c": [],
        }
        # Add top confusers from start=0
        if r["windows"]:
            first_window = r["windows"][0]
            for entry in first_window.get("narrowing", [])[:3]:
                for c in entry.get("top", [])[:2]:
                    ckey = f"{c['surah']}:{c['ayah']}"
                    if ckey not in [x[0] for x in compact[key]["c"]]:
                        compact[key]["c"].append([ckey, c["score"]])

    compact_path = Path(__file__).parent / "ambiguity-compact.json"
    with open(str(compact_path), "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False)
    print(f"Compact version saved to {compact_path}")
    print(f"  Size: {os.path.getsize(str(compact_path)) / 1024:.0f} KB")
