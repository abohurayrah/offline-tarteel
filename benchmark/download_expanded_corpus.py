"""
Download diverse Quran recitation recordings from HuggingFace
for the expanded test corpus.

Source: ashraf-ali/quran-data (Tarteel.io user recordings)
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "ashraf-ali/quran-data"
OUTPUT_DIR = Path(__file__).parent / "test_corpus_expanded"
OUTPUT_DIR.mkdir(exist_ok=True)

# Quran verse word counts for categorization (approximate)
# Format: {(surah, ayah): word_count}
# We'll define categories based on known verse lengths
VERSE_WORD_COUNTS = {
    # Surah 1 - Al-Fatiha
    (1, 1): 4, (1, 2): 4, (1, 3): 2, (1, 4): 3,
    (1, 5): 5, (1, 6): 4, (1, 7): 9,
    # Surah 2 - selected verses
    (2, 1): 1, (2, 2): 7, (2, 3): 7, (2, 4): 10,
    (2, 5): 9, (2, 6): 8, (2, 7): 19,
    (2, 21): 8, (2, 22): 20, (2, 30): 17,
    (2, 31): 14, (2, 127): 11, (2, 128): 17,
    (2, 152): 6, (2, 153): 8, (2, 155): 12,
    (2, 156): 8, (2, 163): 6, (2, 183): 14,
    (2, 185): 38, (2, 186): 21, (2, 197): 27,
    (2, 201): 11, (2, 255): 50, (2, 256): 23,
    (2, 257): 26, (2, 261): 19, (2, 262): 16,
    (2, 267): 22, (2, 269): 15, (2, 282): 128,
    (2, 284): 18, (2, 285): 26, (2, 286): 32,
    # Surah 3 - Al Imran
    (3, 1): 1, (3, 2): 5, (3, 3): 9, (3, 4): 9,
    (3, 7): 34, (3, 8): 12, (3, 14): 28, (3, 18): 14,
    (3, 19): 17, (3, 26): 17, (3, 27): 13,
    (3, 31): 9, (3, 32): 7, (3, 33): 9,
    (3, 97): 14, (3, 102): 10, (3, 103): 22,
    (3, 104): 12, (3, 110): 18, (3, 133): 11,
    (3, 134): 15, (3, 159): 27, (3, 185): 20,
    (3, 190): 11, (3, 191): 20, (3, 200): 7,
    # Surah 36 - Ya-Sin
    (36, 1): 1, (36, 2): 2, (36, 3): 3, (36, 4): 3,
    (36, 5): 4, (36, 6): 9, (36, 7): 7,
    (36, 8): 11, (36, 9): 9, (36, 10): 6,
    (36, 11): 9, (36, 12): 12, (36, 13): 8,
    (36, 14): 10, (36, 15): 10, (36, 20): 7,
    (36, 30): 9, (36, 36): 10, (36, 40): 11,
    (36, 55): 6, (36, 58): 4, (36, 60): 10,
    (36, 65): 9, (36, 70): 8, (36, 77): 9,
    (36, 78): 6, (36, 79): 8, (36, 80): 7,
    (36, 81): 10, (36, 82): 8, (36, 83): 10,
    # Surah 55 - Ar-Rahman
    (55, 1): 1, (55, 2): 2, (55, 3): 2, (55, 4): 2,
    (55, 5): 4, (55, 6): 4, (55, 7): 4, (55, 8): 3,
    (55, 9): 5, (55, 10): 3, (55, 13): 4,
    (55, 14): 5, (55, 15): 5, (55, 17): 4,
    (55, 19): 2, (55, 20): 2, (55, 26): 4,
    (55, 27): 5, (55, 29): 6, (55, 33): 11,
    (55, 46): 5, (55, 56): 7, (55, 60): 5,
    (55, 64): 2, (55, 68): 5, (55, 72): 3,
    (55, 78): 5,
    # Surah 67 - Al-Mulk
    (67, 1): 8, (67, 2): 10, (67, 3): 14,
    (67, 4): 9, (67, 5): 12, (67, 6): 7,
    (67, 7): 8, (67, 8): 6, (67, 9): 10,
    (67, 10): 7, (67, 11): 5, (67, 12): 8,
    (67, 13): 6, (67, 14): 6, (67, 15): 12,
    (67, 22): 9, (67, 23): 11, (67, 24): 6,
    (67, 26): 7, (67, 29): 9, (67, 30): 8,
    # Surah 78 - An-Naba
    (78, 1): 2, (78, 2): 3, (78, 3): 4, (78, 4): 3,
    (78, 5): 3, (78, 6): 3, (78, 7): 3, (78, 8): 2,
    (78, 9): 3, (78, 10): 3, (78, 11): 3, (78, 12): 3,
    (78, 13): 3, (78, 14): 4, (78, 15): 4, (78, 16): 2,
    (78, 17): 4, (78, 18): 4, (78, 19): 4, (78, 20): 3,
    (78, 21): 3, (78, 22): 3, (78, 23): 3, (78, 24): 4,
    (78, 25): 3, (78, 31): 4, (78, 32): 3, (78, 33): 3,
    (78, 34): 2, (78, 35): 4, (78, 36): 3, (78, 37): 10,
    (78, 38): 13, (78, 39): 10, (78, 40): 12,
    # Surah 93 - Ad-Duha
    (93, 1): 1, (93, 2): 3, (93, 3): 5,
    (93, 4): 5, (93, 5): 4, (93, 6): 4,
    (93, 7): 3, (93, 8): 3, (93, 9): 4,
    (93, 10): 3, (93, 11): 4,
    # Surah 103 - Al-Asr
    (103, 1): 1, (103, 2): 4, (103, 3): 8,
    # Surah 110 - An-Nasr
    (110, 1): 5, (110, 2): 6, (110, 3): 5,
    # Surah 112 - Al-Ikhlas
    (112, 1): 4, (112, 2): 2, (112, 3): 3, (112, 4): 4,
    # Surah 113 - Al-Falaq
    (113, 1): 4, (113, 2): 4, (113, 3): 4,
    (113, 4): 4, (113, 5): 5,
    # Surah 114 - An-Nas
    (114, 1): 4, (114, 2): 3, (114, 3): 3,
    (114, 4): 5, (114, 5): 6, (114, 6): 4,
    # Additional surahs for diversity
    # Surah 18 - Al-Kahf
    (18, 1): 12, (18, 10): 10, (18, 28): 20,
    (18, 29): 27, (18, 39): 13, (18, 46): 13,
    (18, 109): 15, (18, 110): 22,
    # Surah 19 - Maryam
    (19, 1): 1, (19, 2): 3, (19, 3): 5,
    (19, 18): 7, (19, 19): 7, (19, 30): 8,
    (19, 65): 9, (19, 96): 8,
    # Surah 56 - Al-Waqia
    (56, 1): 3, (56, 2): 4, (56, 3): 2,
    (56, 4): 3, (56, 7): 3, (56, 10): 3,
    (56, 57): 3, (56, 75): 4, (56, 77): 4,
    (56, 78): 3, (56, 79): 6, (56, 80): 4,
    # Surah 72 - Al-Jinn
    (72, 1): 11, (72, 2): 8, (72, 18): 6,
    # Surah 73 - Al-Muzzammil
    (73, 1): 2, (73, 4): 5, (73, 8): 5, (73, 20): 60,
    # Surah 79 - An-Naziat
    (79, 1): 2, (79, 2): 2, (79, 3): 2, (79, 4): 2,
    (79, 5): 2, (79, 27): 5, (79, 34): 3, (79, 46): 8,
    # Surah 84 - Al-Inshiqaq
    (84, 1): 3, (84, 2): 3, (84, 6): 6,
    # Surah 87 - Al-Ala
    (87, 1): 3, (87, 14): 3, (87, 15): 3,
    # Surah 91 - Ash-Shams
    (91, 1): 2, (91, 7): 3, (91, 8): 3,
    # Surah 96 - Al-Alaq
    (96, 1): 4, (96, 2): 4, (96, 3): 3, (96, 4): 4, (96, 5): 5,
    # Surah 99 - Az-Zalzalah
    (99, 1): 4, (99, 7): 5, (99, 8): 5,
    # Surah 100 - Al-Adiyat
    (100, 1): 2, (100, 2): 2,
    # Surah 101 - Al-Qariah
    (101, 1): 1, (101, 2): 2, (101, 3): 3,
    # Surah 102 - At-Takathur
    (102, 1): 2, (102, 2): 3,
    # Surah 104 - Al-Humazah
    (104, 1): 4, (104, 2): 5,
    # Surah 105 - Al-Fil
    (105, 1): 5, (105, 2): 4, (105, 3): 4,
    # Surah 107 - Al-Maun
    (107, 1): 4, (107, 2): 4,
    # Surah 109 - Al-Kafirun
    (109, 1): 4, (109, 2): 4, (109, 3): 5,
    (109, 4): 4, (109, 5): 5, (109, 6): 3,
    # Surah 111 - Al-Masad
    (111, 1): 4, (111, 2): 6, (111, 3): 5,
    (111, 4): 4, (111, 5): 4,
}


def get_category(surah: int, ayah: int) -> str:
    wc = VERSE_WORD_COUNTS.get((surah, ayah))
    if wc is None:
        # Default heuristic: short surahs have short verses
        if surah >= 100:
            return "short"
        elif surah >= 50:
            return "medium"
        else:
            return "medium"
    if wc <= 4:
        return "short"
    elif wc <= 15:
        return "medium"
    else:
        return "long"


def main():
    print("Listing files from ashraf-ali/quran-data...")
    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    user_files = [f for f in all_files if f.startswith("data/audio/User/")]

    # Parse into structured data
    file_index = defaultdict(list)  # (surah, ayah) -> [filepath, ...]
    for f in user_files:
        fname = f.split("/")[-1]
        parts = fname.replace(".wav", "").split("_")
        if len(parts) >= 2:
            try:
                surah = int(parts[0])
                ayah = int(parts[1])
                file_index[(surah, ayah)].append(f)
            except ValueError:
                continue

    # Target: 200+ diverse samples
    # Strategy:
    # 1. Primary surahs (high priority): 1, 2, 3, 36, 55, 67, 78, 93, 103, 110, 112, 113, 114
    # 2. Secondary surahs (for diversity): 18, 19, 56, 72, 73, 79, 84, 87, 91, 96, 99-105, 107, 109, 111
    # Pick ~15 samples per primary surah, ~3-5 per secondary surah
    # Ensure mix of different ayahs within each surah
    # Ensure different user IDs for diversity

    primary_surahs = [1, 2, 3, 36, 55, 67, 78, 93, 103, 110, 112, 113, 114]
    secondary_surahs = [18, 19, 56, 72, 73, 79, 84, 87, 91, 96, 99, 100, 101, 102, 104, 105, 107, 109, 111]

    selected = []
    random.seed(42)  # Reproducibility

    # Primary surahs: up to 15 samples each, spread across ayahs
    for surah in primary_surahs:
        surah_ayahs = {(s, a): files for (s, a), files in file_index.items() if s == surah}
        if not surah_ayahs:
            print(f"  WARNING: No user recordings for surah {surah}")
            continue

        # Pick diverse ayahs
        ayah_keys = sorted(surah_ayahs.keys())
        target_count = min(15, len(ayah_keys) * 2)  # Up to 15, but limited by available ayahs

        # First, get one recording per unique ayah (up to target)
        sampled_ayahs = ayah_keys if len(ayah_keys) <= target_count else random.sample(ayah_keys, target_count)
        for key in sampled_ayahs:
            files = surah_ayahs[key]
            chosen = random.choice(files)
            selected.append((key[0], key[1], chosen))

        # If we need more, get additional recordings from different users for the same ayahs
        remaining = target_count - len(sampled_ayahs)
        if remaining > 0:
            # Pick ayahs that have multiple recordings
            multi_ayahs = [(k, v) for k, v in surah_ayahs.items() if len(v) > 1]
            random.shuffle(multi_ayahs)
            for key, files in multi_ayahs[:remaining]:
                # Pick a different recording than already selected
                already_selected = {s[2] for s in selected if s[0] == key[0] and s[1] == key[1]}
                available = [f for f in files if f not in already_selected]
                if available:
                    selected.append((key[0], key[1], random.choice(available)))

        print(f"  Surah {surah:3d}: selected {sum(1 for s in selected if s[0] == surah)} samples")

    # Secondary surahs: 3-5 samples each
    for surah in secondary_surahs:
        surah_ayahs = {(s, a): files for (s, a), files in file_index.items() if s == surah}
        if not surah_ayahs:
            print(f"  WARNING: No user recordings for surah {surah}")
            continue

        ayah_keys = sorted(surah_ayahs.keys())
        target_count = min(5, len(ayah_keys))
        sampled_ayahs = random.sample(ayah_keys, target_count) if len(ayah_keys) > target_count else ayah_keys

        for key in sampled_ayahs:
            files = surah_ayahs[key]
            selected.append((key[0], key[1], random.choice(files)))

        print(f"  Surah {surah:3d}: selected {sum(1 for s in selected if s[0] == surah)} samples")

    print(f"\nTotal selected: {len(selected)} recordings")

    # Download files
    manifest_samples = []
    downloaded = 0
    failed = 0

    for idx, (surah, ayah, filepath) in enumerate(selected):
        local_name = f"tarteel_{idx:03d}.wav"
        local_path = OUTPUT_DIR / local_name
        category = get_category(surah, ayah)
        sample_id = f"tarteel_{idx:03d}"

        manifest_samples.append({
            "id": sample_id,
            "file": local_name,
            "surah": surah,
            "ayah": ayah,
            "ayah_end": None,
            "category": category,
            "source": "tarteel",
            "expected_verses": [{"surah": surah, "ayah": ayah}],
        })

        if local_path.exists():
            print(f"  [{idx+1}/{len(selected)}] {local_name} already exists, skipping")
            downloaded += 1
            continue

        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filepath,
                repo_type="dataset",
                local_dir=str(OUTPUT_DIR / "_hf_cache"),
                local_dir_use_symlinks=False,
            )
            # Move from cache to proper location
            cached_path = OUTPUT_DIR / "_hf_cache" / filepath
            cached_path.rename(local_path)
            downloaded += 1
            print(f"  [{idx+1}/{len(selected)}] Downloaded {local_name} (surah {surah}, ayah {ayah}, {category})")
        except Exception as e:
            print(f"  [{idx+1}/{len(selected)}] FAILED {filepath}: {e}")
            failed += 1
            # Remove from manifest if download failed
            manifest_samples.pop()

    # Write manifest
    manifest = {"samples": manifest_samples}
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Clean up cache
    cache_dir = OUTPUT_DIR / "_hf_cache"
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)

    # Summary
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"  Successfully downloaded: {downloaded}")
    print(f"  Failed: {failed}")
    print(f"  Manifest entries: {len(manifest_samples)}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Manifest: {manifest_path}")

    # Category breakdown
    from collections import Counter
    cats = Counter(s["category"] for s in manifest_samples)
    print(f"\nCategory breakdown:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")

    # Surah breakdown
    surahs = Counter(s["surah"] for s in manifest_samples)
    print(f"\nSurah breakdown:")
    for s, count in sorted(surahs.items()):
        print(f"  Surah {s:3d}: {count}")


if __name__ == "__main__":
    main()
