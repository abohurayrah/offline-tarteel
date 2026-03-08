"""
Comprehensive Quran Disambiguation Analysis — All 7 Research Questions

Reads the output of disambiguate.py (ambiguity-map.json) and produces:
  - All figures (13 plots saved as PNG)
  - All dataset artifacts (JSON files)
  - Console output with all statistics and tables

Requires: matplotlib, numpy, networkx
"""

import json
import math
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

QURAN_PATH = SCRIPT_DIR.parent / "web" / "frontend" / "public" / "quran.json"
MAP_PATH = SCRIPT_DIR / "ambiguity-map.json"
COMPACT_PATH = SCRIPT_DIR / "ambiguity-compact.json"
WAQAR_PATH = SCRIPT_DIR / "waqar144_mutashabihat.json"
BOUNDARY_PATH = SCRIPT_DIR / "boundary-analysis.json"

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


def load_data():
    """Load all required data files."""
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(QURAN_PATH, "r", encoding="utf-8") as f:
        quran_raw = json.load(f)
    return results, quran_raw


# ===================================================================
# RQ1: Disambiguation Length Distribution
# ===================================================================
def rq1_disambiguation_distribution(results):
    print("\n" + "=" * 70)
    print("RQ1: DISAMBIGUATION LENGTH DISTRIBUTION")
    print("=" * 70)

    from_start = [r["from_start"] for r in results]
    positive = [x for x in from_start if x > 0]
    never_unique = [r for r in results if r["from_start"] == -1]
    total = len(results)

    # Summary statistics
    print(f"\nTotal ayahs: {total}")
    print(f"Unique from start: {len(positive)} ({100*len(positive)/total:.1f}%)")
    print(f"Never unique from start: {len(never_unique)} ({100*len(never_unique)/total:.1f}%)")
    if positive:
        print(f"\nDisambiguation from start (for uniquely identifiable ayahs):")
        print(f"  Mean:   {statistics.mean(positive):.2f}")
        print(f"  Median: {statistics.median(positive):.1f}")
        print(f"  Stdev:  {statistics.stdev(positive):.2f}")
        print(f"  Min:    {min(positive)}")
        print(f"  Max:    {max(positive)}")

    # Cumulative distribution
    print(f"\nCumulative distribution:")
    for n in range(1, 20):
        count = sum(1 for x in positive if x <= n)
        pct = 100 * count / total
        pct_of_unique = 100 * count / len(positive) if positive else 0
        print(f"  <= {n:>2} words: {count:>5} ayahs ({pct:.1f}% of all, {pct_of_unique:.1f}% of unique)")
        if count == len(positive):
            break

    # Never-unique list
    print(f"\nAll {len(never_unique)} never-unique ayahs (from start):")
    never_unique_list = []
    for r in never_unique:
        remaining = r["windows"][0].get("remaining", 0) if r["windows"] else 0
        # Get confusers from narrowing
        confuser_refs = []
        if r["windows"] and r["windows"][0].get("narrowing"):
            last_narrowing = r["windows"][0]["narrowing"][-1]
            confuser_refs = [f"{c['s']}:{c['a']}" for c in last_narrowing.get("refs", [])]
        entry = {
            "key": f"{r['surah']}:{r['ayah']}",
            "surah_name_en": r["surah_name_en"],
            "total_words": r["total_words"],
            "remaining_confusers": remaining,
            "sample_confusers": confuser_refs[:5],
        }
        never_unique_list.append(entry)
        print(f"  {entry['key']:>7} ({entry['surah_name_en']:>20}) "
              f"— {entry['total_words']}w, {remaining} confusers, "
              f"e.g. {', '.join(confuser_refs[:3])}")

    # --- Figure 1: Histogram ---
    fig, ax = plt.subplots()
    bins = list(range(1, max(positive) + 2)) if positive else [1]
    counts, edges, patches = ax.hist(positive, bins=bins, color="#2196F3",
                                      edgecolor="white", alpha=0.85, align="left")
    # Add never-unique bar
    ax.bar(max(positive) + 1 if positive else 1, len(never_unique),
           color="#F44336", alpha=0.85, label=f"Never unique ({len(never_unique)})")
    ax.set_xlabel("Words needed to disambiguate (from start)")
    ax.set_ylabel("Number of ayahs")
    ax.set_title("RQ1: Disambiguation Length Distribution from Start")
    ax.legend()
    ax.set_xticks(range(1, (max(positive) + 3 if positive else 3)))
    labels = [str(i) for i in range(1, (max(positive) + 2 if positive else 2))]
    labels.append("N/A")
    ax.set_xticklabels(labels, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig01_histogram_from_start.png")
    plt.close(fig)
    print(f"\n  -> Saved fig01_histogram_from_start.png")

    # --- Figure 2: CDF ---
    fig, ax = plt.subplots()
    sorted_pos = sorted(positive)
    cumulative = np.arange(1, len(sorted_pos) + 1) / total * 100
    ax.plot(sorted_pos, cumulative, color="#2196F3", linewidth=2, label="Cumulative % of all ayahs")
    # Mark key thresholds
    for thresh in [50, 75, 90, 95]:
        idx = np.searchsorted(cumulative, thresh)
        if idx < len(sorted_pos):
            ax.axhline(thresh, color="gray", linestyle="--", alpha=0.3)
            ax.annotate(f"{thresh}% at {sorted_pos[idx]}w",
                       xy=(sorted_pos[idx], cumulative[idx]),
                       fontsize=9, ha="left")
    # Add never-unique as ceiling
    ax.axhline(100 * len(positive) / total, color="#F44336", linestyle="--",
               alpha=0.5, label=f"Max identifiable: {100*len(positive)/total:.1f}%")
    ax.set_xlabel("Maximum words needed")
    ax.set_ylabel("Cumulative % of ayahs identifiable")
    ax.set_title("RQ1: Cumulative Distribution — Words to Disambiguate")
    ax.legend()
    ax.set_xlim(0, max(positive) + 1 if positive else 2)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig02_cdf_from_start.png")
    plt.close(fig)
    print(f"  -> Saved fig02_cdf_from_start.png")

    # --- Table 13: Summary statistics ---
    best_all = [r["best"] for r in results if r["best"] > 0]
    worst_all = [r["worst"] for r in results if r["worst"] > 0]
    summary_table = {
        "total_ayahs": total,
        "unique_from_start": len(positive),
        "never_unique_from_start": len(never_unique),
        "from_start": {
            "mean": round(statistics.mean(positive), 2) if positive else None,
            "median": statistics.median(positive) if positive else None,
            "stdev": round(statistics.stdev(positive), 2) if len(positive) > 1 else None,
            "min": min(positive) if positive else None,
            "max": max(positive) if positive else None,
        },
        "best_position": {
            "mean": round(statistics.mean(best_all), 2) if best_all else None,
            "median": statistics.median(best_all) if best_all else None,
        },
        "worst_position": {
            "mean": round(statistics.mean(worst_all), 2) if worst_all else None,
            "median": statistics.median(worst_all) if worst_all else None,
            "max": max(worst_all) if worst_all else None,
        },
        "never_unique_ayahs": never_unique_list,
    }

    return summary_table


# ===================================================================
# RQ2: Positional Asymmetry
# ===================================================================
def rq2_positional_asymmetry(results):
    print("\n" + "=" * 70)
    print("RQ2: POSITIONAL ASYMMETRY")
    print("=" * 70)

    # For each ayah, compute best/worst and relative-position stats
    asymmetry_data = []
    # Buckets for relative position: 0%, 25%, 50%, 75%, 100%
    pos_buckets = {0: [], 25: [], 50: [], 75: [], 100: []}

    for r in results:
        if r["total_words"] < 2:
            continue
        windows = r["windows"]
        d_values = [w["d"] for w in windows]
        best = r["best"]
        worst = r["worst"]
        gap = worst - best if worst > 0 and best > 0 else None

        asymmetry_data.append({
            "key": f"{r['surah']}:{r['ayah']}",
            "surah_name_en": r["surah_name_en"],
            "total_words": r["total_words"],
            "best": best,
            "worst": worst,
            "gap": gap,
            "from_start": r["from_start"],
            "never_unique_positions": r["never_unique_positions"],
        })

        # Relative position disambiguation lengths
        total_w = len(windows)
        for w in windows:
            if w["d"] <= 0:
                continue
            rel_pos = w["start"] / (total_w - 1) * 100 if total_w > 1 else 0
            # Assign to nearest bucket
            nearest = min(pos_buckets.keys(), key=lambda b: abs(b - rel_pos))
            pos_buckets[nearest].append(w["d"])

    # Gap statistics
    gaps = [a["gap"] for a in asymmetry_data if a["gap"] is not None]
    if gaps:
        print(f"\nPositional asymmetry (worst - best disambiguation length):")
        print(f"  Mean gap:   {statistics.mean(gaps):.2f}")
        print(f"  Median gap: {statistics.median(gaps):.1f}")
        print(f"  Max gap:    {max(gaps)}")

    # Relative position means
    print(f"\nMean disambiguation length by relative position in ayah:")
    bucket_means = {}
    for pct in sorted(pos_buckets.keys()):
        vals = pos_buckets[pct]
        if vals:
            m = statistics.mean(vals)
            bucket_means[pct] = m
            print(f"  {pct:>3}%: {m:.2f} ({len(vals)} measurements)")

    # Top 20 most asymmetric
    asymmetry_data.sort(key=lambda x: -(x["gap"] or 0))
    print(f"\nTop 20 most asymmetric ayahs (largest best-worst gap):")
    top20_asym = asymmetry_data[:20]
    for a in top20_asym:
        print(f"  {a['key']:>7} ({a['surah_name_en']:>20}) "
              f"— best={a['best']}, worst={a['worst']}, gap={a['gap']}, "
              f"{a['never_unique_positions']} never-unique positions")

    # --- Figure 3: Box plots by relative position ---
    fig, ax = plt.subplots()
    box_data = [pos_buckets[p] for p in sorted(pos_buckets.keys()) if pos_buckets[p]]
    box_labels = [f"{p}%" for p in sorted(pos_buckets.keys()) if pos_buckets[p]]
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    showfliers=False, medianprops={"color": "red", "linewidth": 2})
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xlabel("Relative position in ayah")
    ax.set_ylabel("Words needed to disambiguate")
    ax.set_title("RQ2: Disambiguation Length by Starting Position")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig03_boxplot_position.png")
    plt.close(fig)
    print(f"\n  -> Saved fig03_boxplot_position.png")

    return {
        "gap_stats": {
            "mean": round(statistics.mean(gaps), 2) if gaps else None,
            "median": statistics.median(gaps) if gaps else None,
            "max": max(gaps) if gaps else None,
        },
        "bucket_means": {str(k): round(v, 2) for k, v in bucket_means.items()},
        "top20_asymmetric": top20_asym,
    }


# ===================================================================
# RQ3: Surah-Level Patterns
# ===================================================================
def rq3_surah_patterns(results):
    print("\n" + "=" * 70)
    print("RQ3: SURAH-LEVEL PATTERNS")
    print("=" * 70)

    # Aggregate per surah
    surah_data = defaultdict(lambda: {
        "from_start": [], "best": [], "worst": [],
        "word_counts": [], "name": "", "never_unique": 0, "total": 0,
    })
    for r in results:
        s = r["surah"]
        surah_data[s]["name"] = r["surah_name_en"]
        surah_data[s]["total"] += 1
        surah_data[s]["word_counts"].append(r["total_words"])
        if r["from_start"] > 0:
            surah_data[s]["from_start"].append(r["from_start"])
        else:
            surah_data[s]["never_unique"] += 1
        if r["best"] > 0:
            surah_data[s]["best"].append(r["best"])
        if r["worst"] > 0:
            surah_data[s]["worst"].append(r["worst"])

    # Rank by mean from_start
    surah_rankings = []
    for s in sorted(surah_data.keys()):
        d = surah_data[s]
        entry = {
            "surah": s,
            "name": d["name"],
            "total_ayahs": d["total"],
            "mean_from_start": round(statistics.mean(d["from_start"]), 2) if d["from_start"] else None,
            "mean_best": round(statistics.mean(d["best"]), 2) if d["best"] else None,
            "mean_worst": round(statistics.mean(d["worst"]), 2) if d["worst"] else None,
            "mean_word_count": round(statistics.mean(d["word_counts"]), 1),
            "never_unique": d["never_unique"],
            "never_unique_pct": round(100 * d["never_unique"] / d["total"], 1),
        }
        surah_rankings.append(entry)

    # Hardest surahs
    ranked_hard = sorted(
        [s for s in surah_rankings if s["mean_from_start"] is not None],
        key=lambda x: -x["mean_from_start"],
    )
    print(f"\nTop 20 hardest surahs (by mean disambiguation from start):")
    for s in ranked_hard[:20]:
        print(f"  Surah {s['surah']:>3} ({s['name']:>20}) — "
              f"avg {s['mean_from_start']:.1f}w, "
              f"{s['never_unique']} never-unique, "
              f"{s['total_ayahs']} ayahs, "
              f"avg length {s['mean_word_count']:.0f}w")

    # Easiest surahs
    ranked_easy = sorted(
        [s for s in surah_rankings if s["mean_from_start"] is not None],
        key=lambda x: x["mean_from_start"],
    )
    print(f"\nTop 20 easiest surahs:")
    for s in ranked_easy[:20]:
        print(f"  Surah {s['surah']:>3} ({s['name']:>20}) — "
              f"avg {s['mean_from_start']:.1f}w, "
              f"{s['total_ayahs']} ayahs")

    # --- Figure 5: Bar chart surahs ranked by difficulty ---
    fig, ax = plt.subplots(figsize=(14, 6))
    surahs_with_data = [s for s in surah_rankings if s["mean_from_start"] is not None]
    surahs_with_data.sort(key=lambda x: -x["mean_from_start"])
    names = [f"{s['surah']}" for s in surahs_with_data]
    vals = [s["mean_from_start"] for s in surahs_with_data]
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(vals)))
    ax.bar(range(len(vals)), vals, color=colors, width=0.8)
    ax.set_xlabel("Surah number (sorted by difficulty)")
    ax.set_ylabel("Mean words to disambiguate from start")
    ax.set_title("RQ3: Surahs Ranked by Disambiguation Difficulty")
    # Label top 5 and bottom 5
    for i in range(min(5, len(surahs_with_data))):
        ax.annotate(surahs_with_data[i]["name"], (i, vals[i]),
                    fontsize=7, rotation=45, ha="left", va="bottom")
    for i in range(max(0, len(surahs_with_data) - 5), len(surahs_with_data)):
        ax.annotate(surahs_with_data[i]["name"], (i, vals[i]),
                    fontsize=7, rotation=45, ha="left", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig05_surah_difficulty_ranking.png")
    plt.close(fig)
    print(f"\n  -> Saved fig05_surah_difficulty_ranking.png")

    # --- Figure 6: Scatter — ayah length vs. disambiguation length ---
    fig, ax = plt.subplots()
    word_lens = []
    disambig_lens = []
    for r in results:
        if r["from_start"] > 0:
            word_lens.append(r["total_words"])
            disambig_lens.append(r["from_start"])
    ax.scatter(word_lens, disambig_lens, alpha=0.15, s=8, c="#2196F3")
    # Add trend line
    if word_lens:
        z = np.polyfit(word_lens, disambig_lens, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(word_lens), max(word_lens), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7,
                label=f"Trend: y = {z[0]:.3f}x + {z[1]:.2f}")
    corr = np.corrcoef(word_lens, disambig_lens)[0, 1] if len(word_lens) > 2 else 0
    ax.set_xlabel("Ayah length (words)")
    ax.set_ylabel("Words to disambiguate from start")
    ax.set_title(f"RQ3: Ayah Length vs. Disambiguation Length (r = {corr:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig06_length_vs_disambig.png")
    plt.close(fig)
    print(f"  -> Saved fig06_length_vs_disambig.png")

    # --- Figure 4: Heatmaps for 10 most interesting surahs ---
    # Pick: top 5 hardest + some with known patterns
    interesting_surahs = [s["surah"] for s in ranked_hard[:5]]
    for candidate in [55, 26, 77, 2, 12]:  # Ar-Rahman, Ash-Shu'ara, Al-Mursalat, Al-Baqara, Yusuf
        if candidate not in interesting_surahs:
            interesting_surahs.append(candidate)
    interesting_surahs = interesting_surahs[:10]

    for surah_num in interesting_surahs:
        surah_results = [r for r in results if r["surah"] == surah_num]
        if not surah_results:
            continue
        name = surah_results[0]["surah_name_en"]
        max_words = max(r["total_words"] for r in surah_results)
        n_ayahs = len(surah_results)

        # Build heatmap matrix: ayah index (y) x start position (x)
        matrix = np.full((n_ayahs, max_words), np.nan)
        for i, r in enumerate(surah_results):
            for w in r["windows"]:
                val = w["d"] if w["d"] > 0 else max_words + 1  # never-unique = max+1
                matrix[i, w["start"]] = val

        fig, ax = plt.subplots(figsize=(max(8, max_words * 0.3), max(4, n_ayahs * 0.15)))
        cmap = plt.cm.RdYlGn_r.copy()
        cmap.set_bad("white")
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest",
                       vmin=1, vmax=min(15, max_words))
        ax.set_xlabel("Starting word position")
        ax.set_ylabel("Ayah index within surah")
        ax.set_title(f"RQ3: Surah {surah_num} ({name}) — Disambiguation Heatmap")
        plt.colorbar(im, ax=ax, label="Words to disambiguate")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"fig04_heatmap_surah_{surah_num}.png")
        plt.close(fig)

    print(f"  -> Saved fig04_heatmap_surah_*.png for {len(interesting_surahs)} surahs")

    return surah_rankings


# ===================================================================
# RQ4: Confuser Pair Analysis
# ===================================================================
def rq4_confuser_pairs(results):
    print("\n" + "=" * 70)
    print("RQ4: CONFUSER PAIR ANALYSIS")
    print("=" * 70)

    # Build confuser pairs with position metadata
    pairs: dict[tuple[str, str], dict] = {}
    ayah_lookup = {f"{r['surah']}:{r['ayah']}": r for r in results}

    for r in results:
        key_a = f"{r['surah']}:{r['ayah']}"
        for w in r["windows"]:
            for entry in w["narrowing"]:
                for ref in entry["refs"]:
                    key_b = f"{ref['s']}:{ref['a']}"
                    pair_key = tuple(sorted([key_a, key_b]))
                    shared_len = entry["len"]
                    if pair_key not in pairs or shared_len > pairs[pair_key]["shared_len"]:
                        pairs[pair_key] = {
                            "a": pair_key[0],
                            "b": pair_key[1],
                            "shared_len": shared_len,
                            "position_in_a": w["start"],
                        }

    pair_list = sorted(pairs.values(), key=lambda x: -x["shared_len"])
    total_pairs = len(pair_list)
    print(f"\nTotal confuser pairs: {total_pairs}")

    # Cross-surah vs intra-surah
    def get_surah(key):
        return int(key.split(":")[0])

    cross_surah = sum(1 for p in pair_list if get_surah(p["a"]) != get_surah(p["b"]))
    intra_surah = total_pairs - cross_surah
    print(f"Cross-surah pairs: {cross_surah} ({100*cross_surah/total_pairs:.1f}%)")
    print(f"Intra-surah pairs: {intra_surah} ({100*intra_surah/total_pairs:.1f}%)")

    # By threshold
    print(f"\nCross-surah vs intra-surah at various shared_len thresholds:")
    for t in [2, 3, 5, 7, 10, 15]:
        above = [p for p in pair_list if p["shared_len"] >= t]
        if not above:
            break
        cross = sum(1 for p in above if get_surah(p["a"]) != get_surah(p["b"]))
        intra = len(above) - cross
        print(f"  shared_len >= {t:>2}: {len(above):>6} pairs "
              f"({cross} cross-surah = {100*cross/len(above):.0f}%, "
              f"{intra} intra-surah = {100*intra/len(above):.0f}%)")

    # Top 50 pairs
    print(f"\nTop 50 confuser pairs:")
    top50 = []
    for p in pair_list[:50]:
        sa = get_surah(p["a"])
        sb = get_surah(p["b"])
        cross = "CROSS" if sa != sb else "same"
        r_a = ayah_lookup.get(p["a"], {})
        r_b = ayah_lookup.get(p["b"], {})
        name_a = r_a.get("surah_name_en", "")
        name_b = r_b.get("surah_name_en", "")
        top50.append({
            "a": p["a"], "b": p["b"],
            "shared_len": p["shared_len"],
            "cross_surah": sa != sb,
            "name_a": name_a, "name_b": name_b,
        })
        print(f"  {p['a']:>7} ({name_a:>15}) <-> {p['b']:>7} ({name_b:>15}) "
              f"— {p['shared_len']} words [{cross}]")

    # --- Confuser graph analysis ---
    print(f"\nConfuser graph analysis (edges = shared_len >= 5):")
    G = nx.Graph()
    for p in pair_list:
        if p["shared_len"] >= 5:
            G.add_edge(p["a"], p["b"], weight=p["shared_len"])

    if G.number_of_nodes() > 0:
        components = list(nx.connected_components(G))
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Connected components: {len(components)}")
        largest_cc = max(components, key=len)
        print(f"  Largest component: {len(largest_cc)} nodes")

        # Hub ayahs
        degrees = sorted(G.degree(), key=lambda x: -x[1])
        print(f"\n  Top 20 hub ayahs (most confuser connections):")
        for node, deg in degrees[:20]:
            r = ayah_lookup.get(node, {})
            print(f"    {node:>7} ({r.get('surah_name_en', ''):>15}) — {deg} connections")

        # Cliques
        cliques = list(nx.find_cliques(G))
        cliques_3plus = [c for c in cliques if len(c) >= 3]
        print(f"\n  Cliques of size >= 3: {len(cliques_3plus)}")
        cliques_3plus.sort(key=lambda x: -len(x))
        for c in cliques_3plus[:10]:
            print(f"    Size {len(c)}: {', '.join(sorted(c))}")

        # --- Figure 8: Network visualization of largest component ---
        fig, ax = plt.subplots(figsize=(12, 10))
        subG = G.subgraph(largest_cc)
        pos = nx.spring_layout(subG, k=2, iterations=50, seed=42)
        node_sizes = [G.degree(n) * 50 + 100 for n in subG.nodes()]
        edge_weights = [subG[u][v]["weight"] for u, v in subG.edges()]
        nx.draw_networkx_nodes(subG, pos, ax=ax, node_size=node_sizes,
                               node_color="#2196F3", alpha=0.7)
        nx.draw_networkx_edges(subG, pos, ax=ax, width=[w/5 for w in edge_weights],
                               alpha=0.3, edge_color="gray")
        # Label top hubs
        top_nodes = sorted(subG.nodes(), key=lambda n: -G.degree(n))[:15]
        labels = {n: n for n in top_nodes}
        nx.draw_networkx_labels(subG, pos, labels, ax=ax, font_size=7)
        ax.set_title(f"RQ4: Confuser Graph — Largest Component ({len(largest_cc)} ayahs)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "fig08_confuser_network.png")
        plt.close(fig)
        print(f"\n  -> Saved fig08_confuser_network.png")
    else:
        components = []
        cliques_3plus = []

    # --- Figure 7: Table viz as bar chart of top 25 pairs ---
    fig, ax = plt.subplots(figsize=(10, 8))
    top25 = pair_list[:25]
    labels = [f"{p['a']} / {p['b']}" for p in top25]
    vals = [p["shared_len"] for p in top25]
    colors = ["#F44336" if get_surah(p["a"]) != get_surah(p["b"]) else "#2196F3"
              for p in top25]
    y_pos = range(len(top25))
    ax.barh(y_pos, vals, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Shared word-sequence length")
    ax.set_title("RQ4: Top 25 Confuser Pairs (red = cross-surah)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig07_top_confuser_pairs.png")
    plt.close(fig)
    print(f"  -> Saved fig07_top_confuser_pairs.png")

    return {
        "total_pairs": total_pairs,
        "cross_surah": cross_surah,
        "intra_surah": intra_surah,
        "top50": top50,
        "graph": {
            "nodes": G.number_of_nodes() if G else 0,
            "edges": G.number_of_edges() if G else 0,
            "components": len(components),
            "largest_component_size": len(largest_cc) if components else 0,
            "cliques_3plus": len(cliques_3plus),
        },
    }


# ===================================================================
# RQ5: Formulaic Phrase Analysis
# ===================================================================
def rq5_formulaic_phrases(results):
    print("\n" + "=" * 70)
    print("RQ5: FORMULAIC PHRASE ANALYSIS")
    print("=" * 70)

    # Rebuild n-gram index for phrase analysis
    # We need: for each phrase, at each occurrence, how many additional words to disambiguate?
    # Load the raw data and build the index
    _DIACRITIC_RE = re.compile(
        "[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
    )

    def normalize_arabic(text: str) -> str:
        text = text.replace("\u2581", " ").replace("\uFEFF", "")
        text = _DIACRITIC_RE.sub("", text)
        for ch in "أإآٱ":
            text = text.replace(ch, "ا")
        text = text.replace("ة", "ه").replace("ى", "ي").replace("ـ", "")
        text = re.sub(r"[،؟.!:]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # Build verse words for lookup
    verse_words = {}
    for r in results:
        key = f"{r['surah']}:{r['ayah']}"
        verse_words[key] = r["text_norm"].split()

    # Build n-gram index
    ngram_index: dict[tuple, set[str]] = {}
    for r in results:
        key = f"{r['surah']}:{r['ayah']}"
        words = r["text_norm"].split()
        for n in range(1, min(16, len(words) + 1)):
            for start in range(len(words) - n + 1):
                gram = tuple(words[start:start + n])
                if gram not in ngram_index:
                    ngram_index[gram] = set()
                ngram_index[gram].add(key)

    # Find formulaic phrases (appearing in 5+ ayahs, 2+ words)
    phrases = []
    for gram, ayah_set in ngram_index.items():
        if len(gram) >= 2 and len(ayah_set) >= 5:
            phrases.append((gram, ayah_set))

    # For each phrase, compute disambiguation penalty
    phrase_analysis = []
    results_lookup = {f"{r['surah']}:{r['ayah']}": r for r in results}

    for gram, ayah_set in phrases:
        count = len(ayah_set)
        phrase_str = " ".join(gram)
        phrase_len = len(gram)

        # For each ayah containing this phrase, find the disambiguation length
        # at the position where the phrase starts
        additional_words = []
        never_unique_count = 0

        for ayah_key in ayah_set:
            r = results_lookup.get(ayah_key)
            if not r:
                continue
            words = r["text_norm"].split()
            # Find where this phrase starts in the ayah
            for start in range(len(words) - phrase_len + 1):
                if tuple(words[start:start + phrase_len]) == gram:
                    # Look up disambiguation from this position
                    if start < len(r["windows"]):
                        d = r["windows"][start]["d"]
                        if d > 0:
                            # Additional words beyond the phrase itself
                            additional = max(0, d - phrase_len)
                            additional_words.append(additional)
                        else:
                            never_unique_count += 1
                    break  # only first occurrence per ayah

        mean_additional = statistics.mean(additional_words) if additional_words else float("inf")
        worst_additional = max(additional_words) if additional_words else -1
        never_unique_rate = never_unique_count / count if count > 0 else 0
        danger_score = count * mean_additional if additional_words else count * 100

        phrase_analysis.append({
            "phrase": phrase_str,
            "words": phrase_len,
            "count": count,
            "mean_additional": round(mean_additional, 2) if additional_words else None,
            "worst_additional": worst_additional,
            "never_unique_rate": round(never_unique_rate, 3),
            "danger_score": round(danger_score, 1),
        })

    # Remove sub-phrases with same count
    phrase_analysis.sort(key=lambda x: (-x["count"], -x["words"]))

    # Deduplicate — remove shorter phrases with same count as a longer superphrase
    seen_counts: dict[int, list[str]] = {}
    filtered = []
    for p in phrase_analysis:
        # Keep if no longer phrase has the same count
        dominated = False
        if p["count"] in seen_counts:
            for longer in seen_counts[p["count"]]:
                if p["phrase"] in longer:
                    dominated = True
                    break
        if not dominated:
            filtered.append(p)
            if p["count"] not in seen_counts:
                seen_counts[p["count"]] = []
            seen_counts[p["count"]].append(p["phrase"])

    phrase_analysis = filtered

    # Sort by danger score
    phrase_analysis.sort(key=lambda x: -(x["danger_score"] or 0))

    print(f"\nFormulaic phrases found: {len(phrase_analysis)}")
    print(f"\nTop 50 by danger score (frequency x mean additional words):")
    print(f"  {'Phrase':<35} {'Count':>5} {'MeanAdd':>7} {'Worst':>5} {'NeverUniq':>9} {'Danger':>8}")
    print(f"  {'-'*35} {'-'*5} {'-'*7} {'-'*5} {'-'*9} {'-'*8}")
    for p in phrase_analysis[:50]:
        ma = f"{p['mean_additional']:.1f}" if p["mean_additional"] is not None else "N/A"
        print(f"  {p['phrase']:<35} {p['count']:>5} {ma:>7} {p['worst_additional']:>5} "
              f"{p['never_unique_rate']:>8.1%} {p['danger_score']:>8.0f}")

    # Taxonomy
    taxonomy = {
        "theological": [],
        "formulaic_openings": [],
        "eschatological": [],
        "cosmological": [],
        "legislative": [],
        "other": [],
    }
    theological_kw = ["الله", "رب", "لا اله", "سبحان"]
    opening_kw = ["بسم", "يايها", "قل"]
    eschato_kw = ["يوم", "القيمه", "عذاب", "خلدين", "جنات", "نار"]
    cosmo_kw = ["السموت", "الارض", "السماء"]
    legal_kw = ["حرم", "كتب", "عليكم"]

    for p in phrase_analysis[:100]:
        ph = p["phrase"]
        if any(k in ph for k in theological_kw):
            taxonomy["theological"].append(p["phrase"])
        elif any(k in ph for k in opening_kw):
            taxonomy["formulaic_openings"].append(p["phrase"])
        elif any(k in ph for k in eschato_kw):
            taxonomy["eschatological"].append(p["phrase"])
        elif any(k in ph for k in cosmo_kw):
            taxonomy["cosmological"].append(p["phrase"])
        elif any(k in ph for k in legal_kw):
            taxonomy["legislative"].append(p["phrase"])
        else:
            taxonomy["other"].append(p["phrase"])

    print(f"\nPhrase taxonomy (top 100):")
    for cat, items in taxonomy.items():
        print(f"  {cat}: {len(items)} phrases")

    # --- Figure 9: Top 25 formulaic phrases ---
    fig, ax = plt.subplots(figsize=(10, 8))
    top25 = phrase_analysis[:25]
    labels = [p["phrase"][:40] for p in top25]
    vals = [p["danger_score"] or 0 for p in top25]
    freqs = [p["count"] for p in top25]
    y_pos = range(len(top25))
    bars = ax.barh(y_pos, vals, alpha=0.8, color="#FF9800")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8, fontfamily="sans-serif")
    # Annotate with frequency
    for i, (v, f) in enumerate(zip(vals, freqs)):
        ax.annotate(f"n={f}", (v, i), fontsize=7, va="center", ha="left")
    ax.set_xlabel("Danger Score (frequency x mean additional words)")
    ax.set_title("RQ5: Top 25 Formulaic Phrases by Disambiguation Danger")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig09_formulaic_danger.png")
    plt.close(fig)
    print(f"\n  -> Saved fig09_formulaic_danger.png")

    return {
        "total_phrases": len(phrase_analysis),
        "top50": phrase_analysis[:50],
        "taxonomy": {k: len(v) for k, v in taxonomy.items()},
    }


# ===================================================================
# RQ6: Validation Against Waqar144
# ===================================================================
def rq6_waqar144_validation(results):
    print("\n" + "=" * 70)
    print("RQ6: VALIDATION AGAINST WAQAR144 MUTASHABIHAT DATASET")
    print("=" * 70)

    if not WAQAR_PATH.exists():
        print("  Waqar144 dataset not found, skipping.")
        return None

    with open(WAQAR_PATH, "r", encoding="utf-8") as f:
        waqar_data = json.load(f)

    # Build global ayah index: global_ayah_number -> (surah, ayah)
    # Need to map Waqar144's global numbering to surah:ayah
    with open(QURAN_PATH, "r", encoding="utf-8") as f:
        quran_raw = json.load(f)

    global_to_key = {}
    for i, v in enumerate(quran_raw):
        global_to_key[i + 1] = f"{v['surah']}:{v['ayah']}"  # 1-indexed

    # Extract all Waqar144 pairs
    # Some entries have lists of ayah numbers (multi-ayah spans), handle both
    def resolve_ayah_field(field):
        """Convert ayah field (int or list) to list of global ayah numbers."""
        if isinstance(field, int):
            return [field]
        elif isinstance(field, list):
            return field
        return []

    waqar_pairs: set[tuple[str, str]] = set()
    parse_errors = 0
    for juz, entries in waqar_data.items():
        for entry in entries:
            src_globals = resolve_ayah_field(entry["src"]["ayah"])
            src_keys = []
            for sg in src_globals:
                k = global_to_key.get(sg)
                if k:
                    src_keys.append(k)
                else:
                    parse_errors += 1
            if not src_keys:
                continue
            for mut in entry.get("muts", []):
                mut_globals = resolve_ayah_field(mut["ayah"])
                for mg in mut_globals:
                    mut_key = global_to_key.get(mg)
                    if not mut_key:
                        parse_errors += 1
                        continue
                    for sk in src_keys:
                        pair = tuple(sorted([sk, mut_key]))
                        waqar_pairs.add(pair)

    print(f"Waqar144 pairs extracted: {len(waqar_pairs)} (parse errors: {parse_errors})")

    # Build our confuser pairs with shared_len
    our_pairs: dict[tuple[str, str], int] = {}
    for r in results:
        key_a = f"{r['surah']}:{r['ayah']}"
        for w in r["windows"]:
            for entry in w["narrowing"]:
                for ref in entry["refs"]:
                    key_b = f"{ref['s']}:{ref['a']}"
                    pair = tuple(sorted([key_a, key_b]))
                    shared_len = entry["len"]
                    if pair not in our_pairs or shared_len > our_pairs[pair]:
                        our_pairs[pair] = shared_len

    print(f"Our confuser pairs: {len(our_pairs)}")

    # Precision-Recall at varying thresholds
    print(f"\nPrecision-Recall at varying shared_len thresholds:")
    print(f"  {'Threshold':>9} {'Our pairs':>10} {'Recall':>8} {'Precision':>10} {'F1':>6}")
    pr_data = []
    for t in range(1, 20):
        our_at_t = {p for p, l in our_pairs.items() if l >= t}
        tp = len(waqar_pairs & our_at_t)
        recall = tp / len(waqar_pairs) if waqar_pairs else 0
        precision = tp / len(our_at_t) if our_at_t else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        pr_data.append({
            "threshold": t,
            "our_count": len(our_at_t),
            "tp": tp,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
        })
        print(f"  {t:>9} {len(our_at_t):>10} {recall:>7.1%} {precision:>9.1%} {f1:>6.3f}")
        if len(our_at_t) == 0:
            break

    # Unmatched Waqar144 pairs (in their dataset but not in ours at any threshold)
    unmatched = waqar_pairs - set(our_pairs.keys())
    print(f"\nWaqar144 pairs NOT in our map: {len(unmatched)}")
    print("  (These are likely phonemic or semantic confusers, not lexical)")
    for p in sorted(unmatched)[:20]:
        print(f"    {p[0]} <-> {p[1]}")

    # Novel pairs (in ours at high threshold but not in Waqar144)
    novel_threshold = 5
    our_high = {p for p, l in our_pairs.items() if l >= novel_threshold}
    novel = our_high - waqar_pairs
    novel_sorted = sorted(novel, key=lambda p: -our_pairs[p])
    print(f"\nNovel confuser pairs (shared_len >= {novel_threshold}, not in Waqar144): {len(novel)}")
    print(f"Top 20 novel pairs:")
    novel_list = []
    results_lookup = {f"{r['surah']}:{r['ayah']}": r for r in results}
    for p in novel_sorted[:20]:
        ra = results_lookup.get(p[0], {})
        rb = results_lookup.get(p[1], {})
        entry = {
            "a": p[0], "b": p[1],
            "shared_len": our_pairs[p],
            "name_a": ra.get("surah_name_en", ""),
            "name_b": rb.get("surah_name_en", ""),
        }
        novel_list.append(entry)
        print(f"    {p[0]:>7} ({entry['name_a']:>15}) <-> "
              f"{p[1]:>7} ({entry['name_b']:>15}) — {our_pairs[p]} words")

    # --- Figure 10: Precision-Recall curve ---
    fig, ax1 = plt.subplots()
    thresholds = [d["threshold"] for d in pr_data]
    recalls = [d["recall"] for d in pr_data]
    precisions = [d["precision"] for d in pr_data]
    f1s = [d["f1"] for d in pr_data]
    ax1.plot(thresholds, recalls, "b-o", label="Recall", linewidth=2)
    ax1.plot(thresholds, precisions, "r-s", label="Precision", linewidth=2)
    ax1.plot(thresholds, f1s, "g--^", label="F1", linewidth=1.5, alpha=0.7)
    ax1.set_xlabel("Shared word-sequence length threshold")
    ax1.set_ylabel("Score")
    ax1.set_title("RQ6: Precision-Recall vs. Waqar144 Mutashabihat Dataset")
    ax1.legend()
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig10_precision_recall.png")
    plt.close(fig)
    print(f"\n  -> Saved fig10_precision_recall.png")

    validation_output = {
        "waqar144_pairs": len(waqar_pairs),
        "our_pairs": len(our_pairs),
        "precision_recall": pr_data,
        "unmatched_count": len(unmatched),
        "unmatched_pairs": [list(p) for p in sorted(unmatched)[:50]],
        "novel_pairs": novel_list,
    }

    # Save validation artifact
    with open(SCRIPT_DIR / "validation-waqar144.json", "w", encoding="utf-8") as f:
        json.dump(validation_output, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved validation-waqar144.json")

    return validation_output


# ===================================================================
# RQ7: Structural Indexing and Lookup Properties
# ===================================================================
def rq7_structural_indexing(results):
    print("\n" + "=" * 70)
    print("RQ7: STRUCTURAL INDEXING AND LOOKUP PROPERTIES")
    print("=" * 70)

    total = len(results)

    # Minimum index depth D*
    best_disambigs = [r["best"] for r in results if r["best"] > 0]
    d_star = max(best_disambigs) if best_disambigs else -1
    print(f"\nMinimum index depth D* (max of all best disambiguations): {d_star}")
    print(f"  An index of all {d_star}-grams is sufficient to uniquely identify any ayah.")

    # Ayahs with no unique identifying substring at any position
    never_at_any = [r for r in results if r["best"] == -1]
    print(f"  Ayahs with NO unique identifying substring: {len(never_at_any)}")
    for r in never_at_any:
        print(f"    {r['surah']}:{r['ayah']} ({r['surah_name_en']}) — {r['total_words']}w")

    # Identifiability frontier: for n = 1, 2, ..., D*, how many ayahs
    # can be uniquely identified by at least one n-gram of that size?
    print(f"\nIdentifiability frontier:")
    frontier = []
    for n in range(1, d_star + 1):
        identifiable = sum(1 for r in results if r["best"] > 0 and r["best"] <= n)
        pct = 100 * identifiable / total
        frontier.append({"n": n, "identifiable": identifiable, "pct": round(pct, 2)})
        print(f"  n = {n:>2}: {identifiable:>5} ayahs ({pct:.1f}%)")

    # Optimal prefix set: shortest unique substring per ayah
    fingerprints = []
    for r in results:
        best_d = r["best"]
        if best_d <= 0:
            fingerprints.append({
                "key": f"{r['surah']}:{r['ayah']}",
                "fingerprint": None,
                "length": -1,
                "start": -1,
            })
            continue
        # Find the window with the best disambiguation
        for w in r["windows"]:
            if w["d"] == best_d:
                words = r["text_norm"].split()
                fp_words = words[w["start"]:w["start"] + best_d]
                fingerprints.append({
                    "key": f"{r['surah']}:{r['ayah']}",
                    "fingerprint": " ".join(fp_words),
                    "length": best_d,
                    "start": w["start"],
                })
                break

    fp_lengths = [f["length"] for f in fingerprints if f["length"] > 0]
    print(f"\nFingerprint statistics (shortest unique identifying substring):")
    if fp_lengths:
        print(f"  Mean length:   {statistics.mean(fp_lengths):.2f}")
        print(f"  Median length: {statistics.median(fp_lengths):.1f}")
        print(f"  Max length:    {max(fp_lengths)}")
        print(f"  Total fingerprints: {len(fp_lengths)}")

    # Distribution of fingerprint lengths
    print(f"\nFingerprint length distribution:")
    fp_hist: dict[int, int] = {}
    for l in fp_lengths:
        fp_hist[l] = fp_hist.get(l, 0) + 1
    for k in sorted(fp_hist.keys()):
        print(f"  {k:>2} words: {fp_hist[k]:>5} ayahs")

    # Fraction requiring non-prefix identification (start > 0)
    non_prefix = sum(1 for f in fingerprints if f["start"] > 0 and f["length"] > 0)
    prefix_only = sum(1 for f in fingerprints if f["start"] == 0 and f["length"] > 0)
    print(f"\nPrefix vs. non-prefix identification:")
    print(f"  Best fingerprint at start (prefix): {prefix_only} ayahs ({100*prefix_only/total:.1f}%)")
    print(f"  Best fingerprint mid-verse:         {non_prefix} ayahs ({100*non_prefix/total:.1f}%)")
    print(f"  -> {100*non_prefix/(non_prefix+prefix_only):.1f}% of ayahs benefit from mid-verse identification "
          f"that a simple prefix trie would miss")

    # --- Figure 12: Identifiability frontier ---
    fig, ax = plt.subplots()
    ns = [f["n"] for f in frontier]
    pcts = [f["pct"] for f in frontier]
    ax.plot(ns, pcts, "b-o", linewidth=2, markersize=5)
    ax.fill_between(ns, pcts, alpha=0.1, color="blue")
    ax.set_xlabel("N-gram size (n)")
    ax.set_ylabel("% of ayahs uniquely identifiable")
    ax.set_title("RQ7: Identifiability Frontier vs. N-gram Size")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    # Mark 90%, 95%, 99%
    for target in [90, 95, 99]:
        for f in frontier:
            if f["pct"] >= target:
                ax.axhline(target, color="gray", linestyle="--", alpha=0.3)
                ax.annotate(f"{target}% at n={f['n']}", (f["n"], f["pct"]),
                           fontsize=9, ha="right")
                break
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig12_identifiability_frontier.png")
    plt.close(fig)
    print(f"\n  -> Saved fig12_identifiability_frontier.png")

    # Save fingerprints artifact
    with open(SCRIPT_DIR / "fingerprints.json", "w", encoding="utf-8") as f:
        json.dump(fingerprints, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved fingerprints.json ({os.path.getsize(SCRIPT_DIR / 'fingerprints.json') / 1024:.0f} KB)")

    return {
        "d_star": d_star,
        "never_identifiable": len(never_at_any),
        "frontier": frontier,
        "fingerprint_stats": {
            "mean": round(statistics.mean(fp_lengths), 2) if fp_lengths else None,
            "median": statistics.median(fp_lengths) if fp_lengths else None,
            "max": max(fp_lengths) if fp_lengths else None,
        },
        "prefix_vs_nonprefix": {
            "prefix": prefix_only,
            "non_prefix": non_prefix,
            "non_prefix_pct": round(100 * non_prefix / (non_prefix + prefix_only), 1) if (non_prefix + prefix_only) > 0 else 0,
        },
    }


# ===================================================================
# RQ8: Cross-Boundary (Continuous Recitation) Analysis
# ===================================================================
def rq8_cross_boundary():
    print("\n" + "=" * 70)
    print("RQ8: CROSS-BOUNDARY DISAMBIGUATION (CONTINUOUS RECITATION)")
    print("=" * 70)

    if not BOUNDARY_PATH.exists():
        print("  boundary-analysis.json not found, skipping.")
        return None

    with open(BOUNDARY_PATH, "r", encoding="utf-8") as f:
        boundary_data = json.load(f)

    total = len(boundary_data)
    print(f"\nTotal ayah boundaries: {total}")

    # Unique vs never-unique
    unique = [b for b in boundary_data if b["best_crossing"] is not None]
    never = [b for b in boundary_data if b["best_crossing"] is None]
    print(f"Uniquely identifiable: {len(unique)} ({100*len(unique)/total:.1f}%)")
    print(f"Never unique (within 10w context): {len(never)} ({100*len(never)/total:.1f}%)")

    # Best crossing stats
    if unique:
        best_totals = [b["best_crossing"]["total"] for b in unique]
        best_tails = [b["best_crossing"]["tail"] for b in unique]
        best_heads = [b["best_crossing"]["head"] for b in unique]
        print(f"\nBest crossing window (total words):")
        print(f"  Mean:   {statistics.mean(best_totals):.2f}")
        print(f"  Median: {statistics.median(best_totals):.1f}")
        print(f"  Max:    {max(best_totals)}")
        print(f"  Tail (from prev ayah): mean={statistics.mean(best_tails):.2f}")
        print(f"  Head (into next ayah): mean={statistics.mean(best_heads):.2f}")

    # From-last-word stats
    from_last = [b["from_last_word"] for b in boundary_data if b["from_last_word"] is not None]
    no_from_last = sum(1 for b in boundary_data if b["from_last_word"] is None)
    if from_last:
        print(f"\nFrom last word of previous ayah:")
        print(f"  Mean words to unique: {statistics.mean(from_last):.2f}")
        print(f"  Median:               {statistics.median(from_last):.1f}")
        print(f"  Max:                  {max(from_last)}")
        print(f"  Never unique:         {no_from_last}")

    # Surah transition vs intra-surah
    surah_trans = [b for b in boundary_data if b["is_surah_transition"]]
    intra_surah = [b for b in boundary_data if not b["is_surah_transition"]]
    print(f"\nSurah transitions: {len(surah_trans)}")
    print(f"Intra-surah boundaries: {len(intra_surah)}")

    st_unique = [b for b in surah_trans if b["best_crossing"] is not None]
    is_unique = [b for b in intra_surah if b["best_crossing"] is not None]
    if st_unique:
        st_totals = [b["best_crossing"]["total"] for b in st_unique]
        print(f"  Surah transitions: {len(st_unique)}/{len(surah_trans)} unique, "
              f"mean={statistics.mean(st_totals):.2f}")
    if is_unique:
        is_totals = [b["best_crossing"]["total"] for b in is_unique]
        print(f"  Intra-surah:       {len(is_unique)}/{len(intra_surah)} unique, "
              f"mean={statistics.mean(is_totals):.2f}")

    # Distribution of best crossing totals
    print(f"\nCrossing length distribution:")
    if best_totals:
        cross_hist: dict[int, int] = {}
        for v in best_totals:
            cross_hist[v] = cross_hist.get(v, 0) + 1
        for k in sorted(cross_hist.keys()):
            pct = 100 * cross_hist[k] / len(unique)
            bar = "#" * int(pct)
            print(f"  {k:>2} words: {cross_hist[k]:>5} ({pct:.1f}%) {bar}")

    # Never-unique boundaries — what are they?
    print(f"\nNever-unique boundaries ({len(never)}):")
    for b in never[:20]:
        print(f"  {b['prev_ayah']:>7} -> {b['next_ayah']:<7} "
              f"({b['prev_name']}/{b['next_name']})"
              f"{' [SURAH]' if b['is_surah_transition'] else ''}")

    # Hardest boundaries
    hardest = sorted(unique, key=lambda b: -b["best_crossing"]["total"])[:25]
    print(f"\nTop 25 hardest boundaries:")
    hardest_list = []
    for b in hardest:
        bc = b["best_crossing"]
        entry = {
            "prev_ayah": b["prev_ayah"],
            "next_ayah": b["next_ayah"],
            "prev_name": b["prev_name"],
            "next_name": b["next_name"],
            "tail": bc["tail"],
            "head": bc["head"],
            "total": bc["total"],
            "is_surah_transition": b["is_surah_transition"],
        }
        hardest_list.append(entry)
        print(f"  {b['prev_ayah']:>7} -> {b['next_ayah']:<7} "
              f"— {bc['tail']} tail + {bc['head']} head = {bc['total']} total"
              f"{' [SURAH]' if b['is_surah_transition'] else ''}")

    # Per-surah boundary difficulty
    surah_boundary_stats = defaultdict(list)
    for b in boundary_data:
        if not b["is_surah_transition"] and b["best_crossing"] is not None:
            s = int(b["prev_ayah"].split(":")[0])
            surah_boundary_stats[s].append(b["best_crossing"]["total"])

    surah_boundary_ranking = []
    for s, vals in surah_boundary_stats.items():
        if len(vals) >= 3:
            surah_boundary_ranking.append({
                "surah": s,
                "mean": round(statistics.mean(vals), 2),
                "max": max(vals),
                "count": len(vals),
            })
    surah_boundary_ranking.sort(key=lambda x: -x["mean"])
    print(f"\nSurahs by mean boundary difficulty (intra-surah, 3+ boundaries):")
    for s in surah_boundary_ranking[:15]:
        print(f"  Surah {s['surah']:>3} — mean={s['mean']:.2f}, max={s['max']}, "
              f"n={s['count']}")

    # --- Figure 13: Histogram of crossing lengths ---
    fig, ax = plt.subplots()
    if best_totals:
        bins = list(range(1, max(best_totals) + 2))
        ax.hist(best_totals, bins=bins, color="#4CAF50", edgecolor="white",
                alpha=0.85, align="left")
        # Add never-unique bar
        if never:
            ax.bar(max(best_totals) + 1, len(never), color="#F44336", alpha=0.85,
                   label=f"Never unique ({len(never)})")
            ax.legend()
    ax.set_xlabel("Total words in crossing window")
    ax.set_ylabel("Number of boundaries")
    ax.set_title("RQ8: Cross-Boundary Disambiguation Length Distribution")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig13_boundary_histogram.png")
    plt.close(fig)
    print(f"\n  -> Saved fig13_boundary_histogram.png")

    # --- Figure 14: Tail vs Head scatter ---
    fig, ax = plt.subplots()
    if unique:
        tails = [b["best_crossing"]["tail"] for b in unique]
        heads = [b["best_crossing"]["head"] for b in unique]
        ax.scatter(tails, heads, alpha=0.3, s=15, c="#2196F3")
        ax.set_xlabel("Words from previous ayah (tail)")
        ax.set_ylabel("Words from next ayah (head)")
        ax.set_title("RQ8: Crossing Window Composition (tail vs head)")
        ax.set_xlim(0, max(tails) + 1)
        ax.set_ylim(0, max(heads) + 1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig14_boundary_tail_vs_head.png")
    plt.close(fig)
    print(f"  -> Saved fig14_boundary_tail_vs_head.png")

    # --- Figure 15: Surah boundary difficulty heatmap ---
    fig, ax = plt.subplots(figsize=(14, 5))
    surah_means = []
    surah_labels = []
    for s in range(1, 115):
        if s in surah_boundary_stats:
            vals = surah_boundary_stats[s]
            surah_means.append(statistics.mean(vals))
        else:
            surah_means.append(0)
        surah_labels.append(str(s))

    colors_map = plt.cm.RdYlGn_r(np.array(surah_means) / max(surah_means) if max(surah_means) > 0 else np.zeros(len(surah_means)))
    ax.bar(range(len(surah_means)), surah_means, color=colors_map, width=0.8)
    ax.set_xlabel("Surah number")
    ax.set_ylabel("Mean boundary crossing length")
    ax.set_title("RQ8: Mean Boundary Disambiguation Difficulty by Surah")
    ax.set_xticks(range(0, 114, 10))
    ax.set_xticklabels([str(i+1) for i in range(0, 114, 10)])
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig15_surah_boundary_difficulty.png")
    plt.close(fig)
    print(f"  -> Saved fig15_surah_boundary_difficulty.png")

    # --- Figure 16: Comparison bar — isolated vs boundary ---
    # Compare: mean isolated from-start vs mean boundary crossing
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["From Start\n(isolated)", "Best Position\n(isolated)",
                   "Boundary\n(cross-ayah)", "From Last Word\n(continuous)"]
    # Load isolated stats
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        map_data = json.load(f)
    fwd_start = [r["from_start"] for r in map_data if r["from_start"] > 0]
    fwd_best = [r["best"] for r in map_data if r["best"] > 0]
    vals = [
        statistics.mean(fwd_start) if fwd_start else 0,
        statistics.mean(fwd_best) if fwd_best else 0,
        statistics.mean(best_totals) if best_totals else 0,
        statistics.mean(from_last) if from_last else 0,
    ]
    bar_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    bars = ax.bar(categories, vals, color=bar_colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean words to disambiguate")
    ax.set_title("Disambiguation Difficulty: Isolated vs. Continuous Recitation")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig16_isolated_vs_boundary.png")
    plt.close(fig)
    print(f"  -> Saved fig16_isolated_vs_boundary.png")

    return {
        "total_boundaries": total,
        "unique_boundaries": len(unique),
        "never_unique_boundaries": len(never),
        "best_crossing_stats": {
            "mean": round(statistics.mean(best_totals), 2) if best_totals else None,
            "median": statistics.median(best_totals) if best_totals else None,
            "max": max(best_totals) if best_totals else None,
        },
        "from_last_word_stats": {
            "mean": round(statistics.mean(from_last), 2) if from_last else None,
            "median": statistics.median(from_last) if from_last else None,
            "max": max(from_last) if from_last else None,
        },
        "surah_transitions": len(surah_trans),
        "hardest_boundaries": hardest_list,
        "surah_boundary_ranking": surah_boundary_ranking[:15],
    }


# ===================================================================
# Summary Table (Figure 13)
# ===================================================================
def generate_summary_table(rq1, rq2, rq3, rq4, rq5, rq6, rq7, rq8):
    print("\n" + "=" * 70)
    print("TABLE 13: CORPUS-LEVEL SUMMARY STATISTICS")
    print("=" * 70)

    rows = [
        ("Total ayahs", str(rq1["total_ayahs"])),
        ("Unique from start", f"{rq1['unique_from_start']} ({100*rq1['unique_from_start']/rq1['total_ayahs']:.1f}%)"),
        ("Never unique from start", f"{rq1['never_unique_from_start']} ({100*rq1['never_unique_from_start']/rq1['total_ayahs']:.1f}%)"),
        ("Mean disambig. from start", f"{rq1['from_start']['mean']} words"),
        ("Median disambig. from start", f"{rq1['from_start']['median']} words"),
        ("Max disambig. from start", f"{rq1['from_start']['max']} words"),
        ("Mean best disambig. (any pos.)", f"{rq1['best_position']['mean']} words"),
        ("Mean worst disambig.", f"{rq1['worst_position']['mean']} words"),
        ("Mean positional gap", f"{rq2['gap_stats']['mean']} words"),
        ("Max positional gap", f"{rq2['gap_stats']['max']} words"),
        ("Total confuser pairs", f"{rq4['total_pairs']:,}"),
        ("Cross-surah confuser pairs", f"{rq4['cross_surah']:,} ({100*rq4['cross_surah']/rq4['total_pairs']:.1f}%)"),
        ("Confuser graph nodes (>=5 shared)", f"{rq4['graph']['nodes']:,}"),
        ("Confuser graph cliques (>=3)", f"{rq4['graph']['cliques_3plus']}"),
        ("Formulaic phrases (>=5 ayahs)", f"{rq5['total_phrases']}"),
        ("Min. index depth D*", f"{rq7['d_star']} words"),
        ("Non-prefix identifications", f"{rq7['prefix_vs_nonprefix']['non_prefix_pct']}%"),
    ]

    if rq6:
        rows.append(("Waqar144 pairs", str(rq6["waqar144_pairs"])))
        best_f1 = max(rq6["precision_recall"], key=lambda x: x["f1"])
        rows.append(("Best F1 vs Waqar144", f"{best_f1['f1']:.3f} at threshold {best_f1['threshold']}"))

    if rq8:
        rows.append(("Boundary crossings unique", f"{rq8['unique_boundaries']}/{rq8['total_boundaries']} ({100*rq8['unique_boundaries']/rq8['total_boundaries']:.1f}%)"))
        if rq8["best_crossing_stats"]["mean"]:
            rows.append(("Mean boundary crossing length", f"{rq8['best_crossing_stats']['mean']} words"))
        if rq8["from_last_word_stats"]["mean"]:
            rows.append(("Mean from-last-word", f"{rq8['from_last_word_stats']['mean']} words"))

    for label, value in rows:
        print(f"  {label:<40} {value}")

    return rows


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 70)
    print("COMPREHENSIVE QURAN DISAMBIGUATION ANALYSIS")
    print("Answering all 7 Research Questions")
    print("=" * 70)

    results, quran_raw = load_data()
    print(f"Loaded {len(results)} ayah results from {MAP_PATH.name}")

    # Run all RQs
    rq1 = rq1_disambiguation_distribution(results)
    rq2 = rq2_positional_asymmetry(results)
    rq3 = rq3_surah_patterns(results)
    rq4 = rq4_confuser_pairs(results)
    rq5 = rq5_formulaic_phrases(results)
    rq6 = rq6_waqar144_validation(results)
    rq7 = rq7_structural_indexing(results)
    rq8 = rq8_cross_boundary()

    # Summary table
    summary = generate_summary_table(rq1, rq2, rq3, rq4, rq5, rq6, rq7, rq8)

    # Save master analysis output
    analysis_output = {
        "rq1_distribution": rq1,
        "rq2_asymmetry": rq2,
        "rq3_surah_rankings": rq3,
        "rq4_confuser_pairs": rq4,
        "rq5_formulaic_phrases": rq5,
        "rq6_waqar144_validation": rq6,
        "rq7_structural_indexing": rq7,
        "rq8_cross_boundary": rq8,
    }
    with open(SCRIPT_DIR / "analysis-results.json", "w", encoding="utf-8") as f:
        json.dump(analysis_output, f, ensure_ascii=False, indent=2, default=str)

    # List all output files
    print("\n" + "=" * 70)
    print("ALL OUTPUT FILES")
    print("=" * 70)
    for p in sorted(SCRIPT_DIR.glob("*.json")):
        size = os.path.getsize(p)
        if size > 1024 * 1024:
            print(f"  {p.name}: {size / 1024 / 1024:.1f} MB")
        else:
            print(f"  {p.name}: {size / 1024:.0f} KB")
    print(f"\nFigures:")
    for p in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  {p.name}: {os.path.getsize(p) / 1024:.0f} KB")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
