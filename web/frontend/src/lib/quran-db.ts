import { ratio, fragmentScore } from "./levenshtein";
import type { QuranVerse } from "./types";

const _BSM_PHONEMES_JOINED = "bismi allahi arraHmaani arraHiimi";

export function partialRatio(short: string, long: string): number {
  if (!short || !long) return 0.0;
  if (short.length > long.length) [short, long] = [long, short];
  const window = short.length;
  let best = 0.0;
  for (let i = 0; i <= Math.max(0, long.length - window); i++) {
    const r = ratio(short, long.slice(i, i + window));
    if (r > best) {
      best = r;
      if (best === 1.0) break;
    }
  }
  return best;
}

export class QuranDB {
  verses: QuranVerse[];
  private _byRef: Map<string, QuranVerse> = new Map();
  private _bySurah: Map<number, QuranVerse[]> = new Map();
  private _trigramIndex: Map<string, number[]> = new Map();

  constructor(data: QuranVerse[]) {
    this.verses = data;
    for (const v of data) {
      this._byRef.set(`${v.surah}:${v.ayah}`, v);
      const arr = this._bySurah.get(v.surah) ?? [];
      arr.push(v);
      this._bySurah.set(v.surah, arr);

      // Pre-compute bismillah-stripped phonemes for verse 1 of each surah
      // (Al-Fatiha 1:1 IS the bismillah, At-Tawbah 9 has none)
      if (
        v.ayah === 1 &&
        v.surah !== 1 &&
        v.surah !== 9 &&
        v.phonemes_joined.startsWith(_BSM_PHONEMES_JOINED)
      ) {
        const stripped = v.phonemes_joined.slice(_BSM_PHONEMES_JOINED.length).trim();
        v.phonemes_joined_no_bsm = stripped || null;
      } else {
        v.phonemes_joined_no_bsm = null;
      }

      // Pre-compute no-space versions for fragment scoring
      v.phonemes_joined_ns = v.phonemes_joined.replace(/ /g, "");
      v.phonemes_joined_no_bsm_ns = v.phonemes_joined_no_bsm
        ? v.phonemes_joined_no_bsm.replace(/ /g, "")
        : null;
    }
    this._buildTrigramIndex();
  }

  /** Build trigram inverted index: trigram → verse indices */
  private _buildTrigramIndex(): void {
    for (let idx = 0; idx < this.verses.length; idx++) {
      const v = this.verses[idx];
      const text = v.phonemes_joined_ns!;
      const seen = new Set<string>();
      for (let i = 0; i <= text.length - 3; i++) {
        const tri = text.slice(i, i + 3);
        if (seen.has(tri)) continue;
        seen.add(tri);
        const arr = this._trigramIndex.get(tri);
        if (arr) arr.push(idx);
        else this._trigramIndex.set(tri, [idx]);
      }
      // Also index bismillah-stripped version
      if (v.phonemes_joined_no_bsm_ns) {
        const noBsm = v.phonemes_joined_no_bsm_ns;
        for (let i = 0; i <= noBsm.length - 3; i++) {
          const tri = noBsm.slice(i, i + 3);
          if (seen.has(tri)) continue;
          seen.add(tri);
          const arr = this._trigramIndex.get(tri);
          if (arr) arr.push(idx);
          else this._trigramIndex.set(tri, [idx]);
        }
      }
    }
  }

  /** Get top candidates by trigram overlap with query */
  private _getCandidates(text: string, maxCandidates = 200): Set<number> {
    const noSpace = text.replace(/ /g, "");
    if (noSpace.length < 3) {
      // Too short for trigrams — return all
      return new Set(this.verses.map((_, i) => i));
    }
    const queryTrigrams = new Set<string>();
    for (let i = 0; i <= noSpace.length - 3; i++) {
      queryTrigrams.add(noSpace.slice(i, i + 3));
    }
    const hits = new Map<number, number>();
    for (const tri of queryTrigrams) {
      const posting = this._trigramIndex.get(tri);
      if (!posting) continue;
      for (const idx of posting) {
        hits.set(idx, (hits.get(idx) ?? 0) + 1);
      }
    }
    const sorted = [...hits.entries()].sort((a, b) => b[1] - a[1]);
    const candidates = new Set<number>();
    for (let i = 0; i < Math.min(sorted.length, maxCandidates); i++) {
      candidates.add(sorted[i][0]);
    }
    return candidates;
  }

  get totalVerses(): number {
    return this.verses.length;
  }

  get surahCount(): number {
    return this._bySurah.size;
  }

  getVerse(surah: number, ayah: number): QuranVerse | undefined {
    return this._byRef.get(`${surah}:${ayah}`);
  }

  getSurah(surah: number): QuranVerse[] {
    return this._bySurah.get(surah) ?? [];
  }

  getNextVerse(surah: number, ayah: number): QuranVerse | undefined {
    const verses = this._bySurah.get(surah) ?? [];
    for (let i = 0; i < verses.length; i++) {
      if (verses[i].ayah === ayah) {
        if (i + 1 < verses.length) return verses[i + 1];
        const nextSurah = this._bySurah.get(surah + 1) ?? [];
        return nextSurah[0];
      }
    }
    return undefined;
  }

  search(text: string, topK = 5): (QuranVerse & { score: number })[] {
    const scored: (QuranVerse & { score: number })[] = [];
    for (const v of this.verses) {
      const score = ratio(text, v.phonemes_joined);
      scored.push({ ...v, score });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }

  /**
   * Bayesian continuation priors — multiplicative instead of additive.
   * Returns multipliers: next verse → 1.35, +2 → 1.20, +3 → 1.10.
   * This makes it much harder for coincidentally similar unrelated verses
   * to outcompete the actual next verse in a sequence.
   */
  private _continuationPriors(
    hint: [number, number] | null,
  ): Map<string, number> {
    const priors = new Map<string, number>();
    if (!hint) return priors;

    const [hSurah, hAyah] = hint;
    const nv = this._byRef.get(`${hSurah}:${hAyah + 1}`);
    if (nv) {
      priors.set(`${hSurah}:${hAyah + 1}`, 1.35);
      if (this._byRef.has(`${hSurah}:${hAyah + 2}`))
        priors.set(`${hSurah}:${hAyah + 2}`, 1.20);
      if (this._byRef.has(`${hSurah}:${hAyah + 3}`))
        priors.set(`${hSurah}:${hAyah + 3}`, 1.10);
    } else {
      // Last ayah in surah — priors carry to first ayah(s) of next surah
      const nextVerses = this._bySurah.get(hSurah + 1) ?? [];
      const priorValues = [1.35, 1.20, 1.10];
      for (let i = 0; i < Math.min(nextVerses.length, 3); i++) {
        priors.set(
          `${nextVerses[i].surah}:${nextVerses[i].ayah}`,
          priorValues[i],
        );
      }
    }
    return priors;
  }

  /**
   * Length-aware scoring: selects ratio() or fragmentScore() based on
   * the length ratio between transcript and verse.
   *   - Similar lengths (0.7–1.3): ratio() — symmetric is fair
   *   - Query shorter  (< 0.7):    fragmentScore() blended — asymmetric partial
   *   - Query longer   (> 1.3):    ratio() — likely multi-verse
   */
  private static _smartScore(
    textNoSpace: string,
    verseNs: string,
  ): number {
    const lengthRatio = textNoSpace.length / verseNs.length;

    if (lengthRatio >= 0.7 && lengthRatio <= 1.3) {
      return ratio(textNoSpace, verseNs);
    } else if (lengthRatio < 0.7) {
      const frag = fragmentScore(textNoSpace, verseNs);
      const r = ratio(textNoSpace, verseNs);
      return Math.max(frag, 0.6 * frag + 0.4 * r);
    } else {
      return ratio(textNoSpace, verseNs);
    }
  }

  private static _suffixPrefixScore(text: string, verseText: string): number {
    const wordsT = text.split(" ");
    const wordsV = verseText.split(" ");
    if (wordsT.length < 2 || wordsV.length < 2) return 0.0;

    let best = 0.0;
    const maxTrim = Math.min(Math.floor(wordsT.length / 2), 4);
    for (let trim = 1; trim <= maxTrim; trim++) {
      const suffix = wordsT.slice(trim).join(" ");
      const n = wordsT.length - trim;
      const prefix = wordsV.slice(0, Math.min(n, wordsV.length)).join(" ");
      best = Math.max(best, ratio(suffix, prefix));
    }
    return best;
  }

  matchVerse(
    text: string,
    threshold = 0.3,
    maxSpan = 3,
    hint: [number, number] | null = null,
    returnTopK = 0,
  ): Record<string, any> | null {
    if (!text.trim()) return null;

    const priors = this._continuationPriors(hint);
    const noSpaceText = text.replace(/ /g, "");

    // Pass 1: trigram-pruned candidates scored with adaptive _smartScore
    const candidates = this._getCandidates(text, 200);
    // Always include continuation-prior verses in candidate set
    for (const key of priors.keys()) {
      const [s, a] = key.split(":").map(Number);
      const idx = this.verses.findIndex(v => v.surah === s && v.ayah === a);
      if (idx >= 0) candidates.add(idx);
    }

    const scored: [QuranVerse, number, number, number][] = [];
    for (const idx of candidates) {
      const v = this.verses[idx];
      let raw = QuranDB._smartScore(noSpaceText, v.phonemes_joined_ns!);
      if (v.phonemes_joined_no_bsm_ns) {
        raw = Math.max(raw, QuranDB._smartScore(noSpaceText, v.phonemes_joined_no_bsm_ns));
      }
      // Also check spaced ratio for same-length cases (smartScore uses no-space)
      const spacedRatio = ratio(text, v.phonemes_joined);
      raw = Math.max(raw, spacedRatio);
      if (v.phonemes_joined_no_bsm) {
        raw = Math.max(raw, ratio(text, v.phonemes_joined_no_bsm));
      }
      const prior = priors.get(`${v.surah}:${v.ayah}`) ?? 1.0;
      if (prior > 1.0) {
        const sp = QuranDB._suffixPrefixScore(text, v.phonemes_joined);
        raw = Math.max(raw, sp);
      }
      scored.push([v, raw, prior, Math.min(raw * prior, 1.0)]);
    }
    scored.sort((a, b) => b[3] - a[3]);

    // Top-20 surahs for Pass 2 multi-ayah spans
    const pass2Surahs = new Set<number>();
    for (let idx = 0; idx < Math.min(scored.length, 20); idx++) {
      pass2Surahs.add(scored[idx][0].surah);
    }

    const [bestV, bestRaw, bestBonus, bestScoreInit] = scored[0];
    let bestScore = bestScoreInit;
    let best: Record<string, any> = {
      ...bestV,
      score: bestScore,
      raw_score: bestRaw,
      bonus: bestBonus,
    };

    // Collect single-verse runners-up before span pass
    const topSingles = scored
      .slice(0, Math.max(returnTopK, 5))
      .map(([v, raw, bon, total]) => ({
        surah: v.surah,
        ayah: v.ayah,
        raw_score: Math.round(raw * 1000) / 1000,
        bonus: Math.round(bon * 1000) / 1000,
        score: Math.round(total * 1000) / 1000,
        phonemes_joined: v.phonemes_joined.slice(0, 60),
      }));

    // Pass 2: try multi-ayah spans in surahs from ratio-only top-20
    // (using pass2Surahs to avoid fragmentScore pollution)
    for (const s of pass2Surahs) {
      const verses = this._bySurah.get(s)!;
      for (let i = 0; i < verses.length; i++) {
        for (let span = 2; span <= maxSpan; span++) {
          if (i + span > verses.length) break;
          const chunk = verses.slice(i, i + span);
          // Use no-bismillah text for the first verse in a span
          const firstText =
            chunk[0].phonemes_joined_no_bsm ?? chunk[0].phonemes_joined;
          const combined = [firstText]
            .concat(chunk.slice(1).map((c) => c.phonemes_joined))
            .join(" ");
          const raw = ratio(text, combined);
          const prior =
            priors.get(`${chunk[0].surah}:${chunk[0].ayah}`) ?? 1.0;
          const score = Math.min(raw * prior, 1.0);
          if (score > bestScore) {
            bestScore = score;
            best = {
              surah: s,
              ayah: chunk[0].ayah,
              ayah_end: chunk[chunk.length - 1].ayah,
              text: chunk.map((c) => c.text_uthmani).join(" "),
              phonemes_joined: combined,
              score,
              raw_score: raw,
              bonus: prior,
            };
          }
        }
      }
    }

    if (bestScore >= threshold) {
      if (returnTopK > 0) {
        best.runners_up = topSingles.slice(0, returnTopK);
      }
      return best;
    }
    return null;
  }

  /**
   * Narrow match: only score verses near a known position.
   * Used when we have high confidence about location (post-tracking).
   * Falls back to full matchVerse() if nothing found in window.
   */
  matchVerseNarrow(
    text: string,
    hint: [number, number],
    windowSize = 5,
    threshold = 0.35,
  ): Record<string, any> | null {
    if (!text.trim()) return null;

    const noSpaceText = text.replace(/ /g, "");
    const priors = this._continuationPriors(hint);

    // Collect window: current verse + next windowSize + prev 2
    const window: QuranVerse[] = [];
    let current = this.getVerse(hint[0], hint[1]);
    if (current) window.push(current);

    let ref: [number, number] = [hint[0], hint[1]];
    for (let i = 0; i < windowSize; i++) {
      const next = this.getNextVerse(ref[0], ref[1]);
      if (!next) break;
      window.push(next);
      ref = [next.surah, next.ayah];
    }
    // Also add 2 previous verses (user might repeat)
    const hintSurahVerses = this.getSurah(hint[0]);
    for (const v of hintSurahVerses) {
      if (v.ayah >= hint[1] - 2 && v.ayah < hint[1]) {
        if (!window.find(w => w.surah === v.surah && w.ayah === v.ayah)) {
          window.push(v);
        }
      }
    }

    // Score narrow window with _smartScore
    const scored: [QuranVerse, number, number, number][] = [];
    for (const v of window) {
      let raw = QuranDB._smartScore(noSpaceText, v.phonemes_joined_ns!);
      if (v.phonemes_joined_no_bsm_ns) {
        raw = Math.max(raw, QuranDB._smartScore(noSpaceText, v.phonemes_joined_no_bsm_ns));
      }
      const spacedRatio = ratio(text, v.phonemes_joined);
      raw = Math.max(raw, spacedRatio);
      if (v.phonemes_joined_no_bsm) {
        raw = Math.max(raw, ratio(text, v.phonemes_joined_no_bsm));
      }
      const prior = priors.get(`${v.surah}:${v.ayah}`) ?? 1.0;
      if (prior > 1.0) {
        const sp = QuranDB._suffixPrefixScore(text, v.phonemes_joined);
        raw = Math.max(raw, sp);
      }
      scored.push([v, raw, prior, Math.min(raw * prior, 1.0)]);
    }
    scored.sort((a, b) => b[3] - a[3]);

    if (scored.length > 0 && scored[0][3] >= threshold) {
      const [bestV, bestRaw, bestBonus, bestScore] = scored[0];
      return {
        ...bestV,
        score: bestScore,
        raw_score: bestRaw,
        bonus: bestBonus,
      };
    }

    // Fallback: full matchVerse if narrow window didn't work
    return this.matchVerse(text, threshold, 3, hint);
  }
}
