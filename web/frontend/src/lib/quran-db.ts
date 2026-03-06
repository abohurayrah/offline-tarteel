import { ratio, fragmentScore } from "./levenshtein";
import type { QuranVerse } from "./types";

/**
 * Normalize Arabic text for comparison:
 * - Strip BPE markers, diacritics, normalize hamza/taa/yaa
 */
function normalizeArabic(text: string): string {
  text = text.replace(/\u2581/g, " ");  // BPE marker -> space
  text = text.replace(/\uFEFF/g, "");
  text = text.replace(/[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]/g, "");
  text = text.replace(/[أإآٱ]/g, "ا");
  text = text.replace(/ة/g, "ه");
  text = text.replace(/ى/g, "ي");
  text = text.replace(/ـ/g, "");
  text = text.replace(/[،؟.!:]/g, "");
  text = text.replace(/\s+/g, " ").trim();
  return text;
}

export { normalizeArabic };

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

const BSM_NORM = normalizeArabic("بسم الله الرحمن الرحيم");

export class QuranDB {
  verses: QuranVerse[];
  private _byRef: Map<string, QuranVerse> = new Map();
  private _bySurah: Map<number, QuranVerse[]> = new Map();
  private _trigramIndex: Map<string, number[]> = new Map();

  constructor(data: QuranVerse[]) {
    this.verses = data;
    for (const v of data) {
      // Normalize Arabic text at load time
      const norm = normalizeArabic(v.text_clean || v.text_uthmani);
      v.text_norm = norm;
      v.text_norm_ns = norm.replace(/ /g, "");
      v.text_words = norm.split(" ");

      // Strip bismillah for ayah 1 (except Al-Fatiha 1:1 and At-Tawbah 9:1)
      if (v.ayah === 1 && v.surah !== 1 && v.surah !== 9) {
        const stripped = norm.replace(BSM_NORM, "").trim();
        if (stripped.length > 0) {
          v.text_norm_no_bsm = stripped;
          v.text_norm_no_bsm_ns = stripped.replace(/ /g, "");
        } else {
          v.text_norm_no_bsm = null;
          v.text_norm_no_bsm_ns = null;
        }
      } else {
        v.text_norm_no_bsm = null;
        v.text_norm_no_bsm_ns = null;
      }

      this._byRef.set(`${v.surah}:${v.ayah}`, v);
      const arr = this._bySurah.get(v.surah) ?? [];
      arr.push(v);
      this._bySurah.set(v.surah, arr);
    }
    this._buildTrigramIndex();
  }

  private _buildTrigramIndex(): void {
    for (let idx = 0; idx < this.verses.length; idx++) {
      const v = this.verses[idx];
      const text = v.text_norm_ns!;
      const seen = new Set<string>();
      for (let i = 0; i <= text.length - 3; i++) {
        const tri = text.slice(i, i + 3);
        if (seen.has(tri)) continue;
        seen.add(tri);
        const arr = this._trigramIndex.get(tri);
        if (arr) arr.push(idx);
        else this._trigramIndex.set(tri, [idx]);
      }
      if (v.text_norm_no_bsm_ns) {
        const noBsm = v.text_norm_no_bsm_ns;
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

  private _getCandidates(text: string, maxCandidates = 200): Set<number> {
    const noSpace = text.replace(/ /g, "");
    if (noSpace.length < 3) {
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
    const normText = normalizeArabic(text);
    const scored: (QuranVerse & { score: number })[] = [];
    for (const v of this.verses) {
      const score = ratio(normText, v.text_norm!);
      scored.push({ ...v, score });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }

  private _continuationBonuses(
    hint: [number, number] | null,
  ): Map<string, number> {
    const bonuses = new Map<string, number>();
    if (!hint) return bonuses;

    const [hSurah, hAyah] = hint;
    const nv = this._byRef.get(`${hSurah}:${hAyah + 1}`);
    if (nv) {
      bonuses.set(`${hSurah}:${hAyah + 1}`, 0.22);
      if (this._byRef.has(`${hSurah}:${hAyah + 2}`))
        bonuses.set(`${hSurah}:${hAyah + 2}`, 0.12);
      if (this._byRef.has(`${hSurah}:${hAyah + 3}`))
        bonuses.set(`${hSurah}:${hAyah + 3}`, 0.06);
    } else {
      const nextVerses = this._bySurah.get(hSurah + 1) ?? [];
      const bonusValues = [0.22, 0.12, 0.06];
      for (let i = 0; i < Math.min(nextVerses.length, 3); i++) {
        bonuses.set(
          `${nextVerses[i].surah}:${nextVerses[i].ayah}`,
          bonusValues[i],
        );
      }
    }
    return bonuses;
  }

  /**
   * Length-aware scoring: selects ratio() or fragmentScore() based on
   * the length ratio between transcript and verse.
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

    // Normalize the input transcript
    const normText = normalizeArabic(text);
    const noSpaceText = normText.replace(/ /g, "");

    if (!noSpaceText || noSpaceText.length < 3) return null;

    const bonuses = this._continuationBonuses(hint);

    // Pass 1: trigram-pruned candidates
    const candidates = this._getCandidates(normText, 200);
    for (const key of bonuses.keys()) {
      const [s, a] = key.split(":").map(Number);
      const idx = this.verses.findIndex(v => v.surah === s && v.ayah === a);
      if (idx >= 0) candidates.add(idx);
    }

    const scored: [QuranVerse, number, number, number][] = [];
    for (const idx of candidates) {
      const v = this.verses[idx];
      let raw = QuranDB._smartScore(noSpaceText, v.text_norm_ns!);
      if (v.text_norm_no_bsm_ns) {
        raw = Math.max(raw, QuranDB._smartScore(noSpaceText, v.text_norm_no_bsm_ns));
      }
      const spacedRatio = ratio(normText, v.text_norm!);
      raw = Math.max(raw, spacedRatio);
      if (v.text_norm_no_bsm) {
        raw = Math.max(raw, ratio(normText, v.text_norm_no_bsm));
      }
      const bonus = bonuses.get(`${v.surah}:${v.ayah}`) ?? 0.0;
      if (bonus > 0) {
        const sp = QuranDB._suffixPrefixScore(normText, v.text_norm!);
        raw = Math.max(raw, sp);
      }
      scored.push([v, raw, bonus, Math.min(raw + bonus, 1.0)]);
    }
    scored.sort((a, b) => {
      const diff = b[3] - a[3];
      if (Math.abs(diff) < 0.001) {
        // Tiebreaker: prefer verse closest in length to transcript
        const lenA = a[0].text_norm_ns!.length;
        const lenB = b[0].text_norm_ns!.length;
        return Math.abs(lenA - noSpaceText.length) - Math.abs(lenB - noSpaceText.length);
      }
      return diff;
    });

    // Top-20 surahs for Pass 2 multi-ayah spans
    const pass2Surahs = new Set<number>();
    for (let idx = 0; idx < Math.min(scored.length, 20); idx++) {
      pass2Surahs.add(scored[idx][0].surah);
    }

    const [bestV, bestRaw, bestBonus, bestScoreInit] = scored[0] ?? [null, 0, 0, 0];
    if (!bestV) return null;

    let bestScore = bestScoreInit;
    let best: Record<string, any> = {
      ...bestV,
      score: bestScore,
      raw_score: bestRaw,
      bonus: bestBonus,
    };

    const topSingles = scored
      .slice(0, Math.max(returnTopK, 5))
      .map(([v, raw, bon, total]) => ({
        surah: v.surah,
        ayah: v.ayah,
        raw_score: Math.round(raw * 1000) / 1000,
        bonus: Math.round(bon * 1000) / 1000,
        score: Math.round(total * 1000) / 1000,
        text_norm: (v.text_norm ?? "").slice(0, 60),
      }));

    // Pass 2: multi-ayah spans
    for (const s of pass2Surahs) {
      const verses = this._bySurah.get(s)!;
      for (let i = 0; i < verses.length; i++) {
        for (let span = 2; span <= maxSpan; span++) {
          if (i + span > verses.length) break;
          const chunk = verses.slice(i, i + span);
          const firstText = chunk[0].text_norm_no_bsm ?? chunk[0].text_norm!;
          const combined = [firstText]
            .concat(chunk.slice(1).map((c) => c.text_norm!))
            .join(" ");
          const raw = ratio(normText, combined);
          const bonus =
            bonuses.get(`${chunk[0].surah}:${chunk[0].ayah}`) ?? 0.0;
          const score = Math.min(raw + bonus, 1.0);
          if (score > bestScore) {
            bestScore = score;
            best = {
              surah: s,
              ayah: chunk[0].ayah,
              ayah_end: chunk[chunk.length - 1].ayah,
              text: chunk.map((c) => c.text_uthmani).join(" "),
              text_norm: combined,
              score,
              raw_score: raw,
              bonus,
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
   */
  matchVerseNarrow(
    text: string,
    hint: [number, number],
    windowSize = 5,
    threshold = 0.35,
  ): Record<string, any> | null {
    if (!text.trim()) return null;

    const normText = normalizeArabic(text);
    const noSpaceText = normText.replace(/ /g, "");
    const bonuses = this._continuationBonuses(hint);

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
    const hintSurahVerses = this.getSurah(hint[0]);
    for (const v of hintSurahVerses) {
      if (v.ayah >= hint[1] - 2 && v.ayah < hint[1]) {
        if (!window.find(w => w.surah === v.surah && w.ayah === v.ayah)) {
          window.push(v);
        }
      }
    }

    const scored: [QuranVerse, number, number, number][] = [];
    for (const v of window) {
      let raw = QuranDB._smartScore(noSpaceText, v.text_norm_ns!);
      if (v.text_norm_no_bsm_ns) {
        raw = Math.max(raw, QuranDB._smartScore(noSpaceText, v.text_norm_no_bsm_ns));
      }
      const spacedRatio = ratio(normText, v.text_norm!);
      raw = Math.max(raw, spacedRatio);
      if (v.text_norm_no_bsm) {
        raw = Math.max(raw, ratio(normText, v.text_norm_no_bsm));
      }
      const bonus = bonuses.get(`${v.surah}:${v.ayah}`) ?? 0.0;
      if (bonus > 0) {
        const sp = QuranDB._suffixPrefixScore(normText, v.text_norm!);
        raw = Math.max(raw, sp);
      }
      scored.push([v, raw, bonus, Math.min(raw + bonus, 1.0)]);
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

    return this.matchVerse(text, threshold, 3, hint);
  }
}
