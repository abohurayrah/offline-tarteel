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
      // Last ayah in surah — bonus carries to first ayah(s) of next surah
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

  /**
   * Length-aware scoring: automatically selects the right metric based on
   * the length ratio between transcript and verse.
   *
   * - When lengths are similar (ratio 0.7-1.3): use ratio() — symmetric is fair
   * - When transcript is much shorter (ratio < 0.7): use fragmentScore() — asymmetric,
   *   measures "how much of the transcript does the verse explain?"
   * - When transcript is longer (ratio > 1.3): use ratio() — likely multi-verse
   *
   * Returns the score plus which method was used for debugging.
   */
  private static _smartScore(
    text: string,
    textNoSpace: string,
    verseJoined: string,
    verseNs: string,
  ): number {
    const lengthRatio = textNoSpace.length / verseNs.length;

    if (lengthRatio >= 0.7 && lengthRatio <= 1.3) {
      // Similar lengths: symmetric ratio is fair and accurate
      return ratio(text, verseJoined);
    } else if (lengthRatio < 0.7) {
      // Transcript is shorter: use fragmentScore (directional containment)
      // Weight: 60% fragmentScore + 40% ratio (ratio still matters for short matches)
      const frag = fragmentScore(textNoSpace, verseNs);
      const rat = ratio(text, verseJoined);
      return Math.max(frag, rat * 0.4 + frag * 0.6);
    } else {
      // Transcript is longer: likely multi-verse, ratio handles this
      return ratio(text, verseJoined);
    }
  }

  matchVerse(
    text: string,
    threshold = 0.3,
    maxSpan = 3,
    hint: [number, number] | null = null,
    returnTopK = 0,
  ): Record<string, any> | null {
    if (!text.trim()) return null;

    const bonuses = this._continuationBonuses(hint);
    const noSpaceText = text.replace(/ /g, "");

    // Pass 1: score all single verses with length-aware smart scoring
    const scored: [QuranVerse, number, number, number][] = [];
    for (const v of this.verses) {
      let raw = QuranDB._smartScore(text, noSpaceText, v.phonemes_joined, v.phonemes_joined_ns!);
      if (v.phonemes_joined_no_bsm && v.phonemes_joined_no_bsm_ns) {
        raw = Math.max(raw, QuranDB._smartScore(text, noSpaceText, v.phonemes_joined_no_bsm, v.phonemes_joined_no_bsm_ns));
      }
      const bonus = bonuses.get(`${v.surah}:${v.ayah}`) ?? 0.0;
      if (bonus > 0) {
        const sp = QuranDB._suffixPrefixScore(text, v.phonemes_joined);
        raw = Math.max(raw, sp);
      }
      scored.push([v, raw, bonus, Math.min(raw + bonus, 1.0)]);
    }
    scored.sort((a, b) => b[3] - a[3]);

    // Save top-20 surahs from Pass 1 for Pass 2 span checking
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

    // Pass 2: try multi-ayah spans in top-20 surahs from Pass 1
    for (const s of Array.from(pass2Surahs)) {
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
              phonemes_joined: combined,
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
}
