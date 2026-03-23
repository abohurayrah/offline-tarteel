import { ratio, fragmentScore, sellersWordMatch, phoneticRatio } from "./levenshtein";
import type { QuranVerse, VerseMatch, VerseMatchCandidate } from "./types";

/**
 * Normalize Arabic text for comparison.
 *
 * Handles the gap between Whisper ASR output (modern Arabic) and
 * Quranic text (Uthmani-derived). Strips diacritics, normalizes
 * hamza/taa/yaa variants, removes Quran-specific marks, and handles
 * zero-width Unicode characters.
 */
function normalizeArabic(text: string): string {
  text = text.replace(/\u2581/g, " ");  // BPE marker -> space
  text = text.replace(/\uFEFF/g, "");
  // Unified range U+06D6-U+06ED strips ALL Quranic annotation marks including
  // small waw ۥ (U+06E5, ~990 verses), small yaa ۦ (U+06E6, ~833 verses),
  // rub el hizb ۞ (U+06DE, ~199 verses), sajdah ۩ (U+06E9, ~15 verses)
  text = text.replace(/[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]/g, "");
  text = text.replace(/[أإآٱ]/g, "ا");
  text = text.replace(/ة/g, "ه");    // taa marbuta → haa
  text = text.replace(/ى/g, "ي");    // alif maqsura → yaa
  text = text.replace(/ـ/g, "");     // tatweel
  text = text.replace(/[،؟.!:]/g, ""); // Arabic punctuation
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

/**
 * Fast character-level similarity (Jaccard on character bigrams).
 * Used for fuzzy word matching in the prefix trie walk.
 */
function _charSimilarity(a: string, b: string): number {
  if (a === b) return 1.0;
  if (a.length < 2 || b.length < 2) return a === b ? 1.0 : 0.0;
  const bigramsA = new Set<string>();
  for (let i = 0; i < a.length - 1; i++) bigramsA.add(a[i] + a[i + 1]);
  let intersection = 0;
  const bigramsB = new Set<string>();
  for (let i = 0; i < b.length - 1; i++) {
    const bg = b[i] + b[i + 1];
    bigramsB.add(bg);
    if (bigramsA.has(bg)) intersection++;
  }
  const union = bigramsA.size + bigramsB.size - intersection;
  return union > 0 ? intersection / union : 0.0;
}

// ---------------------------------------------------------------------------
// Disambiguation compact entry (from ambiguity-compact.json, paper research)
// w  = total word count of the verse
// d  = per-starting-position disambiguation length (-1 = never unique alone)
// c  = list of confuser verse refs ("surah:ayah") from the first word position
// ---------------------------------------------------------------------------
export interface DisambiguationEntry {
  w: number;
  d: number[];
  c: string[];
}

// ---------------------------------------------------------------------------
// Word-prefix index node
// Implements a trie over normalized Arabic words so we can narrow the
// candidate set to O(1) per word consumed, rather than scoring all 6 236
// verses with Levenshtein on every cycle.
// ---------------------------------------------------------------------------
interface PrefixNode {
  // verse indices (into QuranDB.verses) that have this word sequence as a prefix
  indices: number[];
  children: Map<string, PrefixNode>;
}

export class QuranDB {
  verses: QuranVerse[];
  private _byRef: Map<string, QuranVerse> = new Map();
  private _bySurah: Map<number, QuranVerse[]> = new Map();
  private _trigramIndex: Map<string, number[]> = new Map();

  // --- new disambiguation / prefix-narrowing state ---
  private _disambig: Map<string, DisambiguationEntry> = new Map();
  private _prefixRoot: PrefixNode = { indices: [], children: new Map() };

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
    this._buildPrefixIndex();
  }

  /**
   * Load the disambiguation compact map (ambiguity-compact.json).
   * Called after construction once the JSON has been fetched.
   */
  loadDisambiguationMap(map: Record<string, DisambiguationEntry>): void {
    this._disambig.clear();
    for (const [key, entry] of Object.entries(map)) {
      this._disambig.set(key, entry);
    }
  }

  /**
   * How many words from the start of this verse are needed to uniquely
   * identify it from the full corpus?  Returns -1 if the verse is never
   * uniquely identifiable in isolation (needs boundary context).
   */
  getDisambiguationLength(surah: number, ayah: number): number {
    const entry = this._disambig.get(`${surah}:${ayah}`);
    if (!entry || !entry.d.length) return -1;
    return entry.d[0];
  }

  /**
   * Is this verse one of the 339 that are never uniquely identifiable
   * from their opening words alone (bismillah, refrains, muqattaat)?
   */
  isAmbiguousInIsolation(surah: number, ayah: number): boolean {
    return this.getDisambiguationLength(surah, ayah) === -1;
  }

  /**
   * Get the full disambiguation entry for a verse (word count, disambiguation
   * lengths per starting position, and confuser verse refs).
   */
  getDisambiguationEntry(surah: number, ayah: number): DisambiguationEntry | null {
    return this._disambig.get(`${surah}:${ayah}`) ?? null;
  }

  /**
   * Count how many verses start with the given word prefix sequence.
   * Walks the prefix trie and returns the candidate count at each depth.
   * Returns an array of { word, count } for each word consumed.
   */
  prefixNarrowingCascade(
    words: string[],
  ): { word: string; count: number }[] {
    if (!words.length) return [];

    const result: { word: string; count: number }[] = [];
    let node = this._prefixRoot;

    for (const word of words) {
      // Exact match first
      let child = node.children.get(word);

      // Fuzzy fallback (same logic as narrowByPrefix)
      if (!child && node.children.size > 0 && word.length >= 3) {
        let bestSim = 0.74;
        let bestChild: PrefixNode | undefined;
        for (const [childWord, childNode] of node.children) {
          const sim = _charSimilarity(word, childWord);
          if (sim > bestSim) {
            bestSim = sim;
            bestChild = childNode;
          }
        }
        child = bestChild;
      }

      if (!child) break;
      node = child;
      result.push({ word, count: node.indices.length });
    }

    return result;
  }

  /**
   * Build a word-prefix trie over all verse texts (normalized).
   * Each node tracks the set of verse indices whose text matches the path
   * from root to that node as a word prefix.
   *
   * For bismillah-stripped ayah-1 verses we insert BOTH the full text and
   * the stripped text so the trie can match whether or not the user recites
   * the opening bismillah.
   */
  private _buildPrefixIndex(): void {
    this._prefixRoot = { indices: [], children: new Map() };

    for (let idx = 0; idx < this.verses.length; idx++) {
      const v = this.verses[idx];
      if (!v.text_words?.length) continue;

      // Insert full text prefix
      this._insertPrefixPath(v.text_words, idx);

      // Insert no-bismillah prefix as an alternative entry for the same verse
      if (v.text_norm_no_bsm) {
        const noBsmWords = v.text_norm_no_bsm.split(" ").filter(Boolean);
        if (noBsmWords.length > 0) {
          this._insertPrefixPath(noBsmWords, idx);
        }
      }
    }
  }

  private _insertPrefixPath(words: string[], verseIdx: number): void {
    let node = this._prefixRoot;
    // Root tracks all verses (universe set — not populated for performance)
    for (const word of words) {
      let child = node.children.get(word);
      if (!child) {
        child = { indices: [], children: new Map() };
        node.children.set(word, child);
      }
      // Only track indices at nodes deeper than 1 word to avoid huge sets
      // at the root level while still being useful for narrowing
      child.indices.push(verseIdx);
      node = child;
    }
  }

  /**
   * Given a sequence of recognized words (from the CTC transcript),
   * walk the prefix trie and return the narrowed candidate set.
   *
   * Returns null if not enough words to narrow, otherwise returns the
   * set of verse indices whose text starts with the given word sequence.
   *
   * Fuzzy matching: if an exact word is not found as a child, try the
   * best Levenshtein-similar child word (similarity >= 0.75).  This
   * handles the CTC producing slight variations like alif variants.
   */
  narrowByPrefix(
    words: string[],
    maxCandidates = 50,
  ): number[] | null {
    if (!words.length) return null;

    let node = this._prefixRoot;
    let depth = 0;

    for (const word of words) {
      // Exact match first
      let child = node.children.get(word);

      // Fuzzy fallback: try Levenshtein-similar child words
      if (!child && node.children.size > 0 && word.length >= 3) {
        let bestSim = 0.74;
        let bestChild: PrefixNode | undefined;
        for (const [childWord, childNode] of node.children) {
          const sim = _charSimilarity(word, childWord);
          if (sim > bestSim) {
            bestSim = sim;
            bestChild = childNode;
          }
        }
        child = bestChild;
      }

      if (!child) break; // No match at this depth — stop narrowing
      node = child;
      depth++;
    }

    if (depth === 0) return null;

    const candidates = node.indices;
    if (candidates.length === 0 || candidates.length > maxCandidates) return null;
    return candidates;
  }

  /**
   * Score a pre-narrowed candidate set against a transcript.
   *
   * This is the second phase of prefix-narrowing: after the trie has reduced
   * the search space to a small set of verses, pick the best match using the
   * same scoring functions as matchVerse but only over those candidates.
   *
   * Returns null if no candidate scores above the threshold.
   */
  matchVerseFromCandidates(
    text: string,
    candidateIndices: number[],
    threshold = 0.35,
    hint: [number, number] | null = null,
  ): Record<string, any> | null {
    if (!text.trim() || !candidateIndices.length) return null;

    const normText = normalizeArabic(text);
    const noSpaceText = normText.replace(/ /g, "");
    if (noSpaceText.length < 2) return null;

    const bonuses = hint ? this._continuationBonuses(hint) : new Map<string, number>();
    const textWords = normText.split(" ");

    const scored: [QuranVerse, number, number, number][] = [];
    for (const idx of candidateIndices) {
      if (idx < 0 || idx >= this.verses.length) continue;
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

      // For short verses / short transcripts: also do a direct word-prefix match.
      // If the first N recognized words all match the verse's first N words, boost.
      if (v.text_words && textWords.length <= v.text_words.length) {
        let prefixMatch = 0;
        for (let i = 0; i < textWords.length; i++) {
          if (i >= v.text_words.length) break;
          const sim = _charSimilarity(textWords[i], v.text_words[i]);
          if (sim >= 0.75) prefixMatch++;
          else break;
        }
        if (prefixMatch >= textWords.length * 0.8 && prefixMatch >= 1) {
          // Prefix matched well — boost based on coverage
          const boostFactor = 0.1 * (prefixMatch / Math.max(textWords.length, 1));
          raw = Math.min(raw + boostFactor, 1.0);
        }
      }

      // Sellers' word-level matching: for mid-ayah recognition
      if (v.text_words && v.text_words.length >= 5 && textWords.length < v.text_words.length * 0.8) {
        const sellersScore = QuranDB._sellersScore(normText, v.text_words);
        raw = Math.max(raw, sellersScore * 0.95);
        if (v.text_norm_no_bsm) {
          const noBsmWords = v.text_norm_no_bsm.split(" ");
          if (noBsmWords.length >= 3) {
            raw = Math.max(raw, QuranDB._sellersScore(normText, noBsmWords) * 0.95);
          }
        }
      }

      const bonus = bonuses.get(`${v.surah}:${v.ayah}`) ?? 0.0;
      scored.push([v, raw, bonus, Math.min(raw + bonus, 1.0)]);
    }

    if (!scored.length) return null;
    scored.sort((a, b) => b[3] - a[3]);

    const [bestV, bestRaw, bestBonus, bestScore] = scored[0];
    if (bestScore < threshold) return null;

    const runnersUp = scored.slice(0, 5).map(([v, raw, bon, total]) => ({
      surah: v.surah,
      ayah: v.ayah,
      raw_score: Math.round(raw * 1000) / 1000,
      bonus: Math.round(bon * 1000) / 1000,
      score: Math.round(total * 1000) / 1000,
      text_norm: (v.text_norm ?? "").slice(0, 60),
      surah_name: v.surah_name,
      surah_name_en: v.surah_name_en,
      text_uthmani: v.text_uthmani.slice(0, 80),
    }));

    return {
      ...bestV,
      score: bestScore,
      raw_score: bestRaw,
      bonus: bestBonus,
      runners_up: runnersUp,
      _via_prefix_narrowing: true,
    };
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
    const effectiveMax = noSpace.length < 15 ? Math.max(maxCandidates, 400) : maxCandidates;
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
    for (let i = 0; i < Math.min(sorted.length, effectiveMax); i++) {
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
   * Enhanced with Sellers' character-level semi-global alignment for fragments.
   */
  private static _smartScore(
    textNoSpace: string,
    verseNs: string,
  ): number {
    const lengthRatio = textNoSpace.length / verseNs.length;

    let baseScore: number;
    if (lengthRatio >= 0.7 && lengthRatio <= 1.3) {
      baseScore = ratio(textNoSpace, verseNs);
    } else if (lengthRatio < 0.7) {
      // Sellers' semi-global alignment via fragmentScore
      const frag = fragmentScore(textNoSpace, verseNs);
      const r = ratio(textNoSpace, verseNs);
      baseScore = Math.max(frag, 0.6 * frag + 0.4 * r);
    } else {
      baseScore = ratio(textNoSpace, verseNs);
    }

    const phonetic = phoneticRatio(textNoSpace, verseNs);
    const isShortText = textNoSpace.length < 20;
    const phoneticWeight = isShortText ? 0.4 : 0.2;
    if (phonetic > baseScore) {
      return (1 - phoneticWeight) * baseScore + phoneticWeight * phonetic;
    }
    return baseScore;
  }

  /**
   * Sellers' word-level approximate substring matching.
   * Finds the best contiguous word sequence in the verse that matches the transcript.
   * Only applied when transcript is significantly shorter than verse (mid-ayah scenario).
   */
  private static _sellersScore(
    normText: string,
    verseWords: string[],
  ): number {
    const textWords = normText.split(" ");
    if (textWords.length < 2 || verseWords.length < 3) return 0;

    const { score } = sellersWordMatch(textWords, verseWords);
    return score;
  }

  /**
   * Sliding word-window score: slides transcript across verse words,
   * comparing transcript against each window of the verse.
   * Catches mid-ayah starts where user begins reading from the middle.
   */
  private static _slidingWordWindowScore(
    normText: string,
    verseWords: string[],
  ): number {
    if (verseWords.length < 3) return 0;
    const textWords = normText.split(" ");
    if (textWords.length < 2) return 0;

    const windowSize = textWords.length;
    let bestScore = 0;

    // Slide transcript-sized word window across verse words
    for (let start = 0; start <= verseWords.length - Math.min(windowSize, verseWords.length); start++) {
      const end = Math.min(start + windowSize + 1, verseWords.length); // +1 to allow slight overrun
      const window = verseWords.slice(start, end).join(" ");
      const score = ratio(normText, window);
      if (score > bestScore) bestScore = score;
    }

    // Also try no-space comparison for spaceless model output
    const noSpaceText = normText.replace(/ /g, "");
    const noSpaceVerse = verseWords.join("");
    if (noSpaceText.length >= 5 && noSpaceText.length < noSpaceVerse.length * 0.85) {
      const frag = fragmentScore(noSpaceText, noSpaceVerse);
      bestScore = Math.max(bestScore, frag);
    }

    return bestScore;
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
    maxSpan = 6,
    hint: [number, number] | null = null,
    returnTopK = 0,
    surahContext: number | null = null,
  ): Record<string, any> | null {
    if (!text.trim()) return null;

    // Normalize the input transcript
    const normText = normalizeArabic(text);
    const noSpaceText = normText.replace(/ /g, "");

    // Muqattaat (disconnected letters) — very short verses that need special handling
    const MUQATTAAT_VERSES: Map<string, [number, number]> = new Map([
      ["الم", [2, 1]],      // Al-Baqarah, Aal-Imran, Al-Ankabut, Ar-Rum, Luqman, As-Sajdah
      ["المص", [7, 1]],     // Al-A'raf
      ["الر", [10, 1]],     // Yunus, Hud, Yusuf, Ibrahim, Al-Hijr
      ["المر", [13, 1]],    // Ar-Ra'd
      ["كهيعص", [19, 1]],   // Maryam
      ["طه", [20, 1]],      // Ta-Ha
      ["طسم", [26, 1]],     // Ash-Shu'ara, Al-Qasas
      ["طس", [27, 1]],      // An-Naml
      ["يس", [36, 1]],      // Ya-Sin
      ["ص", [38, 1]],       // Sad
      ["حم", [40, 1]],      // Ghafir, Fussilat, Az-Zukhruf, Ad-Dukhan, Al-Jathiyah, Al-Ahqaf
      ["حمعسق", [42, 1]],   // Ash-Shura (42:1-2)
      ["ق", [50, 1]],       // Qaf
      ["ن", [68, 1]],       // Al-Qalam
    ]);

    if (noSpaceText && noSpaceText.length >= 1 && noSpaceText.length <= 6) {
      const muq = MUQATTAAT_VERSES.get(noSpaceText);
      if (muq) {
        const v = this.getVerse(muq[0], muq[1]);
        if (v) {
          return {
            ...v,
            score: 1.0,
            raw_score: 1.0,
            bonus: 0,
          };
        }
      }
    }

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
      // Sellers' word-level matching: for mid-ayah recognition
      // Apply when transcript is significantly shorter than verse
      const textWords = normText.split(" ");
      if (v.text_words && v.text_words.length >= 5 && textWords.length < v.text_words.length * 0.8) {
        const sellersScore = QuranDB._sellersScore(normText, v.text_words);
        raw = Math.max(raw, sellersScore * 0.95); // slight discount to avoid false positives

        // Also try against no-bismillah version
        if (v.text_norm_no_bsm) {
          const noBsmWords = v.text_norm_no_bsm.split(" ");
          if (noBsmWords.length >= 3) {
            const sellersNoBsm = QuranDB._sellersScore(normText, noBsmWords);
            raw = Math.max(raw, sellersNoBsm * 0.95);
          }
        }
      }
      // Mid-ayah sliding window: catches partial/mid-verse transcripts
      if (v.text_words && v.text_words.length >= 5 && noSpaceText.length < v.text_norm_ns!.length * 0.8) {
        const swScore = QuranDB._slidingWordWindowScore(normText, v.text_words);
        raw = Math.max(raw, swScore * 0.92); // slight discount vs full-verse match
      }
      // First-word boost for short transcripts
      if (textWords.length <= 4 && v.text_words && v.text_words.length > 0) {
        const effectiveFirstWord = (v.ayah === 1 && v.surah !== 1 && v.surah !== 9 && v.text_norm_no_bsm)
          ? v.text_norm_no_bsm.split(" ")[0]
          : v.text_words[0];
        if (effectiveFirstWord) {
          const firstWordSim = ratio(textWords[0], effectiveFirstWord);
          if (firstWordSim > 0.7) {
            raw = Math.max(raw, raw + 0.05);
          }
        }
      }
      let bonus = bonuses.get(`${v.surah}:${v.ayah}`) ?? 0.0;
      if (bonus > 0) {
        const sp = QuranDB._suffixPrefixScore(normText, v.text_norm!);
        raw = Math.max(raw, sp);
      }
      if (surahContext !== null && v.surah === surahContext && bonus === 0) {
        bonus = 0.06;
      }
      scored.push([v, raw, bonus, Math.min(raw + bonus, 1.0)]);
    }
    scored.sort((a, b) => {
      const diff = b[3] - a[3];
      if (Math.abs(diff) < 0.001) {
        // Tiebreaker: prefer verse closest in length to transcript
        const lenA = a[0].text_norm_ns!.length;
        const lenB = b[0].text_norm_ns!.length;
        const lenDiff = Math.abs(lenA - noSpaceText.length) - Math.abs(lenB - noSpaceText.length);
        if (lenDiff !== 0) return lenDiff;
        // Secondary tiebreaker: prefer earlier surah:ayah
        if (a[0].surah !== b[0].surah) return a[0].surah - b[0].surah;
        return a[0].ayah - b[0].ayah;
      }
      return diff;
    });

    // Top-20 surahs for Pass 2 multi-ayah spans
    const pass2Surahs = new Set<number>();
    for (let idx = 0; idx < Math.min(scored.length, 30); idx++) {
      pass2Surahs.add(scored[idx][0].surah);
    }
    // Also include all surahs from trigram candidate set
    for (const idx of candidates) {
      pass2Surahs.add(this.verses[idx].surah);
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
        surah_name: v.surah_name,
        surah_name_en: v.surah_name_en,
        text_uthmani: v.text_uthmani.slice(0, 80),
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
          const combinedNs = combined.replace(/ /g, "");
          let raw = ratio(normText, combined);
          raw = Math.max(raw, ratio(noSpaceText, combinedNs));
          if (noSpaceText.length < combinedNs.length * 0.85) {
            const frag = fragmentScore(noSpaceText, combinedNs);
            raw = Math.max(raw, frag * 0.95);
          }
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

    // Two-pass surah identification: if not confident yet, try identifying
    // the surah first then searching within it
    if (bestScore < 0.95) {
      const textWords = normText.split(" ");
      const twoPassResult = this._twoPassMatch(normText, noSpaceText, textWords, maxSpan, bonuses, surahContext);
      if (twoPassResult && twoPassResult.score > bestScore + 0.03) {
        bestScore = twoPassResult.score;
        best = twoPassResult;
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
   * Two-pass surah identification method.
   * Pass 1: scores each surah by averaging top-3 verse scores to identify most likely surah.
   * Pass 2: searches only within the best surah for more thorough matching.
   */
  private _twoPassMatch(
    normText: string,
    noSpaceText: string,
    textWords: string[],
    maxSpan: number,
    bonuses: Map<string, number>,
    surahContext: number | null,
  ): Record<string, any> | null {
    // Pass 1: score each surah by its top-3 verse scores
    const surahScores = new Map<number, number[]>();
    const candidates = this._getCandidates(normText, 200);
    for (const idx of candidates) {
      const v = this.verses[idx];
      let raw = QuranDB._smartScore(noSpaceText, v.text_norm_ns!);
      if (v.text_norm_no_bsm_ns) {
        raw = Math.max(raw, QuranDB._smartScore(noSpaceText, v.text_norm_no_bsm_ns));
      }
      const arr = surahScores.get(v.surah) ?? [];
      arr.push(raw);
      surahScores.set(v.surah, arr);
    }

    // Average top-3 scores per surah
    const surahRanking: [number, number][] = [];
    for (const [surah, scores] of surahScores) {
      scores.sort((a, b) => b - a);
      const top3 = scores.slice(0, 3);
      const avg = top3.reduce((s, v) => s + v, 0) / top3.length;
      let adjusted = avg;
      if (surahContext !== null && surah === surahContext) {
        adjusted += 0.06;
      }
      surahRanking.push([surah, adjusted]);
    }
    surahRanking.sort((a, b) => b[1] - a[1]);

    if (surahRanking.length === 0) return null;

    // Pass 2: search within top surah
    const bestSurah = surahRanking[0][0];
    const surahVerses = this._bySurah.get(bestSurah);
    if (!surahVerses) return null;

    let bestScore = 0;
    let best: Record<string, any> | null = null;

    // Score individual verses in the surah
    for (const v of surahVerses) {
      let raw = QuranDB._smartScore(noSpaceText, v.text_norm_ns!);
      if (v.text_norm_no_bsm_ns) {
        raw = Math.max(raw, QuranDB._smartScore(noSpaceText, v.text_norm_no_bsm_ns));
      }
      const spacedRatio = ratio(normText, v.text_norm!);
      raw = Math.max(raw, spacedRatio);
      if (v.text_norm_no_bsm) {
        raw = Math.max(raw, ratio(normText, v.text_norm_no_bsm));
      }
      // First-word boost for short transcripts
      if (textWords.length <= 4 && v.text_words && v.text_words.length > 0) {
        const effectiveFirstWord = (v.ayah === 1 && v.surah !== 1 && v.surah !== 9 && v.text_norm_no_bsm)
          ? v.text_norm_no_bsm.split(" ")[0]
          : v.text_words[0];
        if (effectiveFirstWord) {
          const firstWordSim = ratio(textWords[0], effectiveFirstWord);
          if (firstWordSim > 0.7) {
            raw = Math.max(raw, raw + 0.05);
          }
        }
      }
      const bonus = bonuses.get(`${v.surah}:${v.ayah}`) ?? 0.0;
      const score = Math.min(raw + bonus, 1.0);
      if (score > bestScore) {
        bestScore = score;
        best = {
          ...v,
          score,
          raw_score: raw,
          bonus,
        };
      }
    }

    // Also try multi-ayah spans within the surah
    for (let i = 0; i < surahVerses.length; i++) {
      for (let span = 2; span <= maxSpan; span++) {
        if (i + span > surahVerses.length) break;
        const chunk = surahVerses.slice(i, i + span);
        const firstText = chunk[0].text_norm_no_bsm ?? chunk[0].text_norm!;
        const combined = [firstText]
          .concat(chunk.slice(1).map((c) => c.text_norm!))
          .join(" ");
        const combinedNs = combined.replace(/ /g, "");
        let raw = ratio(normText, combined);
        raw = Math.max(raw, ratio(noSpaceText, combinedNs));
        if (noSpaceText.length < combinedNs.length * 0.85) {
          const frag = fragmentScore(noSpaceText, combinedNs);
          raw = Math.max(raw, frag * 0.95);
        }
        const bonus = bonuses.get(`${chunk[0].surah}:${chunk[0].ayah}`) ?? 0.0;
        const score = Math.min(raw + bonus, 1.0);
        if (score > bestScore) {
          bestScore = score;
          best = {
            surah: bestSurah,
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

    return best;
  }

  /**
   * Alias for matchVerse — used when caller wants to pass surahContext.
   */
  matchVerseWithSurahId(
    text: string,
    threshold = 0.3,
    maxSpan = 6,
    hint: [number, number] | null = null,
    returnTopK = 0,
    surahContext: number | null = null,
  ): Record<string, any> | null {
    return this.matchVerse(text, threshold, maxSpan, hint, returnTopK, surahContext);
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
      // Sellers' word-level matching: for mid-ayah recognition
      // Apply when transcript is significantly shorter than verse
      const textWords = normText.split(" ");
      if (v.text_words && v.text_words.length >= 5 && textWords.length < v.text_words.length * 0.8) {
        const sellersScore = QuranDB._sellersScore(normText, v.text_words);
        raw = Math.max(raw, sellersScore * 0.95); // slight discount to avoid false positives

        // Also try against no-bismillah version
        if (v.text_norm_no_bsm) {
          const noBsmWords = v.text_norm_no_bsm.split(" ");
          if (noBsmWords.length >= 3) {
            const sellersNoBsm = QuranDB._sellersScore(normText, noBsmWords);
            raw = Math.max(raw, sellersNoBsm * 0.95);
          }
        }
      }
      // Mid-ayah sliding window: catches partial/mid-verse transcripts
      if (v.text_words && v.text_words.length >= 5 && noSpaceText.length < v.text_norm_ns!.length * 0.8) {
        const swScore = QuranDB._slidingWordWindowScore(normText, v.text_words);
        raw = Math.max(raw, swScore * 0.92);
      }
      // First-word boost for short transcripts
      if (textWords.length <= 4 && v.text_words && v.text_words.length > 0) {
        const effectiveFirstWord = (v.ayah === 1 && v.surah !== 1 && v.surah !== 9 && v.text_norm_no_bsm)
          ? v.text_norm_no_bsm.split(" ")[0]
          : v.text_words[0];
        if (effectiveFirstWord) {
          const firstWordSim = ratio(textWords[0], effectiveFirstWord);
          if (firstWordSim > 0.7) {
            raw = Math.max(raw, raw + 0.05);
          }
        }
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
      const result: Record<string, any> = {
        ...bestV,
        score: bestScore,
        raw_score: bestRaw,
        bonus: bestBonus,
      };
      result.runners_up = scored.slice(0, 10).map(([v, raw, bon, total]) => ({
        surah: v.surah,
        ayah: v.ayah,
        raw_score: Math.round(raw * 1000) / 1000,
        bonus: Math.round(bon * 1000) / 1000,
        score: Math.round(total * 1000) / 1000,
        text_norm: (v.text_norm ?? "").slice(0, 60),
        surah_name: v.surah_name,
        surah_name_en: v.surah_name_en,
        text_uthmani: v.text_uthmani.slice(0, 80),
      }));
      return result;
    }

    return this.matchVerse(text, threshold, 3, hint, 10);
  }
}
