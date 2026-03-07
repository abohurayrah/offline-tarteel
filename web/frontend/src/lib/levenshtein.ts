/**
 * Levenshtein edit distance between two strings.
 * Uses a single-row DP approach for O(min(m,n)) space.
 */
export function distance(a: string, b: string): number {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;

  // Ensure a is the shorter string for space efficiency
  if (a.length > b.length) [a, b] = [b, a];

  const m = a.length;
  const n = b.length;
  let prev = new Uint16Array(m + 1);
  let curr = new Uint16Array(m + 1);

  for (let i = 0; i <= m; i++) prev[i] = i;

  for (let j = 1; j <= n; j++) {
    curr[0] = j;
    for (let i = 1; i <= m; i++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[i] = Math.min(
        prev[i] + 1, // deletion
        curr[i - 1] + 1, // insertion
        prev[i - 1] + cost, // substitution
      );
    }
    [prev, curr] = [curr, prev];
  }

  return prev[m];
}

/**
 * Normalized Levenshtein similarity ratio.
 * Returns 1.0 for identical strings, 0.0 for completely different.
 * Matches python-Levenshtein's `ratio()` behavior:
 *   ratio = (len(a) + len(b) - distance) / (len(a) + len(b))
 */
export function ratio(a: string, b: string): number {
  const lenSum = a.length + b.length;
  if (lenSum === 0) return 1.0;
  return (lenSum - distance(a, b)) / lenSum;
}

/**
 * Semi-global edit distance: finds the minimum edit distance to align
 * the entire query against any substring of ref.
 * Free gaps at start and end of ref (row 0 initialized to 0, take min of last row).
 * Use case: "how well does this transcript fragment match somewhere inside this verse?"
 */
export function semiGlobalDistance(query: string, ref: string): number {
  if (query.length === 0) return 0;
  if (ref.length === 0) return query.length;
  const m = query.length;
  const n = ref.length;
  let prev = new Uint16Array(m + 1);
  let curr = new Uint16Array(m + 1);
  for (let i = 0; i <= m; i++) prev[i] = i;
  let best = prev[m];
  for (let j = 1; j <= n; j++) {
    curr[0] = 0; // Free to start anywhere in ref
    for (let i = 1; i <= m; i++) {
      const cost = query[i - 1] === ref[j - 1] ? 0 : 1;
      curr[i] = Math.min(prev[i] + 1, curr[i - 1] + 1, prev[i - 1] + cost);
    }
    best = Math.min(best, curr[m]); // Free to end anywhere in ref
    [prev, curr] = [curr, prev];
  }
  return best;
}

/**
 * Fragment score: how well does the query match as a fragment of ref?
 * Returns 0.0-1.0. Score of 1.0 means query is an exact substring of ref.
 * Directional: measures "how much of the query does the ref explain?"
 */
export function fragmentScore(query: string, ref: string): number {
  if (query.length === 0) return 1.0;
  return Math.max(0, 1 - semiGlobalDistance(query, ref) / query.length);
}

/**
 * Word similarity score using Levenshtein ratio.
 * Returns 1.0 for identical words, 0.0 for completely different.
 */
function wordSimilarity(w1: string, w2: string): number {
  if (w1 === w2) return 1.0;
  if (w1.length === 0 || w2.length === 0) return 0.0;
  return ratio(w1, w2);
}

/**
 * Sellers' word-level approximate substring matching.
 * Finds the best contiguous subsequence of verseWords that matches transcriptWords.
 * Returns { score: 0-1, startIdx, endIdx } where startIdx/endIdx are word indices.
 *
 * Algorithm: For each possible starting position in verse, extract a window
 * of verse words matching the transcript length, compute similarity, and
 * return the best-scoring window.
 */
export function sellersWordMatch(
  transcriptWords: string[],
  verseWords: string[],
  wordSimilarityThreshold = 0.7
): { score: number; startIdx: number; endIdx: number } {
  if (transcriptWords.length === 0 || verseWords.length === 0) {
    return { score: 0, startIdx: 0, endIdx: 0 };
  }

  const tLen = transcriptWords.length;
  let bestScore = 0;
  let bestStart = 0;
  let bestEnd = 0;

  // Try all possible starting positions in verse
  for (let start = 0; start <= verseWords.length - tLen; start++) {
    const end = start + tLen;
    const window = verseWords.slice(start, end);

    // Score this window: count words that match with similarity >= threshold
    let matches = 0;
    let totalSim = 0;
    for (let i = 0; i < tLen; i++) {
      const sim = wordSimilarity(transcriptWords[i], window[i]);
      totalSim += sim;
      if (sim >= wordSimilarityThreshold) {
        matches++;
      }
    }

    // Score combines match ratio and average similarity
    const matchRatio = matches / tLen;
    const avgSim = totalSim / tLen;
    const score = 0.6 * matchRatio + 0.4 * avgSim;

    if (score > bestScore) {
      bestScore = score;
      bestStart = start;
      bestEnd = end;
    }
  }

  // Also try windows slightly longer than transcript (±1 word) for flexibility
  for (let start = 0; start < verseWords.length; start++) {
    for (let windowSize of [tLen - 1, tLen + 1]) {
      if (windowSize < 1 || start + windowSize > verseWords.length) continue;

      const end = start + windowSize;
      const window = verseWords.slice(start, end);

      // Use string-based comparison for variable-length windows
      const windowText = window.join(" ");
      const transcriptText = transcriptWords.join(" ");
      const score = ratio(transcriptText, windowText);

      if (score > bestScore) {
        bestScore = score;
        bestStart = start;
        bestEnd = end;
      }
    }
  }

  return { score: bestScore, startIdx: bestStart, endIdx: bestEnd };
}
