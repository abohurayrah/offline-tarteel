/**
 * Phoneme-level alignment using edit distance.
 *
 * Port of shared/phoneme_aligner.py to TypeScript.
 * Aligns predicted phoneme sequences against reference sequences and
 * classifies each position as correct, substitution, deletion, or insertion.
 */

export interface AlignmentError {
  type: "substitution" | "deletion" | "insertion";
  position: number;
  expected: string | null;
  got: string | null;
}

export interface AlignmentResult {
  errors: AlignmentError[];
  per: number;
  correctRate: number;
  alignment: [string | null, string | null][];
}

export function alignPhonemes(
  predicted: string[],
  reference: string[],
): AlignmentResult {
  const n = reference.length;
  const m = predicted.length;

  if (n === 0 && m === 0) {
    return { errors: [], per: 0, correctRate: 1, alignment: [] };
  }
  if (n === 0) {
    const errors: AlignmentError[] = predicted.map((p) => ({
      type: "insertion" as const,
      position: 0,
      expected: null,
      got: p,
    }));
    return {
      errors,
      per: m,
      correctRate: 0,
      alignment: predicted.map((p) => [null, p]),
    };
  }
  if (m === 0) {
    const errors: AlignmentError[] = reference.map((r, i) => ({
      type: "deletion" as const,
      position: i,
      expected: r,
      got: null,
    }));
    return {
      errors,
      per: 1,
      correctRate: 0,
      alignment: reference.map((r) => [r, null]),
    };
  }

  // DP matrix
  const dp: number[][] = [];
  const bt: string[][] = [];
  for (let i = 0; i <= n; i++) {
    dp.push(new Array(m + 1).fill(0));
    bt.push(new Array(m + 1).fill(""));
  }

  for (let i = 1; i <= n; i++) {
    dp[i][0] = i;
    bt[i][0] = "D";
  }
  for (let j = 1; j <= m; j++) {
    dp[0][j] = j;
    bt[0][j] = "I";
  }

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const cost = reference[i - 1] === predicted[j - 1] ? 0 : 1;
      const sub = dp[i - 1][j - 1] + cost;
      const ins = dp[i][j - 1] + 1;
      const del = dp[i - 1][j] + 1;
      const best = Math.min(sub, ins, del);
      dp[i][j] = best;
      if (best === sub) bt[i][j] = "S";
      else if (best === del) bt[i][j] = "D";
      else bt[i][j] = "I";
    }
  }

  // Backtrace
  const alignment: [string | null, string | null][] = [];
  let i = n;
  let j = m;
  while (i > 0 || j > 0) {
    if (i === 0) {
      alignment.push([null, predicted[j - 1]]);
      j--;
    } else if (j === 0) {
      alignment.push([reference[i - 1], null]);
      i--;
    } else {
      const move = bt[i][j];
      if (move === "S") {
        alignment.push([reference[i - 1], predicted[j - 1]]);
        i--;
        j--;
      } else if (move === "D") {
        alignment.push([reference[i - 1], null]);
        i--;
      } else {
        alignment.push([null, predicted[j - 1]]);
        j--;
      }
    }
  }
  alignment.reverse();

  // Classify
  const errors: AlignmentError[] = [];
  let correct = 0;
  let refPos = 0;

  for (const [refTok, predTok] of alignment) {
    if (refTok !== null && predTok !== null) {
      if (refTok === predTok) {
        correct++;
      } else {
        errors.push({
          type: "substitution",
          position: refPos,
          expected: refTok,
          got: predTok,
        });
      }
      refPos++;
    } else if (refTok !== null) {
      errors.push({
        type: "deletion",
        position: refPos,
        expected: refTok,
        got: null,
      });
      refPos++;
    } else {
      errors.push({
        type: "insertion",
        position: refPos,
        expected: null,
        got: predTok,
      });
    }
  }

  return {
    errors,
    per: errors.length / n,
    correctRate: correct / n,
    alignment,
  };
}
