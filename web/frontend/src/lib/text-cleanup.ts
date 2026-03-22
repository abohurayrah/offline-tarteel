/**
 * Clean up common Whisper ASR artifacts:
 * - Repetition loops (model gets stuck repeating tokens)
 * - Leading/trailing whitespace
 * - Consecutive duplicate words/phrases
 */
export function cleanWhisperOutput(text: string): string {
  text = text.trim();
  if (!text) return text;

  // Detect and fix repetition loops: if a word or short phrase repeats 3+ times
  // consecutively, collapse to a single occurrence.
  // e.g., "لله لله لله لله الرحمن" → "لله الرحمن"
  const words = text.split(/\s+/);
  if (words.length <= 2) return words.join(" ");

  const cleaned: string[] = [];
  let i = 0;
  while (i < words.length) {
    // Check for repeating sequences of length 1-3 words
    let foundRepeat = false;
    for (let seqLen = 1; seqLen <= 3 && seqLen <= words.length - i; seqLen++) {
      const seq = words.slice(i, i + seqLen).join(" ");
      let repeatCount = 1;
      let j = i + seqLen;
      while (j + seqLen <= words.length) {
        const next = words.slice(j, j + seqLen).join(" ");
        if (next === seq) {
          repeatCount++;
          j += seqLen;
        } else {
          break;
        }
      }
      if (repeatCount >= 3) {
        // Collapse: keep one occurrence of the repeated sequence
        cleaned.push(...words.slice(i, i + seqLen));
        i = j;
        foundRepeat = true;
        break;
      }
    }
    if (!foundRepeat) {
      cleaned.push(words[i]);
      i++;
    }
  }

  return cleaned.join(" ");
}
