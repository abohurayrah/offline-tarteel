import { ratio as levRatio } from "./levenshtein";
import { QuranDB, partialRatio, normalizeArabic } from "./quran-db";
import type { QuranVerse, WorkerOutbound, SurroundingVerse, CandidateVerse, VerseMatch, VerseMatchCandidate } from "./types";
import {
  SAMPLE_RATE,
  TRIGGER_SAMPLES,
  FIRST_TRIGGER_SAMPLES,
  MAX_WINDOW_SAMPLES,
  SILENCE_RMS_THRESHOLD,
  VERSE_MATCH_THRESHOLD,
  FIRST_MATCH_THRESHOLD,
  RAW_TRANSCRIPT_THRESHOLD,
  SURROUNDING_CONTEXT,
  TRACKING_TRIGGER_SAMPLES,
  TRACKING_SILENCE_SAMPLES,
  TRACKING_MAX_WINDOW_SAMPLES,
  STALE_CYCLE_LIMIT,
  LOOKAHEAD,
  MIN_DISCOVERY_WORDS,
  PREFIX_NARROW_THRESHOLD,
  PREFIX_NARROW_MAX_CANDIDATES,
} from "./types";

export interface TranscribeResult {
  text: string;
  rawTokens: string;
}

type TranscribeFn = (audio: Float32Array, prompt?: string) => Promise<TranscribeResult>;

function concatFloat32(a: Float32Array, b: Float32Array): Float32Array {
  const result = new Float32Array(a.length + b.length);
  result.set(a);
  result.set(b, a.length);
  return result;
}

function isSilence(audio: Float32Array): boolean {
  let sumSq = 0;
  for (let i = 0; i < audio.length; i++) {
    sumSq += audio[i] * audio[i];
  }
  const rms = Math.sqrt(sumSq / audio.length);
  return rms < SILENCE_RMS_THRESHOLD;
}

function wordsMatch(w1: string, w2: string, threshold = 0.7): boolean {
  if (w1 === w2) return true;
  if (w1.length <= 2 || w2.length <= 2) return w1 === w2;
  return levRatio(w1, w2) >= threshold;
}

function alignPosition(
  recognizedWords: string[],
  verseWords: string[],
  startFrom = 0,
): { position: number; matchedIndices: number[] } {
  if (!recognizedWords.length || !verseWords.length) {
    return { position: 0, matchedIndices: [] };
  }

  const matchedIndices: number[] = [];
  let versePtr = startFrom;

  for (const rec of recognizedWords) {
    if (versePtr >= verseWords.length) break;
    const limit = Math.min(versePtr + LOOKAHEAD, verseWords.length);
    for (let j = versePtr; j < limit; j++) {
      if (wordsMatch(rec, verseWords[j])) {
        matchedIndices.push(j);
        versePtr = j + 1;
        break;
      }
    }
  }

  if (matchedIndices.length) {
    return {
      position: matchedIndices[matchedIndices.length - 1] + 1,
      matchedIndices,
    };
  }
  return { position: startFrom, matchedIndices: [] };
}

function getSurroundingVerses(
  db: QuranDB,
  surah: number,
  ayah: number,
): SurroundingVerse[] {
  const verses = db.getSurah(surah);
  const result: SurroundingVerse[] = [];
  for (const v of verses) {
    if (Math.abs(v.ayah - ayah) <= SURROUNDING_CONTEXT) {
      result.push({
        surah: v.surah,
        ayah: v.ayah,
        text: v.text_uthmani,
        is_current: v.ayah === ayah,
      });
    }
  }
  return result;
}

export class RecitationTracker {
  private fullAudio: Float32Array = new Float32Array(0);
  private newAudioCount = 0;
  private lastEmittedRef: [number, number] | null = null;
  private lastEmittedText = "";
  private prevEmittedRef: [number, number] | null = null;
  private prevEmittedText = "";
  private hasEverMatched = false;
  private cyclesSinceEmit = Infinity; // anti-cascade: counts discovery cycles since last emit

  // Tracking mode state
  private trackingVerse: QuranVerse | null = null;
  private trackingVerseWords: string[] = [];
  private trackingLastWordIdx = -1;
  private silenceSamples = 0;
  private staleCycles = 0;

  // Verse transition anti-bounce state
  private transitionCooldown = 0;       // cycles to suppress backward jumps after verse advance
  private transitionGraceCycles = 0;    // grace cycles after entering tracking (skip stale audio)
  private lastConfirmedAyah = -1;       // highest ayah number confirmed in current surah
  private lastConfirmedSurah = -1;

  // Transcript accumulation across cycles (for long verses)
  private accumulatedText = "";
  private accumulatedCycles = 0;

  // Long-verse mode state
  private _longVerseMode = false;
  private _longVerseModeCycles = 0;

  // Ambiguity guard deferral tracking
  private lastDeferredRef: string | null = null;
  private consecutiveDeferrals = 0;

  // Session surah context: persists across tracking resets to prevent surah-level misidentification
  private sessionSurah: number | null = null;

  private db: QuranDB;
  private transcribe: TranscribeFn;

  constructor(db: QuranDB, transcribe: TranscribeFn) {
    this.db = db;
    this.transcribe = transcribe;
  }

  async feed(samples: Float32Array): Promise<WorkerOutbound[]> {
    const messages: WorkerOutbound[] = [];

    // Append audio
    this.fullAudio = concatFloat32(this.fullAudio, samples);
    this.newAudioCount += samples.length;

    // Trim to max window
    const maxSamples =
      this.trackingVerse !== null
        ? TRACKING_MAX_WINDOW_SAMPLES
        : this._longVerseMode
          ? SAMPLE_RATE * 18  // 18 seconds for long verses
          : MAX_WINDOW_SAMPLES;
    if (this.fullAudio.length > maxSamples) {
      this.fullAudio = this.fullAudio.slice(-maxSamples);
    }

    // TRACKING MODE
    if (this.trackingVerse !== null) {
      const trackMsgs = await this._handleTracking(samples);
      messages.push(...trackMsgs);
      return messages;
    }

    // DISCOVERY MODE
    const discMsgs = await this._handleDiscovery();
    messages.push(...discMsgs);
    return messages;
  }

  private async _handleTracking(
    samples: Float32Array,
  ): Promise<WorkerOutbound[]> {
    const messages: WorkerOutbound[] = [];

    // Check silence accumulation
    let sumSq = 0;
    for (let i = 0; i < samples.length; i++) {
      sumSq += samples[i] * samples[i];
    }
    const chunkRms = Math.sqrt(sumSq / samples.length);

    if (chunkRms < SILENCE_RMS_THRESHOLD) {
      this.silenceSamples += samples.length;
      if (this.silenceSamples >= TRACKING_SILENCE_SAMPLES) {
        this._exitTracking("extended silence");
        this.newAudioCount = 0;
        return messages;
      }
    } else {
      this.silenceSamples = 0;
    }

    // Faster trigger in tracking mode
    if (this.newAudioCount < TRACKING_TRIGGER_SAMPLES) {
      return messages;
    }
    this.newAudioCount = 0;

    // Decrement transition cooldown
    if (this.transitionCooldown > 0) {
      this.transitionCooldown--;
    }

    // Grace period after verse transition — skip inference to let stale audio flush
    if (this.transitionGraceCycles > 0) {
      this.transitionGraceCycles--;
      return messages;
    }

    // Transcribe and normalize Arabic
    // Pass current verse text as decoder prompt to bias toward correct vocabulary
    const trackingPrompt = this.trackingVerse?.text_norm?.slice(-80);
    const { text: rawText } = await this.transcribe(
      this.fullAudio.slice(),
      trackingPrompt,
    );
    const text = normalizeArabic(rawText);
    if (!text || text.trim().length < 3) return messages;

    const recognizedWords = text.split(" ");

    // Align against known verse (using normalized Arabic words)
    const resumeFrom = Math.max(this.trackingLastWordIdx, 0);
    let { matchedIndices } = alignPosition(
      recognizedWords,
      this.trackingVerseWords,
      resumeFrom,
    );

    // Fallback: character-level progress when word alignment fails
    // (model sometimes outputs spaceless strings)
    // Only for verses with 10+ words where word-level alignment is unreliable
    if (matchedIndices.length === 0 && text.length >= 5 && this.trackingVerseWords.length >= 10) {
      const charWordIdx = this._charLevelProgress(text);
      if (charWordIdx > this.trackingLastWordIdx) {
        matchedIndices = [charWordIdx];
      }
    }

    // Check for stale tracking
    const advanced =
      matchedIndices.length > 0 &&
      matchedIndices[matchedIndices.length - 1] > this.trackingLastWordIdx;

    if (!advanced) {
      this.staleCycles++;
      // Before giving up, check if user jumped to a nearby verse
      // Apply hysteresis: during cooldown, require much higher score to switch
      // and never switch backward within same surah
      // Only consider verse jump if we've actually tracked some words of the current verse.
      // Without this, a 2-word fragment can cause an immediate jump to the next verse.
      const currentCoverage = this.trackingLastWordIdx >= 0
        ? (this.trackingLastWordIdx + 1) / this.trackingVerseWords.length
        : 0;
      if (this.staleCycles >= 2 && text.length >= 8 && this.trackingVerse && currentCoverage >= 0.3) {
        const jumpHint: [number, number] = [this.trackingVerse.surah, this.trackingVerse.ayah];
        const jumpThreshold = this.transitionCooldown > 0 ? 0.7 : 0.5;
        const jumpMatch = this.db.matchVerseNarrow(text, jumpHint, 3, jumpThreshold);
        if (jumpMatch && (jumpMatch.surah !== this.trackingVerse.surah || jumpMatch.ayah !== this.trackingVerse.ayah)) {
          // Anti-bounce: during cooldown, block backward jumps within same surah
          const isBackward = jumpMatch.surah === this.trackingVerse.surah &&
                            jumpMatch.ayah < this.trackingVerse.ayah;
          if (isBackward && this.transitionCooldown > 0) {
            // Suppress backward jump during cooldown — likely stale audio
          } else {
            const newVerse = this.db.getVerse(jumpMatch.surah, jumpMatch.ayah);
            if (newVerse) {
              this.lastEmittedRef = [this.trackingVerse.surah, this.trackingVerse.ayah];
              this.lastEmittedText = this.trackingVerse.text_norm!;
              this._exitTracking("verse jump detected");
              const ref: [number, number] = [newVerse.surah, newVerse.ayah];
              const surrounding = getSurroundingVerses(this.db, newVerse.surah, newVerse.ayah);
              messages.push({
                type: "verse_match",
                surah: newVerse.surah,
                ayah: newVerse.ayah,
                verse_text: newVerse.text_uthmani,
                surah_name: newVerse.surah_name,
                confidence: Math.round(jumpMatch.score * 100) / 100,
                surrounding_verses: surrounding,
              });
              this.hasEverMatched = true;
              this.cyclesSinceEmit = 0;
              this.prevEmittedRef = this.lastEmittedRef;
              this.prevEmittedText = this.lastEmittedText;
              this.lastEmittedRef = ref;
              this.lastEmittedText = newVerse.text_norm!;
              this._enterTracking(newVerse);
              this.fullAudio = this.fullAudio.slice(-TRIGGER_SAMPLES);
              return messages;
            }
          }
        }
      }
      // Dynamic limit: longer verses get more patience (1 extra cycle per 8 words)
      const effectiveStaleLimit = Math.max(
        STALE_CYCLE_LIMIT,
        Math.floor(this.trackingVerseWords.length / 8),
      );
      if (this.staleCycles >= effectiveStaleLimit) {
        this._exitTracking(
          `stale (${this.staleCycles} cycles, no progress)`,
        );
        // Let audio buffer accumulate naturally before re-attempting discovery
        this.newAudioCount = 0;
        return messages;
      }
    } else {
      this.staleCycles = 0;
    }

    // Coverage-based exit: if we've covered most of the verse but tracking stalled,
    // advance to next verse rather than waiting for full stale timeout
    if (!advanced && this.staleCycles >= 2 && this.trackingLastWordIdx >= 0) {
      const coverage = (this.trackingLastWordIdx + 1) / this.trackingVerseWords.length;
      if (coverage >= 0.85) {
        // Treat as verse complete — high coverage + stalled = user finished
        const curRef: [number, number] = [
          this.trackingVerse!.surah,
          this.trackingVerse!.ayah,
        ];
        this.lastEmittedRef = curRef;
        this.lastEmittedText = this.trackingVerse!.text_norm!;
        this.cyclesSinceEmit = 0;

        if (this.trackingVerse!.surah === this.lastConfirmedSurah) {
          this.lastConfirmedAyah = Math.max(this.lastConfirmedAyah, this.trackingVerse!.ayah);
        } else {
          this.lastConfirmedSurah = this.trackingVerse!.surah;
          this.lastConfirmedAyah = this.trackingVerse!.ayah;
        }

        const nextV = this.db.getNextVerse(curRef[0], curRef[1]);
        this._exitTracking("verse complete");

        if (nextV) {
          const nextRef: [number, number] = [nextV.surah, nextV.ayah];
          const surrounding = getSurroundingVerses(this.db, nextV.surah, nextV.ayah);
          messages.push({
            type: "verse_match",
            surah: nextV.surah,
            ayah: nextV.ayah,
            verse_text: nextV.text_uthmani,
            surah_name: nextV.surah_name,
            confidence: 0.95,
            surrounding_verses: surrounding,
          });
          this.prevEmittedRef = this.lastEmittedRef;
          this.prevEmittedText = this.lastEmittedText;
          this.lastEmittedRef = nextRef;
          this.lastEmittedText = nextV.text_norm!;
          this._enterTracking(nextV);
          this.transitionCooldown = 3;
        }

        this.fullAudio = new Float32Array(0);
        this.accumulatedText = "";
        this.accumulatedCycles = 0;
        return messages;
      }
    }

    // Send word_progress if advanced
    if (advanced) {
      this.trackingLastWordIdx =
        matchedIndices[matchedIndices.length - 1];
      const wordPos = this.trackingLastWordIdx + 1;
      messages.push({
        type: "word_progress",
        surah: this.trackingVerse!.surah,
        ayah: this.trackingVerse!.ayah,
        word_index: wordPos,
        total_words: this.trackingVerseWords.length,
        matched_indices: matchedIndices,
      });

    }

    // Check if verse is complete
    // Require at least 30% word coverage before considering advancement.
    // This prevents jumping to the next verse after only tracking 1-2 words
    // of a long verse (e.g. 13:13 has 19 words — matching 1 word near the end
    // should not trigger advancement).
    const wordCoverageRatio = matchedIndices.length / this.trackingVerseWords.length;
    if (matchedIndices.length > 0 && wordCoverageRatio >= 0.3) {
      const cumulativeCoverage =
        (this.trackingLastWordIdx + 1) / this.trackingVerseWords.length;
      const nearEnd =
        this.trackingLastWordIdx >=
        this.trackingVerseWords.length - 2;

      if (cumulativeCoverage >= 0.8 && nearEnd) {
        // Advance to next verse
        const curRef: [number, number] = [
          this.trackingVerse!.surah,
          this.trackingVerse!.ayah,
        ];
        this.lastEmittedRef = curRef;
        this.lastEmittedText = this.trackingVerse!.text_norm!;
        this.cyclesSinceEmit = 0;

        // Track highest confirmed ayah for anti-bounce
        if (this.trackingVerse!.surah === this.lastConfirmedSurah) {
          this.lastConfirmedAyah = Math.max(this.lastConfirmedAyah, this.trackingVerse!.ayah);
        } else {
          this.lastConfirmedSurah = this.trackingVerse!.surah;
          this.lastConfirmedAyah = this.trackingVerse!.ayah;
        }

        const nextV = this.db.getNextVerse(curRef[0], curRef[1]);
        this._exitTracking("verse complete");

        if (nextV) {
          const nextRef: [number, number] = [nextV.surah, nextV.ayah];
          const surrounding = getSurroundingVerses(
            this.db,
            nextV.surah,
            nextV.ayah,
          );
          messages.push({
            type: "verse_match",
            surah: nextV.surah,
            ayah: nextV.ayah,
            verse_text: nextV.text_uthmani,
            surah_name: nextV.surah_name,
            confidence: 0.99,
            surrounding_verses: surrounding,
          });
          // Save completed verse state for recovery if next-verse tracking fails
          this.prevEmittedRef = this.lastEmittedRef;
          this.prevEmittedText = this.lastEmittedText;
          this.lastEmittedRef = nextRef;
          this.lastEmittedText = nextV.text_norm!;
          this._enterTracking(nextV);

          // Activate anti-bounce cooldown (3 cycles ≈ 1.5s)
          this.transitionCooldown = 3;
        }

        // Clear audio buffer on verse advance to prevent stale audio from
        // the completed verse contaminating next-verse tracking.
        // User typically pauses briefly between verses, so next inference
        // will capture fresh audio from the new verse.
        this.fullAudio = new Float32Array(0);
        this.accumulatedText = "";
        this.accumulatedCycles = 0;
      }
    }

    return messages;
  }

  private async _handleDiscovery(): Promise<WorkerOutbound[]> {
    const messages: WorkerOutbound[] = [];

    // Adaptive trigger:
    //   - Very first attempt of the session: 2.0s (FIRST_TRIGGER_SAMPLES).
    //     Short clips (< 3s) would get zero inference cycles at the old 3.0s
    //     threshold.  FastConformer can produce useful output from 2s of audio.
    //   - Subsequent discovery cycles: 3.0s (TRIGGER_SAMPLES) to avoid being
    //     chatty and to give the model enough context for longer verses.
    const isVeryFirstAttempt = !this.hasEverMatched && this.cyclesSinceEmit === Infinity;
    const triggerThreshold = isVeryFirstAttempt ? FIRST_TRIGGER_SAMPLES : TRIGGER_SAMPLES;
    if (this.newAudioCount < triggerThreshold) return messages;
    this.newAudioCount = 0;
    this.cyclesSinceEmit++;

    // Skip silent chunks
    const tail = this.fullAudio.slice(-TRIGGER_SAMPLES);
    if (isSilence(tail)) {
      // Reset accumulated text on silence
      this.accumulatedText = "";
      this.accumulatedCycles = 0;
      return messages;
    }

    // Transcribe and normalize Arabic
    const { text: rawText } = await this.transcribe(this.fullAudio.slice());
    const text = normalizeArabic(rawText);
    if (!text || text.trim().length < 5) return messages;

    // Skip if transcription is mostly residual from last emitted verse
    if (this.lastEmittedText) {
      const residual = partialRatio(text, this.lastEmittedText);
      if (residual > 0.7) return messages;
    }

    // Accumulate transcript across discovery cycles for long verses
    // New text replaces accumulated when it's significantly different
    // (the audio window slides, so later transcripts extend the text)
    this.accumulatedCycles++;
    if (this.accumulatedText) {
      // Check if new text extends the accumulated (shares a suffix/prefix overlap)
      const newWords = text.split(" ");
      const accWords = this.accumulatedText.split(" ");
      // Try to find overlap: last N words of accumulated match first N words of new
      let overlapLen = 0;
      const maxCheck = Math.min(accWords.length, newWords.length, 6);
      for (let n = maxCheck; n >= 2; n--) {
        const accTail = accWords.slice(-n).join(" ");
        const newHead = newWords.slice(0, n).join(" ");
        if (partialRatio(accTail, newHead) >= 0.7) {
          overlapLen = n;
          break;
        }
      }
      if (overlapLen > 0) {
        // Merge: accumulated + new text after overlap
        this.accumulatedText = this.accumulatedText + " " + newWords.slice(overlapLen).join(" ");
      } else if (this.accumulatedCycles <= 4) {
        // No overlap found but still early — just append
        this.accumulatedText = this.accumulatedText + " " + text;
      } else {
        // Too many cycles without overlap — reset
        this.accumulatedText = text;
        this.accumulatedCycles = 1;
      }
    } else {
      this.accumulatedText = text;
    }

    // Use accumulated text for matching if it's longer than current transcript
    const matchText = this.accumulatedText.length > text.length * 1.3
      ? this.accumulatedText
      : text;

    // -----------------------------------------------------------------------
    // Phase 1 — Prefix-narrowing (paper algorithm, low latency path)
    //
    // Use the first N recognized words to walk the QuranDB word-prefix trie.
    // If this narrows to a small candidate set (per-paper: mean 3.11 words
    // suffices for 94.6% of verses), score only those candidates with a
    // relaxed threshold.  This replaces the full 6236-verse Levenshtein pass
    // for the discovery step, dramatically lowering the score needed to emit.
    //
    // We attempt prefix-narrowing FIRST because it is faster and more
    // discriminative for short/clean transcripts.  If it fails (transcript
    // too noisy / candidate set too large), we fall back to the original
    // matchVerse path below.
    // -----------------------------------------------------------------------
    let match: VerseMatch | null = null;
    let viaPrefix = false;
    // Set to true when Phase 1 explicitly found a candidate but determined
    // we need more audio before emitting (disambiguation hold).
    // When true, Phase 2 is SKIPPED to prevent the full-corpus search from
    // overriding the Phase 1 hold with an incorrect premature match.
    let prefixDeferred = false;

    {
      const normWords = matchText.split(" ").filter(w => w.length > 0);
      // Use first 8 words for narrowing (paper: 89.4% unique within 6 words;
      // use 8 for margin when the first 6 are still ambiguous)
      const prefixWords = normWords.slice(0, 8);

      if (prefixWords.length >= 1) {
        // Try progressively longer prefixes from longest to shortest.
        // Longest prefix = most discriminative; stop as soon as we get a
        // candidate set small enough to score reliably (up to PREFIX_NARROW_MAX_CANDIDATES).
        for (let depth = prefixWords.length; depth >= 1; depth--) {
          const narrowed = this.db.narrowByPrefix(
            prefixWords.slice(0, depth),
            PREFIX_NARROW_MAX_CANDIDATES,
          );
          if (narrowed && narrowed.length > 0 && narrowed.length <= PREFIX_NARROW_MAX_CANDIDATES) {
            const prefixMatch = this.db.matchVerseFromCandidates(
              matchText,
              narrowed,
              PREFIX_NARROW_THRESHOLD,
              this.lastEmittedRef,
            );
            if (prefixMatch) {
              // Disambiguation-aware hold: the paper tells us exactly how many
              // words are needed from the start of this verse to uniquely
              // identify it.  If we have fewer words than that, the current
              // winner may change once more audio arrives — defer.
              //
              // Exception: if the narrowed set has exactly 1 candidate, the
              // trie itself has already disambiguated (no hold needed).
              const requiresWords = this.db.getDisambiguationLength(prefixMatch.surah, prefixMatch.ayah);
              const haveWords = normWords.length;
              const trieUnique = narrowed.length === 1;

              // requiresWords === -1 means the verse is NEVER uniquely identifiable
              // in isolation (e.g., 55:13 repeats 31 times). Hold unless we have
              // sequential context (previous verse known).
              const neverUnique = requiresWords === -1;
              const needsMoreWords = requiresWords > 0 && haveWords < requiresWords;
              if (!trieUnique && (neverUnique || needsMoreWords)) {
                // Don't emit yet — accumulate more audio and re-try.
                // Mark as deferred so Phase 2 doesn't override this hold.
                prefixDeferred = true;
                break;
              }

              match = prefixMatch as VerseMatch;
              viaPrefix = true;
              break;
            }
          }
        }
      }
    }

    // Phase 2 — Full corpus match (original path, fallback when prefix fails)
    // SKIP when Phase 1 explicitly deferred — the deferral means we found a
    // narrowed candidate set but need more words before we can commit.
    // Running Phase 2 here would pick a premature match from the full corpus.
    if (!match && !prefixDeferred) {
      if (this.lastEmittedRef && this.cyclesSinceEmit <= 3) {
        match = this.db.matchVerseNarrow(
          matchText,
          this.lastEmittedRef,
          5,
          RAW_TRANSCRIPT_THRESHOLD,
        ) as VerseMatch | null;
      } else {
        match = this.db.matchVerse(
          matchText,
          RAW_TRANSCRIPT_THRESHOLD,
          4,
          this.lastEmittedRef,
          10,
          this.sessionSurah,
        ) as VerseMatch | null;
      }
    }

    // Phase 2 disambiguation-aware hold: if Phase 2 returned a match via
    // full-corpus search, apply the same logic as Phase 1 — if the matched
    // verse needs MORE words than the transcript has to be uniquely identified,
    // require a higher confidence score (>= 0.75) to commit.
    //
    // This prevents the low FIRST_MATCH_THRESHOLD (0.55) from causing premature
    // commits on the first 2s trigger for long verses with ambiguous openings.
    //
    // Exception: continue normally when match is a short verse (text_words <= 4)
    // or when we already have enough words per the disambiguation data.
    if (match && !viaPrefix) {
      const requiresWords = this.db.getDisambiguationLength(match.surah, match.ayah);
      const haveWords = matchText.split(" ").filter(w => w.length > 0).length;
      const matchedVerseWordCount2 = this.db.getVerse(match.surah, match.ayah)?.text_words?.length ?? 0;
      const isShortVerse2 = matchedVerseWordCount2 <= 4;

      // Never-unique verses (d=-1, e.g., 55:13 repeated 31×):
      // Only commit if we have sequential context (previous verse → boundary resolution)
      const neverUnique2 = requiresWords === -1;
      if (neverUnique2 && !isShortVerse2 && !this.lastEmittedRef) {
        match = null;
        prefixDeferred = true;
      }
      // Not enough words yet per disambiguation data:
      const needsMore2 = requiresWords > 0 && haveWords < requiresWords;
      if (match && !isShortVerse2 && needsMore2) {
        // Require 0.75 confidence for premature Phase 2 matches
        if (match.score < 0.75) {
          // Not confident enough — treat as deferred, fall through to raw_transcript
          match = null;
          prefixDeferred = true;
        }
      }
    }

    // Emit candidate list for live narrowing UI
    if (match?.runners_up?.length) {
      const candidates: CandidateVerse[] = match.runners_up
        .filter((ru) => ru.score >= 0.15)
        .slice(0, 10)
        .map((ru) => ({
          surah: ru.surah,
          ayah: ru.ayah,
          score: ru.score,
          surah_name: ru.surah_name ?? "",
          surah_name_en: ru.surah_name_en ?? "",
          text_preview: ru.text_uthmani ?? ru.text_norm ?? "",
        }));
      if (candidates.length > 0) {
        messages.push({
          type: "candidate_list",
          candidates,
          transcript: text,
        });
      }
    }

    // Anti-cascade: shortly after an emit, require higher threshold for
    // non-continuation jumps to prevent false positives from cascading
    let effectiveThreshold = this.hasEverMatched
      ? VERSE_MATCH_THRESHOLD
      : FIRST_MATCH_THRESHOLD;

    // When the match came via prefix-narrowing (paper algorithm), the
    // candidate set was already pre-filtered to <= PREFIX_NARROW_MAX_CANDIDATES
    // verses.  The prefix trie provides the same false-positive protection that
    // the high FIRST_MATCH_THRESHOLD was meant to give, so we can use the
    // lower PREFIX_NARROW_THRESHOLD instead.
    if (viaPrefix) {
      effectiveThreshold = Math.min(effectiveThreshold, PREFIX_NARROW_THRESHOLD);
    }

    if (match && this.hasEverMatched && this.cyclesSinceEmit <= 2 && this.lastEmittedRef) {
      const isContinuation =
        match.surah === this.lastEmittedRef[0] &&
        match.ayah >= this.lastEmittedRef[1] + 1 &&
        match.ayah <= this.lastEmittedRef[1] + 3;
      if (!isContinuation) {
        // Keep the anti-cascade bump, but don't let it exceed 0.65 when we
        // have prefix-narrowing confidence (pre-filtered candidates).
        const cascadeBump = viaPrefix ? 0.55 : 0.65;
        effectiveThreshold = Math.max(effectiveThreshold, cascadeBump);
      }
    }

    // Long-verse mode: if top candidate has 20+ words and we've been accumulating,
    // extend the audio window to capture more of the verse
    if (match && match.score >= 0.3 && match.score < effectiveThreshold && this.accumulatedCycles >= 3) {
      const topVerse = this.db.getVerse(match.surah, match.ayah);
      if (topVerse?.text_words && topVerse.text_words.length > 20) {
        if (!this._longVerseMode) {
          this._longVerseMode = true;
          this._longVerseModeCycles = 0;
        }
      }
    }
    // Expire long-verse mode after 8 cycles to prevent unbounded buffer growth
    if (this._longVerseMode) {
      this._longVerseModeCycles++;
      if (this._longVerseModeCycles > 8) {
        this._longVerseMode = false;
        this._longVerseModeCycles = 0;
      }
    }

    // Minimum word count for first discovery match (prevents false positives
    // on very short / ambiguous audio). Once tracking is established, shorter
    // transcripts are fine for continuation.
    //
    // Exception paths:
    //   - Muqatta'at (disconnected letters): always allow (1 word, <= 5 chars)
    //   - Prefix-narrowing hit with short verse: skip MIN_DISCOVERY_WORDS gate
    //     when the matched verse itself is short (text_words.length <= 3) since
    //     there is no more text to wait for.
    //   - Already-established session with context: gate only applies for the
    //     very first match.
    const matchWords = matchText.split(" ").filter((w: string) => w.length > 0);
    const matchedVerse = match ? this.db.getVerse(match.surah, match.ayah) : null;
    const matchedVerseWordCount = matchedVerse?.text_words?.length ?? 999;

    if (!this.hasEverMatched && matchWords.length < MIN_DISCOVERY_WORDS && match && match.score >= effectiveThreshold) {
      // Exception: muqatta'at (isolated letter) verses like يس, طه, الم,
      // كهيعص, حم, المص etc. These are 1 word of <= 5 characters.
      // A 2-word phrase like "بسم الله" (8 chars) is NOT muqatta'at.
      const isMuqattaat = match.score >= 0.95 &&
        matchWords.length === 1 &&
        matchWords[0].length <= 5;
      // Exception: the ENTIRE verse is short (e.g. 114:2 = 2 words).
      // Waiting for more words is pointless — there are none.
      const isShortVerse = matchedVerseWordCount <= MIN_DISCOVERY_WORDS;
      // Exception: prefix-narrowing resolved to a uniquely identifying prefix
      // (the trie confirmed this word sequence belongs to only this verse).
      const isPrefixUnique = viaPrefix && match.score >= PREFIX_NARROW_THRESHOLD;

      if (!isMuqattaat && !isShortVerse && !isPrefixUnique) {
        // Not enough words yet — emit raw transcript and wait for more audio
        messages.push({
          type: "raw_transcript",
          text,
          confidence: Math.round(match.score * 100) / 100,
        });
        return messages;
      }
    }

    // Fragment coverage gate: only block extremely short fragments (1-2 words)
    // of very long verses (15+ words) with low scores.
    // Exception: skip this gate when prefix-narrowing has pre-vetted the
    // candidates — the trie already ensures the short transcript matches this
    // verse's prefix, so the short-fragment check is not needed.
    if (!viaPrefix && match && match.text_words && match.text_words.length > 15 &&
        matchWords.length <= 2 && match.score < 0.95) {
      messages.push({ type: "raw_transcript", text, confidence: Math.round(match.score * 100) / 100 });
      return messages;
    }

    if (match && match.score >= effectiveThreshold) {
      const ref: [number, number] = [match.surah, match.ayah];

      // Ambiguity guard: only suppress when scores are nearly identical
      // and the transcript hasn't clearly differentiated the verses.
      //
      // Key improvement: when we have sequential boundary context (the user
      // has been reciting and we know the previous verse), use that to resolve
      // the 339 never-unique verses (bismillah openers, refrains, muqattaat).
      // The paper shows 328/339 resolve with boundary context.
      const runnersUp: VerseMatchCandidate[] = match.runners_up ?? [];
      const isAmbiguousVerse = this.db.isAmbiguousInIsolation(match.surah, match.ayah);
      const hasSequentialContext = this.lastEmittedRef !== null && this.hasEverMatched;

      // Boundary context resolution: if the matched verse is never-unique in
      // isolation AND we have a previous verse, check if the match is the
      // sequential successor.  If yes, emit with high confidence.
      if (isAmbiguousVerse && hasSequentialContext && this.lastEmittedRef) {
        const [prevS, prevA] = this.lastEmittedRef;
        const isSequentialNext =
          (match.surah === prevS && match.ayah === prevA + 1) ||
          (match.surah === prevS + 1 && match.ayah === 1 && !this.db.getVerse(prevS, prevA + 1));
        const isSameSessionSurah = this.sessionSurah !== null && match.surah === this.sessionSurah;

        if (isSequentialNext || isSameSessionSurah) {
          // Context resolves the ambiguity — skip the ambiguity guard and emit
        } else if (runnersUp.length >= 2) {
          // Still run the ambiguity guard since context doesn't resolve it
          const matchVerse = this.db.getVerse(match.surah, match.ayah);
          let altRunner: VerseMatchCandidate | null = null;
          for (const ru of runnersUp) {
            if (ru.surah !== match.surah || ru.ayah !== match.ayah) {
              altRunner = ru;
              break;
            }
          }
          if (altRunner && altRunner.score >= runnersUp[0].score * 0.97) {
            // Defer — truly ambiguous without context
            const deferKey = `${match.surah}:${match.ayah}`;
            if (this.lastDeferredRef === deferKey) {
              this.consecutiveDeferrals++;
            } else {
              this.consecutiveDeferrals = 1;
              this.lastDeferredRef = deferKey;
            }
            if (this.consecutiveDeferrals <= 2) {
              messages.push({ type: "raw_transcript", text, confidence: Math.round(match.score * 100) / 100 });
              return messages;
            }
            this.consecutiveDeferrals = 0;
            this.lastDeferredRef = null;
          }
        }
      } else if (runnersUp.length >= 2) {
        // Original ambiguity guard path (non-ambiguous verses or no context)
        const matchVerse = this.db.getVerse(match.surah, match.ayah);
        let altRunner: VerseMatchCandidate | null = null;
        for (const ru of runnersUp) {
          if (ru.surah !== match.surah || ru.ayah !== match.ayah) {
            altRunner = ru;
            break;
          }
        }
        // Only guard when alt is within 3% of top score
        if (altRunner && altRunner.score >= runnersUp[0].score * 0.97) {
          const altVerse = this.db.getVerse(altRunner.surah, altRunner.ayah);
          if (matchVerse && altVerse) {
            const w1 = matchVerse.text_norm!.split(" ");
            const w2 = altVerse.text_norm!.split(" ");
            let sharedPrefix = 0;
            for (let i = 0; i < Math.min(w1.length, w2.length); i++) {
              if (w1[i] === w2[i]) sharedPrefix++;
              else break;
            }
            // Adaptive guard: for short transcripts (<=6 words, typical
            // bismillah + 0-2 words), use sharedPrefix threshold of 4;
            // for longer transcripts, keep the original threshold of 8.
            const textWords = text.split(" ").length;
            const minSharedPrefix = textWords <= 6 ? 4 : 8;
            if (sharedPrefix >= minSharedPrefix) {
              // Only defer if match isn't a near-perfect hit.
              // e.g. 1:1 IS just bismillah → scores ~1.0 → allow through.
              // 114:1 at T=2s with only bismillah heard → scores ~0.95 → defer.
              if (textWords <= sharedPrefix + 2 && match.score < 0.98) {
                const deferKey = `${match.surah}:${match.ayah}`;
                if (this.lastDeferredRef === deferKey) {
                  this.consecutiveDeferrals++;
                } else {
                  this.consecutiveDeferrals = 1;
                  this.lastDeferredRef = deferKey;
                }

                // Force emit after 2 consecutive deferrals of the same verse
                // (4+ seconds of audio) — prevents indefinite blocking in
                // sequential/concatenated mode where mixed audio always
                // produces scores < 0.98.
                if (this.consecutiveDeferrals <= 2) {
                  messages.push({
                    type: "raw_transcript",
                    text,
                    confidence: Math.round(match.score * 100) / 100,
                  });
                  return messages;
                }
                // Past deferral limit — fall through to emit verse_match
                this.consecutiveDeferrals = 0;
                this.lastDeferredRef = null;
              }
            }
          }
        }
      }

      // Dedup: skip if same verse was just sent
      if (
        this.lastEmittedRef &&
        this.lastEmittedRef[0] === ref[0] &&
        this.lastEmittedRef[1] === ref[1]
      ) {
        return messages;
      }

      const verse = this.db.getVerse(match.surah, match.ayah);
      const surrounding = getSurroundingVerses(
        this.db,
        match.surah,
        match.ayah,
      );

      messages.push({
        type: "verse_match",
        surah: match.surah,
        ayah: match.ayah,
        verse_text: verse?.text_uthmani ?? match.text ?? "",
        surah_name: verse?.surah_name ?? "",
        confidence: Math.round(match.score * 100) / 100,
        surrounding_verses: surrounding,
      });

      this.hasEverMatched = true;
      this._longVerseMode = false;
      this._longVerseModeCycles = 0;
      this.cyclesSinceEmit = 0;
      this.consecutiveDeferrals = 0;
      this.lastDeferredRef = null;
      this.accumulatedText = "";
      this.accumulatedCycles = 0;

      // Update session surah context (or switch if confident enough)
      if (match.score >= 0.75 || this.sessionSurah === null) {
        this.sessionSurah = match.surah;
      }

      // For multi-verse spans, advance hint to the last verse
      const ayahEnd = match.ayah_end;
      const effectiveRef: [number, number] = ayahEnd
        ? [match.surah, ayahEnd]
        : ref;
      // Save pre-match state for recovery if tracking determines misidentification
      this.prevEmittedRef = this.lastEmittedRef;
      this.prevEmittedText = this.lastEmittedText;
      this.lastEmittedRef = effectiveRef;
      this.lastEmittedText =
        match.text_norm ?? verse?.text_norm ?? "";

      // Enter tracking mode
      if (verse) {
        this._enterTracking(verse);
      } else {
        // No tracking — reset window
        this.fullAudio = tail.slice();
      }
    } else {
      // Send raw transcript
      const score = match ? Math.round(match.score * 100) / 100 : 0;
      messages.push({
        type: "raw_transcript",
        text,
        confidence: score,
      });
    }

    return messages;
  }

  private _charLevelProgress(text: string): number {
    if (!this.trackingVerse) return -1;
    const joined = this.trackingVerse.text_norm!;
    const words = this.trackingVerseWords;
    if (!joined || words.length === 0) return -1;

    // Compare no-space text against no-space verse for spaceless model output
    const noSpaceText = text.replace(/ /g, "");
    const noSpaceJoined = joined.replace(/ /g, "");
    const tLen = noSpaceText.length;
    if (tLen < 3 || tLen >= noSpaceJoined.length) return -1;

    // Slide transcript-sized window across verse, find best match position
    let bestScore = 0;
    let bestEnd = 0;
    // Step by ~10 chars for speed, then refine
    const step = Math.max(1, Math.floor(tLen / 5));
    for (let i = 0; i <= noSpaceJoined.length - tLen; i += step) {
      const span = noSpaceJoined.slice(i, i + tLen);
      const s = levRatio(noSpaceText, span);
      if (s > bestScore) {
        bestScore = s;
        bestEnd = i + tLen;
      }
    }
    // Refine around best position
    if (step > 1) {
      const refStart = Math.max(0, bestEnd - tLen - step);
      const refEnd = Math.min(noSpaceJoined.length - tLen, bestEnd - tLen + step);
      for (let i = refStart; i <= refEnd; i++) {
        const span = noSpaceJoined.slice(i, i + tLen);
        const s = levRatio(noSpaceText, span);
        if (s > bestScore) {
          bestScore = s;
          bestEnd = i + tLen;
        }
      }
    }

    if (bestScore < 0.55) return -1;

    // Map bestEnd position in no-space string back to word index
    // Count chars consumed per word (without spaces) to find which word bestEnd falls in
    let charCount = 0;
    for (let w = 0; w < words.length; w++) {
      charCount += words[w].length;
      if (charCount >= bestEnd) return w;
    }
    return words.length - 1;
  }

  private _enterTracking(verse: QuranVerse): void {
    this.trackingVerse = verse;
    this.trackingVerseWords = verse.text_words!;
    this.trackingLastWordIdx = -1;
    this.silenceSamples = 0;
    this.staleCycles = 0;
    this.accumulatedText = "";
    this.accumulatedCycles = 0;
    // Grace period: skip 1 tracking cycle to let stale audio from
    // the completed verse flush out before starting word alignment
    this.transitionGraceCycles = 1;

    // Initialize confirmed verse tracking on first entry
    if (this.lastConfirmedSurah < 0) {
      this.lastConfirmedSurah = verse.surah;
      this.lastConfirmedAyah = verse.ayah;
    }
  }

  private _exitTracking(reason: string): void {
    const verseLen = this.trackingVerseWords.length;
    const progress =
      verseLen > 0 ? (this.trackingLastWordIdx + 1) / verseLen : 0;

    if (reason === "verse complete") {
      // Caller already updated lastEmittedRef/Text
      this.hasEverMatched = true;
      // Keep anti-bounce state (cooldown, confirmed ayah) — set by caller
    } else if (reason.startsWith("stale") && progress < 0.5) {
      // Low progress + stale = likely misidentification
      this.lastEmittedRef = this.prevEmittedRef;
      this.lastEmittedText = this.prevEmittedText;
      // Reset anti-bounce on misidentification
      this.transitionCooldown = 0;
      this.lastConfirmedAyah = -1;
      this.lastConfirmedSurah = -1;
    } else if (
      reason.startsWith("stale") &&
      this.trackingVerseWords.length > 0 &&
      this.trackingLastWordIdx >= 0
    ) {
      // Good progress + stale = was tracking correctly but user
      // paused or diverged. Trim residual text to tracked portion.
      this.hasEverMatched = true;
      this.lastEmittedText = this.trackingVerseWords
        .slice(0, this.trackingLastWordIdx + 1)
        .join(" ");
    } else if (reason === "extended silence") {
      // Reset anti-bounce on silence exit — user may have stopped
      this.transitionCooldown = 0;
      this.lastConfirmedAyah = -1;
      this.lastConfirmedSurah = -1;
    }

    this.trackingVerse = null;
    this.trackingVerseWords = [];
    this.trackingLastWordIdx = -1;
    this.silenceSamples = 0;
    this.staleCycles = 0;
    this.accumulatedText = "";
    this.accumulatedCycles = 0;
    this._longVerseMode = false;
    this._longVerseModeCycles = 0;
  }
}
