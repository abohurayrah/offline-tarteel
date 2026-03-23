// ---------------------------------------------------------------------------
// Message protocol (Worker <-> Main Thread)
// ---------------------------------------------------------------------------
export interface VerseMatchMessage {
  type: "verse_match";
  surah: number;
  ayah: number;
  verse_text: string;
  surah_name: string;
  confidence: number;
  surrounding_verses: SurroundingVerse[];
}

export interface WordProgressMessage {
  type: "word_progress";
  surah: number;
  ayah: number;
  word_index: number;
  total_words: number;
  matched_indices: number[];
}

export interface RawTranscriptMessage {
  type: "raw_transcript";
  text: string;
  confidence: number;
}

export interface CandidateVerse {
  surah: number;
  ayah: number;
  score: number;
  surah_name: string;
  surah_name_en: string;
  text_preview: string;
}

export interface CandidateListMessage {
  type: "candidate_list";
  candidates: CandidateVerse[];
  transcript: string;
}

// ---------------------------------------------------------------------------
// Forced Alignment message types
// ---------------------------------------------------------------------------
export interface WordAlignedMessage {
  type: "word_aligned";
  surah: number;
  ayah: number;
  word_index: number;
  total_words: number;
  confidence: number;        // 0-1 pronunciation quality
  cumulative_indices: number[]; // all word indices confirmed so far
}

export interface VerseCompleteMessage {
  type: "verse_complete";
  surah: number;
  ayah: number;
  overall_score: number;
  word_scores: number[];     // per-word confidence scores
  next_surah: number;
  next_ayah: number;
}

export interface SurroundingVerse {
  surah: number;
  ayah: number;
  text: string;
  is_current: boolean;
}

// Main -> Worker
export type WorkerInbound =
  | { type: "init" }
  | { type: "audio"; samples: Float32Array }
  | { type: "reset" };

// Worker -> Main
export type WorkerOutbound =
  | { type: "loading"; percent: number }
  | { type: "loading_status"; message: string }
  | { type: "ready" }
  | { type: "error"; message: string }
  | VerseMatchMessage
  | WordProgressMessage
  | RawTranscriptMessage
  | CandidateListMessage
  | WordAlignedMessage
  | VerseCompleteMessage;

// ---------------------------------------------------------------------------
// Verse match results (from QuranDB.matchVerse / matchVerseNarrow)
// ---------------------------------------------------------------------------
export interface VerseMatchCandidate {
  surah: number;
  ayah: number;
  raw_score: number;
  bonus: number;
  score: number;
  text_norm: string;
  surah_name: string;
  surah_name_en: string;
  text_uthmani: string;
}

export interface VerseMatch {
  surah: number;
  ayah: number;
  ayah_end?: number;
  text?: string;
  text_uthmani?: string;
  text_clean?: string;
  surah_name?: string;
  surah_name_en?: string;
  text_norm?: string;
  text_norm_ns?: string;
  text_norm_no_bsm?: string | null;
  text_norm_no_bsm_ns?: string | null;
  text_words?: string[];
  score: number;
  raw_score: number;
  bonus: number;
  runners_up?: VerseMatchCandidate[];
}

// ---------------------------------------------------------------------------
// Quran data (from quran.json)
// ---------------------------------------------------------------------------
export interface QuranVerse {
  surah: number;
  ayah: number;
  text_uthmani: string;
  text_clean: string;
  surah_name: string;
  surah_name_en: string;
  // Normalized Arabic text (computed at load time)
  text_norm?: string;
  text_norm_ns?: string;                     // no-space version
  text_norm_no_bsm?: string | null;          // bismillah stripped
  text_norm_no_bsm_ns?: string | null;       // no-space no-bismillah
  text_words?: string[];                     // words of normalized text
}

// ---------------------------------------------------------------------------
// Constants (matching server.py exactly)
// ---------------------------------------------------------------------------
export const SAMPLE_RATE = 16000;
// First inference attempt after 2.0s (down from 3.0s) so short clips get at
// least one inference cycle.  Subsequent cycles still use TRIGGER_SAMPLES so
// discovery doesn't become chatty once audio is flowing.
export const TRIGGER_SAMPLES = SAMPLE_RATE * 3.0;
export const FIRST_TRIGGER_SAMPLES = SAMPLE_RATE * 2.0;   // very first attempt
export const MAX_WINDOW_SAMPLES = SAMPLE_RATE * 10.0;
export const SILENCE_RMS_THRESHOLD = 0.005;

export const VERSE_MATCH_THRESHOLD = 0.45;
// Lowered from 0.75 — the prefix-narrowing path in the tracker applies
// additional candidate pre-filtering before scoring, so the threshold can
// be lower without increasing false positives.  The original 0.75 was
// calibrated for a full-corpus scan with no pre-filtering.
export const FIRST_MATCH_THRESHOLD = 0.55;
export const RAW_TRANSCRIPT_THRESHOLD = 0.25;
export const SURROUNDING_CONTEXT = 2;

export const TRACKING_TRIGGER_SAMPLES = SAMPLE_RATE * 1.0;
export const TRACKING_SILENCE_SAMPLES = SAMPLE_RATE * 2.0;
export const TRACKING_MAX_WINDOW_SAMPLES = SAMPLE_RATE * 8.0;
export const STALE_CYCLE_LIMIT = 4;
export const LOOKAHEAD = 8;

// Word-level similarity threshold for tracking mode.
// Lower than the default 0.7 because in tracking mode we already know which
// verse the user is reciting, so partial/noisy BPE output is expected.
export const TRACKING_WORD_THRESHOLD = 0.55;

// Discovery mode: minimum words before first verse match.
// Lowered to 2 — the paper shows 44.3% of verses are uniquely identifiable
// in 2 words.  The old value of 3 blocked all 2-word verses (e.g. 114:2
// ملك الناس) permanently.
export const MIN_DISCOVERY_WORDS = 2;

// Prefix-narrowing: threshold to use when disambiguation data has pre-vetted
// the candidate set to <= PREFIX_NARROW_MAX_CANDIDATES verses.
export const PREFIX_NARROW_THRESHOLD = 0.40;
export const PREFIX_NARROW_MAX_CANDIDATES = 5;

// Forced alignment constants
export const FA_CONFIDENCE_GOOD = 0.7;
export const FA_CONFIDENCE_WARN = 0.4;
