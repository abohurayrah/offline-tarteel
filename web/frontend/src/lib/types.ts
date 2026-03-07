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

export interface WordCorrection {
  word_index: number;
  expected: string;
  got: string;
  error_type: "substitution" | "deletion" | "insertion";
}

export interface WordCorrectionMessage {
  type: "word_correction";
  surah: number;
  ayah: number;
  corrections: WordCorrection[];
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
  | WordCorrectionMessage
  | RawTranscriptMessage
  | CandidateListMessage
  | WordAlignedMessage
  | VerseCompleteMessage;

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

export interface SurahData {
  surah: number;
  surah_name: string;
  surah_name_en: string;
  verses: { ayah: number; text_uthmani: string }[];
}

// ---------------------------------------------------------------------------
// Constants (matching server.py exactly)
// ---------------------------------------------------------------------------
export const SAMPLE_RATE = 16000;
export const TRIGGER_SECONDS = 2.0;
export const TRIGGER_SAMPLES = SAMPLE_RATE * TRIGGER_SECONDS;
export const MAX_WINDOW_SECONDS = 10.0;
export const MAX_WINDOW_SAMPLES = SAMPLE_RATE * MAX_WINDOW_SECONDS;
export const SILENCE_RMS_THRESHOLD = 0.005;

export const VERSE_MATCH_THRESHOLD = 0.45;
export const FIRST_MATCH_THRESHOLD = 0.75;
export const RAW_TRANSCRIPT_THRESHOLD = 0.25;
export const SURROUNDING_CONTEXT = 2;

export const TRACKING_TRIGGER_SECONDS = 0.5;
export const TRACKING_TRIGGER_SAMPLES = SAMPLE_RATE * TRACKING_TRIGGER_SECONDS;
export const TRACKING_SILENCE_TIMEOUT = 4.0;
export const TRACKING_SILENCE_SAMPLES = SAMPLE_RATE * TRACKING_SILENCE_TIMEOUT;
export const TRACKING_MAX_WINDOW_SECONDS = 5.0;
export const TRACKING_MAX_WINDOW_SAMPLES =
  SAMPLE_RATE * TRACKING_MAX_WINDOW_SECONDS;
export const STALE_CYCLE_LIMIT = 4;
export const LOOKAHEAD = 5;

// Forced alignment constants
export const FA_TRIGGER_SAMPLES = SAMPLE_RATE * 0.3;  // 300ms — fast FA updates
export const FA_CONFIDENCE_GOOD = 0.7;
export const FA_CONFIDENCE_WARN = 0.4;
export const FA_VERSE_COMPLETE_THRESHOLD = 0.85;  // fraction of words needed
