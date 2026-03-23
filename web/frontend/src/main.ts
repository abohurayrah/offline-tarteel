import "@fontsource/amiri/400.css";
import "@fontsource/amiri/700.css";
import "@fontsource/amiri-quran/400.css";
import "./style.css";

import { initFeedback, showWrongButton, hideWrongButton } from "./feedback";
import { encodeWav } from "./lib/wav-encoder";
import { QuranDB } from "./lib/quran-db";

import type {
  VerseMatchMessage,
  RawTranscriptMessage,
  WordProgressMessage,
  CandidateListMessage,
  WordAlignedMessage,
  VerseCompleteMessage,
  WorkerOutbound,
  QuranVerse,
} from "./lib/types";

import {
  FA_CONFIDENCE_GOOD,
  FA_CONFIDENCE_WARN,
} from "./lib/types";

import type { MushafPageData } from "./lib/mushaf-renderer";
import {
  renderPage as renderMushafPage,
  revealVerse as mushafRevealVerse,
  highlightWord as mushafHighlightWord,
  highlightErrors as mushafHighlightErrors,
  clearErrors as mushafClearErrors,
  revealAll as mushafRevealAll,
  hideUnrevealed as mushafHideUnrevealed,
  getPageVerses,
} from "./lib/mushaf-renderer";

// ---------------------------------------------------------------------------
// Types (UI-only)
// ---------------------------------------------------------------------------
interface SurahVerse {
  ayah: number;
  text_uthmani: string;
}

interface SurahData {
  surah: number;
  surah_name: string;
  surah_name_en: string;
  verses: SurahVerse[];
}

interface VerseGroup {
  surah: number;
  surahName: string;
  surahNameEn: string;
  currentAyah: number;
  verses: SurahVerse[];
  element: HTMLElement;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
interface DiagnosticEvent {
  timestamp: number;
  type: string;
  data: Record<string, unknown>;
}

const MAX_DIAGNOSTIC_EVENTS = 50;
const DIAGNOSTIC_COOLDOWN_MS = 30_000;

const state = {
  groups: [] as VerseGroup[],
  worker: null as Worker | null,
  audioCtx: null as AudioContext | null,
  stream: null as MediaStream | null,
  isActive: false,
  hasFirstMatch: false,
  modelReady: false,
  surahCache: new Map<number, SurahData>(),
  quranData: null as QuranVerse[] | null,
  sessionAudioChunks: [] as Float32Array[],
  lastModelPrediction: null as { surah: number; ayah: number; confidence: number } | null,
  lastCandidates: [] as { surah: number; ayah: number; score: number; surah_name: string; surah_name_en: string; text_preview: string }[],
  diagnosticEvents: [] as DiagnosticEvent[],
  lastDiagnosticSentAt: 0,
  recentVerseMatches: [] as { surah: number; ayah: number; timestamp: number }[],
  practiceMode: false,
  /** Raw transcript text shown in the candidate status bar during listening */
  lastRawTranscript: "" as string,
  /** Timer for fading the candidate bar after a confirmed match */
  candidateMatchFadeTimer: null as ReturnType<typeof setTimeout> | null,
  // Mushaf page mode
  mushafPages: null as MushafPageData[] | null,
  verseToPage: null as Record<string, number> | null,
  currentMushafPage: 1,
  revealedVerses: new Set<string>(),
  mushafDataReady: false,
  lastVerseTransitionTime: 0,
  // Algorithm view
  algorithmMode: false,
  algorithmDB: null as QuranDB | null,
  algorithmDBReady: false,
  /** Narrowing cascade history for current recognition cycle */
  narrowingHistory: [] as { word: string; count: number }[],
  /** Timestamp when the current algorithm cycle started */
  algorithmCycleStart: 0,
  /** Last identified verse info for the algorithm view */
  algorithmIdentified: null as {
    surah: number;
    ayah: number;
    surahName: string;
    confidence: number;
    wordsNeeded: number;
    timeMs: number;
    text: string;
  } | null,
};

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const $verses = document.getElementById("verses")!;
const $rawTranscript = document.getElementById("raw-transcript")!;
const $indicator = document.getElementById("listening-indicator")!;
const $permissionPrompt = document.getElementById("permission-prompt")!;
const $modelStatus = document.getElementById("model-status")!;
const $loadingStatus = document.getElementById("loading-status")!;
const $loadingProgress = document.getElementById("loading-progress")!;
const $loadingDetail = document.getElementById("loading-detail")!;
const $postRecording = document.getElementById("post-recording")!;
const $btnRecToggle = document.getElementById("btn-rec-toggle")!;
const $btnRestart = document.getElementById("btn-restart")!;
const $btnPractice = document.getElementById("btn-practice")!;
const $candidateList = document.getElementById("candidate-list")!;
const $app = document.getElementById("app")!;
// Algorithm view
const $algorithmView = document.getElementById("algorithm-view")!;
const $avTranscriptText = document.getElementById("av-transcript-text")!;
const $avNarrowingCascade = document.getElementById("av-narrowing-cascade")!;
const $avCandidateList = document.getElementById("av-candidate-list")!;
const $avDisambigInfo = document.getElementById("av-disambig-info")!;
const $avIdentified = document.getElementById("av-identified")!;
const $avIdentifiedRef = document.getElementById("av-identified-ref")!;
const $avIdentifiedText = document.getElementById("av-identified-text")!;
const $avIdentifiedStats = document.getElementById("av-identified-stats")!;
const $btnAlgoView = document.getElementById("btn-algo-view")!;
// Mushaf page mode
const $mushafContainer = document.getElementById("mushaf-container")!;
const $mushafPage = document.getElementById("mushaf-page")!;
const $btnPagePrev = document.getElementById("btn-page-prev")!;
const $btnPageNext = document.getElementById("btn-page-next")!;
const $pageInfo = document.getElementById("page-info")!;

// ---------------------------------------------------------------------------
// Arabic numeral converter
// ---------------------------------------------------------------------------
const arabicNumerals = ["٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"];
function toArabicNum(n: number): string {
  return String(n)
    .split("")
    .map((d) => arabicNumerals[parseInt(d)])
    .join("");
}

// ---------------------------------------------------------------------------
// Surah data (loaded from quran.json, no server needed)
// ---------------------------------------------------------------------------
async function loadQuranData(): Promise<void> {
  if (state.quranData) return;
  const res = await fetch("/quran.json");
  if (!res.ok) throw new Error(`Failed to load Quran data (HTTP ${res.status})`);
  state.quranData = await res.json();
  initFeedback(state.quranData!);
}

async function fetchSurah(surahNum: number): Promise<SurahData> {
  const cached = state.surahCache.get(surahNum);
  if (cached) return cached;

  await loadQuranData();
  const verses = state.quranData!.filter((v) => v.surah === surahNum);
  if (!verses.length) throw new Error(`Surah ${surahNum} not found`);

  const data: SurahData = {
    surah: surahNum,
    surah_name: verses[0].surah_name,
    surah_name_en: verses[0].surah_name_en,
    verses: verses.map((v) => ({
      ayah: v.ayah,
      text_uthmani: v.text_uthmani,
    })),
  };
  state.surahCache.set(surahNum, data);
  return data;
}

// ---------------------------------------------------------------------------
// Mushaf page mode
// ---------------------------------------------------------------------------
async function loadMushafData(): Promise<void> {
  if (state.mushafDataReady) return;
  const [pagesRes, vtpRes] = await Promise.all([
    fetch("/mushaf-pages.json"),
    fetch("/verse-to-page.json"),
  ]);
  if (!pagesRes.ok) throw new Error(`Failed to load mushaf pages (HTTP ${pagesRes.status})`);
  if (!vtpRes.ok) throw new Error(`Failed to load verse-to-page map (HTTP ${vtpRes.status})`);
  state.mushafPages = await pagesRes.json();
  state.verseToPage = await vtpRes.json();
  state.mushafDataReady = true;
}

function getPageForVerse(surah: number, ayah: number): number | null {
  if (!state.verseToPage) return null;
  return state.verseToPage[`${surah}:${ayah}`] ?? null;
}

async function navigateToMushafPage(pageNum: number): Promise<void> {
  if (!state.mushafPages || pageNum < 1 || pageNum > 604) return;
  if (pageNum === state.currentMushafPage && $mushafPage.children.length > 0) return;

  state.currentMushafPage = pageNum;
  $pageInfo.textContent = `${pageNum} / 604`;
  ($btnPagePrev as HTMLButtonElement).disabled = pageNum >= 604;
  ($btnPageNext as HTMLButtonElement).disabled = pageNum <= 1;

  // Transition: fade out using CSS transition (synced via transitionend, not setTimeout).
  // Lock the container height to prevent layout shifts during re-render.
  const currentHeight = $mushafPage.offsetHeight;
  if (currentHeight > 0) {
    $mushafPage.style.minHeight = `${currentHeight}px`;
  }

  $mushafPage.classList.add("mushaf-page--exit");
  await new Promise<void>((resolve) => {
    const onEnd = () => {
      $mushafPage.removeEventListener("transitionend", onEnd);
      resolve();
    };
    $mushafPage.addEventListener("transitionend", onEnd);
    // Fallback timeout in case transitionend doesn't fire (e.g. no transition)
    setTimeout(resolve, 350);
  });

  const page = state.mushafPages[pageNum - 1];
  await renderMushafPage($mushafPage, page, state.revealedVerses, state.practiceMode);

  // After re-rendering, restore word highlights for ALL revealed verses on this page.
  // The DOM was recreated by renderPage, so mp-word--spoken classes were lost for
  // previously completed verses. Restore them so previous verses stay fully visible.
  _restoreRevealedVerseHighlights(pageNum);

  // Also restore the currently-tracked verse's word-by-word progress
  if (_mushafTrackingKey && _mushafMatchedWords.size > 0) {
    const [ts, ta] = _mushafTrackingKey.split(":");
    const trackingPage = getPageForVerse(parseInt(ts), parseInt(ta));
    if (trackingPage === pageNum) {
      const accumulated = Array.from(_mushafMatchedWords).sort((a, b) => a - b);
      mushafHighlightWord($mushafPage, parseInt(ts), parseInt(ta), accumulated);
    }
  }

  // Release height lock and fade in
  $mushafPage.style.minHeight = "";
  $mushafPage.classList.remove("mushaf-page--exit");
  $mushafPage.classList.add("mushaf-page--enter");

  // Use animationend to clean up the enter class (synced with CSS animation)
  const onAnimEnd = () => {
    $mushafPage.removeEventListener("animationend", onAnimEnd);
    $mushafPage.classList.remove("mushaf-page--enter");
  };
  $mushafPage.addEventListener("animationend", onAnimEnd);
  // Fallback in case animationend doesn't fire
  setTimeout(() => $mushafPage.classList.remove("mushaf-page--enter"), 400);
}

// Restore mp-word--spoken and mp-word--revealed state for all revealed verses on a page.
// Called after renderPage re-creates the DOM, which strips runtime classes.
function _restoreRevealedVerseHighlights(pageNum: number): void {
  if (!state.mushafPages) return;
  const pageData = state.mushafPages[pageNum - 1];
  const pageVerses = getPageVerses(pageData);

  for (const vk of pageVerses) {
    if (state.revealedVerses.has(vk)) {
      const [s, a] = vk.split(":");
      mushafRevealVerse($mushafPage, parseInt(s), parseInt(a));
      // Also mark as spoken so they stay fully visible in practice mode
      const words = $mushafPage.querySelectorAll<HTMLElement>(
        `.mp-word[data-surah="${s}"][data-ayah="${a}"]`,
      );
      for (const w of words) {
        w.classList.add("mp-word--spoken");
      }
    }
  }
}

// Track which verses actually had word progress (not just verse_match)
const _wordTrackedVerses = new Set<string>();
// Track whether we've done the initial prior-verse reveal for the current page
let _priorRevealDoneForPage = 0;

// Handle verse match in mushaf mode — navigate + conditionally reveal prior verses
async function handleMushafVerseMatch(msg: VerseMatchMessage): Promise<void> {
  const prevPrediction = state.lastModelPrediction;
  state.lastModelPrediction = { surah: msg.surah, ayah: msg.ayah, confidence: msg.confidence };

  // Clear error state from previous verse to prevent stale red highlighting
  _mushafErrorWords.clear();
  _mushafErrorKey = "";

  // Clear word progress accumulator from previous verse to prevent stale highlights
  // from the old verse flashing during the transition to the new verse.
  // The accumulator will be re-populated by handleMushafWordProgress for the new verse.
  const newKey = `${msg.surah}:${msg.ayah}`;
  if (newKey !== _mushafTrackingKey) {
    _mushafMatchedWords = new Set<number>();
    _mushafTrackingKey = newKey;
    _mushafBismillahOffset = _computeBismillahOffset(msg.surah, msg.ayah);
    _mushafTrackingTotal = 0; // will be set by first word_progress
  }

  // Track verse transition time — suppress gap detection for 3s after transitions
  // (increased from 2s to cover the grace cycle gap in tracker.ts)
  state.lastVerseTransitionTime = Date.now();

  console.log(
    `%c[VERSE_MATCH] ${msg.surah}:${msg.ayah} (conf: ${(msg.confidence * 100).toFixed(1)}%)` +
    (prevPrediction ? ` prev: ${prevPrediction.surah}:${prevPrediction.ayah}` : ` (first)`),
    "color: #C2A05B; font-weight: bold",
  );

  if (!state.hasFirstMatch) {
    state.hasFirstMatch = true;
    $indicator.classList.add("has-verses");
    console.log("[MUSHAF] First match — indicator activated");
  }

  const targetPage = getPageForVerse(msg.surah, msg.ayah);
  if (!targetPage) {
    console.warn(`[MUSHAF] No page found for ${msg.surah}:${msg.ayah}`);
    return;
  }

  // Previous verse stays as-is — spoken words are already visible via
  // mp-word--spoken from word_progress. Only mark fully-read verses as
  // "revealed" (done in handleMushafWordProgress when ALL words matched).
  // This prevents the cascade where model jumping to next verse auto-reveals
  // the entire previous verse even if only a few words were spoken.

  const isNewPage = targetPage !== state.currentMushafPage;
  const isFirstOnPage = isNewPage || _priorRevealDoneForPage !== targetPage;

  // Navigate if needed
  if (isNewPage) {
    console.log(`[MUSHAF] Navigating: page ${state.currentMushafPage} → ${targetPage}`);
  }

  // Reveal prior verses ONLY on first match on this page (user started mid-page)
  if (isFirstOnPage && state.mushafPages) {
    _priorRevealDoneForPage = targetPage;
    const pageData = state.mushafPages[targetPage - 1];
    const pageVerses = getPageVerses(pageData);
    const matchKey = `${msg.surah}:${msg.ayah}`;
    const priorRevealed: string[] = [];
    for (const vk of pageVerses) {
      if (vk === matchKey) break;
      if (!state.revealedVerses.has(vk)) priorRevealed.push(vk);
      state.revealedVerses.add(vk);
    }
    if (priorRevealed.length > 0) {
      console.log(`[MUSHAF] First match on page — revealing ${priorRevealed.length} prior verses:`, priorRevealed.join(", "));
    }
  }

  // Navigate (re-renders with updated revealedVerses including priors)
  if (isNewPage) {
    await navigateToMushafPage(targetPage);
  } else if (isFirstOnPage && state.mushafPages) {
    // Already on page but first match — reveal priors in DOM
    const pageData = state.mushafPages[targetPage - 1];
    const pageVerses = getPageVerses(pageData);
    const matchKey = `${msg.surah}:${msg.ayah}`;
    for (const vk of pageVerses) {
      if (vk === matchKey) break;
      const [s, a] = vk.split(":");
      mushafRevealVerse($mushafPage, parseInt(s), parseInt(a));
    }
  }

  // DON'T reveal the current verse fully — let word_progress reveal words
  // progressively as the tracker confirms each word. Only prior verses are
  // fully revealed. The current verse shows word-by-word in practice mode.

  console.log(
    `[MUSHAF] State: page=${state.currentMushafPage}, revealed=${state.revealedVerses.size} verses, wordTracked=${_wordTrackedVerses.size}, practice=${state.practiceMode}`,
  );
}

// Mushaf word progress accumulator (same pattern as flowing mode)
// IMPORTANT: This set is monotonic — indices are only ever added, never removed,
// for the lifetime of a single verse. It is only cleared on verse transition.
// NOTE: indices stored here are MUSHAF-space (0-indexed into the DOM words for
// the verse, with bismillah offset already subtracted for ayah 1 verses).
let _mushafMatchedWords = new Set<number>();
let _mushafTrackingKey = "";
// Total word count for the current tracking verse in MUSHAF-space
let _mushafTrackingTotal = 0;
// Bismillah word offset: for ayah 1 of surahs with separate basmala, the tracker
// sends indices into text_words which includes bismillah, but the mushaf DOM does
// not include those words (they are on a separate basmala line). This offset is
// subtracted from tracker indices to align them with the mushaf DOM.
let _mushafBismillahOffset = 0;
// Track error words — blocks progression past them (indices in MUSHAF-space)
let _mushafErrorWords = new Set<number>();
let _mushafErrorKey = "";

// Compute the bismillah offset for a given verse. For ayah 1 of surahs
// with a separate basmala line in the mushaf, the tracker's text_words includes
// the 4 bismillah words, but the mushaf DOM has them on a separate line.
// Returns the number of words to skip (0 or 4).
function _computeBismillahOffset(surah: number, ayah: number): number {
  if (ayah !== 1 || surah === 1 || surah === 9) return 0;
  // Check if this page has a basmala line for this surah
  if (!state.mushafPages) return 0;
  const targetPage = getPageForVerse(surah, ayah);
  if (!targetPage) return 0;
  const pageData = state.mushafPages[targetPage - 1];
  const hasBasmala = pageData.lines.some((l) => l.type === "basmala");
  return hasBasmala ? 4 : 0;
}

// Handle word progress in mushaf mode — reveal words one at a time
async function handleMushafWordProgress(msg: WordProgressMessage): Promise<void> {
  const targetPage = getPageForVerse(msg.surah, msg.ayah);

  if (!targetPage) {
    console.warn(`[WORD] No page for ${msg.surah}:${msg.ayah}`);
    return;
  }
  if (targetPage !== state.currentMushafPage) {
    // Cross-page verse continuation: auto-navigate to the correct page.
    // This handles the case where verse N+1 starts on the next page.
    console.log(
      `[WORD] ${msg.surah}:${msg.ayah} word ${msg.word_index}/${msg.total_words} — auto-navigating p${state.currentMushafPage} → p${targetPage}`,
    );
    await navigateToMushafPage(targetPage);
  }

  // Accumulate matched indices across events for the same verse.
  // The accumulator is MONOTONIC: indices are only added, never removed.
  // It is cleared on verse transition (in handleMushafVerseMatch or here on key change).
  const key = `${msg.surah}:${msg.ayah}`;
  if (key !== _mushafTrackingKey) {
    _mushafMatchedWords = new Set<number>();
    _mushafTrackingKey = key;
    // Compute bismillah offset: tracker indices include bismillah words for ayah 1,
    // but mushaf DOM has them on a separate basmala line.
    _mushafBismillahOffset = _computeBismillahOffset(msg.surah, msg.ayah);
    _mushafTrackingTotal = msg.total_words - _mushafBismillahOffset;
  }
  // Monotonic add: only add new indices, never recreate the set.
  // Apply bismillah offset to convert tracker-space indices to mushaf-space indices.
  // Skip any indices that fall within the bismillah range (they map to the basmala line).
  for (const idx of msg.matched_indices) {
    const mushafIdx = idx - _mushafBismillahOffset;
    if (mushafIdx >= 0) {
      _mushafMatchedWords.add(mushafIdx);
    }
  }

  // Track that this verse had actual word progress (used by verse_match to decide reveals)
  _wordTrackedVerses.add(key);
  const accumulated = Array.from(_mushafMatchedWords).sort((a, b) => a - b);

  // Clear errors for this verse when new word progress comes in (user retrying)
  if (_mushafErrorKey === key && _mushafErrorWords.size > 0) {
    // Only clear errors for words that are now matched (in mushaf-space)
    for (const idx of msg.matched_indices) {
      const mushafIdx = idx - _mushafBismillahOffset;
      if (mushafIdx >= 0) {
        _mushafErrorWords.delete(mushafIdx);
      }
    }
    if (_mushafErrorWords.size === 0) {
      mushafClearErrors($mushafPage);
    }
  }

  // Build contiguous progress from word 0 forward.
  // Follow confirmed words, allowing small gaps (1-2 words = alignment noise).
  // Stop at gaps of 3+ unmatched words, or if a gap has no matched word after it
  // within the tolerance window (prevents spurious gap extension at the frontier).
  const totalMushafWords = msg.total_words - _mushafBismillahOffset;
  const allConfirmed = new Set([..._mushafMatchedWords, ..._mushafErrorWords]);
  let contiguousMax = -1;
  let gapCount = 0;
  for (let i = 0; i < totalMushafWords; i++) {
    if (allConfirmed.has(i)) {
      contiguousMax = i;
      gapCount = 0;
    } else {
      gapCount++;
      if (gapCount > 2) break; // stop at gaps of 3+
      // Small gap — only extend if a confirmed word follows within the window
      if (contiguousMax >= 0) {
        let hasBridge = false;
        for (let k = i + 1; k <= i + (3 - gapCount) && k < totalMushafWords; k++) {
          if (allConfirmed.has(k)) {
            hasBridge = true;
            break;
          }
        }
        if (hasBridge) {
          contiguousMax = i;
        } else {
          break;
        }
      }
    }
  }

  // Detect skipped words (gaps) — these are likely misreads
  // e.g., accumulated=[0,1,2,5,6] with contiguousMax=2 → words 3,4 were skipped
  // Suppress during verse transitions (3s grace) to avoid false positives from stale audio.
  // Also require contiguousMax >= 2 (at least 3 words matched contiguously) to avoid
  // false positives at the very start of a verse where alignment is still settling.
  const beyondContiguous = accumulated.filter((i) => i > contiguousMax + 1);
  const isRecentTransition = Date.now() - state.lastVerseTransitionTime < 3000;
  if (beyondContiguous.length >= 2 && contiguousMax >= 2 && !isRecentTransition) {
    const firstBeyond = beyondContiguous[0];
    const skippedIndices: number[] = [];
    for (let i = contiguousMax + 1; i < firstBeyond; i++) {
      if (!_mushafMatchedWords.has(i) && !_mushafErrorWords.has(i)) {
        skippedIndices.push(i);
      }
    }
    if (skippedIndices.length > 0) {
      _mushafErrorKey = key;
      for (const idx of skippedIndices) {
        _mushafErrorWords.add(idx);
      }
      mushafHighlightErrors($mushafPage, msg.surah, msg.ayah, Array.from(_mushafErrorWords));
      console.log(
        `%c[MISREAD] ${msg.surah}:${msg.ayah} words [${skippedIndices.join(",")}] skipped — marking as errors`,
        "color: #ff6b6b; font-weight: bold",
      );
    }
  }

  // Look up Arabic word text from page data for logging
  const verseWordMap: Record<number, string> = {};
  if (state.mushafPages) {
    const pageData = state.mushafPages[targetPage - 1];
    for (const line of pageData.lines) {
      if (line.type === "text" && line.words) {
        for (const w of line.words) {
          const [s, a, widx] = w.location.split(":");
          if (s === String(msg.surah) && a === String(msg.ayah)) {
            verseWordMap[parseInt(widx) - 1] = w.word; // 0-indexed
          }
        }
      }
    }
  }

  // Build spoken text so far (contiguous words from 0, skip error words)
  const spokenWords: string[] = [];
  for (let i = 0; i <= contiguousMax; i++) {
    if (!_mushafErrorWords.has(i)) {
      spokenWords.push(verseWordMap[i] || `[${i}]`);
    }
  }

  const mushafWordIdx = msg.word_index - _mushafBismillahOffset;
  const currentWord = verseWordMap[mushafWordIdx >= 0 ? mushafWordIdx : msg.word_index] || "";
  console.log(
    `[WORD] ${msg.surah}:${msg.ayah} word ${msg.word_index}/${msg.total_words}` +
    (_mushafBismillahOffset > 0 ? ` (mushaf ${mushafWordIdx}/${_mushafTrackingTotal}, bsmOffset=${_mushafBismillahOffset})` : "") +
    (currentWord ? ` "${currentWord}"` : "") +
    ` new=[${msg.matched_indices.join(",")}] accumulated=[${accumulated.join(",")}] contiguous=0..${contiguousMax}` +
    (_mushafErrorWords.size > 0 ? ` errors=[${Array.from(_mushafErrorWords).join(",")}]` : ""),
  );
  if (spokenWords.length > 0) {
    console.log(
      `%c[READING] ${spokenWords.join(" ")}`,
      "color: #7ec8e3; font-size: 14px",
    );
  }

  // Highlight using accumulated mushaf-space indices (not just this event's)
  mushafHighlightWord($mushafPage, msg.surah, msg.ayah, accumulated);

  // Mark verse as revealed only when ALL mushaf words are matched
  if (_mushafTrackingTotal > 0 && _mushafMatchedWords.size >= _mushafTrackingTotal) {
    state.revealedVerses.add(key);
    console.log(
      `%c[VERSE_COMPLETE] ${msg.surah}:${msg.ayah} — all ${_mushafTrackingTotal} mushaf words matched`,
      "color: #7a9a5a; font-weight: bold",
    );
  }
}

// ---------------------------------------------------------------------------
// Forced Alignment message handlers
// ---------------------------------------------------------------------------

// Track FA state for mushaf word highlighting
let _faTrackingKey = "";
let _faConfirmedWords = new Set<number>();
let _faBismillahOffset = 0;

function handleMushafWordAligned(msg: WordAlignedMessage): void {
  const targetPage = getPageForVerse(msg.surah, msg.ayah);
  if (!targetPage || targetPage !== state.currentMushafPage) return;

  const key = `${msg.surah}:${msg.ayah}`;
  if (key !== _faTrackingKey) {
    _faConfirmedWords = new Set<number>();
    _faTrackingKey = key;
    _faBismillahOffset = _computeBismillahOffset(msg.surah, msg.ayah);
  }

  // Add confirmed words (apply bismillah offset to convert to mushaf-space)
  for (const idx of msg.cumulative_indices) {
    const mushafIdx = idx - _faBismillahOffset;
    if (mushafIdx >= 0) {
      _faConfirmedWords.add(mushafIdx);
    }
  }

  // Track that this verse had progress
  _wordTrackedVerses.add(key);

  // Get words for this verse from page data for confidence coloring
  const allMushafWords = $mushafPage.querySelectorAll<HTMLElement>(
    `.mp-word[data-surah="${msg.surah}"][data-ayah="${msg.ayah}"]`,
  );

  // The msg.word_index is in tracker-space; convert to mushaf-space
  const mushafWordIdx = msg.word_index - _faBismillahOffset;

  // Highlight all confirmed words up to current position
  for (let i = 0; i < allMushafWords.length; i++) {
    const w = allMushafWords[i];
    if (_faConfirmedWords.has(i)) {
      w.classList.remove("mp-word--hidden");
      w.classList.add("mp-word--spoken");

      // Clear previous confidence classes
      w.classList.remove("mp-word--fa-good", "mp-word--fa-warn", "mp-word--fa-error");

      // Apply confidence-based color only for the current word
      if (i === mushafWordIdx) {
        w.classList.add("mp-word--current");
        if (msg.confidence >= FA_CONFIDENCE_GOOD) {
          w.classList.add("mp-word--fa-good");
        } else if (msg.confidence >= FA_CONFIDENCE_WARN) {
          w.classList.add("mp-word--fa-warn");
        } else {
          w.classList.add("mp-word--fa-error");
        }
      } else {
        w.classList.remove("mp-word--current");
      }
    }
  }

  // Look up Arabic word text for logging
  let wordText = "";
  if (state.mushafPages && targetPage > 0) {
    const pageData = state.mushafPages[targetPage - 1];
    for (const line of pageData.lines) {
      if (line.type === "text" && line.words) {
        for (const w of line.words) {
          const [s, a, widx] = w.location.split(":");
          if (s === String(msg.surah) && a === String(msg.ayah) && parseInt(widx) - 1 === mushafWordIdx) {
            wordText = w.word;
          }
        }
      }
    }
  }

  console.log(
    `%c[FA] ${msg.surah}:${msg.ayah} word ${msg.word_index}/${msg.total_words}` +
    (wordText ? ` "${wordText}"` : "") +
    ` conf=${(msg.confidence * 100).toFixed(1)}%` +
    ` words=[${msg.cumulative_indices.join(",")}]`,
    msg.confidence >= FA_CONFIDENCE_GOOD ? "color: #7a9a5a; font-weight: bold" :
    msg.confidence >= FA_CONFIDENCE_WARN ? "color: #C2A05B; font-weight: bold" :
    "color: #D64545; font-weight: bold",
  );
}

async function handleMushafVerseComplete(msg: VerseCompleteMessage): Promise<void> {
  const key = `${msg.surah}:${msg.ayah}`;
  state.revealedVerses.add(key);
  _wordTrackedVerses.add(key);

  // Clear current highlights on completed verse
  const words = $mushafPage.querySelectorAll<HTMLElement>(
    `.mp-word[data-surah="${msg.surah}"][data-ayah="${msg.ayah}"]`,
  );
  for (const w of words) {
    w.classList.remove("mp-word--hidden", "mp-word--current");
    w.classList.add("mp-word--spoken", "mp-word--fa-good");
  }

  console.log(
    `%c[FA VERSE COMPLETE] ${msg.surah}:${msg.ayah} score=${(msg.overall_score * 100).toFixed(1)}% → ${msg.next_surah}:${msg.next_ayah}`,
    "color: #7a9a5a; font-weight: bold; font-size: 14px",
  );

  // Navigate to next verse's page if needed
  const nextPage = getPageForVerse(msg.next_surah, msg.next_ayah);
  if (nextPage && nextPage !== state.currentMushafPage) {
    await navigateToMushafPage(nextPage);
  }

  // Reset FA tracking for next verse
  _faConfirmedWords = new Set<number>();
  _faTrackingKey = `${msg.next_surah}:${msg.next_ayah}`;
  _faBismillahOffset = _computeBismillahOffset(msg.next_surah, msg.next_ayah);
}

// ---------------------------------------------------------------------------
// Verse rendering
// ---------------------------------------------------------------------------
const WAQF_MARKS = new Set([
  "\u06D6", "\u06D7", "\u06D8", "\u06D9", "\u06DA", "\u06DB", "\u06DC",
]);

function isWaqfToken(token: string): boolean {
  return token.length <= 2 && [...token].every((c) => WAQF_MARKS.has(c));
}

interface WordToken {
  text: string;
  isRealWord: boolean;
}

function splitUthmaniWords(text: string): WordToken[] {
  const raw = text.split(/\s+/).filter((w) => w.length > 0);
  const result: WordToken[] = [];

  for (const token of raw) {
    if (isWaqfToken(token) && result.length > 0) {
      result[result.length - 1].text += " " + token;
    } else {
      result.push({ text: token, isRealWord: true });
    }
  }

  return result;
}

const BISMILLAH_WORD_COUNT = 4;
const BISMILLAH_BASE = "بسم الله الرحمن الرحيم";

function stripDiacritics(s: string): string {
  return s.replace(/[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]/g, "");
}

function normalizeArabic(s: string): string {
  return stripDiacritics(s)
    .replace(/\u0671/g, "\u0627")  // ٱ (alef wasla) → ا
    .replace(/[\u0622\u0623\u0625]/g, "\u0627"); // أ إ آ → ا
}

function startsWithBismillah(text: string): boolean {
  const normalized = normalizeArabic(text);
  const base = normalizeArabic(BISMILLAH_BASE);
  return normalized.startsWith(base);
}

function createVerseGroupElement(group: VerseGroup): HTMLElement {
  const el = document.createElement("div");
  el.className = "verse-group";
  el.setAttribute("data-surah", String(group.surah));

  // Ornate surah header cartouche
  const header = document.createElement("div");
  header.className = "surah-header";

  const ornL = document.createElement("span");
  ornL.className = "surah-header-ornament";
  ornL.textContent = "\uFD3E"; // ﴾

  const content = document.createElement("div");
  content.className = "surah-header-content";

  const arName = document.createElement("div");
  arName.className = "surah-header-ar";
  arName.textContent = group.surahName;

  const enName = document.createElement("div");
  enName.className = "surah-header-en";
  enName.textContent = group.surahNameEn;

  content.appendChild(arName);
  content.appendChild(enName);

  const ornR = document.createElement("span");
  ornR.className = "surah-header-ornament";
  ornR.textContent = "\uFD3F"; // ﴿

  header.appendChild(ornL);
  header.appendChild(content);
  header.appendChild(ornR);
  el.appendChild(header);

  const hasBismillah =
    group.surah !== 1 &&
    group.surah !== 9 &&
    startsWithBismillah(group.verses[0]?.text_uthmani ?? "");
  if (hasBismillah) {
    const words = group.verses[0].text_uthmani.split(/\s+/);
    const bsmText = words.slice(0, BISMILLAH_WORD_COUNT).join(" ");
    const bsmEl = document.createElement("div");
    bsmEl.className = "bismillah";
    bsmEl.dir = "rtl";
    bsmEl.lang = "ar";
    bsmEl.textContent = bsmText;
    el.appendChild(bsmEl);

    const bsmSep = document.createElement("div");
    bsmSep.className = "bismillah-separator";
    el.appendChild(bsmSep);
  }

  const body = document.createElement("div");
  body.className = "verse-body";
  body.dir = "rtl";
  body.lang = "ar";

  for (const v of group.verses) {
    const verseEl = document.createElement("span");
    verseEl.className = "verse verse--upcoming";
    verseEl.setAttribute("data-ayah", String(v.ayah));

    const allWords = splitUthmaniWords(v.text_uthmani);
    const skipBsm = hasBismillah && v.ayah === 1;
    const startIdx = skipBsm ? BISMILLAH_WORD_COUNT : 0;

    const textEl = document.createElement("span");
    textEl.className = "verse-text";
    for (let i = startIdx; i < allWords.length; i++) {
      const wordEl = document.createElement("span");
      wordEl.className = "word";
      wordEl.setAttribute("data-word-idx", String(i));
      wordEl.textContent = allWords[i].text;
      textEl.appendChild(wordEl);
      if (i < allWords.length - 1) {
        textEl.appendChild(document.createTextNode(" "));
      }
    }
    verseEl.appendChild(textEl);

    const markerEl = document.createElement("span");
    markerEl.className = "verse-marker";
    markerEl.textContent = ` \u06DD${toArabicNum(v.ayah)} `;
    verseEl.appendChild(markerEl);

    body.appendChild(verseEl);
  }

  el.appendChild(body);
  return el;
}

function updateVerseHighlight(group: VerseGroup, newAyah: number): void {
  const el = group.element;
  const oldAyah = group.currentAyah;

  const verses = el.querySelectorAll<HTMLElement>(".verse");
  for (const verseEl of verses) {
    const ayah = parseInt(verseEl.getAttribute("data-ayah") || "0");
    if (ayah === newAyah) {
      verseEl.className = "verse verse--active";
    } else if (ayah <= newAyah && (ayah >= oldAyah || ayah < oldAyah)) {
      if (
        verseEl.classList.contains("verse--active") ||
        (ayah > oldAyah && ayah < newAyah) ||
        ayah <= oldAyah
      ) {
        verseEl.className = "verse verse--recited";
      }
    }
  }

  group.currentAyah = newAyah;
  scrollToActiveVerse();
}

function scrollToActiveVerse(): void {
  const active = document.querySelector(".verse--active");
  if (active) {
    active.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

// ---------------------------------------------------------------------------
// Message handlers
// ---------------------------------------------------------------------------
async function handleVerseMatch(msg: VerseMatchMessage): Promise<void> {
  $rawTranscript.textContent = "";
  $rawTranscript.classList.remove("visible");

  state.lastModelPrediction = { surah: msg.surah, ayah: msg.ayah, confidence: msg.confidence };

  if (!state.hasFirstMatch) {
    state.hasFirstMatch = true;
    $indicator.classList.add("has-verses");
  }

  const lastGroup = state.groups[state.groups.length - 1];

  if (lastGroup && lastGroup.surah === msg.surah) {
    updateVerseHighlight(lastGroup, msg.ayah);
    return;
  }

  if (lastGroup) {
    lastGroup.element.classList.add("verse-group--exiting");
    const oldEl = lastGroup.element;
    setTimeout(() => oldEl.remove(), 400);
  }

  const surahData = await fetchSurah(msg.surah);

  const group: VerseGroup = {
    surah: msg.surah,
    surahName: surahData.surah_name,
    surahNameEn: surahData.surah_name_en,
    currentAyah: 0,
    verses: surahData.verses,
    element: document.createElement("div"),
  };
  group.element = createVerseGroupElement(group);
  state.groups.push(group);
  $verses.appendChild(group.element);

  updateVerseHighlight(group, msg.ayah);
}

let _matchedWordIndices = new Set<number>();
let _trackingKey = "";

function handleWordProgress(msg: WordProgressMessage): void {
  const lastGroup = state.groups[state.groups.length - 1];
  if (!lastGroup || lastGroup.surah !== msg.surah) return;

  const verseEl = lastGroup.element.querySelector<HTMLElement>(
    `.verse[data-ayah="${msg.ayah}"]`,
  );
  if (!verseEl) return;

  if (!verseEl.classList.contains("verse--active")) {
    updateVerseHighlight(lastGroup, msg.ayah);
  }

  const key = `${msg.surah}:${msg.ayah}`;
  if (key !== _trackingKey) {
    _matchedWordIndices = new Set<number>();
    _trackingKey = key;
  }

  for (const idx of msg.matched_indices) {
    _matchedWordIndices.add(idx);
  }

  let contiguousMax = -1;
  for (let i = 0; i <= msg.total_words; i++) {
    if (_matchedWordIndices.has(i)) {
      contiguousMax = i;
    } else {
      break;
    }
  }

  const wordEls = verseEl.querySelectorAll<HTMLElement>(".word");
  for (const wordEl of wordEls) {
    const idx = parseInt(wordEl.getAttribute("data-word-idx") || "-1");
    wordEl.classList.remove("word--current");
    if (idx <= contiguousMax) {
      wordEl.classList.add("word--spoken");
      if (idx === contiguousMax) {
        wordEl.classList.add("word--current");
      }
    }
  }
}

function handleRawTranscript(msg: RawTranscriptMessage): void {
  $rawTranscript.textContent = msg.text;
  $rawTranscript.classList.add("visible");

  // Update last raw transcript for the candidate status bar
  state.lastRawTranscript = msg.text;

  // If no candidates are showing yet, render the "listening" state in the bar
  if (!$candidateList.querySelector(".candidate-item")) {
    renderCandidateListening(msg.text);
  } else {
    // Update the transcript line inside the existing candidate bar
    const transcriptEl = $candidateList.querySelector(".cbar-transcript");
    if (transcriptEl) {
      transcriptEl.textContent = `"${msg.text}"`;
    }
  }

  // Algorithm view: update transcript and narrowing cascade
  if (state.algorithmMode) {
    updateAlgoTranscript(msg.text);
    updateAlgoNarrowing(msg.text);
  }
}

/** Render the "Listening..." state in the candidate status bar */
function renderCandidateListening(text: string): void {
  if (!state.isActive) return;
  // Clear any pending match fade
  if (state.candidateMatchFadeTimer) {
    clearTimeout(state.candidateMatchFadeTimer);
    state.candidateMatchFadeTimer = null;
  }

  $candidateList.innerHTML = "";
  $candidateList.className = "cbar cbar--listening";

  const row = document.createElement("div");
  row.className = "cbar-listening-row";

  const icon = document.createElement("span");
  icon.className = "cbar-icon";
  icon.textContent = "\uD83C\uDFA4"; // microphone emoji

  const label = document.createElement("span");
  label.className = "cbar-label";
  label.textContent = "Listening...";

  row.appendChild(icon);
  row.appendChild(label);
  $candidateList.appendChild(row);

  if (text) {
    const transcript = document.createElement("div");
    transcript.className = "cbar-transcript";
    transcript.dir = "rtl";
    transcript.lang = "ar";
    transcript.textContent = `"${text}"`;
    $candidateList.appendChild(transcript);
  }

  $candidateList.classList.add("visible");
}

// ---------------------------------------------------------------------------
// Live narrowing (candidate list)
// ---------------------------------------------------------------------------
function handleCandidateList(msg: CandidateListMessage): void {
  if (!state.isActive) return;

  // Clear any pending match fade timer (we have new candidates)
  if (state.candidateMatchFadeTimer) {
    clearTimeout(state.candidateMatchFadeTimer);
    state.candidateMatchFadeTimer = null;
  }

  const container = $candidateList;
  container.innerHTML = "";

  if (msg.candidates.length === 0) {
    // No candidates -- show listening state with last transcript
    if (state.lastRawTranscript) {
      renderCandidateListening(state.lastRawTranscript);
    }
    return;
  }

  // Narrowing state -- show top 3 candidates
  container.className = "cbar cbar--narrowing";

  const topScore = msg.candidates[0].score;
  const shown = msg.candidates.slice(0, 3);

  // Header row
  const header = document.createElement("div");
  header.className = "cbar-header";

  const icon = document.createElement("span");
  icon.className = "cbar-icon";
  icon.textContent = "\uD83D\uDCD6"; // open book emoji

  const label = document.createElement("span");
  label.className = "cbar-label";
  label.textContent = `${msg.candidates.length} candidate${msg.candidates.length !== 1 ? "s" : ""}`;

  header.appendChild(icon);
  header.appendChild(label);
  container.appendChild(header);

  // Candidate rows
  for (const c of shown) {
    const item = document.createElement("div");
    item.className = "candidate-item";

    // Highlight confidence relative to top
    const relScore = topScore > 0 ? c.score / topScore : 0;
    if (relScore >= 0.97) {
      item.classList.add("candidate--top");
    } else if (relScore >= 0.85) {
      item.classList.add("candidate--likely");
    }

    // Make tappable -- navigate to this verse
    item.style.cursor = "pointer";
    item.addEventListener("click", () => {
      handleCandidateTap(c.surah, c.ayah, c.surah_name_en);
    });

    // Left side: label + score bar
    const info = document.createElement("div");
    info.className = "candidate-info";

    const labelEl = document.createElement("span");
    labelEl.className = "candidate-label";
    labelEl.textContent = `${c.surah_name_en} ${c.surah}:${c.ayah}`;

    const scoreEl = document.createElement("span");
    scoreEl.className = "candidate-score";
    const pct = Math.round(c.score * 100);
    scoreEl.textContent = `${pct}%`;

    info.appendChild(labelEl);
    info.appendChild(scoreEl);
    item.appendChild(info);

    // Progress bar
    const barTrack = document.createElement("div");
    barTrack.className = "candidate-bar-track";

    const barFill = document.createElement("div");
    barFill.className = "candidate-bar-fill";
    barFill.style.width = `${pct}%`;
    barTrack.appendChild(barFill);
    item.appendChild(barTrack);

    container.appendChild(item);
  }

  // Show transcript at bottom if available
  const transcriptText = msg.transcript || state.lastRawTranscript;
  if (transcriptText) {
    const transcript = document.createElement("div");
    transcript.className = "cbar-transcript";
    transcript.dir = "rtl";
    transcript.lang = "ar";
    transcript.textContent = `"${transcriptText}"`;
    container.appendChild(transcript);
  }

  container.classList.add("visible");

  // Algorithm view: update candidates
  if (state.algorithmMode && msg.candidates.length > 0) {
    updateAlgoCandidates(msg.candidates);
    // Also update narrowing cascade from transcript if available
    const transcript = msg.transcript || state.lastRawTranscript;
    if (transcript) {
      updateAlgoNarrowing(transcript);
    }
  }
}

/** Handle tapping a candidate -- navigate to that verse's page and enter tracking */
async function handleCandidateTap(surah: number, ayah: number, surahNameEn: string): Promise<void> {
  if (!state.mushafDataReady) return;

  const targetPage = getPageForVerse(surah, ayah);
  if (!targetPage) return;

  console.log(`[CANDIDATE_TAP] User selected ${surahNameEn} ${surah}:${ayah} → page ${targetPage}`);

  // Update tracking state as if this were a verse_match
  state.lastModelPrediction = { surah, ayah, confidence: 1.0 };

  // Navigate to the page
  if (targetPage !== state.currentMushafPage) {
    await navigateToMushafPage(targetPage);
  }

  // Clear candidate bar with matched state
  renderCandidateMatched(surahNameEn, surah, ayah, "");

  // Tell the worker to lock onto this verse
  state.worker?.postMessage({ type: "hint_verse", surah, ayah });
}

/** Render the "Matched" confirmation state in the candidate bar */
function renderCandidateMatched(surahNameEn: string, surah: number, ayah: number, textPreview: string): void {
  // Clear any existing fade timer
  if (state.candidateMatchFadeTimer) {
    clearTimeout(state.candidateMatchFadeTimer);
    state.candidateMatchFadeTimer = null;
  }

  $candidateList.innerHTML = "";
  $candidateList.className = "cbar cbar--matched";

  const row = document.createElement("div");
  row.className = "cbar-matched-row";

  const check = document.createElement("span");
  check.className = "cbar-icon cbar-icon--check";
  check.textContent = "\u2713"; // checkmark

  const label = document.createElement("span");
  label.className = "cbar-label";
  label.textContent = `${surahNameEn} ${surah}:${ayah}`;

  row.appendChild(check);
  row.appendChild(label);

  if (textPreview) {
    const preview = document.createElement("span");
    preview.className = "cbar-matched-preview";
    preview.dir = "rtl";
    preview.lang = "ar";
    // Truncate long previews
    preview.textContent = textPreview.length > 40 ? textPreview.slice(0, 40) + "\u2026" : textPreview;
    row.appendChild(preview);
  }

  $candidateList.appendChild(row);
  $candidateList.classList.add("visible");

  // Fade out after 2 seconds
  state.candidateMatchFadeTimer = setTimeout(() => {
    $candidateList.classList.add("cbar--fading");
    setTimeout(() => {
      $candidateList.classList.remove("visible", "cbar--fading");
      $candidateList.innerHTML = "";
      $candidateList.className = "";
    }, 400);
  }, 2000);
}

// ---------------------------------------------------------------------------
// Algorithm View — main-thread QuranDB + visualization
// ---------------------------------------------------------------------------

/** Load a QuranDB instance on the main thread for algorithm view queries */
async function loadAlgorithmDB(): Promise<void> {
  if (state.algorithmDBReady) return;

  await loadQuranData();
  const db = new QuranDB(state.quranData!);

  // Load disambiguation map
  try {
    const res = await fetch("/ambiguity-compact.json");
    if (res.ok) {
      const disambigData = await res.json();
      db.loadDisambiguationMap(disambigData);
    }
  } catch {
    console.warn("[ALGO_VIEW] ambiguity-compact.json not available");
  }

  state.algorithmDB = db;
  state.algorithmDBReady = true;
}

/** Toggle algorithm view on/off */
function toggleAlgorithmView(): void {
  state.algorithmMode = !state.algorithmMode;
  $app.classList.toggle("algorithm-mode", state.algorithmMode);

  if (state.algorithmMode) {
    $algorithmView.hidden = false;
    $mushafContainer.querySelector(".mushaf-page-wrap")?.classList.add("av-hidden");
    $mushafContainer.querySelector(".mushaf-nav")?.classList.add("av-hidden");
    clearAlgorithmView();
  } else {
    $algorithmView.hidden = true;
    $mushafContainer.querySelector(".mushaf-page-wrap")?.classList.remove("av-hidden");
    $mushafContainer.querySelector(".mushaf-nav")?.classList.remove("av-hidden");
  }
}

/** Clear all algorithm view state for a new cycle */
function clearAlgorithmView(): void {
  state.narrowingHistory = [];
  state.algorithmIdentified = null;
  state.algorithmCycleStart = Date.now();
  $avTranscriptText.textContent = "";
  $avNarrowingCascade.innerHTML = "";
  $avCandidateList.innerHTML = "";
  $avDisambigInfo.textContent = "";
  $avIdentified.hidden = true;
}

/** Update the algorithm view transcript display */
function updateAlgoTranscript(text: string): void {
  if (!state.algorithmMode) return;
  $avTranscriptText.textContent = text || "";

  // When we get a new transcript, start the cycle timer if not already started
  if (state.algorithmCycleStart === 0) {
    state.algorithmCycleStart = Date.now();
  }
}

/** Normalize Arabic text (same as QuranDB) for prefix trie queries */
function normalizeForTrie(text: string): string {
  let t = text.replace(/\u2581/g, " ");
  t = t.replace(/\uFEFF/g, "");
  t = t.replace(/[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]/g, "");
  t = t.replace(/[أإآٱ]/g, "ا");
  t = t.replace(/ة/g, "ه");
  t = t.replace(/ى/g, "ي");
  t = t.replace(/ـ/g, "");
  t = t.replace(/[،؟.!:]/g, "");
  t = t.replace(/\s+/g, " ").trim();
  return t;
}

/** Update the narrowing cascade visualization */
function updateAlgoNarrowing(transcript: string): void {
  if (!state.algorithmMode || !state.algorithmDB) return;

  const normalized = normalizeForTrie(transcript);
  const words = normalized.split(" ").filter(Boolean);
  if (words.length === 0) return;

  const cascade = state.algorithmDB.prefixNarrowingCascade(words);
  if (cascade.length === 0) return;

  // Only update if cascade has changed
  const cascadeChanged = cascade.length !== state.narrowingHistory.length ||
    cascade.some((c, i) => state.narrowingHistory[i]?.count !== c.count);
  if (!cascadeChanged) return;
  state.narrowingHistory = cascade;

  // Rebuild cascade display
  $avNarrowingCascade.innerHTML = "";

  for (let i = 0; i < cascade.length; i++) {
    const step = cascade[i];
    const row = document.createElement("div");
    row.className = "av-narrow-row";
    if (i === cascade.length - 1) row.classList.add("av-narrow-row--latest");

    // Animate new rows sliding in
    row.style.animationDelay = `${i * 50}ms`;

    const wordNum = document.createElement("span");
    wordNum.className = "av-narrow-word-num";
    wordNum.textContent = `Word ${i + 1}:`;

    const wordText = document.createElement("span");
    wordText.className = "av-narrow-word-text";
    wordText.dir = "rtl";
    wordText.lang = "ar";
    wordText.textContent = `"${step.word}"`;

    const arrow = document.createElement("span");
    arrow.className = "av-narrow-arrow";
    arrow.textContent = "\u2192";

    const count = document.createElement("span");
    count.className = "av-narrow-count";
    if (step.count === 1) {
      count.classList.add("av-narrow-count--unique");
      count.textContent = "1 candidate \u2713";
    } else {
      count.textContent = `${step.count} candidates`;
    }

    row.appendChild(wordNum);
    row.appendChild(wordText);
    row.appendChild(arrow);
    row.appendChild(count);
    $avNarrowingCascade.appendChild(row);
  }

  // Auto-scroll to bottom of cascade
  $avNarrowingCascade.scrollTop = $avNarrowingCascade.scrollHeight;
}

/** Update the top candidates visualization */
function updateAlgoCandidates(candidates: { surah: number; ayah: number; score: number; surah_name_en: string; text_preview: string }[]): void {
  if (!state.algorithmMode) return;

  $avCandidateList.innerHTML = "";

  if (candidates.length === 0) return;

  const topScore = candidates[0].score;
  const shown = candidates.slice(0, 5);

  for (let i = 0; i < shown.length; i++) {
    const c = shown[i];
    const row = document.createElement("div");
    row.className = "av-candidate-row";
    if (i === 0) row.classList.add("av-candidate-row--top");

    const ref = document.createElement("span");
    ref.className = "av-candidate-ref";
    ref.textContent = `${c.surah_name_en} ${c.surah}:${c.ayah}`;

    const bar = document.createElement("div");
    bar.className = "av-candidate-bar";

    const fill = document.createElement("div");
    fill.className = "av-candidate-bar-fill";
    const pct = Math.round(c.score * 100);
    fill.style.width = `${pct}%`;
    bar.appendChild(fill);

    const score = document.createElement("span");
    score.className = "av-candidate-score";
    score.textContent = `${pct}%`;

    const indicator = document.createElement("span");
    indicator.className = "av-candidate-indicator";
    if (i === 0 && c.score > 0.7) {
      indicator.textContent = "\u2190 most likely";
      indicator.classList.add("av-candidate-indicator--likely");
    }

    row.appendChild(ref);
    row.appendChild(bar);
    row.appendChild(score);
    row.appendChild(indicator);
    $avCandidateList.appendChild(row);
  }

  // Update disambiguation info for the top candidate
  updateAlgoDisambig(shown[0].surah, shown[0].ayah);
}

/** Update disambiguation info for a verse */
function updateAlgoDisambig(surah: number, ayah: number): void {
  if (!state.algorithmMode || !state.algorithmDB) return;

  const db = state.algorithmDB;
  const disambigLen = db.getDisambiguationLength(surah, ayah);
  const isAmbiguous = db.isAmbiguousInIsolation(surah, ayah);
  const entry = db.getDisambiguationEntry(surah, ayah);

  if (isAmbiguous && entry) {
    // Count how many verses share this opening
    const confuserCount = entry.c.length;
    $avDisambigInfo.innerHTML = "";

    const warning = document.createElement("span");
    warning.className = "av-disambig-warning";
    warning.textContent = "\u26A0\uFE0F";

    const text = document.createElement("span");
    text.textContent = ` This verse shares its opening with ${confuserCount} other verse${confuserCount !== 1 ? "s" : ""} \u2014 needs boundary context`;

    $avDisambigInfo.appendChild(warning);
    $avDisambigInfo.appendChild(text);
    $avDisambigInfo.className = "av-disambig-info av-disambig-info--warning";
  } else if (disambigLen > 0) {
    $avDisambigInfo.innerHTML = "";

    const icon = document.createElement("span");
    icon.className = "av-disambig-icon";
    icon.textContent = "\u2139\uFE0F";

    const text = document.createElement("span");
    text.textContent = ` This verse needs ${disambigLen} word${disambigLen !== 1 ? "s" : ""} to identify (paper average: 3.11)`;

    $avDisambigInfo.appendChild(icon);
    $avDisambigInfo.appendChild(text);
    $avDisambigInfo.className = "av-disambig-info av-disambig-info--info";
  } else {
    $avDisambigInfo.textContent = "";
    $avDisambigInfo.className = "av-disambig-info";
  }
}

/** Show the identified state in algorithm view */
function showAlgoIdentified(msg: VerseMatchMessage): void {
  if (!state.algorithmMode || !state.algorithmDB) return;

  const timeMs = state.algorithmCycleStart > 0
    ? Date.now() - state.algorithmCycleStart
    : 0;

  const db = state.algorithmDB;
  const disambigLen = db.getDisambiguationLength(msg.surah, msg.ayah);
  const wordsNeeded = disambigLen > 0 ? disambigLen : -1;

  state.algorithmIdentified = {
    surah: msg.surah,
    ayah: msg.ayah,
    surahName: msg.surah_name,
    confidence: msg.confidence,
    wordsNeeded,
    timeMs,
    text: msg.verse_text,
  };

  $avIdentified.hidden = false;
  $avIdentifiedRef.textContent = `${msg.surah_name} ${msg.surah}:${msg.ayah}`;

  // Show verse text (truncated for display)
  const displayText = msg.verse_text.length > 120
    ? msg.verse_text.slice(0, 120) + "\u2026"
    : msg.verse_text;
  $avIdentifiedText.textContent = displayText;

  // Stats line
  const confPct = Math.round(msg.confidence * 100);
  const timeSec = (timeMs / 1000).toFixed(1);
  const wordsStr = wordsNeeded > 0 ? `${wordsNeeded}` : "n/a";
  $avIdentifiedStats.textContent = `Confidence: ${confPct}% | Words needed: ${wordsStr} | Time: ${timeSec}s`;
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------
function pushDiagnosticEvent(type: string, data: Record<string, unknown>): void {
  state.diagnosticEvents.push({ timestamp: Date.now(), type, data });
  if (state.diagnosticEvents.length > MAX_DIAGNOSTIC_EVENTS) {
    state.diagnosticEvents.shift();
  }
}

function checkAnomalyAndSend(msg: VerseMatchMessage): void {
  const now = Date.now();

  // Track recent verse matches for rapid switching detection
  state.recentVerseMatches.push({ surah: msg.surah, ayah: msg.ayah, timestamp: now });
  // Keep only last 10 seconds
  state.recentVerseMatches = state.recentVerseMatches.filter(
    (m) => now - m.timestamp < 10_000,
  );

  let trigger: string | null = null;

  // Surah jump: different surah than previous match
  const prev = state.lastModelPrediction;
  if (prev && prev.surah !== msg.surah) {
    trigger = "surah_jump";
  }

  // Rapid switching: 3+ different verses in 10 seconds
  if (!trigger) {
    const unique = new Set(
      state.recentVerseMatches.map((m) => `${m.surah}:${m.ayah}`),
    );
    if (unique.size >= 3) {
      trigger = "rapid_switching";
    }
  }

  if (!trigger) return;

  // Cooldown
  if (now - state.lastDiagnosticSentAt < DIAGNOSTIC_COOLDOWN_MS) return;
  state.lastDiagnosticSentAt = now;

  sendDiagnosticReport(trigger);
}

async function sendDiagnosticReport(trigger: string): Promise<void> {
  try {
    // Build audio WAV from session chunks
    const totalLen = state.sessionAudioChunks.reduce((s, c) => s + c.length, 0);
    const merged = new Float32Array(totalLen);
    let offset = 0;
    for (const chunk of state.sessionAudioChunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }

    // Only send last 30s of audio max
    const maxSamples = 16000 * 30;
    const audioSlice = merged.length > maxSamples ? merged.slice(-maxSamples) : merged;
    const wavBlob = encodeWav(audioSlice, 16000);

    const form = new FormData();
    form.append("audio", wavBlob, "diagnostic.wav");
    form.append("events", JSON.stringify(state.diagnosticEvents));
    form.append("trigger", trigger);

    await fetch("/api/diagnostics", { method: "POST", body: form });
  } catch (err) {
    console.error("Failed to send diagnostic report:", err);
  }
}

// ---------------------------------------------------------------------------
// Worker message handler
// ---------------------------------------------------------------------------
function handleWorkerMessage(msg: WorkerOutbound): void {
  if (msg.type === "loading") {
    $modelStatus.textContent = `Loading model... ${msg.percent}%`;
    $modelStatus.classList.remove("ready");
    $loadingProgress.style.width = `${msg.percent}%`;
    $loadingDetail.textContent = `Downloading model — ${msg.percent}%`;
  } else if (msg.type === "loading_status") {
    $loadingDetail.textContent = msg.message;
  } else if (msg.type === "error") {
    $modelStatus.textContent = "Error";
    console.error("Worker reported error:", msg.message);
    // Show actionable error with retry on the loading screen
    $loadingProgress.style.width = "0%";
    $loadingDetail.innerHTML = "";
    const errText = document.createElement("span");
    errText.textContent = `Failed to load: ${msg.message}. `;
    const retryBtn = document.createElement("button");
    retryBtn.textContent = "Retry";
    retryBtn.style.cssText = "cursor:pointer;text-decoration:underline;background:none;border:none;color:inherit;font:inherit;padding:0;";
    retryBtn.addEventListener("click", () => {
      $loadingDetail.textContent = "Retrying...";
      $loadingProgress.style.width = "0%";
      state.worker?.postMessage({ type: "init" });
    });
    $loadingDetail.appendChild(errText);
    $loadingDetail.appendChild(retryBtn);
  } else if (msg.type === "ready") {
    $modelStatus.textContent = "Model ready";
    $modelStatus.classList.add("ready");
    state.modelReady = true;
    $loadingStatus.hidden = true;
    // Show mushaf directly — no "Begin" screen
    if (state.mushafDataReady) {
      $mushafContainer.hidden = false;
      $verses.hidden = true;
      state.currentMushafPage = 0; // force re-render
      navigateToMushafPage(1);
    }
  } else if (msg.type === "verse_match") {
    // Show confirmed match in candidate bar (fades after 2s)
    renderCandidateMatched(
      msg.surah_name,
      msg.surah,
      msg.ayah,
      msg.verse_text,
    );
    pushDiagnosticEvent("verse_match", {
      surah: msg.surah, ayah: msg.ayah, confidence: msg.confidence,
    });
    checkAnomalyAndSend(msg);
    // Algorithm view: show identified state
    if (state.algorithmMode) {
      showAlgoIdentified(msg);
    }
    // Show "Wrong?" feedback button
    showWrongButton({
      audioChunks: state.sessionAudioChunks,
      transcript: state.lastRawTranscript,
      matched: { surah: msg.surah, ayah: msg.ayah, confidence: msg.confidence },
      candidates: state.lastCandidates,
      quranData: state.quranData!,
    });
    // Route to mushaf mode or flowing mode
    if (state.mushafDataReady) {
      handleMushafVerseMatch(msg);
    } else {
      handleVerseMatch(msg);
    }
  } else if (msg.type === "word_progress") {
    pushDiagnosticEvent("word_progress", {
      surah: msg.surah, ayah: msg.ayah,
      word_index: msg.word_index, total_words: msg.total_words,
    });
    // Route to mushaf mode or flowing mode
    if (state.mushafDataReady) {
      void handleMushafWordProgress(msg);
    } else {
      handleWordProgress(msg);
    }
  } else if (msg.type === "raw_transcript") {
    console.log(
      `%c[HEARD] "${msg.text}" %c(conf: ${(msg.confidence * 100).toFixed(1)}%)`,
      "color: #e8b339; font-size: 13px",
      "color: #999",
    );
    pushDiagnosticEvent("raw_transcript", {
      text: msg.text, confidence: msg.confidence,
    });
    handleRawTranscript(msg);
  } else if (msg.type === "candidate_list") {
    if (msg.candidates.length > 0) {
      console.log(
        `[CANDIDATES] ${msg.candidates.length} candidates, top: ${msg.candidates[0].surah}:${msg.candidates[0].ayah} (${(msg.candidates[0].score * 100).toFixed(0)}%)`,
      );
    }
    handleCandidateList(msg);
    // Store candidates for feedback reporting
    if (msg.candidates.length > 0) {
      state.lastCandidates = msg.candidates;
    }
  } else if (msg.type === "word_aligned") {
    // Forced Alignment: word confirmed with confidence
    if (state.mushafDataReady) {
      handleMushafWordAligned(msg);
    }
  } else if (msg.type === "verse_complete") {
    // Forced Alignment: entire verse completed
    if (state.mushafDataReady) {
      handleMushafVerseComplete(msg);
    }
  }
}

// ---------------------------------------------------------------------------
// Audio capture
// ---------------------------------------------------------------------------

// Cap session audio to ~60 seconds (16kHz * 60s = 960000 samples).
// Each chunk is 4800 samples (300ms). 60s = 200 chunks.
const MAX_SESSION_CHUNKS = 200;

async function startAudio(): Promise<void> {
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
  } catch (err) {
    console.error("Microphone access denied or failed:", err);
    const isDenied =
      err instanceof DOMException &&
      (err.name === "NotAllowedError" || err.name === "PermissionDeniedError");
    if (isDenied) {
      $permissionPrompt.textContent =
        "Microphone access was denied. Please allow microphone access in your browser settings, then try again.";
    } else {
      $permissionPrompt.textContent =
        "Could not access the microphone. Please check that your device has a working microphone.";
    }
    $permissionPrompt.hidden = false;
    // Signal failure to the caller so it can revert button state
    throw err;
  }

  state.stream = stream;
  $permissionPrompt.hidden = true;

  const audioCtx = new AudioContext();
  state.audioCtx = audioCtx;

  // Resume AudioContext if suspended (browser autoplay policy on mobile)
  if (audioCtx.state === "suspended") {
    await audioCtx.resume();
  }

  await audioCtx.audioWorklet.addModule("/audio-processor.js");
  const source = audioCtx.createMediaStreamSource(stream);
  const processor = new AudioWorkletNode(audioCtx, "audio-stream-processor");

  processor.port.onmessage = (e: MessageEvent) => {
    const samples = new Float32Array(e.data as ArrayBuffer);
    // Save copy to session buffer (capped to prevent unbounded memory growth)
    state.sessionAudioChunks.push(samples.slice());
    if (state.sessionAudioChunks.length > MAX_SESSION_CHUNKS) {
      state.sessionAudioChunks.shift();
    }
    // Send to worker for recognition
    if (state.worker) {
      state.worker.postMessage(
        { type: "audio", samples },
        [samples.buffer],
      );
    }
  };

  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  source.connect(analyser);
  source.connect(processor);

  const levelBuf = new Float32Array(analyser.fftSize);
  const checkLevel = () => {
    if (!state.isActive) return;
    analyser.getFloatTimeDomainData(levelBuf);
    let sum = 0;
    for (let i = 0; i < levelBuf.length; i++) {
      sum += levelBuf[i] * levelBuf[i];
    }
    const rms = Math.sqrt(sum / levelBuf.length);
    if (rms > 0.01) {
      $indicator.classList.add("audio-detected");
      $indicator.classList.remove("silence");
    } else {
      $indicator.classList.remove("audio-detected");
      $indicator.classList.add("silence");
    }
    requestAnimationFrame(checkLevel);
  };
  checkLevel();

  state.isActive = true;
  $indicator.classList.add("active");
}

// ---------------------------------------------------------------------------
// Stop audio capture
// ---------------------------------------------------------------------------
function stopAudio(): void {
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
  if (state.audioCtx) {
    state.audioCtx.close().catch(() => {/* ignore close errors */});
    state.audioCtx = null;
  }
  state.isActive = false;
  $indicator.classList.remove("active", "audio-detected", "silence", "has-verses");
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  // Create inference worker (FastConformer CTC with constrained decoding)
  const worker = new Worker(
    new URL("./worker/inference-fastconformer.ts", import.meta.url),
    { type: "module" },
  );
  state.worker = worker;

  worker.onmessage = (e: MessageEvent<WorkerOutbound>) => {
    handleWorkerMessage(e.data);
  };

  worker.onerror = (e) => {
    console.error("Worker error:", e);
    $loadingDetail.textContent = `Worker error: ${e.message || "unknown"}`;
  };

  // Initialize worker (loads model, vocab, quranDB)
  worker.postMessage({ type: "init" });

  // Load mushaf layout data in background
  loadMushafData()
    .then(() => {
      // If model was already ready before mushaf data loaded, show mushaf now
      if (state.modelReady && $mushafContainer.hidden) {
        $mushafContainer.hidden = false;
        $verses.hidden = true;
        state.currentMushafPage = 0;
        navigateToMushafPage(1);
      }
    })
    .catch((err) =>
      console.warn("Mushaf data not available, using flowing mode:", err),
    );

  // Algorithm view toggle
  $btnAlgoView.addEventListener("click", async () => {
    // Ensure the algorithm DB is loaded before toggling
    if (!state.algorithmDBReady) {
      await loadAlgorithmDB();
    }
    toggleAlgorithmView();
  });

  // Load algorithm DB in background (low priority, after mushaf data)
  loadAlgorithmDB().catch((err) =>
    console.warn("Algorithm DB not available:", err),
  );

  // Practice mode toggle (applies to both mushaf and flowing)
  $btnPractice.addEventListener("click", () => {
    state.practiceMode = !state.practiceMode;
    $app.classList.toggle("practice-mode", state.practiceMode);
    // Update mushaf page if in mushaf mode
    if (state.mushafDataReady && !$mushafContainer.hidden) {
      if (state.practiceMode) {
        mushafHideUnrevealed($mushafPage, state.revealedVerses);
      } else {
        mushafRevealAll($mushafPage);
      }
    }
  });

  // Mushaf page navigation
  $btnPagePrev.addEventListener("click", () => {
    if (state.currentMushafPage < 604) {
      navigateToMushafPage(state.currentMushafPage + 1);
    }
  });
  $btnPageNext.addEventListener("click", () => {
    if (state.currentMushafPage > 1) {
      navigateToMushafPage(state.currentMushafPage - 1);
    }
  });

  // Keyboard navigation for mushaf pages
  document.addEventListener("keydown", (e) => {
    if ($mushafContainer.hidden) return;
    if (e.key === "ArrowRight" && state.currentMushafPage < 604) {
      navigateToMushafPage(state.currentMushafPage + 1);
    } else if (e.key === "ArrowLeft" && state.currentMushafPage > 1) {
      navigateToMushafPage(state.currentMushafPage - 1);
    }
  });

  // Swipe gesture for mushaf page navigation (mobile)
  // In RTL mushaf: swipe left = next page (higher number), swipe right = prev page (lower number)
  let _touchStartX = 0;
  let _touchStartY = 0;
  const SWIPE_THRESHOLD = 50;

  $mushafPage.addEventListener("touchstart", (e: TouchEvent) => {
    if (e.touches.length !== 1) return;
    _touchStartX = e.touches[0].clientX;
    _touchStartY = e.touches[0].clientY;
  }, { passive: true });

  $mushafPage.addEventListener("touchend", (e: TouchEvent) => {
    if ($mushafContainer.hidden) return;
    if (e.changedTouches.length !== 1) return;
    const dx = e.changedTouches[0].clientX - _touchStartX;
    const dy = e.changedTouches[0].clientY - _touchStartY;
    // Only trigger if horizontal swipe is dominant
    if (Math.abs(dx) < SWIPE_THRESHOLD || Math.abs(dx) < Math.abs(dy)) return;
    if (dx < 0 && state.currentMushafPage < 604) {
      // Swipe left = next page (RTL: forward in mushaf)
      navigateToMushafPage(state.currentMushafPage + 1);
    } else if (dx > 0 && state.currentMushafPage > 1) {
      // Swipe right = prev page (RTL: backward in mushaf)
      navigateToMushafPage(state.currentMushafPage - 1);
    }
  }, { passive: true });

  // Record toggle (single button: mic ↔ stop)
  $btnRecToggle.addEventListener("click", async () => {
    if (!state.modelReady) return; // Ignore clicks before model is loaded
    if (!state.isActive) {
      // --- Start / Resume recording ---

      // Keep revealedVerses, currentMushafPage, lastModelPrediction, _wordTrackedVerses
      // so the user can continue where they left off
      state.sessionAudioChunks = [];
      state.hasFirstMatch = false;
      state.diagnosticEvents = [];
      state.recentVerseMatches = [];
      state.lastCandidates = [];
      hideWrongButton();
      // Reset per-event accumulators but keep verse-level tracking
      _mushafMatchedWords = new Set<number>();
      _mushafTrackingKey = "";
      _mushafErrorWords = new Set<number>();
      _mushafErrorKey = "";
      _faConfirmedWords = new Set<number>();
      _faTrackingKey = "";
      state.practiceMode = true;
      $app.classList.add("practice-mode");
      $rawTranscript.textContent = "";
      $rawTranscript.classList.remove("visible");
      $postRecording.hidden = true;
      state.lastRawTranscript = "";

      // Clear algorithm view for new recording session
      if (state.algorithmMode) {
        clearAlgorithmView();
      }

      if (state.mushafDataReady) {
        $mushafContainer.hidden = false;
        $verses.hidden = true;
        // Re-render current page to apply practice mode with preserved reveals
        const pg = state.currentMushafPage >= 1 ? state.currentMushafPage : 1;
        state.currentMushafPage = 0; // force re-render
        await navigateToMushafPage(pg);
      }

      state.worker?.postMessage({ type: "reset" });

      // Attempt to start audio — if mic access fails, revert button state
      try {
        await startAudio();
        // Only switch to stop-button state after audio starts successfully
        $btnRecToggle.classList.remove("mc-btn--rec");
        $btnRecToggle.classList.add("mc-btn--stop", "recording");
        $btnRecToggle.title = "Stop";
        // Show initial listening state in the candidate bar
        renderCandidateListening("");
      } catch {
        // startAudio already set the permission prompt;
        // revert button to mic state so user can retry
        $btnRecToggle.classList.remove("mc-btn--stop", "recording");
        $btnRecToggle.classList.add("mc-btn--rec");
        $btnRecToggle.title = "Start recitation";
      }
    } else {
      // --- Stop recording (pause — keep state) ---
      stopAudio();
      $btnRecToggle.classList.remove("mc-btn--stop", "recording");
      $btnRecToggle.classList.add("mc-btn--rec");
      $btnRecToggle.title = "Start recitation";

      // Keep practice mode and mushaf visible so user sees their progress
      // Hide feedback button
      hideWrongButton();
      // Clear candidate bar and any pending fade timers
      if (state.candidateMatchFadeTimer) {
        clearTimeout(state.candidateMatchFadeTimer);
        state.candidateMatchFadeTimer = null;
      }
      $candidateList.innerHTML = "";
      $candidateList.className = "";
      $candidateList.classList.remove("visible");
    }
  });

  $btnRestart.addEventListener("click", () => {
    // Full reset — clear everything
    state.sessionAudioChunks = [];
    state.lastModelPrediction = null;
    state.hasFirstMatch = false;
    state.lastCandidates = [];
    state.groups = [];
    state.revealedVerses = new Set<string>();
    hideWrongButton();
    // Reset mushaf tracking state
    _wordTrackedVerses.clear();
    _priorRevealDoneForPage = 0;
    _mushafMatchedWords = new Set<number>();
    _mushafTrackingKey = "";
    _mushafErrorWords = new Set<number>();
    _mushafErrorKey = "";
    _faConfirmedWords = new Set<number>();
    _faTrackingKey = "";
    $verses.innerHTML = "";
    $rawTranscript.textContent = "";
    $rawTranscript.classList.remove("visible");
    $postRecording.hidden = true;
    state.practiceMode = false;
    state.lastRawTranscript = "";
    $app.classList.remove("practice-mode");
    // Clear candidate bar and any pending fade timers
    if (state.candidateMatchFadeTimer) {
      clearTimeout(state.candidateMatchFadeTimer);
      state.candidateMatchFadeTimer = null;
    }
    $candidateList.innerHTML = "";
    $candidateList.className = "";
    $candidateList.classList.remove("visible");

    // Reset toggle button to mic state
    $btnRecToggle.classList.remove("mc-btn--stop", "recording");
    $btnRecToggle.classList.add("mc-btn--rec");
    $btnRecToggle.title = "Start recitation";

    // Show mushaf page 1
    if (state.mushafDataReady) {
      $mushafContainer.hidden = false;
      $verses.hidden = true;
      state.currentMushafPage = 0;
      navigateToMushafPage(1);
    }
  });

});
