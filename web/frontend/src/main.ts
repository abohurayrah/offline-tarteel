import "@fontsource/amiri/400.css";
import "@fontsource/amiri/700.css";
import "@fontsource/amiri-quran/400.css";
import "./style.css";

import { initSurahDropdown, openReportDialog } from "./report-dialog";

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
  diagnosticEvents: [] as DiagnosticEvent[],
  lastDiagnosticSentAt: 0,
  recentVerseMatches: [] as { surah: number; ayah: number; timestamp: number }[],
  practiceMode: false,
  narrowingMode: false,
  // Mushaf page mode
  mushafPages: null as MushafPageData[] | null,
  verseToPage: null as Record<string, number> | null,
  currentMushafPage: 1,
  revealedVerses: new Set<string>(),
  mushafDataReady: false,
  lastVerseTransitionTime: 0,
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
const $btnReport = document.getElementById("btn-report")!;
const $btnRestart = document.getElementById("btn-restart")!;
const $btnPractice = document.getElementById("btn-practice")!;
const $btnNarrowing = document.getElementById("btn-narrowing")!;
const $candidateList = document.getElementById("candidate-list")!;
const $app = document.getElementById("app")!;
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
  state.quranData = await res.json();
  initSurahDropdown(state.quranData!);
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

  // Transition: fade out, render, fade in
  $mushafPage.classList.add("mushaf-page--exit");
  await new Promise((r) => setTimeout(r, 150));

  const page = state.mushafPages[pageNum - 1];
  await renderMushafPage($mushafPage, page, state.revealedVerses, state.practiceMode);
  $mushafPage.classList.remove("mushaf-page--exit");
  $mushafPage.classList.add("mushaf-page--enter");
  setTimeout(() => $mushafPage.classList.remove("mushaf-page--enter"), 350);
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

  // Track verse transition time — suppress gap detection for 2s after transitions
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

  console.log(
    `[MUSHAF] State: page=${state.currentMushafPage}, revealed=${state.revealedVerses.size} verses, wordTracked=${_wordTrackedVerses.size}, practice=${state.practiceMode}`,
  );
}

// Mushaf word progress accumulator (same pattern as flowing mode)
let _mushafMatchedWords = new Set<number>();
let _mushafTrackingKey = "";
// Track error words — blocks progression past them
let _mushafErrorWords = new Set<number>();
let _mushafErrorKey = "";

// Handle word progress in mushaf mode — reveal words one at a time
function handleMushafWordProgress(msg: WordProgressMessage): void {
  const targetPage = getPageForVerse(msg.surah, msg.ayah);

  if (!targetPage) {
    console.warn(`[WORD] No page for ${msg.surah}:${msg.ayah}`);
    return;
  }
  if (targetPage !== state.currentMushafPage) {
    console.warn(
      `[WORD] ${msg.surah}:${msg.ayah} word ${msg.word_index}/${msg.total_words} — wrong page (verse on p${targetPage}, showing p${state.currentMushafPage})`,
    );
    return;
  }

  // Accumulate matched indices across events for the same verse
  const key = `${msg.surah}:${msg.ayah}`;
  if (key !== _mushafTrackingKey) {
    _mushafMatchedWords = new Set<number>();
    _mushafTrackingKey = key;
  }
  for (const idx of msg.matched_indices) {
    _mushafMatchedWords.add(idx);
  }

  // Track that this verse had actual word progress (used by verse_match to decide reveals)
  _wordTrackedVerses.add(key);
  const accumulated = Array.from(_mushafMatchedWords).sort((a, b) => a - b);

  // Clear errors for this verse when new word progress comes in (user retrying)
  if (_mushafErrorKey === key && _mushafErrorWords.size > 0) {
    // Only clear errors for words that are now matched
    for (const idx of msg.matched_indices) {
      _mushafErrorWords.delete(idx);
    }
    if (_mushafErrorWords.size === 0) {
      mushafClearErrors($mushafPage);
    }
  }

  // Compute contiguous from 0, stopping at error words
  let contiguousMax = -1;
  for (let i = 0; i < msg.total_words; i++) {
    if (_mushafErrorWords.has(i)) break; // Block at error word
    if (_mushafMatchedWords.has(i)) contiguousMax = i;
    else break;
  }

  // Detect skipped words (gaps) — these are likely misreads
  // e.g., accumulated=[0,1,2,5,6] with contiguousMax=2 → words 3,4 were skipped
  // Suppress during verse transitions (2s grace) to avoid false positives from stale audio
  const beyondContiguous = accumulated.filter((i) => i > contiguousMax + 1);
  const isRecentTransition = Date.now() - state.lastVerseTransitionTime < 2000;
  if (beyondContiguous.length >= 2 && contiguousMax >= 1 && !isRecentTransition) {
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

  // Build spoken text so far (contiguous words from 0)
  const spokenWords: string[] = [];
  for (let i = 0; i <= contiguousMax; i++) {
    spokenWords.push(verseWordMap[i] || `[${i}]`);
  }

  const currentWord = verseWordMap[msg.word_index] || "";
  console.log(
    `[WORD] ${msg.surah}:${msg.ayah} word ${msg.word_index}/${msg.total_words}` +
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

  // Highlight using accumulated indices (not just this event's)
  mushafHighlightWord($mushafPage, msg.surah, msg.ayah, accumulated);

  // Mark verse as revealed only when ALL words are matched
  if (_mushafMatchedWords.size >= msg.total_words) {
    state.revealedVerses.add(key);
    console.log(
      `%c[VERSE_COMPLETE] ${msg.surah}:${msg.ayah} — all ${msg.total_words} words matched`,
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

function handleMushafWordAligned(msg: WordAlignedMessage): void {
  const targetPage = getPageForVerse(msg.surah, msg.ayah);
  if (!targetPage || targetPage !== state.currentMushafPage) return;

  const key = `${msg.surah}:${msg.ayah}`;
  if (key !== _faTrackingKey) {
    _faConfirmedWords = new Set<number>();
    _faTrackingKey = key;
  }

  // Add confirmed words
  for (const idx of msg.cumulative_indices) {
    _faConfirmedWords.add(idx);
  }

  // Track that this verse had progress
  _wordTrackedVerses.add(key);

  // Get words for this verse from page data for confidence coloring
  const allMushafWords = $mushafPage.querySelectorAll<HTMLElement>(
    `.mp-word[data-surah="${msg.surah}"][data-ayah="${msg.ayah}"]`,
  );

  // Highlight all confirmed words up to current position
  for (let i = 0; i < allMushafWords.length; i++) {
    const w = allMushafWords[i];
    if (_faConfirmedWords.has(i)) {
      w.classList.remove("mp-word--hidden");
      w.classList.add("mp-word--spoken");

      // Clear previous confidence classes
      w.classList.remove("mp-word--fa-good", "mp-word--fa-warn", "mp-word--fa-error");

      // Apply confidence-based color only for the current word
      if (i === msg.word_index) {
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
          if (s === String(msg.surah) && a === String(msg.ayah) && parseInt(widx) - 1 === msg.word_index) {
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
}

// ---------------------------------------------------------------------------
// Live narrowing (candidate list)
// ---------------------------------------------------------------------------
function handleCandidateList(msg: CandidateListMessage): void {
  if (!state.narrowingMode) return;

  const container = $candidateList;
  container.innerHTML = "";

  if (msg.candidates.length === 0) {
    container.classList.remove("visible");
    return;
  }

  // Show top score for reference
  const topScore = msg.candidates[0].score;

  for (const c of msg.candidates) {
    const item = document.createElement("div");
    item.className = "candidate-item";

    // Highlight confidence relative to top
    const relScore = topScore > 0 ? c.score / topScore : 0;
    if (relScore >= 0.97) {
      item.classList.add("candidate--top");
    } else if (relScore >= 0.85) {
      item.classList.add("candidate--likely");
    }

    const pct = Math.round(c.score * 100);
    const bar = document.createElement("div");
    bar.className = "candidate-bar";
    bar.style.width = `${pct}%`;
    item.appendChild(bar);

    const info = document.createElement("div");
    info.className = "candidate-info";

    const label = document.createElement("span");
    label.className = "candidate-label";
    label.textContent = `${c.surah_name_en} ${c.surah}:${c.ayah}`;

    const score = document.createElement("span");
    score.className = "candidate-score";
    score.textContent = `${pct}%`;

    info.appendChild(label);
    info.appendChild(score);
    item.appendChild(info);

    const preview = document.createElement("div");
    preview.className = "candidate-preview";
    preview.dir = "rtl";
    preview.lang = "ar";
    preview.textContent = c.text_preview;
    item.appendChild(preview);

    container.appendChild(item);
  }

  container.classList.add("visible");
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
    const wavBlob = float32ToWav(audioSlice, 16000);

    const form = new FormData();
    form.append("audio", wavBlob, "diagnostic.wav");
    form.append("events", JSON.stringify(state.diagnosticEvents));
    form.append("trigger", trigger);

    await fetch("/api/diagnostics", { method: "POST", body: form });
  } catch (err) {
    console.error("Failed to send diagnostic report:", err);
  }
}

function float32ToWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeStr(off: number, str: string) {
    for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
  }

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let off = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    off += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
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
    $loadingDetail.textContent = `Error: ${msg.message}`;
    $modelStatus.textContent = "Error";
    console.error("Worker reported error:", msg.message);
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
    // Hide candidates once a verse is confirmed
    $candidateList.innerHTML = "";
    $candidateList.classList.remove("visible");
    pushDiagnosticEvent("verse_match", {
      surah: msg.surah, ayah: msg.ayah, confidence: msg.confidence,
    });
    checkAnomalyAndSend(msg);
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
      handleMushafWordProgress(msg);
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
async function startAudio(): Promise<void> {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
    state.stream = stream;
    $permissionPrompt.hidden = true;

    const audioCtx = new AudioContext();
    state.audioCtx = audioCtx;

    await audioCtx.audioWorklet.addModule("/audio-processor.js");
    const source = audioCtx.createMediaStreamSource(stream);
    const processor = new AudioWorkletNode(audioCtx, "audio-stream-processor");

    processor.port.onmessage = (e: MessageEvent) => {
      const samples = new Float32Array(e.data as ArrayBuffer);
      // Save copy to session buffer
      state.sessionAudioChunks.push(samples.slice());
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
  } catch (err) {
    console.error("Failed to start audio:", err);
    $permissionPrompt.hidden = false;
  }
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
    state.audioCtx.close();
    state.audioCtx = null;
  }
  state.isActive = false;
  $indicator.classList.remove("active", "audio-detected", "silence", "has-verses");
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  // Create inference worker
  const worker = new Worker(
    new URL("./worker/inference.ts", import.meta.url),
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

  // Narrowing mode toggle
  $btnNarrowing.addEventListener("click", () => {
    state.narrowingMode = !state.narrowingMode;
    $btnNarrowing.classList.toggle("active", state.narrowingMode);
    if (!state.narrowingMode) {
      $candidateList.innerHTML = "";
      $candidateList.classList.remove("visible");
    }
  });

  // Record toggle (single button: mic ↔ stop)
  $btnRecToggle.addEventListener("click", async () => {
    if (!state.isActive) {
      // --- Start / Resume recording ---
      $btnRecToggle.classList.remove("mc-btn--rec");
      $btnRecToggle.classList.add("mc-btn--stop", "recording");
      $btnRecToggle.title = "Stop";

      // Keep revealedVerses, currentMushafPage, lastModelPrediction, _wordTrackedVerses
      // so the user can continue where they left off
      state.sessionAudioChunks = [];
      state.hasFirstMatch = false;
      state.diagnosticEvents = [];
      state.recentVerseMatches = [];
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

      if (state.mushafDataReady) {
        $mushafContainer.hidden = false;
        $verses.hidden = true;
        // Re-render current page to apply practice mode with preserved reveals
        const pg = state.currentMushafPage >= 1 ? state.currentMushafPage : 1;
        state.currentMushafPage = 0; // force re-render
        await navigateToMushafPage(pg);
      }

      state.worker?.postMessage({ type: "reset" });
      await startAudio();
    } else {
      // --- Stop recording (pause — keep state) ---
      stopAudio();
      $btnRecToggle.classList.remove("mc-btn--stop", "recording");
      $btnRecToggle.classList.add("mc-btn--rec");
      $btnRecToggle.title = "Start recitation";

      // Keep practice mode and mushaf visible so user sees their progress
      $candidateList.innerHTML = "";
      $candidateList.classList.remove("visible");
    }
  });

  $btnRestart.addEventListener("click", () => {
    // Full reset — clear everything
    state.sessionAudioChunks = [];
    state.lastModelPrediction = null;
    state.hasFirstMatch = false;
    state.groups = [];
    state.revealedVerses = new Set<string>();
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
    state.narrowingMode = false;
    $app.classList.remove("practice-mode");
    $btnNarrowing.classList.remove("active");
    $candidateList.innerHTML = "";
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

  $btnReport.addEventListener("click", () => {
    openReportDialog({
      audioChunks: state.sessionAudioChunks,
      modelPrediction: state.lastModelPrediction,
      quranData: state.quranData!,
    });
  });
});
