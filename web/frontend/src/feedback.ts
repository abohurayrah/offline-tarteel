import { encodeWav, concatChunks } from "./lib/wav-encoder";
import type { QuranVerse, CandidateVerse } from "./lib/types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface FeedbackContext {
  audioChunks: Float32Array[];
  transcript: string;
  matched: { surah: number; ayah: number; confidence: number } | null;
  candidates: CandidateVerse[];
  quranData: QuranVerse[];
}

interface SurahInfo {
  num: number;
  name: string;
  nameEn: string;
  ayahCount: number;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let currentContext: FeedbackContext | null = null;
let surahList: SurahInfo[] = [];
let quranVerses: QuranVerse[] = [];
let selectedSurah: number | null = null;
let selectedAyah: number | null = null;
let wrongButtonTimer: ReturnType<typeof setTimeout> | null = null;
let isSubmitting = false;

// DOM refs (lazily created)
let $wrongButton: HTMLButtonElement | null = null;
let $overlay: HTMLDivElement | null = null;
let $panel: HTMLDivElement | null = null;

// ---------------------------------------------------------------------------
// Initialization — build surah index from quranData
// ---------------------------------------------------------------------------
export function initFeedback(data: QuranVerse[]): void {
  quranVerses = data;
  const map = new Map<number, SurahInfo>();
  for (const v of data) {
    if (!map.has(v.surah)) {
      map.set(v.surah, {
        num: v.surah,
        name: v.surah_name,
        nameEn: v.surah_name_en,
        ayahCount: 0,
      });
    }
    const info = map.get(v.surah)!;
    if (v.ayah > info.ayahCount) info.ayahCount = v.ayah;
  }
  surahList = Array.from(map.values()).sort((a, b) => a.num - b.num);
}

// ---------------------------------------------------------------------------
// "Wrong?" pill button — shown after verse_match, auto-hides after 10s
// ---------------------------------------------------------------------------
export function showWrongButton(ctx: FeedbackContext): void {
  currentContext = ctx;

  // Clear any existing timer
  if (wrongButtonTimer) {
    clearTimeout(wrongButtonTimer);
    wrongButtonTimer = null;
  }

  if (!$wrongButton) {
    $wrongButton = document.createElement("button");
    $wrongButton.id = "feedback-wrong-btn";
    $wrongButton.className = "feedback-wrong-btn";
    $wrongButton.textContent = "Wrong?";
    $wrongButton.addEventListener("click", () => {
      hideWrongButton();
      openFeedbackPanel();
    });
    document.getElementById("app")!.appendChild($wrongButton);
  }

  // Reset state for fresh display
  $wrongButton.classList.remove("feedback-wrong-btn--hiding");
  $wrongButton.hidden = false;

  // Force reflow then add visible class for fade-in
  void $wrongButton.offsetWidth;
  $wrongButton.classList.add("feedback-wrong-btn--visible");

  // Auto-hide after 10 seconds
  wrongButtonTimer = setTimeout(() => {
    hideWrongButton();
  }, 10_000);
}

/** Open the feedback panel directly (from bottom bar wrong button) */
export function openFeedbackPanelDirect(): void {
  openFeedbackPanel();
}

export function hideWrongButton(): void {
  if (wrongButtonTimer) {
    clearTimeout(wrongButtonTimer);
    wrongButtonTimer = null;
  }
  if ($wrongButton) {
    $wrongButton.classList.add("feedback-wrong-btn--hiding");
    $wrongButton.classList.remove("feedback-wrong-btn--visible");
    setTimeout(() => {
      if ($wrongButton) $wrongButton.hidden = true;
    }, 400);
  }
}

// ---------------------------------------------------------------------------
// Feedback panel — slide-up from bottom
// ---------------------------------------------------------------------------
function openFeedbackPanel(): void {
  if (!currentContext) return;

  // Reset selection
  selectedSurah = null;
  selectedAyah = null;
  isSubmitting = false;

  // Create overlay
  if (!$overlay) {
    $overlay = document.createElement("div");
    $overlay.className = "feedback-overlay";
    $overlay.addEventListener("click", closeFeedbackPanel);
    document.body.appendChild($overlay);
  }
  $overlay.hidden = false;
  void $overlay.offsetWidth;
  $overlay.classList.add("feedback-overlay--visible");

  // Create panel
  if (!$panel) {
    $panel = document.createElement("div");
    $panel.className = "feedback-panel";
    document.body.appendChild($panel);
  }

  $panel.innerHTML = buildPanelHTML();
  $panel.hidden = false;
  void $panel.offsetWidth;
  $panel.classList.add("feedback-panel--visible");

  // Wire up event listeners
  wireUpPanel();
}

function closeFeedbackPanel(): void {
  if ($overlay) {
    $overlay.classList.remove("feedback-overlay--visible");
    setTimeout(() => {
      if ($overlay) $overlay.hidden = true;
    }, 300);
  }
  if ($panel) {
    $panel.classList.remove("feedback-panel--visible");
    setTimeout(() => {
      if ($panel) $panel.hidden = true;
    }, 300);
  }
}

function buildPanelHTML(): string {
  const ctx = currentContext!;
  const matchedLabel = ctx.matched
    ? `Surah ${ctx.matched.surah}, Ayah ${ctx.matched.ayah} (${Math.round(ctx.matched.confidence * 100)}%)`
    : "No prediction";
  const transcriptLabel = ctx.transcript || "(no transcript)";

  return `
    <div class="feedback-panel-handle"></div>
    <div class="feedback-panel-content">
      <h3 class="feedback-title">Report Incorrect Match</h3>

      <div class="feedback-section feedback-captured">
        <div class="feedback-label">What we captured</div>
        <div class="feedback-captured-row">
          <span class="feedback-captured-icon">&#x1F50A;</span>
          <span class="feedback-captured-text" dir="rtl" lang="ar">${escapeHtml(transcriptLabel)}</span>
        </div>
        <div class="feedback-captured-row">
          <span class="feedback-captured-icon">&#x1F4CD;</span>
          <span class="feedback-captured-text">${escapeHtml(matchedLabel)}</span>
        </div>
      </div>

      <div class="feedback-section">
        <div class="feedback-label">What were you reading? <span class="feedback-optional">(optional)</span></div>
        <div class="feedback-search-wrap">
          <input
            type="text"
            id="feedback-surah-search"
            class="feedback-search-input"
            placeholder="Search: surah name, number, or Arabic..."
            autocomplete="off"
            autocorrect="off"
            autocapitalize="off"
            spellcheck="false"
          />
          <div id="feedback-search-results" class="feedback-search-results" hidden></div>
        </div>
        <div id="feedback-ayah-section" class="feedback-ayah-section" hidden>
          <div class="feedback-ayah-header">
            <span id="feedback-selected-surah-label" class="feedback-selected-surah"></span>
            <button id="feedback-change-surah" class="feedback-change-surah">Change</button>
          </div>
          <div class="feedback-ayah-row">
            <label for="feedback-ayah-input" class="feedback-ayah-label">Ayah</label>
            <input
              type="number"
              id="feedback-ayah-input"
              class="feedback-ayah-input"
              min="1"
              placeholder="#"
            />
          </div>
          <div id="feedback-verse-preview" class="feedback-verse-preview" hidden></div>
        </div>
      </div>

      <div class="feedback-actions">
        <button id="feedback-submit-btn" class="feedback-submit-btn">Submit Report</button>
        <button id="feedback-skip-btn" class="feedback-skip-btn">Just wrong, don't know the correct verse</button>
      </div>

      <div id="feedback-status" class="feedback-status" hidden></div>
    </div>
  `;
}

function wireUpPanel(): void {
  const $search = document.getElementById("feedback-surah-search") as HTMLInputElement;
  const $results = document.getElementById("feedback-search-results")!;
  const $ayahSection = document.getElementById("feedback-ayah-section")!;
  const $ayahInput = document.getElementById("feedback-ayah-input") as HTMLInputElement;
  const $preview = document.getElementById("feedback-verse-preview")!;
  const $surahLabel = document.getElementById("feedback-selected-surah-label")!;
  const $changeSurah = document.getElementById("feedback-change-surah")!;
  const $submitBtn = document.getElementById("feedback-submit-btn")!;
  const $skipBtn = document.getElementById("feedback-skip-btn")!;
  const $status = document.getElementById("feedback-status")!;

  // Surah search
  $search.addEventListener("input", () => {
    const query = $search.value.trim();
    if (!query) {
      $results.hidden = true;
      return;
    }
    const filtered = filterSurahs(query);
    renderSearchResults($results, filtered);
    $results.hidden = filtered.length === 0;
  });

  // Focus shows results if there's a query
  $search.addEventListener("focus", () => {
    const query = $search.value.trim();
    if (query) {
      const filtered = filterSurahs(query);
      renderSearchResults($results, filtered);
      $results.hidden = filtered.length === 0;
    }
  });

  // Delegate click on search results
  $results.addEventListener("click", (e) => {
    const target = (e.target as HTMLElement).closest("[data-surah]") as HTMLElement | null;
    if (!target) return;
    const surahNum = parseInt(target.dataset.surah!);
    selectSurah(surahNum, $search, $results, $ayahSection, $ayahInput, $surahLabel, $preview);
  });

  // Change surah button
  $changeSurah.addEventListener("click", () => {
    selectedSurah = null;
    selectedAyah = null;
    $ayahSection.hidden = true;
    $search.value = "";
    $search.hidden = false;
    $search.focus();
  });

  // Ayah input — preview verse on change
  $ayahInput.addEventListener("input", () => {
    const val = parseInt($ayahInput.value);
    if (!selectedSurah || isNaN(val) || val < 1) {
      $preview.hidden = true;
      selectedAyah = null;
      return;
    }
    selectedAyah = val;
    const verse = quranVerses.find(
      (v) => v.surah === selectedSurah && v.ayah === val,
    );
    if (verse) {
      $preview.textContent = verse.text_uthmani;
      $preview.hidden = false;
    } else {
      $preview.textContent = "Ayah not found";
      $preview.hidden = false;
    }
  });

  // Submit
  $submitBtn.addEventListener("click", async () => {
    if (isSubmitting) return;
    isSubmitting = true;
    $submitBtn.setAttribute("disabled", "true");
    $submitBtn.textContent = "Submitting...";
    $status.hidden = true;

    const correction =
      selectedSurah && selectedAyah
        ? { surah: selectedSurah, ayah: selectedAyah }
        : null;

    const success = await submitFeedback(correction);

    if (success) {
      $status.textContent = "Thank you for your feedback!";
      $status.className = "feedback-status feedback-status--success";
      $status.hidden = false;
      setTimeout(() => closeFeedbackPanel(), 1200);
    } else {
      $status.textContent = "Failed to submit. Please try again.";
      $status.className = "feedback-status feedback-status--error";
      $status.hidden = false;
      $submitBtn.removeAttribute("disabled");
      $submitBtn.textContent = "Submit Report";
      isSubmitting = false;
    }
  });

  // Skip — submit without correction
  $skipBtn.addEventListener("click", async () => {
    if (isSubmitting) return;
    isSubmitting = true;
    $skipBtn.setAttribute("disabled", "true");
    $skipBtn.textContent = "Submitting...";
    $status.hidden = true;

    const success = await submitFeedback(null);

    if (success) {
      $status.textContent = "Thank you for your feedback!";
      $status.className = "feedback-status feedback-status--success";
      $status.hidden = false;
      setTimeout(() => closeFeedbackPanel(), 1200);
    } else {
      $status.textContent = "Failed to submit. Please try again.";
      $status.className = "feedback-status feedback-status--error";
      $status.hidden = false;
      $skipBtn.removeAttribute("disabled");
      $skipBtn.textContent = "Just wrong, don't know the correct verse";
      isSubmitting = false;
    }
  });
}

// ---------------------------------------------------------------------------
// Surah search / filter
// ---------------------------------------------------------------------------
function filterSurahs(query: string): SurahInfo[] {
  const q = query.toLowerCase().trim();

  // Number match (exact prefix)
  const num = parseInt(q);
  if (!isNaN(num)) {
    return surahList.filter((s) => String(s.num).startsWith(q));
  }

  // Text match — Arabic or English
  return surahList.filter((s) => {
    const en = s.nameEn.toLowerCase();
    const ar = s.name;
    // Loose English match: strip common prefixes for easier searching
    const enBare = en.replace(/^al-/i, "").replace(/^al /i, "");
    return (
      en.includes(q) ||
      enBare.includes(q) ||
      ar.includes(query) || // Arabic: use original case
      String(s.num) === q
    );
  });
}

function renderSearchResults(container: HTMLElement, results: SurahInfo[]): void {
  container.innerHTML = "";
  const shown = results.slice(0, 10);
  for (const s of shown) {
    const row = document.createElement("div");
    row.className = "feedback-search-row";
    row.dataset.surah = String(s.num);
    row.innerHTML = `
      <span class="feedback-search-num">${s.num}</span>
      <span class="feedback-search-name">
        <span class="feedback-search-name-en">${escapeHtml(s.nameEn)}</span>
        <span class="feedback-search-name-ar">${escapeHtml(s.name)}</span>
      </span>
      <span class="feedback-search-ayahs">${s.ayahCount} ayahs</span>
    `;
    container.appendChild(row);
  }
}

function selectSurah(
  surahNum: number,
  $search: HTMLInputElement,
  $results: HTMLElement,
  $ayahSection: HTMLElement,
  $ayahInput: HTMLInputElement,
  $surahLabel: HTMLElement,
  $preview: HTMLElement,
): void {
  const info = surahList.find((s) => s.num === surahNum);
  if (!info) return;

  selectedSurah = surahNum;
  selectedAyah = null;

  // Hide search, show ayah section
  $search.hidden = true;
  $results.hidden = true;
  $ayahSection.hidden = false;
  $surahLabel.textContent = `${info.num}. ${info.nameEn} — ${info.name}`;
  $ayahInput.max = String(info.ayahCount);
  $ayahInput.value = "";
  $preview.hidden = true;
  $ayahInput.focus();
}

// ---------------------------------------------------------------------------
// Submit feedback to /api/reports
// ---------------------------------------------------------------------------
async function submitFeedback(
  correction: { surah: number; ayah: number } | null,
): Promise<boolean> {
  if (!currentContext) return false;

  const ctx = currentContext;

  // Encode last 30s of audio (16kHz * 30s = 480000 samples)
  const MAX_SAMPLES = 16000 * 30;
  const combined = concatChunks(ctx.audioChunks);
  const trimmed =
    combined.length > MAX_SAMPLES
      ? combined.slice(combined.length - MAX_SAMPLES)
      : combined;
  const audioBlob = encodeWav(trimmed);

  const metadata = {
    matched: ctx.matched
      ? { surah: ctx.matched.surah, ayah: ctx.matched.ayah, score: ctx.matched.confidence }
      : null,
    transcript: ctx.transcript,
    candidates: ctx.candidates.slice(0, 10).map((c) => ({
      surah: c.surah,
      ayah: c.ayah,
      score: c.score,
    })),
    correction: correction,
    device: navigator.userAgent,
    screen: `${window.screen.width}x${window.screen.height}`,
    timestamp: new Date().toISOString(),
  };

  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.wav");
  formData.append("metadata", JSON.stringify(metadata));

  try {
    const res = await fetch("/api/reports", { method: "POST", body: formData });
    return res.ok;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function escapeHtml(str: string): string {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
