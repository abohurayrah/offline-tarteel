// ---------------------------------------------------------------------------
// Mushaf Page Renderer — exact 15-line Madani Mushaf layout with QPC V1 fonts
// ---------------------------------------------------------------------------

// ── Types ──
export interface MushafWord {
  location: string; // "surah:ayah:word"
  word: string;
  qpcV1: string;
}

export interface MushafLine {
  line: number;
  type: string; // "surah-header" | "basmala" | "text"
  text?: string;
  surah?: string;
  verseRange?: string;
  words?: MushafWord[];
  qpcV1?: string;
}

export interface MushafPageData {
  page: number;
  lines: MushafLine[];
}

// ── Font management ──
const _loaded = new Set<string>();

function _fam(p: number): string {
  return `QCF_P${String(p).padStart(3, "0")}`;
}

async function _loadFont(family: string, url: string): Promise<void> {
  if (_loaded.has(family)) return;
  const face = new FontFace(family, `url(${url})`);
  await face.load();
  document.fonts.add(face);
  _loaded.add(family);
}

export async function loadPageFont(page: number): Promise<void> {
  const fam = _fam(page);
  await _loadFont(fam, `/fonts/qpc-v1/${fam}.woff2`);
}

export async function loadBismillahFont(): Promise<void> {
  await _loadFont("QCF_BSML", "/fonts/qpc-v1/QCF_BSML.woff2");
}

export function preloadAdjacentFonts(page: number): void {
  if (page > 1) loadPageFont(page - 1).catch(() => {});
  if (page < 604) loadPageFont(page + 1).catch(() => {});
}

// ── Page rendering ──
export async function renderPage(
  container: HTMLElement,
  page: MushafPageData,
  revealedVerses: Set<string>,
  practiceMode: boolean,
): Promise<void> {
  const pageNum = page.page;
  const fam = _fam(pageNum);

  // Load fonts in parallel
  const tasks: Promise<void>[] = [loadPageFont(pageNum)];
  if (page.lines.some((l) => l.type === "basmala")) {
    tasks.push(loadBismillahFont());
  }
  await Promise.all(tasks);

  // Clear and set up container
  container.innerHTML = "";
  container.className = "mushaf-page";
  container.setAttribute("data-page", String(pageNum));

  for (const line of page.lines) {
    const el = document.createElement("div");
    el.className = "mp-line";
    el.setAttribute("data-line", String(line.line));

    if (line.type === "surah-header") {
      el.classList.add("mp-line--header");
      _renderHeader(el, line);
    } else if (line.type === "basmala") {
      el.classList.add("mp-line--basmala");
      _renderBasmala(el, line);
    } else if (line.type === "text") {
      el.classList.add("mp-line--text");
      if (line.verseRange) {
        el.setAttribute("data-verse-range", line.verseRange);
      }
      _renderText(el, line, fam, revealedVerses, practiceMode);
    }

    container.appendChild(el);
  }

  // Preload adjacent page fonts in background
  preloadAdjacentFonts(pageNum);
}

function _renderHeader(el: HTMLElement, line: MushafLine): void {
  const box = document.createElement("div");
  box.className = "mp-header-box";

  const ornL = document.createElement("span");
  ornL.className = "mp-ornament";
  ornL.textContent = "\uFD3E"; // ﴾

  const name = document.createElement("span");
  name.className = "mp-header-name";
  name.textContent = line.text || "";

  const ornR = document.createElement("span");
  ornR.className = "mp-ornament";
  ornR.textContent = "\uFD3F"; // ﴿

  box.append(ornL, name, ornR);
  el.appendChild(box);
}

function _renderBasmala(el: HTMLElement, line: MushafLine): void {
  const span = document.createElement("span");
  span.className = "mp-basmala-text";
  span.style.fontFamily = "QCF_BSML";
  span.textContent = line.qpcV1 || "";
  el.appendChild(span);
}

function _renderText(
  el: HTMLElement,
  line: MushafLine,
  fam: string,
  revealedVerses: Set<string>,
  practiceMode: boolean,
): void {
  if (!line.words) return;

  for (const word of line.words) {
    const span = document.createElement("span");
    span.className = "mp-word";
    span.setAttribute("data-location", word.location);
    span.style.fontFamily = fam;
    span.textContent = word.qpcV1;

    const parts = word.location.split(":");
    const s = parts[0];
    const a = parts[1];
    span.setAttribute("data-surah", s);
    span.setAttribute("data-ayah", a);

    const key = `${s}:${a}`;
    if (practiceMode && !revealedVerses.has(key)) {
      span.classList.add("mp-word--hidden");
    }

    el.appendChild(span);
  }
}

// ── Verse reveal ──
export function revealVerse(
  container: HTMLElement,
  surah: number,
  ayah: number,
): void {
  const words = container.querySelectorAll<HTMLElement>(
    `.mp-word[data-surah="${surah}"][data-ayah="${ayah}"]`,
  );
  for (const w of words) {
    w.classList.remove("mp-word--hidden");
    w.classList.add("mp-word--revealed");
  }
}

// ── Word highlighting ──
export function highlightWord(
  container: HTMLElement,
  surah: number,
  ayah: number,
  matchedIndices: number[],
): void {
  // Clear previous current highlights
  for (const w of container.querySelectorAll<HTMLElement>(".mp-word--current")) {
    w.classList.remove("mp-word--current");
  }

  // Get all words for this verse, in DOM order
  const words = Array.from(
    container.querySelectorAll<HTMLElement>(
      `.mp-word[data-surah="${surah}"][data-ayah="${ayah}"]`,
    ),
  );
  if (!words.length) return;

  // Build contiguous max from matched indices
  const matched = new Set(matchedIndices);
  let contiguousMax = -1;
  for (let i = 0; i < words.length; i++) {
    if (matched.has(i)) contiguousMax = i;
    else break;
  }

  for (let i = 0; i < words.length; i++) {
    if (i <= contiguousMax) {
      words[i].classList.add("mp-word--spoken");
      words[i].classList.remove("mp-word--hidden");
      if (i === contiguousMax) {
        words[i].classList.add("mp-word--current");
      }
    }
  }
}

// ── Error highlighting (misread words) ──
export function highlightErrors(
  container: HTMLElement,
  surah: number,
  ayah: number,
  errorIndices: number[],
): void {
  const words = Array.from(
    container.querySelectorAll<HTMLElement>(
      `.mp-word[data-surah="${surah}"][data-ayah="${ayah}"]`,
    ),
  );
  if (!words.length) return;

  const errors = new Set(errorIndices);
  for (let i = 0; i < words.length; i++) {
    if (errors.has(i)) {
      words[i].classList.add("mp-word--error");
    } else {
      words[i].classList.remove("mp-word--error");
    }
  }
}

export function clearErrors(container: HTMLElement): void {
  for (const w of container.querySelectorAll<HTMLElement>(".mp-word--error")) {
    w.classList.remove("mp-word--error");
  }
}

// ── Reveal all (toggle practice mode off) ──
export function revealAll(container: HTMLElement): void {
  for (const w of container.querySelectorAll<HTMLElement>(".mp-word--hidden")) {
    w.classList.remove("mp-word--hidden");
  }
}

// ── Hide unrevealed (toggle practice mode on) ──
export function hideUnrevealed(
  container: HTMLElement,
  revealedVerses: Set<string>,
): void {
  for (const w of container.querySelectorAll<HTMLElement>(".mp-word")) {
    const s = w.getAttribute("data-surah") || "";
    const a = w.getAttribute("data-ayah") || "";
    const key = `${s}:${a}`;
    if (
      !revealedVerses.has(key) &&
      !w.classList.contains("mp-word--spoken")
    ) {
      w.classList.add("mp-word--hidden");
    }
  }
}

// ── Utility: get all verses on a page ──
export function getPageVerses(page: MushafPageData): string[] {
  const verses: string[] = [];
  const seen = new Set<string>();
  for (const line of page.lines) {
    if (line.type === "text" && line.words) {
      for (const w of line.words) {
        const parts = w.location.split(":");
        const key = `${parts[0]}:${parts[1]}`;
        if (!seen.has(key)) {
          seen.add(key);
          verses.push(key);
        }
      }
    }
  }
  return verses;
}
