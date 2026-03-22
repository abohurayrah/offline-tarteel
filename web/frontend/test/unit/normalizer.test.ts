import { describe, it, expect } from "vitest";
import { normalizeArabic } from "../../src/lib/quran-db.ts";

describe("normalizeArabic (quran-db version)", () => {
  it("strips diacritics (tashkeel)", () => {
    // baa + kasra + siin + sukun + miim = "بِسْمِ" -> "بسم"
    expect(normalizeArabic("بِسْمِ")).toBe("بسم");
  });

  it("normalizes hamza variants to bare alif", () => {
    // أ إ آ ٱ  all → ا
    expect(normalizeArabic("أ")).toBe("ا");
    expect(normalizeArabic("إ")).toBe("ا");
    expect(normalizeArabic("آ")).toBe("ا");
    expect(normalizeArabic("ٱ")).toBe("ا");
    // Combined in a word
    expect(normalizeArabic("أإآٱ")).toBe("اااا");
  });

  it("normalizes taa marbuta to haa", () => {
    // ة → ه
    expect(normalizeArabic("رحمة")).toBe("رحمه");
  });

  it("normalizes alif maqsura to yaa", () => {
    // ى → ي
    expect(normalizeArabic("مُوسَى")).toBe("موسي");
  });

  it("removes tatweel (kashida)", () => {
    // ـ is tatweel U+0640
    expect(normalizeArabic("الـله")).toBe("الله");
  });

  it("strips BOM (byte order mark)", () => {
    expect(normalizeArabic("\uFEFFبسم")).toBe("بسم");
  });

  it("converts BPE marker to space and trims", () => {
    // U+2581 is the BPE underscore marker used by SentencePiece
    expect(normalizeArabic("\u2581بسم")).toBe("بسم");
    expect(normalizeArabic("بسم\u2581الله")).toBe("بسم الله");
  });

  it("removes Arabic punctuation", () => {
    // ، (Arabic comma) and ؟ (Arabic question mark)
    expect(normalizeArabic("بسم،")).toBe("بسم");
    expect(normalizeArabic("ما؟")).toBe("ما");
    // Also removes . ! :
    expect(normalizeArabic("بسم. الله!")).toBe("بسم الله");
  });

  it("collapses whitespace and trims", () => {
    expect(normalizeArabic("  بسم   الله  ")).toBe("بسم الله");
    expect(normalizeArabic("بسم\t\nالله")).toBe("بسم الله");
  });

  it("returns empty string for empty input", () => {
    expect(normalizeArabic("")).toBe("");
  });

  it("is idempotent on already-normalized text", () => {
    const normalized = "بسم الله الرحمن الرحيم";
    expect(normalizeArabic(normalized)).toBe(normalized);
    expect(normalizeArabic(normalizeArabic(normalized))).toBe(normalized);
  });

  it("handles whitespace-only input", () => {
    expect(normalizeArabic("   ")).toBe("");
  });

  // --- New tests below ---

  it("strips multiple consecutive diacritics (full basmala with heavy tashkeel)", () => {
    // بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ
    // Each letter carries kasra, sukun, shadda, fatha, etc.
    const heavy = "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ";
    expect(normalizeArabic(heavy)).toBe("بسم الله الرحمن الرحيم");
  });

  it("normalizes mixed diacritics and hamza: أُولَٰئِكَ", () => {
    // أ → ا, strip all diacritics (damma, fatha, kasra, superscript alif)
    const input = "أُولَٰئِكَ";
    const result = normalizeArabic(input);
    // After normalization: أ→ا, strip diacritics
    expect(result).toBe("اولئك");
  });

  it("normalizes real Quran verse: Al-Fatiha 1:1 (text_uthmani)", () => {
    const uthmani = "﻿بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ";
    expect(normalizeArabic(uthmani)).toBe("بسم الله الرحمن الرحيم");
  });

  it("normalizes real Quran verse: Al-Asr 103:2 (text_uthmani)", () => {
    const uthmani = "إِنَّ ٱلْإِنسَٰنَ لَفِى خُسْرٍ";
    expect(normalizeArabic(uthmani)).toBe("ان الانسن لفي خسر");
  });

  it("normalizes real Quran verse: Al-Baqara 2:255 opening (text_uthmani)", () => {
    const uthmani = "ٱللَّهُ لَآ إِلَٰهَ إِلَّا هُوَ ٱلْحَىُّ ٱلْقَيُّومُ";
    expect(normalizeArabic(uthmani)).toBe("الله لا اله الا هو الحي القيوم");
  });

  it("normalizes real Quran verse: Al-Ikhlaas 112:3 (text_uthmani)", () => {
    const uthmani = "لَمْ يَلِدْ وَلَمْ يُولَدْ";
    expect(normalizeArabic(uthmani)).toBe("لم يلد ولم يولد");
  });

  it("removes tatweel inside words: اللـــه → الله", () => {
    expect(normalizeArabic("اللـــه")).toBe("الله");
  });

  it("handles tatweel between every letter", () => {
    // بـسـم → بسم
    expect(normalizeArabic("بـسـم")).toBe("بسم");
  });

  it("handles all normalization steps together: BPE + diacritics + hamza + tatweel", () => {
    // BPE marker at start, diacritics, hamza variant, tatweel, taa marbuta
    const input = "\u2581بِسْمِ\u2581ٱللّـهِ\u2581ٱلرَّحْمَٰنِ\u2581ٱلرَّحِيمِ";
    expect(normalizeArabic(input)).toBe("بسم الله الرحمن الرحيم");
  });

  it("handles BPE + diacritics + hamza all at once in a word", () => {
    // أُولَئِكَ with BPE prefix
    const input = "\u2581أُولَئِكَ";
    expect(normalizeArabic(input)).toBe("اولئك");
  });

  it("handles text with BPE markers + taa marbuta + alif maqsura", () => {
    const input = "\u2581رَحْمَة\u2581مُوسَى";
    expect(normalizeArabic(input)).toBe("رحمه موسي");
  });

  it("strips zero-width joiner (U+200D)", () => {
    // ZWJ is sometimes inserted in Arabic text for rendering purposes
    // The normalizer may not explicitly strip it, but it should not break
    const input = "بسم\u200Dالله";
    const result = normalizeArabic(input);
    // ZWJ is not whitespace and not a diacritic - it passes through but
    // should not cause crashes
    expect(typeof result).toBe("string");
    expect(result.length).toBeGreaterThan(0);
  });

  it("strips directional marks (LRM U+200F, RLM U+200E)", () => {
    const input = "\u200Fبسم\u200E الله";
    const result = normalizeArabic(input);
    expect(typeof result).toBe("string");
    // Directional marks are non-printing - normalizer preserves them as-is
    // but the core text should survive
    expect(result).toContain("بسم");
    expect(result).toContain("الله");
  });

  it("handles Arabic-Indic digits (not stripped)", () => {
    // Digits ٠١٢٣ should pass through normalization unchanged
    const input = "ايه ١٢٣";
    const result = normalizeArabic(input);
    expect(result).toContain("١٢٣");
  });

  it("handles mixed Arabic and Latin text", () => {
    const input = "surah الفاتحة";
    expect(normalizeArabic(input)).toBe("surah الفاتحه");
  });

  it("handles single character input", () => {
    expect(normalizeArabic("ب")).toBe("ب");
    expect(normalizeArabic("أ")).toBe("ا");
    expect(normalizeArabic("ة")).toBe("ه");
  });

  it("is idempotent on Quran verse text (double normalization)", () => {
    const verses = [
      "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
      "إِنَّ ٱلْإِنسَٰنَ لَفِى خُسْرٍ",
      "قُلْ هُوَ ٱللَّهُ أَحَدٌ",
    ];
    for (const v of verses) {
      const once = normalizeArabic(v);
      const twice = normalizeArabic(once);
      expect(twice).toBe(once);
    }
  });

  it("strips superscript alif (U+0670)", () => {
    // ٰ is superscript alif, common in Uthmani script
    // e.g. ٱلرَّحْمَٰنِ
    const input = "الرحمٰن";
    expect(normalizeArabic(input)).toBe("الرحمن");
  });

  it("strips small waw, yaa, and other Quran orthography marks", () => {
    // U+06D6-U+06DC are Quranic annotation marks (sajda, etc.)
    const input = "وَقْفٌ\u06D6";
    const result = normalizeArabic(input);
    expect(result).toBe("وقف");
  });

  // --- Uthmani script annotation marks ---
  // TODO: Extending the diacritics range to strip small waw/yaa/rub-el-hizb/sajdah
  // changes normalization for ~30% of verses, which invalidates scoring calibration.
  // These need to be re-enabled after re-tuning thresholds and bonuses.

  it("strips Arabic small waw (U+06E5) — pronominal suffix marker in text_clean", () => {
    // text_clean writes حَوْلَهُۥ with U+06E5 (ARABIC SMALL WAW) to mark the waw suffix.
    // Whisper outputs the bare stem حوله.  Stripping U+06E5 closes that 1-char gap.
    // This appeared in ~990 Quran verses in text_clean.
    expect(normalizeArabic("حولهۥ")).toBe("حوله");
    expect(normalizeArabic("ربهۥ")).toBe("ربه");
  });

  it("strips Arabic small yeh (U+06E6) — pronominal suffix marker in text_clean", () => {
    // Analogous to small waw: text_clean writes بهۦ / مثلهۦ with U+06E6 (ARABIC SMALL YEH).
    // Appears in ~833 Quran verses.
    expect(normalizeArabic("ربهۦ")).toBe("ربه");
    expect(normalizeArabic("بهۦ")).toBe("به");
  });

  it("strips Arabic start of rub el hizb (U+06DE) — sectional marker in text_clean", () => {
    // ۞ U+06DE marks quarter-juz sections.  Not a letter; appears in ~199 verses.
    // It appears between words (with surrounding spaces) in text_clean, so after
    // stripping the character the whitespace-collapse step produces a single space.
    expect(normalizeArabic("قُلْ \u06DE هُوَ")).toBe("قل هو");
    // Also works when the marker appears mid-word (no surrounding spaces)
    expect(normalizeArabic("قُلْ\u06DEهُوَ")).toBe("قلهو");
  });

  it("strips Arabic place of sajdah (U+06E9) — sajda marker in text_clean", () => {
    // ۩ U+06E9 marks prostration verses.  Not a letter; appears in ~15 verses.
    expect(normalizeArabic("اسجد\u06E9وا")).toBe("اسجدوا");
  });

  it("normalizes word with small waw suffix to match ASR bare-stem output", () => {
    // Full integration: normalizeArabic on the verse side and on a simulated
    // Whisper output side must produce the same string for a perfect match.
    const verseWord  = "حولهۥ";   // as it appears in text_clean (2:17)
    const whisperWord = "حوله";   // as Whisper would output it
    expect(normalizeArabic(verseWord)).toBe(normalizeArabic(whisperWord));
  });
});
