import { describe, it, expect } from "vitest";
import { cleanWhisperOutput } from "../../src/lib/text-cleanup.ts";

describe("cleanWhisperOutput", () => {
  it("returns empty string for empty input", () => {
    expect(cleanWhisperOutput("")).toBe("");
    expect(cleanWhisperOutput("  ")).toBe("");
  });

  it("trims whitespace", () => {
    expect(cleanWhisperOutput("  بسم الله  ")).toBe("بسم الله");
  });

  it("returns short text unchanged", () => {
    expect(cleanWhisperOutput("بسم")).toBe("بسم");
    expect(cleanWhisperOutput("بسم الله")).toBe("بسم الله");
  });

  it("collapses 3+ repetitions of a single word", () => {
    expect(cleanWhisperOutput("لله لله لله لله الرحمن")).toBe("لله الرحمن");
  });

  it("collapses 3+ repetitions of a two-word phrase", () => {
    expect(cleanWhisperOutput("بسم الله بسم الله بسم الله الرحمن")).toBe("بسم الله الرحمن");
  });

  it("collapses 3+ repetitions of a three-word phrase", () => {
    expect(cleanWhisperOutput("بسم الله الرحمن بسم الله الرحمن بسم الله الرحمن الرحيم")).toBe("بسم الله الرحمن الرحيم");
  });

  it("does NOT collapse 2 repetitions (only 3+)", () => {
    expect(cleanWhisperOutput("الله الله الرحمن")).toBe("الله الله الرحمن");
  });

  it("handles text with no repetitions unchanged", () => {
    const text = "بسم الله الرحمن الرحيم";
    expect(cleanWhisperOutput(text)).toBe(text);
  });

  it("handles multiple different repetition groups", () => {
    expect(cleanWhisperOutput("الله الله الله الرحمن الرحمن الرحمن الرحيم")).toBe("الله الرحمن الرحيم");
  });

  it("preserves non-Arabic text", () => {
    expect(cleanWhisperOutput("hello hello hello world")).toBe("hello world");
  });

  it("handles repetition at end of text", () => {
    expect(cleanWhisperOutput("بسم الله الله الله")).toBe("بسم الله");
  });

  it("handles entire text being one repeated word", () => {
    expect(cleanWhisperOutput("الله الله الله الله الله")).toBe("الله");
  });
});
