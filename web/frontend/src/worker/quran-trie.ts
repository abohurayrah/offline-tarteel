import { BPETokenizer } from "./forced-alignment";

interface TrieNode {
  children: Map<number, TrieNode>;
  isVerseEnd: boolean;
  verseRefs: Array<{ surah: number; ayah: number }>;
}

export class QuranTrie {
  root: TrieNode;
  private tokenizer: BPETokenizer;
  private totalNodes = 0;

  constructor(vocabJson: Record<string, string>) {
    this.root = { children: new Map(), isVerseEnd: false, verseRefs: [] };
    this.tokenizer = new BPETokenizer(vocabJson);
    this.totalNodes = 1;
  }

  /**
   * Add a verse as a BPE token sequence to the trie.
   */
  addVerse(tokenIDs: number[], surah: number, ayah: number): void {
    let node = this.root;

    for (const tokenID of tokenIDs) {
      if (!node.children.has(tokenID)) {
        node.children.set(tokenID, {
          children: new Map(),
          isVerseEnd: false,
          verseRefs: [],
        });
        this.totalNodes++;
      }
      node = node.children.get(tokenID)!;
    }

    node.isVerseEnd = true;
    node.verseRefs.push({ surah, ayah });
  }

  /**
   * Check if token sequence is a valid prefix of any verse.
   */
  isValidPrefix(tokenIDs: number[]): boolean {
    let node = this.root;
    for (const tokenID of tokenIDs) {
      if (!node.children.has(tokenID)) {
        return false;
      }
      node = node.children.get(tokenID)!;
    }
    return true;
  }

  /**
   * Get valid next token IDs from current prefix.
   * Returns all token IDs that can extend this prefix.
   */
  getValidNextTokens(tokenIDs: number[]): number[] {
    let node = this.root;
    for (const tokenID of tokenIDs) {
      if (!node.children.has(tokenID)) {
        return [];
      }
      node = node.children.get(tokenID)!;
    }
    return Array.from(node.children.keys());
  }

  /**
   * Navigate to trie node for a given token sequence.
   * Returns null if path doesn't exist.
   */
  getNode(tokenIDs: number[]): TrieNode | null {
    let node = this.root;
    for (const tokenID of tokenIDs) {
      if (!node.children.has(tokenID)) {
        return null;
      }
      node = node.children.get(tokenID)!;
    }
    return node;
  }

  /**
   * Build trie from QuranDB verses.
   */
  buildFromVerses(verses: Array<{ text_norm: string; surah: number; ayah: number }>): void {
    console.log(`[Trie] Building from ${verses.length} verses...`);

    for (const verse of verses) {
      if (!verse.text_norm) continue;

      const { tokenIDs } = this.tokenizer.tokenize(verse.text_norm);
      this.addVerse(tokenIDs, verse.surah, verse.ayah);
    }

    console.log(`[Trie] Built: ${this.totalNodes} nodes, ${verses.length} verses`);
  }

  get nodeCount(): number {
    return this.totalNodes;
  }
}
