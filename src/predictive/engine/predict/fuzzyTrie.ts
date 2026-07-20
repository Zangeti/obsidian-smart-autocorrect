/**
 * Fuzzy candidate index (#1) - a trie walked with a rolling weighted-edit DP
 * row (Ukkonen / Levenshtein-automaton-over-DAWG, the approach named in the
 * proposal). Unlike the old first-character bucketing it has COMPLETE recall
 * within a cost budget, so first-char errors it used to miss are found:
 *   "hte"->"the", "eth"->"the", "he"->"the", "tthe"->"the".
 *
 * One structure serves both:
 *   - "full"   : whole-word correction (autocorrect / space).
 *   - "prefix" : autocomplete - typed matches a prefix of a longer word.
 *
 * Cost uses the pluggable CostModel (geometry or empirical/adaptive), so the
 * personalised confusion model flows straight through.
 */
import type { CostModel } from "../channel/costModel.ts";
import { phoneticCost } from "../channel/phonetic.ts";

class TrieNode {
  children = new Map<string, TrieNode>();
  word: string | null = null;
  /** cheapest terminal words in this subtree (for prefix collection). */
}

export interface Neighbour {
  word: string;
  cost: number;
}

export class FuzzyTrie {
  private root = new TrieNode();
  private words: string[] = [];

  constructor(words?: Iterable<string>) {
    if (words) for (const w of words) this.insert(w);
  }

  insert(word: string): void {
    if (!word || word === "<s>") return;
    const w = word.toLowerCase();
    let node = this.root;
    for (const ch of w) {
      let next = node.children.get(ch);
      if (!next) {
        next = new TrieNode();
        node.children.set(ch, next);
      }
      node = next;
    }
    if (node.word === null) {
      node.word = w;
      this.words.push(w);
    }
  }

  get size(): number {
    return this.words.length;
  }

  /**
   * Exact prefix completions: all words that start with `prefix`. This is the
   * correct, cheap operation for autocomplete (descend to the prefix node, take
   * its subtree) - unlike the bounded fuzzy walk, it can't miss real completions
   * or return words that merely contain the prefix.
   */
  completions(prefix: string, limit: number): string[] {
    const p = prefix.toLowerCase();
    let node = this.root;
    for (const ch of p) {
      const next = node.children.get(ch);
      if (!next) return [];
      node = next;
    }
    const out: string[] = [];
    const stack: TrieNode[] = [node];
    while (stack.length && out.length < limit) {
      const n = stack.pop()!;
      if (n.word !== null) out.push(n.word);
      for (const child of n.children.values()) stack.push(child);
    }
    return out;
  }

  /**
   * Words within `maxCost` of `typed`. In prefix mode, the cost is that of
   * aligning typed to a *prefix* of the word (no penalty for the unseen suffix).
   */
  neighbours(
    typed: string,
    mode: "prefix" | "full",
    maxCost: number,
    cm: CostModel,
    opts: { limit?: number; usePhonetic?: boolean } = {},
  ): Neighbour[] {
    const t = typed.toLowerCase();
    const n = t.length;
    const limit = opts.limit ?? 64;
    if (n === 0) return [];

    // Column 0 (empty intended): typed vs "" = user typed n extra chars.
    const col0 = new Array(n + 1);
    col0[0] = 0;
    for (let i = 1; i <= n; i++) col0[i] = col0[i - 1] + cm.ins(t[i - 1], (i - 1) / Math.max(1, n - 1));

    const found = new Map<string, number>();
    const push = (word: string, cost: number) => {
      const prev = found.get(word);
      if (prev === undefined || cost < prev) found.set(word, cost);
    };

    // DFS carrying prevCol (parent) and prevPrevCol (grandparent, for transposition).
    const walk = (
      node: TrieNode,
      depth: number,
      prevCol: number[],
      prevPrevCol: number[] | null,
      prevChar: string,
    ) => {
      if (found.size >= limit * 4) return;
      for (const [ch, child] of node.children) {
        const posFrac = Math.min(1, (depth) / 6);
        const col = new Array(n + 1);
        col[0] = prevCol[0] + cm.del(ch, posFrac); // intended char, empty typed prefix
        let rowMin = col[0];
        for (let i = 1; i <= n; i++) {
          const ti = t[i - 1];
          const down = prevCol[i] + cm.del(ch, posFrac); // advance intended (user missed ch)
          const right = col[i - 1] + cm.ins(ti, posFrac); // user typed extra ti
          const diag = prevCol[i - 1] + cm.sub(ch, ti, posFrac, prevChar || undefined);
          let best = down < right ? down : right;
          if (diag < best) best = diag;
          if (
            prevPrevCol && i > 1 && ch === t[i - 2] && prevChar === ti
          ) {
            const tr = prevPrevCol[i - 2] + cm.trans(ch, prevChar, posFrac);
            if (tr < best) best = tr;
          }
          col[i] = best;
          if (best < rowMin) rowMin = best;
        }

        const alignCost = col[n];
        if (mode === "full") {
          if (child.word !== null && alignCost <= maxCost) push(child.word, alignCost);
        } else {
          // prefix: typed aligns to the prefix built so far; every terminal in
          // this subtree is a completion with (at most) this cost.
          if (alignCost <= maxCost) collectSubtree(child, alignCost, push, limit);
        }

        // Prune: if the whole column already exceeds budget, no descendant helps.
        if (rowMin <= maxCost) walk(child, depth + 1, col, prevCol, ch);
      }
    };

    walk(this.root, 1, col0, null, "");

    // Optional phonetic recall boost (whole-word only).
    if (opts.usePhonetic && mode === "full") {
      for (const w of this.words) {
        if (found.has(w)) continue;
        const pc = phoneticCost(t, w);
        if (pc <= maxCost) push(w, pc);
      }
    }

    const out = [...found.entries()].map(([word, cost]) => ({ word, cost }));
    out.sort((a, b) => a.cost - b.cost);
    return out.slice(0, limit);
  }
}

function collectSubtree(
  node: TrieNode,
  cost: number,
  push: (w: string, c: number) => void,
  limit: number,
): void {
  const stack: TrieNode[] = [node];
  let count = 0;
  while (stack.length && count < limit) {
    const n = stack.pop()!;
    if (n.word !== null) {
      push(n.word, cost);
      count++;
    }
    for (const child of n.children.values()) stack.push(child);
  }
}
