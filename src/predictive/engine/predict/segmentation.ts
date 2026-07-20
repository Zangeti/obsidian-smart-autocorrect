/**
 * Split/join and real-word error correction (#4) - errors the within-word
 * matcher structurally cannot see.
 *
 *  - segment():   Viterbi over unigram log-probs to split a run-on token
 *                 ("thebank" -> "the bank", "alot" -> "a lot").
 *  - RealWordCorrector: a *valid* word can still be wrong ("form"/"from",
 *                 "their"/"there"); only context reveals it. We gather confusion
 *                 peers (built-in sets + phonetic homophones) and let the LM pick.
 */
import type { LanguageModel } from "../ngram/model.ts";
import { phoneticKey, phoneticCost } from "../channel/phonetic.ts";

export interface Segmentation {
  words: string[];
  logProb: number;
}

/** Best split of a lower-cased run-on token using unigram log-probs. */
export function segment(
  token: string,
  model: LanguageModel,
  opts: { maxPieces?: number; minPieceLen?: number } = {},
): Segmentation {
  const t = token.toLowerCase();
  const N = t.length;
  const minLen = opts.minPieceLen ?? 1;
  // dp[i] = best {logProb, prev} for prefix t[0..i]
  const best = new Array<number>(N + 1).fill(-Infinity);
  const back = new Array<number>(N + 1).fill(-1);
  best[0] = 0;
  for (let i = 1; i <= N; i++) {
    for (let j = Math.max(0, i - 20); j < i; j++) {
      if (best[j] === -Infinity) continue;
      const piece = t.slice(j, i);
      if (piece.length < minLen && !(piece === "a" || piece === "i")) continue;
      // penalise unknown pieces so we don't over-split into gibberish.
      const lp = model.hasWord(piece)
        ? model.logProb(piece, [])
        : model.logProb(piece, []) - 6;
      if (best[j] + lp > best[i]) {
        best[i] = best[j] + lp;
        back[i] = j;
      }
    }
  }
  const words: string[] = [];
  let i = N;
  while (i > 0) {
    const j = back[i];
    if (j < 0) return { words: [t], logProb: model.logProb(t, []) };
    words.unshift(t.slice(j, i));
    i = j;
  }
  return { words, logProb: best[N] };
}

/**
 * Joint log-probability of reading `pieces` as consecutive words after `context`, each scored
 * in the context the ones before it establish. Shared with the re-spacing decision so that
 * "one token or two" is always judged the same way.
 */
export function sequenceLogProb(
  model: LanguageModel,
  pieces: string[],
  context: string[],
): number {
  let ctx = context;
  let total = 0;
  for (const p of pieces) {
    total += model.logProb(p, ctx);
    ctx = [...ctx, p];
  }
  return total;
}

/**
 * Suggest a split only when it's clearly better than the whole token AND the
 * pieces are all real words (avoids splitting genuine words).
 *
 * Both readings are scored IN CONTEXT. Scoring them by context-free unigrams (the old
 * behaviour) compared the pieces' raw frequency against the model's unknown-word floor, so a
 * split only won when a piece was very common indeed: "thebank" split, but "morehesitant"
 * cleared the floor by 1.4 nats and stayed broken. What actually makes a split obvious is that
 * the pieces read well HERE - "i am more hesitant" - which is exactly what context supplies.
 *
 * A lone letter other than "a"/"i" is never a word in English, however happily the model
 * scores it as a token - so a "split" that strands one is not a reading of the text at all.
 * Without this guard "youd" split to "you d" instead of reaching the contraction fix, which is
 * the worst kind of correction: it turns a near-miss into nonsense.
 */
export function suggestSplit(
  token: string,
  model: LanguageModel,
  context: string[] = [],
  margin = 2.0,
): string[] | null {
  const t = token.toLowerCase();
  if (t.length < 3) return null;
  const seg = segment(t, model);
  if (seg.words.length < 2) return null;
  if (seg.words.some((w) => !model.hasWord(w))) return null;
  if (seg.words.some((w) => w.length === 1 && w !== "a" && w !== "i")) return null;
  const whole = model.logProb(t, context);
  if (sequenceLogProb(model, seg.words, context) > whole + margin) return seg.words;
  return null;
}

const BUILTIN_CONFUSION: string[][] = [
  ["their", "there", "theyre"],
  ["your", "youre"],
  ["its", "it's"],
  ["to", "too", "two"],
  ["form", "from"],
  ["then", "than"],
  ["affect", "effect"],
  ["accept", "except"],
  ["lose", "loose"],
  ["were", "where", "wear"],
  ["quiet", "quite"],
  ["of", "off"],
];

export class RealWordCorrector {
  private byPhonetic = new Map<string, string[]>();
  private confusion = new Map<string, Set<string>>();

  constructor(vocabulary: Iterable<string>) {
    for (const w of vocabulary) {
      if (!w || w === "<s>") continue;
      const key = phoneticKey(w);
      if (!key) continue;
      const arr = this.byPhonetic.get(key);
      if (arr) arr.push(w);
      else this.byPhonetic.set(key, [w]);
    }
    for (const set of BUILTIN_CONFUSION) {
      for (const w of set) {
        let peers = this.confusion.get(w);
        if (!peers) {
          peers = new Set();
          this.confusion.set(w, peers);
        }
        for (const p of set) if (p !== w) peers.add(p);
      }
    }
  }

  private peers(word: string): string[] {
    const out = new Set<string>();
    const c = this.confusion.get(word);
    if (c) for (const p of c) out.add(p);
    const ph = this.byPhonetic.get(phoneticKey(word));
    if (ph) for (const p of ph) if (p !== word) out.add(p);
    return [...out];
  }

  /**
   * If `word` is valid but a confusion peer is much more likely in context,
   * return that peer (else null). `margin` in nats.
   */
  bestAlternative(
    word: string,
    context: string[],
    model: LanguageModel,
    margin = 2.5,
  ): string | null {
    const w = word.toLowerCase();
    const base = model.logProb(w, context);
    let best: string | null = null;
    let bestScore = base + margin;
    for (const peer of this.peers(w)) {
      if (!model.hasWord(peer)) continue;
      // Weigh each peer by how plausible the confusion actually is: its context
      // probability discounted by the channel cost of the two words being
      // confused. A genuine homophone ("from" for "form") is a cheap confusion
      // and competes on context; a far same-sounding word ("whereof" for "worf")
      // carries a large channel cost and loses to the fuzzy spelling fix instead.
      const score = model.logProb(peer, context) - phoneticCost(w, peer);
      if (score > bestScore) {
        bestScore = score;
        best = peer;
      }
    }
    return best;
  }
}
