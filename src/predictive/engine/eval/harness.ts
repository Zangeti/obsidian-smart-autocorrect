/**
 * Evaluation harness (#D8). Measures how well the system actually corrects typos
 * and predicts words, so β / thresholds / the reranker can be tuned with data
 * instead of guesswork, and regressions are caught.
 *
 * Method: take clean held-out text, use a keyboard-realistic corruption model to
 * inject typos, then run the predictor/autocorrect and score recovery:
 *   - correction accuracy: top candidate equals the original word
 *   - recall@k:            original word appears in the top-k
 *   - autocorrect precision/recall on the words we chose to change
 *   - keystroke savings:   fraction of characters a user avoids typing
 *
 * Deterministic (seeded RNG) so results are reproducible in tests.
 */
import { tokenizeWords } from "../text/tokenize.ts";
import { keyDistance, type KeyboardLayoutName } from "../channel/keyboard.ts";
import type { LanguageModel } from "../ngram/model.ts";
import { type CandidateIndex, predict } from "./../predict/predictor.ts";
import type { CostModel } from "../channel/costModel.ts";

/** Small deterministic PRNG (mulberry32). */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const QWERTY = "qwertyuiopasdfghjklzxcvbnm";

function adjacentKey(ch: string, rand: () => number, layout: KeyboardLayoutName): string {
  const candidates = QWERTY.split("").filter((k) => k !== ch && keyDistance(ch, k, layout) <= 1.2);
  if (candidates.length === 0) return ch;
  return candidates[Math.floor(rand() * candidates.length)];
}

export interface CorruptOptions {
  /** probability of applying an error to a given word. */
  errorRate?: number;
  layout?: KeyboardLayoutName;
}

/** Inject a single realistic typo into `word` with probability errorRate. */
export function corrupt(word: string, rand: () => number, opts: CorruptOptions = {}): string {
  const errorRate = opts.errorRate ?? 1.0;
  const layout = opts.layout ?? "qwerty";
  if (word.length < 2 || rand() > errorRate) return word;
  const i = Math.floor(rand() * word.length);
  const roll = rand();
  const chars = word.split("");
  if (roll < 0.4) {
    // substitution with an adjacent key
    chars[i] = adjacentKey(chars[i], rand, layout);
  } else if (roll < 0.6) {
    // deletion (missed char)
    chars.splice(i, 1);
  } else if (roll < 0.8) {
    // insertion of an adjacent key
    chars.splice(i, 0, adjacentKey(chars[i], rand, layout));
  } else if (i < word.length - 1) {
    // transposition
    const t = chars[i];
    chars[i] = chars[i + 1];
    chars[i + 1] = t;
  } else {
    chars[i] = adjacentKey(chars[i], rand, layout);
  }
  return chars.join("");
}

export interface EvalResult {
  words: number;
  corrupted: number;
  correctionAccuracy: number; // of corrupted words, fraction whose top candidate == original
  recallAtK: number; // of corrupted words, fraction with original in top-k
  keystrokeSavings: number; // fraction of chars saved by accepting top prediction
}

export interface EvalOptions {
  seed?: number;
  errorRate?: number;
  k?: number;
  beta?: number;
  costModel?: CostModel;
  layout?: KeyboardLayoutName;
}

/**
 * Run the evaluation over `text`. Each word (with up to 2 words of left context)
 * is corrupted and the predictor is asked to recover it (full-word mode).
 */
export function evaluate(
  model: LanguageModel,
  index: CandidateIndex,
  text: string,
  opts: EvalOptions = {},
): EvalResult {
  const rand = rng(opts.seed ?? 12345);
  const k = opts.k ?? 5;
  const words = tokenizeWords(text);
  let corrupted = 0;
  let corrected = 0;
  let recalled = 0;
  let savedChars = 0;
  let totalChars = 0;

  for (let i = 0; i < words.length; i++) {
    const original = words[i];
    if (original.length < 2) continue;
    const context = words.slice(Math.max(0, i - 2), i);
    totalChars += original.length;

    // keystroke savings: how many chars until the model's top prediction == original?
    for (let p = 1; p <= original.length; p++) {
      const typedPrefix = original.slice(0, p);
      const cands = predict(model, index, {
        context,
        typed: typedPrefix,
        k: 1,
        mode: "prefix",
        beta: opts.beta,
        costModel: opts.costModel,
      });
      if (cands.length && cands[0].word === original) {
        savedChars += original.length - p;
        break;
      }
    }

    const typo = corrupt(original, rand, { errorRate: opts.errorRate ?? 1.0, layout: opts.layout });
    if (typo === original) continue;
    corrupted++;
    const cands = predict(model, index, {
      context,
      typed: typo,
      k,
      mode: "full",
      beta: opts.beta,
      costModel: opts.costModel,
    });
    if (cands.length && cands[0].word === original) corrected++;
    if (cands.some((c) => c.word === original)) recalled++;
  }

  return {
    words: words.length,
    corrupted,
    correctionAccuracy: corrupted ? corrected / corrupted : 0,
    recallAtK: corrupted ? recalled / corrupted : 0,
    keystrokeSavings: totalChars ? savedChars / totalChars : 0,
  };
}
