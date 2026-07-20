/**
 * Candidate generation + scoring pipeline.
 *
 *   candidates = top LM continuations of the context  ∪  fuzzy-trie neighbours
 *                of what was typed (complete recall within a cost budget, incl.
 *                a phonetic path).
 *   score      = Reranker over interpretable features, initialised to reproduce
 *                logP(w|context) − β·channelCost, then adapts to the user.
 *
 * Backward compatible: `predict(model, index, opts)` still works. `index` may be
 * a FuzzyTrie (preferred, complete recall) or the legacy FuzzyIndex. When no
 * CostModel/Reranker is supplied it falls back to geometry + the fixed linear
 * score, so older callers/tests behave exactly as before.
 */
import { channelCost, prefixCost, weightedEdit } from "../channel/editDistance.ts";
import {
  type ChannelConfig,
  DEFAULT_CHANNEL,
  keyDistance,
} from "../channel/keyboard.ts";
import { type CostModel, GeometryCostModel } from "../channel/costModel.ts";
import { phoneticCost } from "../channel/phonetic.ts";
import type { LanguageModel } from "../ngram/model.ts";
import { FEATURE_NAMES, type FeatureVector, Reranker } from "./reranker.ts";

export interface Candidate {
  word: string;
  score: number;
  logProbLM: number;
  channelCost: number;
  features?: FeatureVector;
}

/** Anything that can enumerate fuzzy neighbours + exact completions of a string. */
export interface CandidateIndex {
  neighbours(
    typed: string,
    mode: "prefix" | "full",
    maxCost: number,
    cm: CostModel,
    opts?: { limit?: number; usePhonetic?: boolean },
  ): { word: string; cost: number }[];
  /** Exact prefix completions of `prefix`. */
  completions(prefix: string, limit: number): string[];
}

export interface PredictOptions {
  context: string[];
  typed: string;
  k: number;
  beta?: number;
  channel?: ChannelConfig;
  mode?: "prefix" | "full";
  maxCost?: number;
  lmPool?: number;
  costModel?: CostModel;
  reranker?: Reranker;
  /** weight on the keyboard-geometry channel (higher = keyboard typos cheaper). */
  fuzzyStrength?: number;
  /** weight on the phonetic channel (higher = sound-alike typos cheaper). */
  phoneticStrength?: number;
}

function buildFeatures(
  word: string,
  typed: string,
  logProbLM: number,
  cost: number,
  phonetic: boolean,
  model: LanguageModel,
): FeatureVector {
  const prefixExact = typed && word.startsWith(typed.toLowerCase()) ? 1 : 0;
  const caseMatch = typed && typed[0] === typed[0].toLowerCase() ? 1 : 0;
  const f = new Array(FEATURE_NAMES.length).fill(0);
  f[0] = logProbLM; // lmLogProb
  f[1] = -cost; // negChannel
  f[2] = prefixExact;
  f[3] = model.logProb(word, []); // unigram
  f[4] = caseMatch;
  f[5] = phonetic ? 1 : 0;
  f[6] = -Math.abs(word.length - typed.length);
  f[7] = 1; // bias
  return f;
}

export function predict(
  model: LanguageModel,
  index: CandidateIndex,
  opts: PredictOptions,
): Candidate[] {
  const beta = opts.beta ?? 1.0;
  const channel = opts.channel ?? DEFAULT_CHANNEL;
  const cm = opts.costModel ?? new GeometryCostModel(channel);
  const mode = opts.mode ?? "prefix";
  const maxCost = opts.maxCost ?? 4.0;
  const lmPool = opts.lmPool ?? Math.max(opts.k * 4, 40);
  const reranker = opts.reranker ?? null;
  // Channel-path weights. The two pathways are combined as a weighted MINIMUM
  // (best explanation wins) rather than added, so a strong phonetic match isn't
  // negated by large keyboard distance, and vice-versa. Strength ↑ ⇒ that path's
  // cost ↓ ⇒ it explains more typos; strength 0 turns that path off entirely.
  const fuzzyStrength = opts.fuzzyStrength ?? 1;
  const phoneticStrength = opts.phoneticStrength ?? 1;
  const usePhonetic = phoneticStrength > 0; // gate the trie's phonetic recall

  const scoreOf = (f: FeatureVector, logProbLM: number, cost: number): number =>
    reranker ? reranker.score(f) : logProbLM - beta * cost;

  // Pure next-word prediction when nothing is typed.
  if (!opts.typed) {
    return model.predict(opts.context, opts.k).map((s) => {
      const f = buildFeatures(s.word, "", s.logProb, 0, false, model);
      return { word: s.word, score: scoreOf(f, s.logProb, 0), logProbLM: s.logProb, channelCost: 0, features: f };
    });
  }

  const alignCost = (word: string): { cost: number; phonetic: boolean } => {
    const geo = mode === "prefix" ? prefixCost(opts.typed, word, cm, maxCost) : weightedEdit(opts.typed, word, cm, maxCost);
    let cost = fuzzyStrength > 0 ? geo / fuzzyStrength : Infinity;
    let phonetic = false;
    if (phoneticStrength > 0) {
      // Phonetics explains SOUND substitutions, not the user typing characters the
      // candidate lacks (or omitting ones it has). Those |Δlen| characters are real
      // orthographic evidence that must be inserted/deleted however alike the two words
      // sound, so floor the phonetic path with the channel's own calibrated indel cost for
      // them. Without it a far-commoner sound-alike that DROPS typed letters ("genric" /
      // "genreic" -> the corpus-saturated "genre") beats the true, far closer correction
      // ("generic"), because its bare phonetic cost is tiny and its frequency prior does the
      // rest - the same distorted-prior failure as the casing homographs.
      const lenFloor = Math.abs(opts.typed.length - word.length) * channel.deleteCost;
      const pc = phoneticCost(opts.typed, word) / phoneticStrength + lenFloor;
      if (pc < cost) {
        cost = pc;
        phonetic = true;
      }
    }
    return { cost, phonetic };
  };

  const typedLower = opts.typed.toLowerCase();
  const isPrefixMatch = (w: string) => mode === "prefix" && typedLower.length > 0 && w.startsWith(typedLower);

  const byWord = new Map<string, Candidate>();
  const consider = (word: string, cost: number, phonetic: boolean) => {
    if (cost > maxCost) return;
    const logProbLM = model.logProb(word, opts.context);
    const f = buildFeatures(word, opts.typed, logProbLM, cost, phonetic, model);
    const score = scoreOf(f, logProbLM, cost);
    const existing = byWord.get(word);
    if (!existing || score > existing.score)
      byWord.set(word, { word, score, logProbLM, channelCost: cost, features: f });
  };

  // 1. Context-driven pool.
  for (const s of model.predict(opts.context, lmPool)) {
    const { cost, phonetic } = alignCost(s.word);
    consider(s.word, cost, phonetic);
  }

  // 2. Exact prefix completions (autocomplete). These are what you're typing, so
  // they're free (cost 0) and ranked purely by context probability.
  if (mode === "prefix" && opts.typed) {
    for (const word of index.completions(opts.typed, 400)) consider(word, 0, false);
  }

  // 3. Input-driven fuzzy pool (typo tolerance). Surfaces corrections when what
  // you typed isn't an exact prefix of anything. The trie decides RECALL (which
  // words are near); alignCost re-scores them so the strength weights apply the
  // same way as for the context pool.
  for (const { word } of index.neighbours(opts.typed, mode, maxCost, cm, { usePhonetic })) {
    const { cost, phonetic } = alignCost(word);
    consider(word, cost, phonetic);
  }

  // When you're mid-word and something actually completes what you typed, show
  // ONLY those completions (ranked by context) - never pull in unrelated words.
  // Fuzzy typo matches surface only when nothing you typed is a real prefix.
  let out = [...byWord.values()];
  if (mode === "prefix" && typedLower.length > 0) {
    const exact = out.filter((c) => isPrefixMatch(c.word));
    if (exact.length > 0) out = exact;
  }
  out.sort((a, b) => b.score - a.score);
  return out.slice(0, opts.k);
}

/**
 * Legacy first-character-bucketed fuzzy index. Retained for compatibility;
 * FuzzyTrie is preferred (complete recall). Implements CandidateIndex.
 */
export class FuzzyIndex implements CandidateIndex {
  private byFirst = new Map<string, string[]>();

  constructor(words: Iterable<string>) {
    for (const w of words) {
      if (!w || w === "<s>") continue;
      const f = w[0];
      let bucket = this.byFirst.get(f);
      if (!bucket) {
        bucket = [];
        this.byFirst.set(f, bucket);
      }
      bucket.push(w);
    }
  }

  neighbours(
    typed: string,
    mode: "prefix" | "full",
    maxCost: number,
    cm: CostModel,
  ): { word: string; cost: number }[] {
    if (!typed) return [];
    const first = typed[0].toLowerCase();
    const firsts = new Set<string>([first]);
    for (const [f] of this.byFirst) if (f !== first && keyDistance(f, first) <= 1.2) firsts.add(f);
    const out: { word: string; cost: number }[] = [];
    for (const f of firsts) {
      const bucket = this.byFirst.get(f);
      if (!bucket) continue;
      for (const w of bucket) {
        const target = mode === "prefix" ? w.slice(0, typed.length + 2) : w;
        const cost = weightedEdit(typed, target, cm, maxCost);
        if (cost <= maxCost) out.push({ word: w, cost });
      }
    }
    return out;
  }

  completions(prefix: string, limit: number): string[] {
    const p = prefix.toLowerCase();
    const bucket = this.byFirst.get(p[0]);
    if (!bucket) return [];
    const out: string[] = [];
    for (const w of bucket) {
      if (w.startsWith(p)) out.push(w);
      if (out.length >= limit) break;
    }
    return out;
  }
}

// re-export for convenience
export { channelCost };
