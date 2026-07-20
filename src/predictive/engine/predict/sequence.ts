/**
 * Unified sequence predictor (single-word AND multi-word) via beam search - the
 * same decoding framework Gmail Smart Compose uses: expand the top-k next words,
 * keep the best `beamWidth` partial phrases, and rank completions by a
 * LENGTH-NORMALISED log-probability (so long and short suggestions compete
 * fairly). Diversity is intentionally NOT sought - we want the single most
 * likely continuation, not variety.
 *
 * "Unified architecture": single-word prediction is just beam search with
 * maxWords = 1. Multi-word is the same call with maxWords = N. The scorer is any
 * LanguageModel - the fast n-gram (with within-document cache + personal vault
 * bias, so phrases track what you're actually writing about) by default, or an
 * optional neural model (see NeuralSequenceModel) for transformer-level quality.
 */
import { SOS } from "../text/tokenize.ts";
import type { LanguageModel } from "../ngram/model.ts";

export interface Phrase {
  words: string[];
  /** total (unnormalised) log-probability of the phrase. */
  logProb: number;
  /** length-normalised confidence used for ranking/thresholding. */
  score: number;
}

export interface BeamOptions {
  beamWidth?: number; // Smart Compose uses 3-5
  maxWords?: number; // 1 = single word; 10-15 for phrases
  stepK?: number; // candidates expanded per beam per step
  /**
   * Per-word length reward (GNMT-style): score = logProb + wordReward·length.
   * A phrase is worth extending while each added word's log-prob exceeds
   * −wordReward, i.e. this doubles as a natural "only continue when confident"
   * stopping rule. Higher ⇒ longer phrases.
   */
  wordReward?: number;
  /** minimum per-word average log-prob for a completion to be surfaced. */
  minAvgLogProb?: number;
  /** words that end a phrase (sentence enders); extension stops after them. */
  stops?: Set<string>;
}

const DEFAULT_STOPS = new Set([".", "!", "?", SOS]);

interface Beam {
  words: string[];
  logProb: number;
}

function scoreOf(logProb: number, len: number, reward: number): number {
  if (len <= 0) return -Infinity;
  return logProb + reward * len;
}

/**
 * Beam-search continuations of `context`. Returns ranked phrases of length
 * 1..maxWords (a phrase of every promising length is a candidate, so the best
 * stopping point is chosen automatically).
 */
export function beamSearchPhrases(
  model: LanguageModel,
  context: string[],
  opts: BeamOptions = {},
): Phrase[] {
  const beamWidth = opts.beamWidth ?? 4;
  const maxWords = opts.maxWords ?? 1;
  const stepK = opts.stepK ?? Math.max(beamWidth * 2, 8);
  const reward = opts.wordReward ?? 0.6;
  const minAvg = opts.minAvgLogProb ?? -Infinity;
  const stops = opts.stops ?? DEFAULT_STOPS;

  let beams: Beam[] = [{ words: [], logProb: 0 }];
  const completed: Phrase[] = [];

  for (let step = 0; step < maxWords; step++) {
    const next: Beam[] = [];
    for (const beam of beams) {
      // A beam ending in a stop is finished, not extended.
      const last = beam.words[beam.words.length - 1];
      if (last !== undefined && stops.has(last)) continue;

      const cont = [...context, ...beam.words];
      for (const s of model.predict(cont, stepK)) {
        if (s.word === SOS) continue;
        const words = [...beam.words, s.word];
        const logProb = beam.logProb + s.logProb;
        next.push({ words, logProb });
        // Every length is a candidate completion.
        completed.push({ words, logProb, score: scoreOf(logProb, words.length, reward) });
      }
    }
    if (next.length === 0) break;
    // Prune to the best `beamWidth` partial beams (same length ⇒ by logProb).
    next.sort((a, b) => b.logProb - a.logProb);
    beams = next.slice(0, beamWidth);
  }

  // Rank, threshold by confidence, and de-duplicate (prefer longer when a
  // shorter phrase is a prefix of a higher-ranked longer one is fine to keep).
  completed.sort((a, b) => b.score - a.score);
  const out: Phrase[] = [];
  const seen = new Set<string>();
  for (const p of completed) {
    const avg = p.logProb / p.words.length;
    if (avg < minAvg) continue;
    const key = p.words.join(" ");
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(p);
  }
  return out;
}

/**
 * Optional neural scorer interface. Anything implementing this (e.g. a
 * transformers.js model in the worker) can either replace the n-gram in
 * beam search or directly generate completions. Kept minimal so the two
 * architectures unify behind one call site.
 */
export interface NeuralSequenceModel {
  ready(): boolean;
  /** Generate ranked continuations for the given left-context text. */
  complete(contextText: string, maxWords: number): Promise<Phrase[]>;
}
