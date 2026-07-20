/**
 * Learning-to-rank reranker (#5). The final ordering of the top-K candidates is
 * a linear score over interpretable features. Weights are initialised so the
 * model reproduces the hand-tuned `logP_LM - beta*channelCost` ordering, then
 * adapt online from implicit feedback: whichever candidate the user actually
 * accepts is nudged above the one currently ranked first (perceptron update).
 *
 * Weights are serialisable (personalisation) and resettable.
 */

export const FEATURE_NAMES = [
  "lmLogProb", // blended P(w|context), log
  "negChannel", // -channelCost (higher is better)
  "prefixExact", // 1 if typed is an exact prefix of word
  "unigram", // word frequency, log
  "caseMatch", // 1 if leading case matches the query
  "phonetic", // 1 if match came via phonetic path
  "lengthPenalty", // -abs(len(word)-len(typed))
  "bias",
] as const;

export type FeatureVector = number[]; // aligned to FEATURE_NAMES

export interface RerankerData {
  version: 1;
  weights: number[];
  updates: number;
}

export function defaultRerankerWeights(beta: number): number[] {
  // Reproduce logP_LM - beta*channelCost (negChannel = -channelCost).
  return [1.0, beta, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0];
}

export class Reranker {
  private w: number[];
  private updates: number;
  private lr: number;

  constructor(data?: RerankerData, beta = 1.0, lr = 0.05) {
    this.w = data?.weights ?? defaultRerankerWeights(beta);
    this.updates = data?.updates ?? 0;
    this.lr = lr;
  }

  toJSON(): RerankerData {
    return { version: 1, weights: this.w.slice(), updates: this.updates };
  }
  static fromJSON(data: RerankerData): Reranker {
    return new Reranker(data);
  }
  reset(beta = 1.0): void {
    this.w = defaultRerankerWeights(beta);
    this.updates = 0;
  }

  score(f: FeatureVector): number {
    let s = 0;
    for (let i = 0; i < this.w.length && i < f.length; i++) s += this.w[i] * f[i];
    return s;
  }

  /**
   * Online update: the user accepted `accepted` from a shown list. Nudge the
   * weights so the accepted features outrank the currently top-scoring ones.
   */
  learn(shown: FeatureVector[], acceptedIndex: number): void {
    if (acceptedIndex < 0 || acceptedIndex >= shown.length) return;
    let topIdx = 0;
    let topScore = -Infinity;
    for (let i = 0; i < shown.length; i++) {
      const s = this.score(shown[i]);
      if (s > topScore) {
        topScore = s;
        topIdx = i;
      }
    }
    if (topIdx === acceptedIndex) return; // already ranked first, nothing to do
    const acc = shown[acceptedIndex];
    const top = shown[topIdx];
    for (let i = 0; i < this.w.length; i++) {
      this.w[i] += this.lr * ((acc[i] ?? 0) - (top[i] ?? 0));
    }
    this.updates++;
  }
}
