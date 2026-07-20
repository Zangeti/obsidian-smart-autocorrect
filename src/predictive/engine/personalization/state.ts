/**
 * Serialisable personalisation state: everything the engine learns about THIS
 * user. Kept as one small JSON blob so the Obsidian layer can persist it in the
 * vault, export/import it to share between machines, and reset it - without any
 * of the runtime objects leaking into storage.
 *
 *   - confusion : adaptive keyboard error model (channel/confusion.ts)
 *   - reranker  : learned candidate-ordering weights (predict/reranker.ts)
 *   - learnList : words the user rejected being autocorrected
 *   - stats     : lightweight counters (for the settings UI)
 */
import {
  ConfusionModel,
  emptyConfusionData,
  type ConfusionData,
} from "../channel/confusion.ts";
import { Reranker, type RerankerData, defaultRerankerWeights } from "../predict/reranker.ts";

export const PERSONALIZATION_VERSION = 1;

export interface PersonalizationState {
  version: number;
  confusion: ConfusionData;
  reranker: RerankerData;
  learnList: string[];
  stats: { corrections: number; accepts: number; reverts: number; charsSaved: number };
  updatedAt: number;
}

export function emptyPersonalization(beta = 1.0): PersonalizationState {
  return {
    version: PERSONALIZATION_VERSION,
    confusion: emptyConfusionData(),
    reranker: { version: 1, weights: defaultRerankerWeights(beta), updates: 0 },
    learnList: [],
    stats: { corrections: 0, accepts: 0, reverts: 0, charsSaved: 0 },
    updatedAt: Date.now(),
  };
}

/** Validate + upgrade an untrusted blob (e.g. an imported file) to current shape. */
export function normalizePersonalization(
  raw: unknown,
  beta = 1.0,
): PersonalizationState {
  const base = emptyPersonalization(beta);
  if (!raw || typeof raw !== "object") return base;
  const r = raw as Partial<PersonalizationState>;
  return {
    version: PERSONALIZATION_VERSION,
    confusion: (r.confusion as ConfusionData) ?? base.confusion,
    reranker: (r.reranker as RerankerData) ?? base.reranker,
    learnList: Array.isArray(r.learnList) ? r.learnList.filter((s) => typeof s === "string") : [],
    stats: { ...base.stats, ...(r.stats ?? {}) },
    updatedAt: Date.now(),
  };
}

/**
 * Live bundle of the runtime learners plus (de)serialisation. The controller
 * holds one of these; the confusion model and reranker are shared with the
 * predictor/autocorrect by reference.
 */
export class Personalization {
  confusion: ConfusionModel;
  reranker: Reranker;
  learnList: Set<string>;
  stats: { corrections: number; accepts: number; reverts: number; charsSaved: number };
  private beta: number;

  constructor(state: PersonalizationState, beta = 1.0) {
    this.beta = beta;
    this.confusion = ConfusionModel.fromJSON(state.confusion);
    this.reranker = Reranker.fromJSON(state.reranker);
    this.learnList = new Set(state.learnList.map((s) => s.toLowerCase()));
    this.stats = { ...state.stats };
  }

  static empty(beta = 1.0): Personalization {
    return new Personalization(emptyPersonalization(beta), beta);
  }

  toState(): PersonalizationState {
    return {
      version: PERSONALIZATION_VERSION,
      confusion: this.confusion.toJSON(),
      reranker: this.reranker.toJSON(),
      learnList: [...this.learnList],
      stats: { ...this.stats },
      updatedAt: Date.now(),
    };
  }

  toJSONString(pretty = false): string {
    return JSON.stringify(this.toState(), null, pretty ? 2 : undefined);
  }

  /** Replace everything with a fresh state (reset button). */
  reset(): void {
    this.confusion.reset();
    this.reranker.reset(this.beta);
    this.learnList.clear();
    this.stats = { corrections: 0, accepts: 0, reverts: 0, charsSaved: 0 };
  }

  /** Zero just the tallied statistics, keeping the learned adaptation intact. */
  resetStats(): void {
    this.stats = { corrections: 0, accepts: 0, reverts: 0, charsSaved: 0 };
  }

  /** Load an imported blob, replacing current state. */
  loadFrom(raw: unknown): void {
    const s = normalizePersonalization(raw, this.beta);
    this.confusion = ConfusionModel.fromJSON(s.confusion);
    this.reranker = Reranker.fromJSON(s.reranker);
    this.learnList = new Set(s.learnList.map((x) => x.toLowerCase()));
    this.stats = { ...s.stats };
  }

  /** Merge another user's personalisation into this one (additive). */
  mergeFrom(raw: unknown): void {
    const s = normalizePersonalization(raw, this.beta);
    this.confusion.mergeFrom(s.confusion);
    for (const w of s.learnList) this.learnList.add(w.toLowerCase());
    this.stats.corrections += s.stats.corrections;
    this.stats.accepts += s.stats.accepts;
    this.stats.reverts += s.stats.reverts;
    this.stats.charsSaved += s.stats.charsSaved;
  }
}
