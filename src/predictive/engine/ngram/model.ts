/**
 * In-memory n-gram language model with count-based (Witten-Bell style) linear
 * interpolation across trigram / bigram / unigram.
 *
 * The blend weight for each order is derived from how much evidence the context
 * has: a high-count, low-diversity context trusts the higher order; a sparse
 * context automatically leans on the lower order. This is the "intelligent
 * mixing of 1/2/3-previous-word probabilities" from the design.
 *
 * The public interface (LanguageModel) is deliberately backing-store agnostic
 * so a packed ArrayBuffer/MPHR implementation can replace this later without
 * touching the predictor or channel model.
 */
import { SOS } from "../text/tokenize.ts";

export interface Scored {
  word: string;
  /** natural-log probability P(word | context) after blending. */
  logProb: number;
}

export interface LanguageModel {
  hasWord(word: string): boolean;
  /** Blended P(word | context) as a natural log. Context is previous words. */
  logProb(word: string, context: string[]): number;
  /** Top-k most likely continuations of `context`. */
  predict(context: string[], k: number): Scored[];
  vocabulary(): IterableIterator<string>;
  size(): { vocab: number; bigrams: number; trigrams: number };
}

/** Nested-map counts. Word ids keep the maps compact and fast. */
export class NgramCounts {
  vocab: string[] = [];
  wordId = new Map<string, number>();
  uni: number[] = [];
  totalUni = 0;
  /** w1 -> (next -> count) */
  bi = new Map<number, Map<number, number>>();
  /** w2 -> w1 -> (next -> count) */
  tri = new Map<number, Map<number, Map<number, number>>>();

  intern(word: string): number {
    let id = this.wordId.get(word);
    if (id === undefined) {
      id = this.vocab.length;
      this.vocab.push(word);
      this.wordId.set(word, id);
      this.uni.push(0);
    }
    return id;
  }

  addSentence(words: string[]): void {
    // Prepend two start sentinels so the first words are ordinary trigrams.
    const ids = [this.intern(SOS), this.intern(SOS), ...words.map((w) => this.intern(w))];
    for (let i = 0; i < ids.length; i++) {
      const w = ids[i];
      // Don't let the sentinel pollute the unigram distribution of real words.
      if (this.vocab[w] !== SOS) {
        this.uni[w]++;
        this.totalUni++;
      }
      if (i >= 1) bump2(this.bi, ids[i - 1], w);
      if (i >= 2) bump3(this.tri, ids[i - 2], ids[i - 1], w);
    }
  }
}

function bump2(m: Map<number, Map<number, number>>, a: number, b: number): void {
  let inner = m.get(a);
  if (!inner) {
    inner = new Map();
    m.set(a, inner);
  }
  inner.set(b, (inner.get(b) ?? 0) + 1);
}

function bump3(
  m: Map<number, Map<number, Map<number, number>>>,
  a: number,
  b: number,
  c: number,
): void {
  let mid = m.get(a);
  if (!mid) {
    mid = new Map();
    m.set(a, mid);
  }
  bump2(mid, b, c);
}

/** Tunable smoothing constants. */
export interface BlendConfig {
  /** floor probability for a completely unknown word. */
  unkFloor: number;
  /** additive constant guarding empty contexts in the Witten-Bell weight. */
  wbGuard: number;
  /**
   * Use Kneser-Ney continuation probability for the lowest order instead of raw
   * unigram frequency. Fixes the "San Francisco" problem: a word frequent in
   * only one context gets a low continuation probability. Off by default to
   * preserve legacy behaviour.
   */
  useContinuation?: boolean;
}

export const DEFAULT_BLEND: BlendConfig = { unkFloor: 1e-9, wbGuard: 1e-9 };

export class InMemoryLanguageModel implements LanguageModel {
  private c: NgramCounts;
  private cfg: BlendConfig;
  private topUni: number[]; // ids sorted by unigram count desc (for cold start)
  private contCount: number[]; // N1+(• w): distinct left contexts of w
  private totalBiTypes: number;

  constructor(counts: NgramCounts, cfg: BlendConfig = DEFAULT_BLEND) {
    this.c = counts;
    this.cfg = cfg;
    this.topUni = counts.vocab
      .map((_, id) => id)
      .filter((id) => counts.vocab[id] !== SOS)
      .sort((a, b) => counts.uni[b] - counts.uni[a]);
    // Continuation counts for Kneser-Ney lowest order.
    this.contCount = new Array<number>(counts.vocab.length).fill(0);
    let totalBiTypes = 0;
    for (const inner of counts.bi.values()) {
      for (const next of inner.keys()) {
        this.contCount[next]++;
        totalBiTypes++;
      }
    }
    this.totalBiTypes = totalBiTypes;
  }

  hasWord(word: string): boolean {
    return this.c.wordId.has(word);
  }

  vocabulary(): IterableIterator<string> {
    return this.c.wordId.keys();
  }

  size() {
    let bi = 0;
    for (const m of this.c.bi.values()) bi += m.size;
    let tri = 0;
    for (const mid of this.c.tri.values())
      for (const m of mid.values()) tri += m.size;
    return { vocab: this.c.vocab.length, bigrams: bi, trigrams: tri };
  }

  /** Raw unigram count of a word (0 if unseen) - used to gate rare/typo words. */
  unigramCount(word: string): number {
    const id = this.c.wordId.get(word);
    return id === undefined ? 0 : this.c.uni[id] ?? 0;
  }

  // --- probability model -------------------------------------------------

  private pUnigram(id: number): number {
    if (this.cfg.useContinuation && this.totalBiTypes > 0) {
      const cc = this.contCount[id] ?? 0;
      return cc > 0 ? cc / this.totalBiTypes : this.cfg.unkFloor;
    }
    if (this.c.totalUni === 0) return this.cfg.unkFloor;
    const c = this.c.uni[id] ?? 0;
    return c > 0 ? c / this.c.totalUni : this.cfg.unkFloor;
  }

  /** P(next | w1) interpolated with unigram via Witten-Bell weight. */
  private pBigram(nextId: number, w1: number): number {
    const inner = this.c.bi.get(w1);
    const lower = this.pUnigram(nextId);
    if (!inner) return lower;
    let total = 0;
    for (const v of inner.values()) total += v;
    const distinct = inner.size; // N1+(w1 •)
    const mle = (inner.get(nextId) ?? 0) / total;
    // higher-order weight: more evidence & less diversity => trust bigram more.
    const lambda = total / (total + distinct + this.cfg.wbGuard);
    return lambda * mle + (1 - lambda) * lower;
  }

  /** P(next | w2 w1) interpolated with the bigram estimate. */
  private pTrigram(nextId: number, w2: number, w1: number): number {
    const lower = this.pBigram(nextId, w1);
    const mid = this.c.tri.get(w2);
    const inner = mid?.get(w1);
    if (!inner) return lower;
    let total = 0;
    for (const v of inner.values()) total += v;
    const distinct = inner.size;
    const mle = (inner.get(nextId) ?? 0) / total;
    const lambda = total / (total + distinct + this.cfg.wbGuard);
    return lambda * mle + (1 - lambda) * lower;
  }

  private probById(nextId: number, context: string[]): number {
    const n = context.length;
    const w1 = n >= 1 ? this.c.wordId.get(context[n - 1]) : undefined;
    const w2 = n >= 2 ? this.c.wordId.get(context[n - 2]) : undefined;
    if (w1 !== undefined && w2 !== undefined) return this.pTrigram(nextId, w2, w1);
    if (w1 !== undefined) return this.pBigram(nextId, w1);
    return this.pUnigram(nextId);
  }

  logProb(word: string, context: string[]): number {
    const id = this.c.wordId.get(word);
    if (id === undefined) return Math.log(this.cfg.unkFloor);
    return Math.log(this.probById(id, context));
  }

  // --- prediction --------------------------------------------------------

  predict(context: string[], k: number): Scored[] {
    const n = context.length;
    const w1 = n >= 1 ? this.c.wordId.get(context[n - 1]) : undefined;
    const w2 = n >= 2 ? this.c.wordId.get(context[n - 2]) : undefined;

    // Gather a candidate id set from the richest available context, plus a
    // cold-start pool of frequent unigrams. We only score this bounded set.
    const cand = new Set<number>();
    if (w1 !== undefined && w2 !== undefined) {
      const inner = this.c.tri.get(w2)?.get(w1);
      if (inner) for (const id of inner.keys()) cand.add(id);
    }
    if (w1 !== undefined) {
      const inner = this.c.bi.get(w1);
      if (inner) for (const id of inner.keys()) cand.add(id);
    }
    // Pad with frequent unigrams ONLY for a true cold start (no observed
    // continuation for this context - e.g. sentence start or an unseen context).
    // Otherwise show just the grammatical, observed continuations, even if that
    // means fewer than k - padding with generic words ("the"/"to") that never
    // followed this context would introduce ungrammatical suggestions.
    if (cand.size === 0) {
      for (let i = 0; i < this.topUni.length && cand.size < Math.max(k, 8); i++)
        cand.add(this.topUni[i]);
    }

    const sos = this.c.wordId.get(SOS);
    const scored: Scored[] = [];
    for (const id of cand) {
      if (id === sos) continue;
      scored.push({ word: this.c.vocab[id], logProb: Math.log(this.probById(id, context)) });
    }
    scored.sort((a, b) => b.logProb - a.logProb);
    return scored.slice(0, k);
  }
}

/**
 * Mixture of a large global model and a small personal (vault) model:
 *   P_final = (1-alpha) * P_global + alpha * P_personal
 * alpha is the user's "personal-style bias" slider.
 */
export class MixtureLanguageModel implements LanguageModel {
  private globalModel: LanguageModel;
  private personal: LanguageModel | null;
  private alpha: number;

  constructor(globalModel: LanguageModel, personal: LanguageModel | null, alpha: number) {
    this.globalModel = globalModel;
    this.personal = personal;
    this.alpha = alpha;
  }

  setAlpha(a: number) {
    this.alpha = a;
  }

  hasWord(word: string): boolean {
    return this.globalModel.hasWord(word) || (this.personal?.hasWord(word) ?? false);
  }

  *vocabulary(): IterableIterator<string> {
    const seen = new Set<string>();
    for (const w of this.globalModel.vocabulary()) {
      seen.add(w);
      yield w;
    }
    if (this.personal)
      for (const w of this.personal.vocabulary()) if (!seen.has(w)) yield w;
  }

  size() {
    return this.globalModel.size();
  }

  logProb(word: string, context: string[]): number {
    const g = Math.exp(this.globalModel.logProb(word, context));
    if (!this.personal || this.alpha <= 0) return Math.log(g);
    const p = Math.exp(this.personal.logProb(word, context));
    return Math.log((1 - this.alpha) * g + this.alpha * p);
  }

  predict(context: string[], k: number): Scored[] {
    if (!this.personal || this.alpha <= 0) return this.globalModel.predict(context, k);
    // Union candidate words from both models, then re-score with the mixture.
    const words = new Set<string>();
    for (const s of this.globalModel.predict(context, k * 3)) words.add(s.word);
    for (const s of this.personal.predict(context, k * 3)) words.add(s.word);
    const scored: Scored[] = [];
    for (const w of words) scored.push({ word: w, logProb: this.logProb(w, context) });
    scored.sort((a, b) => b.logProb - a.logProb);
    return scored.slice(0, k);
  }
}

/**
 * Within-document cache / trigger model: boosts words the user has used recently
 * in the CURRENT note. A tiny, strongly-predictive signal for personal writing.
 * Wraps any base model and mixes a within-document unigram cache with weight gamma.
 */
export class CacheLanguageModel implements LanguageModel {
  private base: LanguageModel;
  private gamma: number;
  private minCount: number;
  private cache = new Map<string, number>();
  private total = 0;

  /** `minCount` = how many times a word must appear in the note before it is
   *  boosted. >=2 means a single just-typed word never spikes to the top
   *  (fixing the "word you just accepted sticks as the next suggestion" bug);
   *  only genuinely recurring topical words get boosted. */
  constructor(base: LanguageModel, gamma = 0.15, minCount = 2) {
    this.base = base;
    this.gamma = gamma;
    this.minCount = minCount;
  }

  setGamma(g: number): void {
    this.gamma = g;
  }

  /** Record words as they are typed/seen in the current document. */
  observe(words: string[]): void {
    for (const w of words) {
      this.cache.set(w, (this.cache.get(w) ?? 0) + 1);
      this.total += 1;
    }
  }

  /** Reset and seed from the current document's text tokens. */
  setDocument(words: string[]): void {
    this.cache.clear();
    this.total = 0;
    this.observe(words);
  }

  private pCache(word: string): number {
    if (this.total === 0) return 0;
    const c = this.cache.get(word) ?? 0;
    return c >= this.minCount ? c / this.total : 0;
  }

  hasWord(word: string): boolean {
    return this.base.hasWord(word); // cache never adds new vocabulary
  }
  vocabulary(): IterableIterator<string> {
    return this.base.vocabulary();
  }
  size() {
    return this.base.size();
  }

  /**
   * Adapt the base model to what this document is about, by REWEIGHTING it rather than by
   * mixing something else into it:
   *
   *     log p'(w | context) = log p(w | context) + γ·[log p_cache(w) − log p_unigram(w)]
   *
   * i.e. each word's context-conditional probability is multiplied by how much commoner it is
   * in this document than in the language at large ("unigram rescaling"; the same shape as
   * classic MaxEnt topic adaptation), with γ setting how far we trust that ratio.
   *
   * The previous form was an additive mixture, (1−γ)·p(w|context) + γ·p_cache(w). Its flaw is
   * structural: p_cache is context-FREE, so it contributed the same mass no matter what the
   * context said, giving every topical word a floor the context could not veto. A rare word
   * mentioned a few times in the note therefore kept a sizeable probability in sentences that
   * plainly excluded it - "i went to the reichskommissariat" - and because the suggestion menu
   * ranks by expected keystrokes SAVED, an 18-character word only needs a modest probability to
   * take over the whole menu. Multiplicatively, a word the context rules out stays ruled out:
   * anything times nearly zero is nearly zero. In-topic predictions keep their boost, since
   * there the context-conditional probability is genuinely high to begin with.
   *
   * Two consequences worth being explicit about:
   *  - The result is deliberately NOT renormalised: the normaliser depends only on the context,
   *    so it cancels exactly in every comparison the engine makes (ranking candidates for one
   *    position, information gain between two words at one position). Computing it would cost a
   *    pass over the vocabulary per keystroke to change nothing that is ever compared.
   *  - It is still clamped to a probability, so no adapted word can come out above certainty.
   */
  logProb(word: string, context: string[]): number {
    const b = this.base.logProb(word, context);
    // The cache only BOOSTS words the base model already knows; it never lets an
    // unknown word (e.g. a typo you just typed) gain probability.
    if (this.total === 0 || this.gamma <= 0 || !this.base.hasWord(word)) return b;
    const pc = this.pCache(word);
    if (pc <= 0) return b; // not topical here: the base model stands unchanged
    const unigram = this.base.logProb(word, []);
    return Math.min(0, b + this.gamma * (Math.log(pc) - unigram));
  }

  predict(context: string[], k: number): Scored[] {
    const words = new Set<string>();
    for (const s of this.base.predict(context, k * 2)) words.add(s.word);
    // Only surface cached words that are real (in the base vocabulary) AND
    // recur enough to count as topical - a one-off word is never introduced.
    for (const [w, c] of this.cache) if (c >= this.minCount && this.base.hasWord(w)) words.add(w);
    const scored: Scored[] = [];
    for (const w of words) scored.push({ word: w, logProb: this.logProb(w, context) });
    scored.sort((a, b) => b.logProb - a.logProb);
    return scored.slice(0, k);
  }
}
