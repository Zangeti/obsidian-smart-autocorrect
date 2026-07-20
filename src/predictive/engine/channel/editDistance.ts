/**
 * Weighted Damerau-Levenshtein distance = the "Shannon distance" channel score,
 * generalised to an arbitrary CostModel (geometry, empirical, or adaptive).
 *
 * Returns min-cost edit path in nats = estimate of -log P(typed | intended).
 * Bounded by `maxCost` (bails to Infinity) to stay real-time.
 *
 * `channelCost` keeps its original signature and behaviour (GeometryCostModel),
 * so existing callers/tests are unaffected. `weightedEdit` takes a CostModel.
 * `IncrementalMatcher` extends one column per keystroke for O(m) updates.
 */
import { type ChannelConfig, DEFAULT_CHANNEL } from "./keyboard.ts";
import { type CostModel, GeometryCostModel } from "./costModel.ts";

export function weightedEdit(
  typed: string,
  intended: string,
  cm: CostModel,
  maxCost = Infinity,
): number {
  const n = typed.length;
  const m = intended.length;
  if (n === 0) {
    let c = 0;
    for (let j = 0; j < m; j++) c += cm.del(intended[j], 0);
    return c <= maxCost ? c : Infinity;
  }
  if (m === 0) {
    let c = 0;
    for (let i = 0; i < n; i++) c += cm.ins(typed[i], 0);
    return c <= maxCost ? c : Infinity;
  }

  const denom = Math.max(1, n - 1);
  const INF = Infinity;
  let r0: number[] = new Array(m + 1).fill(INF);
  let r1: number[] = new Array(m + 1);
  let r2: number[] = new Array(m + 1);

  r1[0] = 0;
  for (let j = 1; j <= m; j++) r1[j] = r1[j - 1] + cm.del(intended[j - 1], 0);

  for (let i = 1; i <= n; i++) {
    const posFrac = (i - 1) / denom;
    const ti = typed[i - 1];
    const tiPrev = typed[i - 2];
    r2[0] = r1[0] + cm.ins(ti, posFrac);
    let rowMin = r2[0];
    for (let j = 1; j <= m; j++) {
      const ij = intended[j - 1];
      const ijPrev = intended[j - 2];
      const extra = r1[j] + cm.ins(ti, posFrac); // user typed an extra char
      const missed = r2[j - 1] + cm.del(ij, posFrac); // user missed a char
      const sub = r1[j - 1] + cm.sub(ij, ti, posFrac, ijPrev);
      let best = extra < missed ? extra : missed;
      if (sub < best) best = sub;
      if (i > 1 && j > 1 && ti === ijPrev && tiPrev === ij) {
        const tr = r0[j - 2] + cm.trans(ti, tiPrev, posFrac);
        if (tr < best) best = tr;
      }
      r2[j] = best;
      if (best < rowMin) rowMin = best;
    }
    if (rowMin > maxCost) return Infinity;
    const spare = r0;
    r0 = r1;
    r1 = r2;
    r2 = spare;
  }
  const result = r1[m];
  return result <= maxCost ? result : Infinity;
}

/**
 * Unit-cost Damerau-Levenshtein distance: the NUMBER of edits (sub/ins/del/
 * transposition), independent of keyboard geometry or phonetics. Used as a
 * structural plausibility gate for autocorrect - so a sound-alike fix like
 * "definately"→"definitely" (1 edit, but geometrically far a→i) is judged
 * plausible, while "peoppe"→"pop" (3 edits) is not. `maxDist` bails early.
 */
export function structuralEdit(a: string, b: string, maxDist = Infinity): number {
  const n = a.length, m = b.length;
  if (n === 0) return m;
  if (m === 0) return n;
  let r0: number[] = new Array(m + 1);
  let r1: number[] = new Array(m + 1);
  let r2: number[] = new Array(m + 1);
  for (let j = 0; j <= m; j++) r1[j] = j;
  for (let i = 1; i <= n; i++) {
    r2[0] = i;
    let rowMin = i;
    for (let j = 1; j <= m; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      let best = Math.min(r1[j] + 1, r2[j - 1] + 1, r1[j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1])
        best = Math.min(best, r0[j - 2] + 1);
      r2[j] = best;
      if (best < rowMin) rowMin = best;
    }
    if (rowMin > maxDist) return Infinity;
    const spare = r0; r0 = r1; r1 = r2; r2 = spare;
  }
  return r1[m];
}

/** Backward-compatible geometry-based channel cost (unchanged behaviour). */
export function channelCost(
  typed: string,
  intended: string,
  cfg: ChannelConfig = DEFAULT_CHANNEL,
  maxCost = Infinity,
): number {
  return weightedEdit(typed, intended, new GeometryCostModel(cfg), maxCost);
}

/**
 * Prefix distance: min cost of aligning `typed` to ANY prefix of `intended`.
 * The correct score for autocomplete (the word isn't finished yet).
 */
export function prefixCost(
  typed: string,
  intended: string,
  cm: CostModel,
  maxCost = Infinity,
): number {
  const n = typed.length;
  const m = intended.length;
  if (n === 0) return 0;
  const denom = Math.max(1, n - 1);
  let r0: number[] = new Array(m + 1).fill(Infinity);
  let r1: number[] = new Array(m + 1);
  let r2: number[] = new Array(m + 1);
  r1[0] = 0;
  for (let j = 1; j <= m; j++) r1[j] = 0; // any prefix boundary is free (no suffix penalty)
  for (let i = 1; i <= n; i++) {
    const posFrac = (i - 1) / denom;
    const ti = typed[i - 1];
    const tiPrev = typed[i - 2];
    r2[0] = r1[0] + cm.ins(ti, posFrac);
    let rowMin = r2[0];
    for (let j = 1; j <= m; j++) {
      const ij = intended[j - 1];
      const ijPrev = intended[j - 2];
      const extra = r1[j] + cm.ins(ti, posFrac);
      const missed = r2[j - 1] + cm.del(ij, posFrac);
      const sub = r1[j - 1] + cm.sub(ij, ti, posFrac, ijPrev);
      let best = extra < missed ? extra : missed;
      if (sub < best) best = sub;
      if (i > 1 && j > 1 && ti === ijPrev && tiPrev === ij) {
        const tr = r0[j - 2] + cm.trans(ti, tiPrev, posFrac);
        if (tr < best) best = tr;
      }
      r2[j] = best;
      if (best < rowMin) rowMin = best;
    }
    if (rowMin > maxCost) return Infinity;
    const spare = r0;
    r0 = r1;
    r1 = r2;
    r2 = spare;
  }
  // best over all prefixes of intended = min of the final row.
  let best = Infinity;
  for (let j = 0; j <= m; j++) if (r1[j] < best) best = r1[j];
  return best <= maxCost ? best : Infinity;
}

/**
 * Incremental matcher: matches a growing `typed` against a fixed `intended`,
 * extending one DP row per appended character in O(m). Used to score a
 * candidate as the user types without recomputing the whole matrix.
 */
export class IncrementalMatcher {
  private intended: string;
  private cm: CostModel;
  private rows: number[][] = [];
  private typed = "";

  constructor(intended: string, cm: CostModel) {
    this.intended = intended;
    this.cm = cm;
    const m = intended.length;
    const row0 = new Array(m + 1);
    row0[0] = 0;
    for (let j = 1; j <= m; j++) row0[j] = row0[j - 1] + cm.del(intended[j - 1], 0);
    this.rows = [row0];
  }

  /** Append one typed character; returns current full-word cost. */
  push(ch: string): number {
    const m = this.intended.length;
    const i = this.typed.length + 1;
    const denom = Math.max(1, i - 1);
    const posFrac = (i - 1) / denom;
    const prev = this.rows[i - 1];
    const prev2 = i >= 2 ? this.rows[i - 2] : null;
    const row = new Array(m + 1);
    row[0] = prev[0] + this.cm.ins(ch, posFrac);
    const tiPrev = this.typed[i - 2];
    for (let j = 1; j <= m; j++) {
      const ij = this.intended[j - 1];
      const ijPrev = this.intended[j - 2];
      const extra = prev[j] + this.cm.ins(ch, posFrac);
      const missed = row[j - 1] + this.cm.del(ij, posFrac);
      const sub = prev[j - 1] + this.cm.sub(ij, ch, posFrac, ijPrev);
      let best = extra < missed ? extra : missed;
      if (sub < best) best = sub;
      if (prev2 && j > 1 && ch === ijPrev && tiPrev === ij) {
        const tr = prev2[j - 2] + this.cm.trans(ch, tiPrev, posFrac);
        if (tr < best) best = tr;
      }
      row[j] = best;
    }
    this.rows.push(row);
    this.typed += ch;
    return row[m];
  }

  cost(): number {
    return this.rows[this.rows.length - 1][this.intended.length];
  }
}
