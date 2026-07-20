/**
 * Empirical, position- AND context-conditioned confusion model (#2 + #D9).
 *
 * Substitution cost backs off hierarchically:
 *   context-conditioned P(typed | intended, prevChar)   [most specific]
 *     → single-char P(typed | intended)
 *       → QWERTY-geometry prior                          [least specific]
 * each level's confidence (observation count) decides how much it is trusted.
 *
 * `learn(intended, typed)` aligns the pair and bumps the relevant counts, so the
 * model adapts to THIS user's recurring slips (incl. context-specific ones like
 * "always types 'k' for 'c' after 's'"). Serialisable for personalisation.
 */
import { type ChannelConfig, DEFAULT_CHANNEL, substitutionCost } from "./keyboard.ts";
import type { CostModel } from "./costModel.ts";

export interface ConfusionData {
  version: 1;
  sub: Record<string, number>; // "i>t"
  del: Record<string, number>; // intended
  ins: Record<string, number>; // typed
  trans: Record<string, number>; // "a,b"
  seen: Record<string, number>; // per intended char
  /** context-conditioned substitutions: "prev|i>t". */
  subCtx: Record<string, number>;
  /** per (prev,intended) observation count: "prev|i". */
  seenCtx: Record<string, number>;
  observations: number;
}

export function emptyConfusionData(): ConfusionData {
  return {
    version: 1,
    sub: {},
    del: {},
    ins: {},
    trans: {},
    seen: {},
    subCtx: {},
    seenCtx: {},
    observations: 0,
  };
}

const inc = (m: Record<string, number>, k: string, by = 1) => {
  m[k] = (m[k] ?? 0) + by;
};

export class ConfusionModel implements CostModel {
  private cfg: ChannelConfig;
  private d: ConfusionData;
  private priorStrength: number;
  private firstPosPenalty: number;

  constructor(
    data: ConfusionData = emptyConfusionData(),
    cfg: ChannelConfig = DEFAULT_CHANNEL,
    opts: { priorStrength?: number; firstPosPenalty?: number } = {},
  ) {
    this.cfg = cfg;
    this.d = { ...emptyConfusionData(), ...data };
    this.priorStrength = opts.priorStrength ?? 20;
    this.firstPosPenalty = opts.firstPosPenalty ?? 1.6;
  }

  setChannel(cfg: ChannelConfig): void {
    this.cfg = cfg;
  }
  toJSON(): ConfusionData {
    return this.d;
  }
  static fromJSON(data: ConfusionData, cfg?: ChannelConfig): ConfusionModel {
    return new ConfusionModel(data, cfg);
  }
  reset(): void {
    this.d = emptyConfusionData();
  }
  mergeFrom(other: ConfusionData): void {
    const o = { ...emptyConfusionData(), ...other };
    for (const k in o.sub) inc(this.d.sub, k, o.sub[k]);
    for (const k in o.del) inc(this.d.del, k, o.del[k]);
    for (const k in o.ins) inc(this.d.ins, k, o.ins[k]);
    for (const k in o.trans) inc(this.d.trans, k, o.trans[k]);
    for (const k in o.seen) inc(this.d.seen, k, o.seen[k]);
    for (const k in o.subCtx) inc(this.d.subCtx, k, o.subCtx[k]);
    for (const k in o.seenCtx) inc(this.d.seenCtx, k, o.seenCtx[k]);
    this.d.observations += o.observations;
  }

  private posMul(posFrac: number): number {
    return 1 + (this.firstPosPenalty - 1) * (1 - Math.min(1, posFrac * 3));
  }

  private blend(empProb: number | null, priorCost: number, support: number): number {
    if (empProb === null || support <= 0) return priorCost;
    const w = support / (support + this.priorStrength);
    const empCost = -Math.log(Math.max(empProb, 1e-6));
    return w * empCost + (1 - w) * priorCost;
  }

  sub(intended: string, typed: string, posFrac: number, prevIntended?: string): number {
    if (intended === typed) return 0;
    const li = intended.toLowerCase();
    const lt = typed.toLowerCase();
    if (li === lt) return this.cfg.caseCost;

    // Level 1: single-char empirical blended with geometry prior.
    const prior = substitutionCost(intended, typed, this.cfg);
    const support = this.d.seen[li] ?? 0;
    const emp = support > 0 ? (this.d.sub[li + ">" + lt] ?? 0) / support : null;
    let cost = this.blend(emp, prior, support);

    // Level 2: context-conditioned, using the single-char cost as its prior.
    if (prevIntended) {
      const pc = prevIntended.toLowerCase();
      const ctxSupport = this.d.seenCtx[pc + "|" + li] ?? 0;
      if (ctxSupport > 0) {
        const ctxEmp = (this.d.subCtx[pc + "|" + li + ">" + lt] ?? 0) / ctxSupport;
        cost = this.blend(ctxEmp, cost, ctxSupport);
      }
    }
    return cost * this.posMul(posFrac);
  }

  del(intended: string, posFrac: number): number {
    const li = intended.toLowerCase();
    const support = this.d.seen[li] ?? 0;
    const emp = support > 0 ? (this.d.del[li] ?? 0) / support : null;
    return this.blend(emp, this.cfg.deleteCost, support) * this.posMul(posFrac);
  }

  ins(typed: string, posFrac: number): number {
    const lt = typed.toLowerCase();
    const support = this.d.observations;
    const emp = support > 0 ? (this.d.ins[lt] ?? 0) / support : null;
    return this.blend(emp, this.cfg.insertCost, support) * this.posMul(posFrac);
  }

  trans(a: string, b: string, posFrac: number): number {
    const support = this.d.observations;
    const emp = support > 0 ? (this.d.trans[a.toLowerCase() + "," + b.toLowerCase()] ?? 0) / support : null;
    return this.blend(emp, this.cfg.transposeCost, support) * this.posMul(posFrac);
  }

  learn(intended: string, typed: string): void {
    const a = typed.toLowerCase();
    const b = intended.toLowerCase();
    for (const ch of b) inc(this.d.seen, ch);
    this.d.observations++;
    let prev = "";
    for (const op of backtrace(a, b)) {
      if (op.type === "sub") {
        inc(this.d.sub, op.intended + ">" + op.typed);
        if (prev) {
          inc(this.d.subCtx, prev + "|" + op.intended + ">" + op.typed);
          inc(this.d.seenCtx, prev + "|" + op.intended);
        }
        prev = op.intended;
      } else if (op.type === "del") {
        inc(this.d.del, op.intended);
        prev = op.intended;
      } else if (op.type === "ins") {
        inc(this.d.ins, op.typed);
      } else if (op.type === "trans") {
        inc(this.d.trans, op.a + "," + op.b);
        prev = op.b;
      } else if (op.type === "match") {
        prev = op.intended;
      }
    }
  }
}

type EditOp =
  | { type: "match"; intended: string }
  | { type: "sub"; intended: string; typed: string }
  | { type: "del"; intended: string }
  | { type: "ins"; typed: string }
  | { type: "trans"; a: string; b: string };

function backtrace(typed: string, intended: string): EditOp[] {
  const n = typed.length;
  const m = intended.length;
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = 0; i <= n; i++) dp[i][0] = i;
  for (let j = 0; j <= m; j++) dp[0][j] = j;
  for (let i = 1; i <= n; i++)
    for (let j = 1; j <= m; j++) {
      const c = typed[i - 1] === intended[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c);
      if (i > 1 && j > 1 && typed[i - 1] === intended[j - 2] && typed[i - 2] === intended[j - 1])
        dp[i][j] = Math.min(dp[i][j], dp[i - 2][j - 2] + 1);
    }
  const ops: EditOp[] = [];
  let i = n;
  let j = m;
  while (i > 0 || j > 0) {
    if (
      i > 1 && j > 1 &&
      typed[i - 1] === intended[j - 2] && typed[i - 2] === intended[j - 1] &&
      dp[i][j] === dp[i - 2][j - 2] + 1
    ) {
      ops.push({ type: "trans", a: intended[j - 2], b: intended[j - 1] });
      i -= 2;
      j -= 2;
    } else if (i > 0 && j > 0 && typed[i - 1] === intended[j - 1] && dp[i][j] === dp[i - 1][j - 1]) {
      ops.push({ type: "match", intended: intended[j - 1] });
      i--;
      j--;
    } else if (i > 0 && j > 0 && dp[i][j] === dp[i - 1][j - 1] + 1) {
      ops.push({ type: "sub", intended: intended[j - 1], typed: typed[i - 1] });
      i--;
      j--;
    } else if (i > 0 && dp[i][j] === dp[i - 1][j] + 1) {
      ops.push({ type: "ins", typed: typed[i - 1] });
      i--;
    } else {
      ops.push({ type: "del", intended: intended[j - 1] });
      j--;
    }
  }
  return ops.reverse();
}
