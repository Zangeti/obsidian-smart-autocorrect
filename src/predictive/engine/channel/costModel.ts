/**
 * Cost model abstraction for the weighted edit distance. A CostModel returns
 * the nats-cost of each edit operation, optionally conditioned on the characters
 * involved, the PREVIOUS intended character (bigram error context), and the
 * position within the typed word.
 */
import { type ChannelConfig, DEFAULT_CHANNEL, substitutionCost } from "./keyboard.ts";

export interface CostModel {
  /**
   * intended char typed as `typed`. `prevIntended` is the preceding intended
   * character (for context-conditioned error models); posFrac is 0..1.
   */
  sub(intended: string, typed: string, posFrac: number, prevIntended?: string): number;
  del(intended: string, posFrac: number): number;
  ins(typed: string, posFrac: number): number;
  trans(a: string, b: string, posFrac: number): number;
}

/** Reproduces the original fixed-cost + geometry behaviour exactly. */
export class GeometryCostModel implements CostModel {
  private cfg: ChannelConfig;
  constructor(cfg: ChannelConfig = DEFAULT_CHANNEL) {
    this.cfg = cfg;
  }
  sub(intended: string, typed: string): number {
    return substitutionCost(intended, typed, this.cfg);
  }
  del(): number {
    return this.cfg.deleteCost;
  }
  ins(): number {
    return this.cfg.insertCost;
  }
  trans(): number {
    return this.cfg.transposeCost;
  }
}
