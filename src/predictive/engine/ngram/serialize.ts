/**
 * Compact serialisation of n-gram counts so a model built off the main thread
 * (a Web Worker) can be transferred back and reconstructed. Plain arrays only -
 * cheaply structured-cloneable across the worker boundary.
 */
import { NgramCounts, InMemoryLanguageModel, type BlendConfig, DEFAULT_BLEND } from "./model.ts";

export interface SerializedCounts {
  v: number; // format version
  vocab: string[];
  uni: number[];
  totalUni: number;
  /** [w1, [[next, count], ...]] */
  bi: [number, [number, number][]][];
  /** [w2, [[w1, [[next, count], ...]], ...]] */
  tri: [number, [number, [number, number][]][]][];
}

export function serializeCounts(c: NgramCounts): SerializedCounts {
  const bi: [number, [number, number][]][] = [];
  for (const [w1, inner] of c.bi) bi.push([w1, [...inner.entries()]]);
  const tri: [number, [number, [number, number][]][]][] = [];
  for (const [w2, mid] of c.tri) {
    const midArr: [number, [number, number][]][] = [];
    for (const [w1, inner] of mid) midArr.push([w1, [...inner.entries()]]);
    tri.push([w2, midArr]);
  }
  return { v: 1, vocab: c.vocab, uni: c.uni, totalUni: c.totalUni, bi, tri };
}

export function deserializeCounts(s: SerializedCounts): NgramCounts {
  const c = new NgramCounts();
  c.vocab = s.vocab;
  c.uni = s.uni;
  c.totalUni = s.totalUni;
  for (let id = 0; id < s.vocab.length; id++) c.wordId.set(s.vocab[id], id);
  for (const [w1, entries] of s.bi) c.bi.set(w1, new Map(entries));
  for (const [w2, midArr] of s.tri) {
    const mid = new Map<number, Map<number, number>>();
    for (const [w1, entries] of midArr) mid.set(w1, new Map(entries));
    c.tri.set(w2, mid);
  }
  return c;
}

export function modelFromSerialized(
  s: SerializedCounts,
  cfg: BlendConfig = DEFAULT_BLEND,
): InMemoryLanguageModel {
  return new InMemoryLanguageModel(deserializeCounts(s), cfg);
}
