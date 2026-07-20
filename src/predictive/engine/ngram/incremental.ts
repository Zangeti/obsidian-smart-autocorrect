/**
 * Incremental per-file corpus (#B3). Maintains an aggregate NgramCounts to which
 * individual documents can be added and REMOVED in O(document) time - so when a
 * note is saved we re-tokenise only that note and patch the aggregate, instead
 * of re-reading and re-counting the entire vault.
 *
 * Per-file contributions are stored as SerializedCounts (string-keyed) so they
 * can be subtracted exactly on update/delete/rename.
 */
import { NgramCounts, InMemoryLanguageModel, type BlendConfig, DEFAULT_BLEND } from "./model.ts";
import { buildSerializedCountsFromDocs } from "./workerKernel.ts";
import { pruneCounts } from "./prune.ts";
import type { SerializedCounts } from "./serialize.ts";

function bump(inner: Map<number, number>, key: number, delta: number): void {
  const v = (inner.get(key) ?? 0) + delta;
  if (v <= 0) inner.delete(key);
  else inner.set(key, v);
}

/** Apply a per-file SerializedCounts to the aggregate counts with sign ±1. */
export function applySerializedToCounts(
  agg: NgramCounts,
  s: SerializedCounts,
  sign: number,
): void {
  // Map the serialized file's local ids to aggregate ids via the word strings.
  const idOf = (localId: number): number => agg.intern(s.vocab[localId]);

  for (let lid = 0; lid < s.vocab.length; lid++) {
    const c = s.uni[lid];
    if (!c) continue;
    const id = idOf(lid);
    agg.uni[id] = Math.max(0, (agg.uni[id] ?? 0) + sign * c);
    agg.totalUni += sign * c;
  }
  if (agg.totalUni < 0) agg.totalUni = 0;

  for (const [w1l, entries] of s.bi) {
    const w1 = idOf(w1l);
    let inner = agg.bi.get(w1);
    if (!inner) {
      inner = new Map();
      agg.bi.set(w1, inner);
    }
    for (const [nextL, c] of entries) bump(inner, idOf(nextL), sign * c);
    if (inner.size === 0) agg.bi.delete(w1);
  }

  for (const [w2l, midArr] of s.tri) {
    const w2 = idOf(w2l);
    let mid = agg.tri.get(w2);
    if (!mid) {
      mid = new Map();
      agg.tri.set(w2, mid);
    }
    for (const [w1l, entries] of midArr) {
      const w1 = idOf(w1l);
      let inner = mid.get(w1);
      if (!inner) {
        inner = new Map();
        mid.set(w1, inner);
      }
      for (const [nextL, c] of entries) bump(inner, idOf(nextL), sign * c);
      if (inner.size === 0) mid.delete(w1);
    }
    if (mid.size === 0) agg.tri.delete(w2);
  }
}

export class IncrementalCorpus {
  private agg = new NgramCounts();
  private perFile = new Map<string, SerializedCounts>();
  private blend: BlendConfig;
  private minCount: number;
  /** word -> how many distinct documents contain it. Maintained alongside the aggregate
   *  counts, so it costs one pass over a file's vocabulary on the update we already do. */
  private df = new Map<string, number>();

  /** `minCount` drops words seen fewer than this many times across the vault -
   *  so one-off typos never become suggestions. */
  constructor(blend: BlendConfig = DEFAULT_BLEND, minCount = 1) {
    this.blend = blend;
    this.minCount = minCount;
  }

  /** Precompute a file's counts (can run in a worker) then commit with `set`. */
  static countsFor(text: string): SerializedCounts {
    return buildSerializedCountsFromDocs([text]);
  }

  /** Fold one file's vocabulary into the document-frequency table with sign ±1. */
  private applyDocFreq(counts: SerializedCounts, sign: number): void {
    for (let lid = 0; lid < counts.vocab.length; lid++) {
      if (!counts.uni[lid]) continue; // present in the file's vocab but contributing nothing
      const w = counts.vocab[lid];
      const v = (this.df.get(w) ?? 0) + sign;
      if (v <= 0) this.df.delete(w);
      else this.df.set(w, v);
    }
  }

  /** Add or replace a file's contribution from already-computed counts. */
  set(path: string, counts: SerializedCounts): void {
    const old = this.perFile.get(path);
    if (old) {
      applySerializedToCounts(this.agg, old, -1);
      this.applyDocFreq(old, -1);
    }
    applySerializedToCounts(this.agg, counts, +1);
    this.applyDocFreq(counts, +1);
    this.perFile.set(path, counts);
  }

  /**
   * In how many distinct documents does `word` occur? A far better signal than a raw count
   * for "is this part of the user's vocabulary": a misspelling repeated inside one note is
   * still one mistake, while a word used across several notes is genuinely theirs. Also lets
   * a caller notice the moment a word leaves the vault entirely (frequency reaches 0).
   */
  documentFrequency(word: string): number {
    return this.df.get(word.toLowerCase()) ?? 0;
  }

  /** Convenience: tokenise + count `text` on the current thread, then commit. */
  upsert(path: string, text: string): void {
    this.set(path, IncrementalCorpus.countsFor(text));
  }

  remove(path: string): void {
    const old = this.perFile.get(path);
    if (!old) return;
    applySerializedToCounts(this.agg, old, -1);
    this.applyDocFreq(old, -1);
    this.perFile.delete(path);
  }

  rename(oldPath: string, newPath: string): void {
    const c = this.perFile.get(oldPath);
    if (!c) return;
    this.perFile.delete(oldPath);
    this.perFile.set(newPath, c);
  }

  has(path: string): boolean {
    return this.perFile.has(path);
  }
  get fileCount(): number {
    return this.perFile.size;
  }

  /** Build a language model from the current aggregate (recomputes derived
   *  tables from counts only - no re-tokenising of the vault). Rare words are
   *  pruned so one-off typos never become suggestions. */
  build(): InMemoryLanguageModel {
    const counts = this.minCount > 1 ? pruneCounts(this.agg, this.minCount) : this.agg;
    return new InMemoryLanguageModel(counts, this.blend);
  }
}
