/**
 * Packed binary language model (#B2). A compact, ArrayBuffer-backed
 * implementation of LanguageModel for shipping a large model without RAM blow-up:
 *
 *   - one front-of-mind design: for autocomplete we only ever need the top-K
 *     continuations of a context, so we store exactly those (pruned),
 *   - probabilities are the already-BLENDED log-probs, quantised to 1 byte,
 *   - contexts are looked up by binary search over sorted key arrays (O(log n)),
 *     and the whole thing is a single ArrayBuffer (memory-mappable, GC-free).
 *
 * `packCounts(model, counts, opts)` produces the buffer from an in-memory model;
 * `PackedLanguageModel.fromBuffer(buf)` loads it. Predictions match the
 * in-memory model up to top-K pruning + 8-bit quantisation.
 */
import { SOS } from "../text/tokenize.ts";
import type { LanguageModel, Scored, NgramCounts } from "./model.ts";

const MAGIC = 0x504b4d32; // "PKM2"

interface ContList {
  ids: Uint32Array;
  q: Uint8Array;
}

// ---- byte writer/reader --------------------------------------------------

class ByteWriter {
  private buf = new Uint8Array(1024);
  private view = new DataView(this.buf.buffer);
  len = 0;
  private ensure(n: number) {
    if (this.len + n <= this.buf.length) return;
    let cap = this.buf.length * 2;
    while (cap < this.len + n) cap *= 2;
    const nb = new Uint8Array(cap);
    nb.set(this.buf);
    this.buf = nb;
    this.view = new DataView(this.buf.buffer);
  }
  u8(v: number) { this.ensure(1); this.view.setUint8(this.len, v); this.len += 1; }
  u16(v: number) { this.ensure(2); this.view.setUint16(this.len, v, true); this.len += 2; }
  u32(v: number) { this.ensure(4); this.view.setUint32(this.len, v >>> 0, true); this.len += 4; }
  f32(v: number) { this.ensure(4); this.view.setFloat32(this.len, v, true); this.len += 4; }
  f64(v: number) { this.ensure(8); this.view.setFloat64(this.len, v, true); this.len += 8; }
  bytes(b: Uint8Array) { this.ensure(b.length); this.buf.set(b, this.len); this.len += b.length; }
  toBuffer(): ArrayBuffer { return this.buf.buffer.slice(0, this.len); }
}

class ByteReader {
  private view: DataView;
  private buf: ArrayBuffer;
  off = 0;
  constructor(buf: ArrayBuffer) { this.buf = buf; this.view = new DataView(buf); }
  u8() { const v = this.view.getUint8(this.off); this.off += 1; return v; }
  u16() { const v = this.view.getUint16(this.off, true); this.off += 2; return v; }
  u32() { const v = this.view.getUint32(this.off, true); this.off += 4; return v; }
  f32() { const v = this.view.getFloat32(this.off, true); this.off += 4; return v; }
  f64() { const v = this.view.getFloat64(this.off, true); this.off += 8; return v; }
  bytes(n: number) { const b = new Uint8Array(this.buf, this.off, n); this.off += n; return b; }
}

// ---- packing -------------------------------------------------------------

export interface PackOptions {
  topK?: number;
  unigramTop?: number;
}

export function packCounts(
  model: LanguageModel,
  counts: NgramCounts,
  opts: PackOptions = {},
): ArrayBuffer {
  const topK = opts.topK ?? 16;
  const unigramTop = opts.unigramTop ?? 256;
  const vocab = counts.vocab;
  const wid = counts.wordId;

  // Gather all (context -> top-K [wordId, logprob]) plus the unigram list.
  type Row = { key: number; ids: number[]; lp: number[] };
  const uni = model.predict([], unigramTop);
  const biRows: Row[] = [];
  const triRows: Row[] = [];
  const V = vocab.length;

  let minLp = Infinity;
  let maxLp = -Infinity;
  const track = (lp: number) => {
    if (lp < minLp) minLp = lp;
    if (lp > maxLp) maxLp = lp;
  };
  for (const s of uni) track(s.logProb);

  // Full per-word unigram log-prob (for backoff of ANY word, not just the top
  // few). ~1 byte/word - the fix for "rare-but-real words tie at a floor value".
  const fullUniLp = new Float64Array(V);
  for (let id = 0; id < V; id++) {
    const lp = model.logProb(vocab[id], []);
    fullUniLp[id] = lp;
    track(lp);
  }

  // Score ONLY a context's own observed continuations (from the count map),
  // not the whole vocabulary - O(total n-grams), so this scales to large
  // corpora. Blended log-probs come from model.logProb.
  const rowFrom = (context: string[], key: number, contIds: IterableIterator<number>): Row | null => {
    const scored: { id: number; lp: number }[] = [];
    for (const id of contIds) {
      if (vocab[id] === SOS) continue;
      const lp = model.logProb(vocab[id], context);
      scored.push({ id, lp });
      track(lp);
    }
    if (!scored.length) return null;
    scored.sort((a, b) => b.lp - a.lp);
    const topN = scored.slice(0, topK);
    return { key, ids: topN.map((s) => s.id), lp: topN.map((s) => s.lp) };
  };

  for (const [w1, inner] of counts.bi) {
    const r = rowFrom([vocab[w1]], w1, inner.keys());
    if (r) biRows.push(r);
  }
  for (const [w2, mid] of counts.tri) {
    for (const [w1, inner] of mid) {
      const r = rowFrom([vocab[w2], vocab[w1]], w2 * V + w1, inner.keys());
      if (r) triRows.push(r);
    }
  }
  if (!isFinite(minLp)) { minLp = -20; maxLp = 0; }
  if (minLp === maxLp) maxLp = minLp + 1;

  biRows.sort((a, b) => a.key - b.key);
  triRows.sort((a, b) => a.key - b.key);

  const quant = (lp: number) =>
    Math.max(0, Math.min(255, Math.round(((lp - minLp) / (maxLp - minLp)) * 255)));

  const w = new ByteWriter();
  w.u32(MAGIC);
  w.u32(V);
  w.f32(minLp);
  w.f32(maxLp);

  // vocab
  const enc = new TextEncoder();
  w.u32(V);
  for (const s of vocab) {
    const b = enc.encode(s);
    w.u16(b.length);
    w.bytes(b);
  }

  // unigram list (for cold-start prediction with no context)
  w.u32(uni.length);
  for (const s of uni) {
    const id = wid.get(s.word) ?? 0;
    w.u32(id);
    w.u8(quant(s.logProb));
  }

  // full per-word unigram log-prob (backoff for every word)
  for (let id = 0; id < V; id++) w.u8(quant(fullUniLp[id]));

  const writeRows = (rows: Row[], keyIsF64: boolean) => {
    w.u32(rows.length);
    for (const r of rows) {
      if (keyIsF64) w.f64(r.key);
      else w.u32(r.key);
      w.u16(r.ids.length);
      for (let i = 0; i < r.ids.length; i++) {
        w.u32(r.ids[i]);
        w.u8(quant(r.lp[i]));
      }
    }
  };
  writeRows(biRows, false);
  writeRows(triRows, true);

  return w.toBuffer();
}

// ---- loading -------------------------------------------------------------

export class PackedLanguageModel implements LanguageModel {
  private vocab: string[];
  private wid = new Map<string, number>();
  private V: number;
  private minLp: number;
  private maxLp: number;
  private uni: ContList;
  private fullUni: Uint8Array;
  private biKeys: Uint32Array;
  private bi: ContList[];
  private triKeys: Float64Array;
  private tri: ContList[];

  private constructor(fields: {
    vocab: string[]; V: number; minLp: number; maxLp: number;
    uni: ContList; fullUni: Uint8Array; biKeys: Uint32Array; bi: ContList[];
    triKeys: Float64Array; tri: ContList[];
  }) {
    this.vocab = fields.vocab;
    this.V = fields.V;
    this.minLp = fields.minLp;
    this.maxLp = fields.maxLp;
    this.uni = fields.uni;
    this.fullUni = fields.fullUni;
    this.biKeys = fields.biKeys;
    this.bi = fields.bi;
    this.triKeys = fields.triKeys;
    this.tri = fields.tri;
    for (let i = 0; i < this.vocab.length; i++) this.wid.set(this.vocab[i], i);
  }

  static fromBuffer(buf: ArrayBuffer): PackedLanguageModel {
    const r = new ByteReader(buf);
    if (r.u32() !== MAGIC) throw new Error("bad packed model magic");
    const V = r.u32();
    const minLp = r.f32();
    const maxLp = r.f32();
    const dec = new TextDecoder();
    const vcount = r.u32();
    const vocab: string[] = new Array(vcount);
    for (let i = 0; i < vcount; i++) {
      const len = r.u16();
      vocab[i] = dec.decode(r.bytes(len));
    }
    const readList = (): ContList => {
      const n = r.u32();
      const ids = new Uint32Array(n);
      const q = new Uint8Array(n);
      for (let i = 0; i < n; i++) { ids[i] = r.u32(); q[i] = r.u8(); }
      return { ids, q };
    };
    const uni = readList();
    const fullUni = new Uint8Array(V);
    for (let i = 0; i < V; i++) fullUni[i] = r.u8();
    const readRows = (keyIsF64: boolean) => {
      const n = r.u32();
      const keys = keyIsF64 ? new Float64Array(n) : new Uint32Array(n);
      const lists: ContList[] = new Array(n);
      for (let i = 0; i < n; i++) {
        keys[i] = keyIsF64 ? r.f64() : r.u32();
        const len = r.u16();
        const ids = new Uint32Array(len);
        const q = new Uint8Array(len);
        for (let j = 0; j < len; j++) { ids[j] = r.u32(); q[j] = r.u8(); }
        lists[i] = { ids, q };
      }
      return { keys, lists };
    };
    const bi = readRows(false);
    const tri = readRows(true);
    return new PackedLanguageModel({
      vocab, V, minLp, maxLp, uni, fullUni,
      biKeys: bi.keys as Uint32Array, bi: bi.lists,
      triKeys: tri.keys as Float64Array, tri: tri.lists,
    });
  }

  private deq(q: number): number {
    return this.minLp + (q / 255) * (this.maxLp - this.minLp);
  }

  private find(keys: { length: number; [i: number]: number }, key: number): number {
    let lo = 0;
    let hi = keys.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const k = keys[mid];
      if (k === key) return mid;
      if (k < key) lo = mid + 1;
      else hi = mid - 1;
    }
    return -1;
  }

  private listFor(context: string[]): ContList {
    const n = context.length;
    const w1 = n >= 1 ? this.wid.get(context[n - 1]) : undefined;
    const w2 = n >= 2 ? this.wid.get(context[n - 2]) : undefined;
    if (w1 !== undefined && w2 !== undefined) {
      const idx = this.find(this.triKeys, w2 * this.V + w1);
      if (idx >= 0) return this.tri[idx];
    }
    if (w1 !== undefined) {
      const idx = this.find(this.biKeys, w1);
      if (idx >= 0) return this.bi[idx];
    }
    return this.uni;
  }

  hasWord(word: string): boolean {
    return this.wid.has(word);
  }
  vocabulary(): IterableIterator<string> {
    return this.wid.keys();
  }
  size() {
    return { vocab: this.vocab.length, bigrams: this.biKeys.length, trigrams: this.triKeys.length };
  }

  logProb(word: string, context: string[]): number {
    const id = this.wid.get(word);
    if (id === undefined) return this.minLp - 2;
    // Try richest context first, then back off.
    const tryList = (list: ContList): number | null => {
      for (let i = 0; i < list.ids.length; i++) if (list.ids[i] === id) return this.deq(list.q[i]);
      return null;
    };
    const n = context.length;
    if (n >= 2) {
      const w1 = this.wid.get(context[n - 1]);
      const w2 = this.wid.get(context[n - 2]);
      if (w1 !== undefined && w2 !== undefined) {
        const idx = this.find(this.triKeys, w2 * this.V + w1);
        if (idx >= 0) { const v = tryList(this.tri[idx]); if (v !== null) return v; }
      }
    }
    if (n >= 1) {
      const w1 = this.wid.get(context[n - 1]);
      if (w1 !== undefined) {
        const idx = this.find(this.biKeys, w1);
        if (idx >= 0) { const v = tryList(this.bi[idx]); if (v !== null) return v; }
      }
    }
    // Full-vocabulary unigram backoff (every real word has a real value).
    return this.deq(this.fullUni[id]);
  }

  predict(context: string[], k: number): Scored[] {
    const list = this.listFor(context);
    const out: Scored[] = [];
    for (let i = 0; i < list.ids.length && out.length < k; i++) {
      const word = this.vocab[list.ids[i]];
      if (word === SOS) continue;
      out.push({ word, logProb: this.deq(list.q[i]) });
    }
    return out;
  }
}
