/**
 * Word-level LSTM next-word language model (CPU). Reads the INT8 FACTORED binary
 * (fmt 3) produced by build_model/train_word_lstm_cased.py. Weight-tied (output =
 * embedding), so predicting = run the LSTM over the context, then a single tied
 * projection + softmax over the vocabulary.
 *
 * LOWERCASE IN, CASE-SENSITIVE OUT. The vocab is lowercase and the model takes
 * lowercase ids with NO input case factor, so we lowercase context words and are
 * done - this class never classifies the case of the text it reads. Casing comes
 * back OUT of a separate 4-way head (see renderCased). That asymmetry is the whole
 * point: an input case factor made "THE QUICK BROWN FOX" genuinely predict more
 * caps, so the editor kept shouting no matter what was typed next. Continuing a
 * caps run is caseMode.ts's job, deterministically.
 *
 * INT8 ONLY. The embedding is the model's bulk (V*dim) and, being tied, is also the
 * output projection: the size problem and the CPU hot loop are the same matrix.
 * Stored as int8 with a per-row (per-word) scale it costs ~46 MB, and under SIMD it
 * is *faster* than f32 (16 int8 lanes per instruction vs 4), so quantising wins
 * twice. The LSTM gate matrices stay f32 here (~5 MB).
 *
 * Two execution paths, same results: a WASM SIMD kernel when the platform has it
 * (~8-16x), else the scalar JS fallback.
 */
import type { Scored, LanguageModel } from "../ngram/model.ts";
import { SENTENCE_END } from "../text/tokenize.ts";
import { terms } from "../text/vector.ts";
import { loadKernel, Bump, type Kernel } from "./wasm.ts";

const MAGIC = 0x4c53544d; // "LSTM"
const FMT_FACTORED = 4; // int8 emb + per-row scales + lowercase vocab + case head + bilinear
// fmt 5 = fmt 4 with the three LSTM gate matrices ALSO int8 (per-row scale after each).
// Measured on 19,809 words: f32 gates ppl 64.621 -> int8 gates+activations 64.534, i.e. no
// cost, while the gate matvecs go 1.75 -> 0.83 ms and the file drops 21 MB. Both formats
// load; fmt 4 is quantised on the fly so the runtime only ever has one (int8) code path.
const FMT_I8_GATES = 5;
const OOV_LOGPROB = -20; // words outside the LSTM vocab get a low prior

/**
 * FULL-DOCUMENT CONTEXT, via waypoints. There is NO context window: the state at the
 * cursor is the exact state of every word before it, which is the regime the model is
 * trained in (stateful truncated BPTT carries (h,c) across ~40k-word passages; only
 * the GRADIENT is truncated). A window here would be an inference-only approximation
 * the training never asked for.
 *
 * The problem a window was solving is real - an LSTM state cannot forget its oldest
 * word, so a SLIDING window forces a full replay on every keystroke, which is exactly
 * what the old MAX_CONTEXT=16 did: 16 steps per keystroke in any document past 16
 * words. Waypoints solve it properly instead of by truncating:
 *
 *   - TYPING appends, so the previous context is a prefix of this one: extend the
 *     cached parent by ONE step. This is the hot path and it is O(1) at any length.
 *   - A CURSOR JUMP or an EDIT lands somewhere we have no state for. Restore the
 *     deepest waypoint at or before it and step forward - at most WAYPOINT_EVERY steps.
 *
 * Waypoints are keyed by a rolling hash of the text prefix, which makes invalidation
 * automatic and impossible to get wrong: edit word 100 and every prefix hash after 100
 * changes, so those waypoints simply stop matching, while everything before 100 stays
 * valid. No change events, no versioning, no stale state.
 *
 * Cost: one waypoint is 2 layers x (dim + hid) f32 ~= 9 KB, so a 10k-word note holds
 * ~156 of them (~1.4 MB) - bounded by an LRU. The one unavoidable cost is FIRST touch
 * of a long document with no waypoints yet: that walk is O(words). It happens once,
 * in the worker, and builds the waypoints that make every later position cheap.
 */
const WAYPOINT_EVERY = 64;
/** Bounds waypoint memory across open documents. 512 x ~9 KB ~= 4.6 MB. */
const WAYPOINT_CAP = 512;

// Case classes - MUST match build_model/train_word_lstm_cased.py.
const CASE_LOWER = 0, CASE_TITLE = 1, CASE_UPPER = 2, CASE_OTHER = 3;
const N_CASE = 4;

/** Inverse of the trainer's case_of for the three REGULAR classes. `lower` is
 *  already lowercase, so this matches Python's .capitalize() exactly. */
function applyCase(lower: string, cls: number): string {
  if (cls === CASE_TITLE) return lower.charAt(0).toUpperCase() + lower.slice(1);
  if (cls === CASE_UPPER) return lower.toUpperCase();
  return lower;
}
// The output projection is LINEAR in the number of vocab rows scanned. The vocab
// is frequency-sorted (id 0 = <unk>, then most-common first), so scanning only the
// top SHORTLIST rows captures virtually all softmax mass and every confident
// prediction - a rare word almost never tops the distribution. Nothing is lost: a
// specific rare word still gets an EXACT probability via an on-demand single-row
// dot product in logProb(), and the n-gram/vault side keeps proposing rare words.
const SHORTLIST = 16000;
// Greedy phrase continuations are even more concentrated on very frequent words
// ("of the", "New York", "to be"), so phrase extension scans a tighter shortlist -
// it runs many steps per suggestion, so this is where row count matters most.
const PHRASE_SHORTLIST = 4000;
// Punctuation tokens the trainer emits. A suggestion is words, so phrase extension
// stops when the model wants one (see phrasesFor).
const PUNCT = new Set([".", ",", "!", "?", ";", ":"]);

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/** Cached per-context state: shortlisted logits, their normaliser, and the full
 *  per-layer recurrent state. The hidden state lets a rare word (id >= nTop) be
 *  scored exactly on demand, and lets phrasesFor() continue stepping from this
 *  context without replaying it. */
interface CtxEntry {
  logits: Float32Array; // length nTop
  logZ: number;
  h: Float32Array[]; // per-layer hidden state that produced `logits`
  c: Float32Array[]; // per-layer cell state
}

interface Ptr {
  embQ: number; scaleW: number; bOut: number;
  wih: number[]; wihS: number[]; whh: number[]; whhS: number[];
  bias: number[]; whr: number[]; whrS: number[];
  h: number[]; c: number[];
  // xQ/hQin/hRawQ are the int8 activation buffers the gate matvecs read; gates2 is
  // the scratch the whh term lands in before being added onto `gates` (project_i8
  // writes, it does not accumulate).
  x: number; gates: number; gates2: number; hRaw: number; zero: number;
  xQ: number; hQin: number; hRawQ: number; hQ: number; out: number;
}

/** An int8 matrix with one dequant scale per row - what `project_i8` consumes. */
interface QMat { q: Int8Array; scale: Float32Array; }

export class LstmLanguageModel implements LanguageModel {
  readonly vocab: string[];
  private wid = new Map<string, number>();
  private V!: number;
  private dim!: number;
  private hid!: number;
  private layers!: number;
  private nTop!: number;
  private nPhrase!: number;

  // Weights. When the kernel is active these are VIEWS into wasm linear memory,
  // so both execution paths read the same bytes and nothing is stored twice.
  private embQ!: Int8Array; // [V, dim] int8
  private scaleW!: Float32Array; // [V] per-row dequant scale
  private bOut!: Float32Array; // [V]
  // Both gate matrices have `dim` columns: layer 0 consumes the embedding (dim),
  // later layers consume the previous h - and h is ALWAYS dim-wide, either because
  // hid == dim or because weight_hr projects it back down to dim.
  // Gates are int8 with a per-ROW dequant scale, same scheme as the embedding.
  // Measured: no ppl cost (64.621 f32 -> 64.534 int8 over 19,809 words) at 2.1x the
  // matvec speed. The re-quantisation of h at every step does NOT compound.
  private wih!: Int8Array[]; // per layer [4*hid, dim]
  private wihS!: Float32Array[]; // per layer [4*hid]
  private whh!: Int8Array[]; // per layer [4*hid, dim]
  private whhS!: Float32Array[]; // per layer [4*hid]
  private bias!: Float32Array[]; // per layer [4*hid] = bih + bhh (folded at load)
  /** LSTMP projection [dim, hid] per layer, or null when hid === dim. int8 + per-row scale. */
  private whr: Int8Array[] | null = null;
  private whrS: Float32Array[] | null = null;

  // --- case head: logit(case) = caseCtxW·h + caseCtxB + caseWordW·emb(w) + caseBias[w]
  // Deliberately log-linear and additive so scoring a candidate is a 4x`dim` dot
  // rather than an MLP over a 16k shortlist. The per-word caseBias is what makes
  // "nasa" -> NASA safe: without it the only word-specific signal is a LINEAR map
  // emb(w) -> 4 classes, which cannot pin a deterministic spelling.
  private caseCtxW!: Float32Array; // [N_CASE, dim]
  private caseCtxB!: Float32Array; // [N_CASE]
  private caseWordW!: Float32Array; // [N_CASE, dim]
  private caseBias!: Float32Array; // [V, N_CASE]
  // Low-rank bilinear context x word interaction (fmt 4). caseA gates the hidden
  // state h, caseB gates emb(w); their per-rank dot products multiply and sum. This
  // is the term that lets context override a homograph's strong per-word bias
  // ("how do I polish" -> LOWER despite bias TITLE +3.45). Mirrors case_A/case_B in
  // the trainer's forward() exactly.
  private caseRank = 0;
  private caseA!: Float32Array; // [N_CASE, caseRank, dim]
  private caseB!: Float32Array; // [N_CASE, caseRank, dim]
  /** lowercase -> irregular spelling ("iphone" -> "iPhone"). ONLY consulted for
   *  CASE_OTHER; NASA never touches it (it is CASE_UPPER and rebuilds by rule). */
  private surface = new Map<string, string>();
  private caseScratch = new Float32Array(N_CASE);
  /** Memo for the lowercased context: the predictor hands the same array object to
   *  every candidate, so this makes it one transform per prediction, not per word. */
  private lcIn: string[] | null = null;
  private lcOut: string[] = [];
  /** Memo for the prefix keys of that same array (keys[i] identifies ctx[0..i)). */
  private keyIn: string[] | null = null;
  private keyOut: string[] = [];
  /** prefix key -> recurrent state, every WAYPOINT_EVERY words. State only: logits
   *  are not worth storing for a position the cursor merely passed through. */
  private waypoints = new Map<string, { h: Float32Array[]; c: Float32Array[] }>();

  // Recurrent state (per layer). Allocated ONCE and mutated in place - never
  // reassigned, because when wasm is active these alias its linear memory.
  private h: Float32Array[] = []; // [dim] - projected/output hidden
  private c: Float32Array[] = []; // [hid] - the cell, at full width
  private xView!: Float32Array; // dequantised input embedding row [dim]
  private gatesView!: Float32Array; // [4*hid]
  private gates2View!: Float32Array; // [4*hid] scratch for the whh term
  private xQ!: Int8Array; // [dim] quantised layer input (scalar path only)
  private hQin!: Int8Array; // [dim] quantised previous h (scalar path only)
  private hRawQ!: Int8Array; // [hid] quantised pre-projection hidden (scalar path only)
  private hRawView!: Float32Array; // [hid] pre-projection o*tanh(c)
  private zeroView!: Float32Array; // [dim] zero bias for the projection matvec
  private outView!: Float32Array; // [nTop] projection output

  private k: Kernel | null = null;
  private ptr!: Ptr;

  private stepLogits: Float32Array | null = null;
  private stepLogZ = 0;
  // LRU of context -> logits. Alternating contexts (the real one and the empty
  // context the scorer uses for its unigram feature) both stay warm instead of
  // evicting each other and forcing a softmax per candidate.
  private ctxCache = new Map<string, CtxEntry>();
  private cur: CtxEntry | null = null;
  private static readonly CACHE_CAP = 8;
  // A phrase is a deterministic function of (context, seed, threshold, maxWords),
  // and phrase extension runs on EVERY mid-word keystroke - where the seed set
  // largely repeats as you type ("new" stays the top completion through n-e-w).
  private phraseCache = new Map<string, { words: string[]; extLogProb: number }[]>();
  private static readonly PHRASE_CACHE_CAP = 64;

  /** True when the SIMD kernel is in use (false = scalar JS fallback). */
  get accelerated(): boolean {
    return this.k !== null;
  }

  private constructor() {
    this.vocab = [];
  }

  static fromBuffer(buf: ArrayBuffer, useSimd = true): LstmLanguageModel {
    const m = new LstmLanguageModel();
    const dv = new DataView(buf);
    let o = 0;
    const u32 = () => { const v = dv.getUint32(o, true); o += 4; return v; };
    const u16 = () => { const v = dv.getUint16(o, true); o += 2; return v; };
    const u8 = () => { const v = dv.getUint8(o); o += 1; return v; };
    if (u32() !== MAGIC) throw new Error("bad LSTM magic");
    const V = u32(), dim = u32(), hid = u32(), layers = u32();
    const fmt = u8();
    if (fmt !== FMT_FACTORED && fmt !== FMT_I8_GATES)
      throw new Error(`LSTM model must be factored int8 (fmt ${FMT_FACTORED} or ` +
                      `${FMT_I8_GATES}); got fmt ${fmt}. Re-export with ` +
                      `build_model/train_word_lstm_cased.py - the cased-vocab formats are ` +
                      `gone, not supported.`);
    m.V = V; m.dim = dim; m.hid = hid; m.layers = layers;
    m.nTop = Math.min(SHORTLIST, V);
    m.nPhrase = Math.min(PHRASE_SHORTLIST, V);

    const dec = new TextDecoder();
    const vocab = new Array<string>(V);
    for (let i = 0; i < V; i++) {
      const len = u16();
      vocab[i] = dec.decode(new Uint8Array(buf, o, len));
      o += len;
    }
    (m as { vocab: string[] }).vocab = vocab;
    for (let i = 0; i < V; i++) m.wid.set(vocab[i], i);

    // Irregular-form table: the CASE_OTHER spellings ("iphone" -> "iPhone") that no
    // case tag can reconstruct. Written as a count, then id/len/bytes per entry.
    const nSurface = u32();
    for (let i = 0; i < nSurface; i++) {
      const id = u32();
      const len = u16();
      m.surface.set(vocab[id], dec.decode(new Uint8Array(buf, o, len)));
      o += len;
    }

    // int8 needs no alignment, so this is a zero-copy view on the source. The f32
    // sections follow a variable-length vocab so `o` isn't 4-aligned - slice()
    // gives an aligned copy (only ~10 MB total, vs 46 MB for the embedding).
    const embQSrc = new Int8Array(buf, o, V * dim); o += V * dim;
    const f32 = (n: number) => { const a = new Float32Array(buf.slice(o, o + n * 4)); o += n * 4; return a; };
    const scaleWSrc = f32(V);
    // hid !== dim means the model is PROJECTED (LSTMP) and each layer carries a
    // weight_hr after its biases. The trainer derives this the same way, so an
    // unprojected model stays readable with no format flag.
    const projected = hid !== dim;
    // A gate matrix, in the int8-per-row form `project_i8` consumes. fmt 5 stores it
    // that way; fmt 4 stores f32 and we quantise here, so there is exactly ONE
    // arithmetic path at run time regardless of which file the user has.
    const gateQ = (rows: number, cols: number): QMat => {
      if (fmt === FMT_I8_GATES) {
        const q = new Int8Array(buf, o, rows * cols); o += rows * cols;
        return { q, scale: f32(rows) };
      }
      const w = f32(rows * cols);
      const q = new Int8Array(rows * cols), scale = new Float32Array(rows);
      for (let r = 0; r < rows; r++) {
        const base = r * cols;
        let mx = 1e-12;
        for (let a = 0; a < cols; a++) { const v = Math.abs(w[base + a]); if (v > mx) mx = v; }
        const s = mx / 127, inv = 1 / s;
        scale[r] = s;
        for (let a = 0; a < cols; a++) {
          const v = Math.round(w[base + a] * inv);
          q[base + a] = v > 127 ? 127 : v < -127 ? -127 : v;
        }
      }
      return { q, scale };
    };
    const wihSrc: QMat[] = [], whhSrc: QMat[] = [], whrSrc: QMat[] = [];
    const biasSrc: Float32Array[] = [];
    for (let L = 0; L < layers; L++) {
      wihSrc.push(gateQ(4 * hid, dim));
      whhSrc.push(gateQ(4 * hid, dim));
      const bih = f32(4 * hid), bhh = f32(4 * hid);
      // Fold the two bias vectors into one: PyTorch keeps them separate, but only
      // their sum is ever used, and the kernel takes a single bias operand.
      const b = new Float32Array(4 * hid);
      for (let i = 0; i < b.length; i++) b[i] = bih[i] + bhh[i];
      biasSrc.push(b);
      if (projected) whrSrc.push(gateQ(dim, hid));
    }
    const bOutSrc = f32(V);
    // Case head. Small enough (~2 MB, dominated by the per-word bias) to stay f32 on
    // the JS heap rather than compete for the wasm arena with the hot matrices.
    m.caseCtxW = f32(N_CASE * dim);
    m.caseCtxB = f32(N_CASE);
    m.caseWordW = f32(N_CASE * dim);
    m.caseBias = f32(V * N_CASE);
    // Bilinear interaction (fmt 4): a u32 rank, then the two [N_CASE, rank, dim]
    // tensors. Small (~4*8*384*2 f32 ≈ 100 KB) so it stays f32 on the JS heap.
    const rank = u32();
    m.caseRank = rank;
    m.caseA = f32(N_CASE * rank * dim);
    m.caseB = f32(N_CASE * rank * dim);
    if (o !== buf.byteLength)
      throw new Error(`LSTM model: ${buf.byteLength - o} trailing bytes - the file and ` +
                      `this loader disagree about the layout`);

    m.init(embQSrc, scaleWSrc, bOutSrc, wihSrc, whhSrc, biasSrc, projected ? whrSrc : null, useSimd);
    return m;
  }

  /** Place the weights + working buffers, in wasm memory when we can get it. */
  private init(
    embQ: Int8Array, scaleW: Float32Array, bOut: Float32Array,
    wih: QMat[], whh: QMat[], bias: Float32Array[],
    whr: QMat[] | null,
    useSimd: boolean,
  ): void {
    const { V, dim, hid, layers, nTop } = this;
    const f = 4;
    // `zero` doubles as the bias operand for the whh matvec, which has 4*hid rows.
    const zeroLen = Math.max(dim, 4 * hid);
    // embQ(int8) + scaleW + bOut + out + x + gates + gates2 + hRaw + zero (f32),
    // + xQ/hQin/hQ(dim) and hRawQ(hid) int8, then per layer the int8 gate matrices
    // with their row scales, bias, optional projection, and the h(dim)/c(hid) state.
    let need = embQ.byteLength
      + (V + V + nTop + dim + 4 * hid + 4 * hid + hid + zeroLen) * f
      + dim * 3 + hid;
    for (let L = 0; L < layers; L++)
      need += wih[L].q.byteLength + wih[L].scale.byteLength
            + whh[L].q.byteLength + whh[L].scale.byteLength
            + bias[L].byteLength
            + (whr ? whr[L].q.byteLength + whr[L].scale.byteLength : 0)
            + (dim + hid) * f;
    need += 1 << 16; // slack for alignment padding

    // The SIMD loops step 16 int8 / 4 f32 lanes at a time with no remainder
    // handling, so refuse the kernel unless the shapes divide cleanly. `useSimd` is
    // the user setting (default on): turning it off forces the scalar JS path even
    // where SIMD is available, and `accelerated` then reports false so the settings
    // pane can say so rather than the plugin silently choosing for them.
    const shapesOk = dim % 16 === 0 && hid % 16 === 0;
    this.k = shapesOk && useSimd ? loadKernel(need) : null;

    if (this.k) {
      const b = new Bump(this.k.mem.buffer.byteLength);
      const mem = this.k.mem.buffer;
      const i8 = (n: number) => { const p = b.alloc(n); return { p, v: new Int8Array(mem, p, n) }; };
      const ff = (n: number) => { const p = b.alloc(n * f); return { p, v: new Float32Array(mem, p, n) }; };
      const e = i8(V * dim); e.v.set(embQ); this.embQ = e.v;
      const sw = ff(V); sw.v.set(scaleW); this.scaleW = sw.v;
      const bo = ff(V); bo.v.set(bOut); this.bOut = bo.v;
      const P: Ptr = {
        embQ: e.p, scaleW: sw.p, bOut: bo.p,
        wih: [], wihS: [], whh: [], whhS: [], bias: [], whr: [], whrS: [], h: [], c: [],
        x: 0, gates: 0, gates2: 0, hRaw: 0, zero: 0,
        xQ: 0, hQin: 0, hRawQ: 0, hQ: 0, out: 0,
      };
      const qmat = (m: QMat, into: Int8Array[], intoS: Float32Array[],
                    pq: number[], ps: number[]) => {
        const a = i8(m.q.length); a.v.set(m.q); into.push(a.v); pq.push(a.p);
        const s = ff(m.scale.length); s.v.set(m.scale); intoS.push(s.v); ps.push(s.p);
      };
      this.wih = []; this.wihS = []; this.whh = []; this.whhS = []; this.bias = [];
      this.whr = whr ? [] : null;
      this.whrS = whr ? [] : null;
      for (let L = 0; L < layers; L++) {
        qmat(wih[L], this.wih, this.wihS, P.wih, P.wihS);
        qmat(whh[L], this.whh, this.whhS, P.whh, P.whhS);
        const d = ff(bias[L].length); d.v.set(bias[L]); this.bias.push(d.v); P.bias.push(d.p);
        if (whr) qmat(whr[L], this.whr!, this.whrS!, P.whr, P.whrS);
      }
      for (let L = 0; L < layers; L++) {
        const hh = ff(dim); this.h.push(hh.v); P.h.push(hh.p);
        const cc = ff(hid); this.c.push(cc.v); P.c.push(cc.p);
      }
      const x = ff(dim); this.xView = x.v; P.x = x.p;
      const g = ff(4 * hid); this.gatesView = g.v; P.gates = g.p;
      const g2 = ff(4 * hid); this.gates2View = g2.v; P.gates2 = g2.p;
      const hr = ff(hid); this.hRawView = hr.v; P.hRaw = hr.p;
      const z = ff(zeroLen); this.zeroView = z.v; P.zero = z.p; // stays all-zero
      P.xQ = i8(dim).p;
      P.hQin = i8(dim).p;
      P.hRawQ = i8(hid).p;
      const hq = i8(dim); P.hQ = hq.p;
      const out = ff(nTop); this.outView = out.v; P.out = out.p;
      this.ptr = P;
    } else {
      this.embQ = embQ; this.scaleW = scaleW; this.bOut = bOut;
      this.wih = wih.map((m) => m.q); this.wihS = wih.map((m) => m.scale);
      this.whh = whh.map((m) => m.q); this.whhS = whh.map((m) => m.scale);
      this.bias = bias;
      this.whr = whr ? whr.map((m) => m.q) : null;
      this.whrS = whr ? whr.map((m) => m.scale) : null;
      for (let L = 0; L < layers; L++) {
        this.h.push(new Float32Array(dim));
        this.c.push(new Float32Array(hid));
      }
      this.xView = new Float32Array(dim);
      this.gatesView = new Float32Array(4 * hid);
      this.gates2View = new Float32Array(4 * hid);
      this.hRawView = new Float32Array(hid);
      this.zeroView = new Float32Array(zeroLen);
      this.outView = new Float32Array(nTop);
      this.xQ = new Int8Array(dim);
      this.hQin = new Int8Array(dim);
      this.hRawQ = new Int8Array(hid);
    }
    this.reset();
  }

  hasWord(w: string): boolean {
    return this.wid.has(w);
  }
  get vocabSize(): number {
    return this.V;
  }

  /** Embedding dimensionality - the length of the vectors `embed` returns. */
  get embeddingDim(): number {
    return this.dim;
  }

  /**
   * How distinctive / descriptive a word is, in (0,1]. The vocab is frequency-sorted, so
   * a word's id is its rarity rank: common filler ("people", "thing") sits at a low id and
   * scores near 0, a domain term ("photosynthesis") scores high. Out-of-vocab words are
   * treated as niche (0.9) since they're usually names or specialist terms. Used to bias
   * tag suggestions toward descriptive words rather than generic ones.
   */
  rarity(word: string): number {
    const id = this.wid.get(word.toLowerCase());
    if (id === undefined) return 0.9;
    if (id === 0) return 0; // <unk>
    return Math.log(id + 1) / Math.log(this.V + 1);
  }

  /**
   * A topical fingerprint for a chunk of text: the IDF-weighted mean of its words'
   * learned embedding rows, L2-normalised. Because the embedding is weight-tied to the
   * output, words that the model predicts in similar contexts have similar rows, so the
   * cosine of two such fingerprints measures how related two passages are - the
   * "semantic" half of related-link suggestion.
   *
   * Weighting: the vocab is frequency-sorted (id 0 = <unk>, low ids = common words), so
   * a word's id is a rank. We down-weight common words with a rank-based IDF proxy, which
   * needs no corpus statistics and mirrors what TF-IDF does on the keyword side. Reads the
   * int8 rows directly and does NOT touch the recurrent scratch, so it is safe to call
   * mid-session without disturbing prediction state.
   */
  embed(text: string): Float32Array {
    const dim = this.dim, out = new Float32Array(dim);
    const q = this.embQ, sw = this.scaleW, V = this.V;
    const denom = Math.log(V + 1);
    let wsum = 0;
    for (const w of terms(text)) {
      const id = this.wid.get(w);
      if (id === undefined || id === 0) continue; // OOV / <unk> carry no topic signal
      // rank-based IDF proxy in (0,1]: rarer word (higher id) => weight nearer 1.
      const weight = Math.log(id + 1) / denom;
      if (weight <= 0) continue;
      const base = id * dim, sc = sw[id] * weight;
      for (let a = 0; a < dim; a++) out[a] += q[base + a] * sc;
      wsum += weight;
    }
    if (wsum === 0) return out; // all-zero: caller treats as "no signal"
    let n = 0;
    for (let a = 0; a < dim; a++) n += out[a] * out[a];
    if (n > 0) {
      const inv = 1 / Math.sqrt(n);
      for (let a = 0; a < dim; a++) out[a] *= inv;
    }
    return out;
  }

  /** Clear the recurrent state (start of a new prediction context). */
  reset(): void {
    for (let L = 0; L < this.layers; L++) { this.h[L].fill(0); this.c[L].fill(0); }
  }

  /** Copy a saved state back into the live buffers (never reassigns - the live
   *  buffers may alias wasm memory). */
  private loadState(h: Float32Array[], c: Float32Array[]): void {
    for (let L = 0; L < this.layers; L++) { this.h[L].set(h[L]); this.c[L].set(c[L]); }
  }

  /** Advance the recurrent state by one word (0/<unk> for OOV). The vocab is
   *  lowercase; callers on the hot path pass already-lowercased tokens. */
  stepWord(word: string): void {
    this.step(this.wid.get(word) ?? 0);
  }

  /**
   * Case logits for `id` against hidden state `h`:
   *   logit(c) = caseCtxW[c]·h + caseCtxB[c] + caseWordW[c]·emb(id) + caseBias[id][c]
   * ~4*dim*2 MACs - negligible next to a step, so cased candidates are ~free.
   */
  private caseLogitsInto(h: Float32Array, id: number, out: Float32Array): void {
    const dim = this.dim, sc = this.scaleW[id], base = id * dim;
    const rank = this.caseRank, caseA = this.caseA, caseB = this.caseB;
    for (let c = 0; c < N_CASE; c++) {
      const row = c * dim;
      let ctx = this.caseCtxB[c], wrd = 0;
      for (let a = 0; a < dim; a++) {
        ctx += this.caseCtxW[row + a] * h[a];
        wrd += this.caseWordW[row + a] * this.embQ[base + a];
      }
      // Bilinear term: sum_r (h . A[c,r]) * (e . B[c,r]), where e = emb(w) = embQ*sc.
      // Factor sc out of the B dot exactly as `wrd * sc` does for the word term.
      let bil = 0;
      for (let r = 0; r < rank; r++) {
        const off = (c * rank + r) * dim;
        let ha = 0, eb = 0;
        for (let a = 0; a < dim; a++) {
          ha += caseA[off + a] * h[a];
          eb += caseB[off + a] * this.embQ[base + a];
        }
        bil += ha * eb * sc;
      }
      out[c] = ctx + wrd * sc + this.caseBias[id * N_CASE + c] + bil;
    }
  }

  /**
   * Turn case logits into a surface form. MIRRORS render_cased in the trainer -
   * these two must not drift.
   *
   * CASE_OTHER is only selectable for a word that HAS a stored spelling: it means
   * "irregular, look it up" ("iphone" -> "iPhone"), so picking it for a word with no
   * entry would silently degrade to bare lowercase. Mask it and take the best regular
   * class instead. NASA needs none of this - it is CASE_UPPER and rebuilds by rule.
   */
  private renderFromLogits(lower: string, lg: Float32Array): string {
    // 4 classes: rank by insertion rather than allocating a sort per candidate.
    const order = [0, 1, 2, 3];
    order.sort((x, y) => lg[y] - lg[x]);
    for (const c of order) {
      if (c === CASE_OTHER) {
        const s = this.surface.get(lower);
        if (s) return s;
        continue; // no stored form: this class is not available for this word
      }
      return applyCase(lower, c);
    }
    return lower;
  }

  /**
   * How `word` should be capitalised in this context - the case head's decision.
   * "the polish government" -> "Polish", "i will polish it" -> "polish", and the
   * per-word case bias is what keeps "nasa" -> "NASA" instead of "Nasa".
   */
  renderCased(word: string, context: string[]): string {
    const lower = word.toLowerCase();
    const id = this.wid.get(lower);
    if (id === undefined) return word; // not ours to case
    this.prepare(context);
    this.caseLogitsInto(this.cur!.h[this.layers - 1], id, this.caseScratch);
    return this.renderFromLogits(lower, this.caseScratch);
  }

  /**
   * The surface forms `word` can plausibly take here - usually one, but TWO for a
   * genuine case-homograph whose top case classes are close ("polish"/"Polish"). Case
   * is a per-word property the head is sometimes UNCERTAIN about, and collapsing it to
   * a single argmax is what stopped the popup from ever offering both; returning every
   * class within `margin` logits of the best lets the caller show them as separate
   * candidates (a word in two cases IS two candidates), best-first and deduped.
   * CASE_OTHER is only included when the word has a stored irregular spelling.
   */
  caseVariants(word: string, context: string[], margin = 4.0): string[] {
    const lower = word.toLowerCase();
    const id = this.wid.get(lower);
    if (id === undefined) return [word];
    this.prepare(context);
    this.caseLogitsInto(this.cur!.h[this.layers - 1], id, this.caseScratch);
    const lg = this.caseScratch;
    const best = this.renderFromLogits(lower, lg); // the model's preferred surface here
    // ASYMMETRIC on purpose. When the model CAPITALISES the word we ALSO offer the plain
    // lowercase form IF lowercase is plausible here (within `margin` logits of the top
    // class) - that is exactly the homograph the user picks between ("Polish"+"polish",
    // "March"+"march"), while a confident proper noun ("Paris", whose lowercase logit is
    // far below) stays single. We do NOT add a capitalised alternate to a LOWERCASE
    // prediction: capitalising mid-sentence is the sentence/CAPS rule's job, and adding it
    // here would wrongly offer "The" for a plain "the". So a lowercase prediction is never
    // split, which is what keeps common words clean even on an under-trained case head.
    let bestC = 0;
    for (let c = 1; c < N_CASE; c++) if (lg[c] > lg[bestC]) bestC = c;
    if (bestC !== CASE_LOWER && lg[CASE_LOWER] >= lg[bestC] - margin && lower !== best) {
      return [best, lower];
    }
    return [best];
  }

  /**
   * Case `word` for AUTOCORRECT, which rewrites what the user actually typed - so it
   * must only move OFF lowercase when the head is CONFIDENT. Returns the best up-cased
   * surface only if that class beats CASE_LOWER by >= `margin`; otherwise lowercase
   * stands. This is what keeps a lowercase homograph ("to polish") lowercase while still
   * fixing a confident proper noun ("paris" -> "Paris"). renderCased (used only for
   * display) keeps the plain argmax; the gate lives here so display and rewrite differ.
   */
  casedConfident(word: string, context: string[], margin = 2.5): string {
    const lower = word.toLowerCase();
    const id = this.wid.get(lower);
    if (id === undefined) return word;
    this.prepare(context);
    this.caseLogitsInto(this.cur!.h[this.layers - 1], id, this.caseScratch);
    const lg = this.caseScratch;
    let bestC = CASE_LOWER;
    for (let c = 0; c < N_CASE; c++) if (lg[c] > lg[bestC]) bestC = c;
    if (bestC === CASE_LOWER) return lower;
    if (lg[bestC] - lg[CASE_LOWER] < margin) return lower; // not confident: keep the typed case
    return this.renderFromLogits(lower, lg);
  }

  /** Case logits for a phrase seed against a saved state (phrasesFor's base). */
  private seedCase(baseH: Float32Array[], seed: string): Float32Array {
    this.caseLogitsInto(baseH[this.layers - 1], this.wid.get(seed) ?? 0, this.caseScratch);
    return this.caseScratch;
  }

  /** Set state to represent `context` exactly, from a clean state. Words are
   *  lowercased here so callers can pass editor tokens directly. */
  setContext(context: string[]): void {
    this.reset();
    for (let i = 0; i < context.length; i++) this.stepWord(context[i].toLowerCase());
  }

  /** Dequantise embedding row `wordId` into the x scratch buffer. */
  private dequantRow(wordId: number): void {
    const dim = this.dim, base = wordId * dim, sc = this.scaleW[wordId];
    const x = this.xView, q = this.embQ;
    for (let a = 0; a < dim; a++) x[a] = q[base + a] * sc;
  }

  /** Apply the gate nonlinearity for layer L: updates c in place and writes the
   *  pre-projection hidden (o * tanh(c)) into hRaw. Gate order i(0), f(1), g(2),
   *  o(3), each `hid` wide. Shared by both execution paths. */
  private gate(L: number): void {
    const hid = this.hid, g = this.gatesView, cv = this.c[L], hr = this.hRawView;
    for (let j = 0; j < hid; j++) {
      const ii = sigmoid(g[j]), ff = sigmoid(g[hid + j]);
      const gc = Math.tanh(g[2 * hid + j]), oo = sigmoid(g[3 * hid + j]);
      const cc = ff * cv[j] + ii * gc;
      cv[j] = cc;
      hr[j] = oo * Math.tanh(cc);
    }
  }

  /** Per-vector int8 scale: max|v| / 127. The wasm path needs only this (the kernel
   *  does the rounding in `quantise_vec`); the scalar path also needs the values. */
  private scaleOf(v: Float32Array, n: number): number {
    let mx = 1e-9;
    for (let a = 0; a < n; a++) { const x = Math.abs(v[a]); if (x > mx) mx = x; }
    return mx / 127;
  }

  /** Quantise `v[0..n)` into `dst` and return its scale - the scalar-path twin of
   *  the kernel's `quantise_vec`, so both paths do the SAME arithmetic. */
  private quantVec(v: Float32Array, dst: Int8Array, n: number): number {
    const s = this.scaleOf(v, n), inv = 1 / s;
    for (let a = 0; a < n; a++) {
      const q = Math.round(v[a] * inv);
      dst[a] = q > 127 ? 127 : q < -127 ? -127 : q;
    }
    return s;
  }

  private step(wordId: number): void {
    if (this.k) { this.stepWasm(wordId); return; }
    const dim = this.dim, hid = this.hid;
    this.dequantRow(wordId);
    let x: Float32Array = this.xView;
    for (let L = 0; L < this.layers; L++) {
      const wih = this.wih[L], wihS = this.wihS[L];
      const whh = this.whh[L], whhS = this.whhS[L], bias = this.bias[L];
      const hPrev = this.h[L], g = this.gatesView;
      const sx = this.quantVec(x, this.xQ, dim);
      const sh = this.quantVec(hPrev, this.hQin, dim);
      const xq = this.xQ, hq = this.hQin;
      for (let j = 0; j < 4 * hid; j++) {
        const r = j * dim; // both gate matrices are [4*hid, dim]
        let si = 0, sr = 0; // int accumulators, one per matrix (each has its own scale)
        for (let a = 0; a < dim; a++) si += wih[r + a] * xq[a];
        for (let a = 0; a < dim; a++) sr += whh[r + a] * hq[a];
        g[j] = bias[j] + si * wihS[j] * sx + sr * whhS[j] * sh;
      }
      this.gate(L);
      const hRaw = this.hRawView, hv = this.h[L];
      if (this.whr) {
        // h = Whr . hRaw : project the cell-width hidden back down to dim.
        const w = this.whr[L], ws = this.whrS![L];
        const sq = this.quantVec(hRaw, this.hRawQ, hid), rq = this.hRawQ;
        for (let p = 0; p < dim; p++) {
          let s = 0;
          const r = p * hid;
          for (let a = 0; a < hid; a++) s += w[r + a] * rq[a];
          hv[p] = s * ws[p] * sq;
        }
      } else {
        hv.set(hRaw); // hid === dim: no projection
      }
      x = hv;
    }
  }

  private stepWasm(wordId: number): void {
    const k = this.k!, P = this.ptr, hid = this.hid, dim = this.dim;
    this.dequantRow(wordId);
    let xPtr = P.x, xView: Float32Array = this.xView;
    const rows = 4 * hid;
    const g = this.gatesView, g2 = this.gates2View;
    for (let L = 0; L < this.layers; L++) {
      // gates = bias + wih . x, then += whh . hPrev  (both matrices are dim-wide).
      // project_i8 WRITES its output, so the whh term goes to a scratch and is added
      // on here: 4*hid adds against 2 x (4*hid x dim) MACs, i.e. noise.
      const sx = this.scaleOf(xView, dim);
      k.quantise_vec(xPtr, P.xQ, dim, 1 / sx);
      k.project_i8(P.wih[L], P.wihS[L], P.bias[L], P.xQ, P.gates, rows, dim, sx);
      const sh = this.scaleOf(this.h[L], dim);
      k.quantise_vec(P.h[L], P.hQin, dim, 1 / sh);
      k.project_i8(P.whh[L], P.whhS[L], P.zero, P.hQin, P.gates2, rows, dim, sh);
      for (let j = 0; j < rows; j++) g[j] += g2[j];
      // Nonlinearity stays in JS: wasm has no exp/tanh, and it's only 4*hid elems
      // against the hid*dim MACs above.
      this.gate(L);
      if (this.whr) {
        // h = Whr . hRaw, with a zero bias so project_i8 is a plain matvec.
        const sr = this.scaleOf(this.hRawView, hid);
        k.quantise_vec(P.hRaw, P.hRawQ, hid, 1 / sr);
        k.project_i8(P.whr[L], P.whrS[L], P.zero, P.hRawQ, P.h[L], dim, hid, sr);
      } else {
        this.h[L].set(this.hRawView);
      }
      xPtr = P.h[L];
      xView = this.h[L];
    }
  }

  /** Project the CURRENT state onto the top-`limit` (frequency-sorted) rows into
   *  `out`, returning the log-partition over that shortlist (≈ the true
   *  normaliser, since the tail carries almost no mass). */
  private fillLogits(out: Float32Array, limit: number): number {
    const dim = this.dim;
    const h = this.h[this.layers - 1];
    if (this.k) {
      const P = this.ptr, hPtr = P.h[this.layers - 1];
      let mx = 1e-9;
      for (let a = 0; a < dim; a++) { const v = Math.abs(h[a]); if (v > mx) mx = v; }
      const scaleH = mx / 127;
      this.k.quantise_vec(hPtr, P.hQ, dim, 1 / scaleH);
      this.k.project_i8(P.embQ, P.scaleW, P.bOut, P.hQ, P.out, limit, dim, scaleH);
      const ov = this.outView;
      let max = -Infinity;
      for (let w = 0; w < limit; w++) { const s = ov[w]; out[w] = s; if (s > max) max = s; }
      let sum = 0;
      for (let w = 0; w < limit; w++) sum += Math.exp(out[w] - max);
      return max + Math.log(sum);
    }
    const q = this.embQ, sw = this.scaleW, bOut = this.bOut;
    const d4 = dim & ~3;
    let max = -Infinity;
    for (let w = 0; w < limit; w++) {
      const base = w * dim;
      let s = 0, a = 0;
      for (; a < d4; a += 4)
        s += q[base + a] * h[a] + q[base + a + 1] * h[a + 1]
           + q[base + a + 2] * h[a + 2] + q[base + a + 3] * h[a + 3];
      for (; a < dim; a++) s += q[base + a] * h[a];
      const v = bOut[w] + s * sw[w];
      out[w] = v;
      if (v > max) max = v;
    }
    let sum = 0;
    for (let w = 0; w < limit; w++) sum += Math.exp(out[w] - max);
    return max + Math.log(sum);
  }

  /** Fill the stepping-mode scratch from the current state, over the tighter
   *  PHRASE shortlist (phrase extension calls this many times). */
  private computeLogits(): void {
    if (!this.stepLogits) this.stepLogits = new Float32Array(this.nTop);
    this.stepLogZ = this.fillLogits(this.stepLogits, this.nPhrase);
  }

  private topKFrom(logits: Float32Array, logZ: number, k: number, avoid: string | undefined, limit: number): Scored[] {
    const cap = k + 4;
    const topW = new Int32Array(cap).fill(-1);
    const topL = new Float32Array(cap).fill(-Infinity);
    let minTop = -Infinity, filled = 0;
    for (let w = 0; w < limit; w++) {
      const s = logits[w];
      if (filled < cap || s > minTop) {
        let p = filled < cap ? filled++ : cap - 1;
        while (p > 0 && topL[p - 1] < s) { topL[p] = topL[p - 1]; topW[p] = topW[p - 1]; p--; }
        topL[p] = s; topW[p] = w;
        minTop = topL[filled < cap ? filled - 1 : cap - 1];
      }
    }
    const out: Scored[] = [];
    for (let i = 0; i < filled && out.length < k; i++) {
      const word = this.vocab[topW[i]];
      if (word === "<unk>" || word === avoid) continue;
      out.push({ word, logProb: topL[i] - logZ });
    }
    return out;
  }

  // --- LanguageModel interface (context-cached: one forward pass per context) ---

  /** The LOWERCASE token stream the model actually conditions on. Memoised by array
   *  identity: the predictor passes one array object per prediction and asks about
   *  many candidates, so this runs once per prediction rather than once per word. */
  private lcContext(context: string[]): string[] {
    if (context === this.lcIn) return this.lcOut;
    const out = new Array<string>(context.length);
    for (let i = 0; i < context.length; i++) out[i] = context[i].toLowerCase();
    this.lcIn = context;
    this.lcOut = out;
    return out;
  }

  /**
   * Keys identifying every PREFIX of `ctx`: keys[i] identifies ctx[0..i), so keys[0]
   * is the empty context and keys[n] the whole thing.
   *
   * This replaces joining the context into a string. A join is O(total characters)
   * and allocates the whole prefix on every keystroke; this is one pass of integer
   * work that yields ALL prefixes at once - which is what makes both the parent
   * lookup (keys[n-1]) and the waypoint search free.
   *
   * TWO independent 32-bit polynomial hashes combined into the key. One 32-bit hash
   * over thousands of prefixes would collide by birthday (~50% at 77k entries), and a
   * collision here means silently conditioning on the WRONG document text. Two gives
   * ~64 bits, where a collision is not a practical concern. Math.imul keeps the
   * multiply in int32 instead of drifting into f64.
   */
  private prefixKeys(ctx: string[]): string[] {
    if (ctx === this.keyIn) return this.keyOut;
    const keys = new Array<string>(ctx.length + 1);
    let a = 0x811c9dc5 | 0, b = 0x01000193 | 0;
    keys[0] = "0:0";
    for (let i = 0; i < ctx.length; i++) {
      const w = ctx[i];
      for (let j = 0; j < w.length; j++) {
        const c = w.charCodeAt(j);
        a = (Math.imul(a ^ c, 16777619) | 0);
        b = (Math.imul(b + c, 2654435761) | 0) ^ (b >>> 13);
      }
      // Mix a separator so ["ab","c"] and ["a","bc"] cannot share a key.
      a = (Math.imul(a ^ 32, 16777619) | 0);
      b = (Math.imul(b + 32, 2654435761) | 0) ^ (b >>> 13);
      keys[i + 1] = `${a >>> 0}:${b >>> 0}`;
    }
    this.keyIn = ctx;
    this.keyOut = keys;
    return keys;
  }

  /** Snapshot the live state as a waypoint (LRU-bounded). */
  private saveWaypoint(key: string): void {
    if (this.waypoints.has(key)) return;
    this.waypoints.set(key, {
      h: this.h.map((a) => a.slice()),
      c: this.c.map((a) => a.slice()),
    });
    if (this.waypoints.size > WAYPOINT_CAP) {
      const oldest = this.waypoints.keys().next().value;
      if (oldest !== undefined) this.waypoints.delete(oldest);
    }
  }

  /** Ensure `this.cur` holds the logits for `context`, using the LRU cache. */
  private prepare(context: string[]): void {
    const ctx = this.lcContext(context);
    const keys = this.prefixKeys(ctx);
    const n = ctx.length;
    const key = keys[n];
    const hit = this.ctxCache.get(key);
    if (hit) {
      this.ctxCache.delete(key); // refresh LRU recency
      this.ctxCache.set(key, hit);
      this.cur = hit;
      return;
    }
    // HOT PATH - you typed one more word, so the previous context is this one's
    // parent and we still hold its state: extend it by ONE step. O(1) at ANY context
    // length, because context.ts no longer slices and the context only ever grows.
    let from = -1;
    if (n > 0 && this.ctxCache.has(keys[n - 1])) {
      const parent = this.ctxCache.get(keys[n - 1])!;
      this.loadState(parent.h, parent.c);
      from = n - 1;
    } else {
      // A jump, an edit, or a freshly opened document. Restore the deepest waypoint
      // that still matches the text and walk forward from there - at most
      // WAYPOINT_EVERY steps, except on first touch of a document, where this walk
      // is what BUILDS the waypoints.
      for (let L = n - (n % WAYPOINT_EVERY); L >= 0; L -= WAYPOINT_EVERY) {
        const wp = this.waypoints.get(keys[L]);
        if (wp) {
          this.loadState(wp.h, wp.c);
          from = L;
          break;
        }
      }
      if (from < 0) {
        this.reset(); // nothing usable: the true state of the empty prefix
        from = 0;
      }
    }
    for (let i = from; i < n; i++) {
      this.stepWord(ctx[i]);
      if ((i + 1) % WAYPOINT_EVERY === 0) this.saveWaypoint(keys[i + 1]);
    }
    const logits = new Float32Array(this.nTop);
    const logZ = this.fillLogits(logits, this.nTop);
    const entry: CtxEntry = {
      logits, logZ,
      h: this.h.map((a) => a.slice()),
      c: this.c.map((a) => a.slice()),
    };
    this.ctxCache.set(key, entry);
    if (this.ctxCache.size > LstmLanguageModel.CACHE_CAP) {
      const oldest = this.ctxCache.keys().next().value;
      if (oldest !== undefined) this.ctxCache.delete(oldest);
    }
    this.cur = entry;
  }

  /** Vocab id for `word`, or undefined. Lets callers that marginalise over several
   *  ids (the cased adapter) resolve once and score by id. */
  idOf(word: string): number | undefined {
    return this.wid.get(word);
  }

  /** log P(vocab[id] | context) - the id-keyed form of logProb(). */
  logProbId(id: number, context: string[]): number {
    this.prepare(context);
    return this.scoreId(id);
  }

  logProb(word: string, context: string[]): number {
    this.prepare(context);
    const id = this.wid.get(word);
    if (id === undefined) return OOV_LOGPROB;
    return this.scoreId(id);
  }

  /** Score an id against the CURRENT prepared context. */
  private scoreId(id: number): number {
    if (id < this.nTop) return this.cur!.logits[id] - this.cur!.logZ;
    // Rare word outside the shortlist: score it EXACTLY with a single-row dot
    // product against the cached hidden state (cheap: `dim` mults). logZ is the
    // shortlist normaliser (≈ exact), so this stays a well-calibrated log-prob.
    const logZ = this.cur!.logZ, h = this.cur!.h[this.layers - 1];
    const dim = this.dim, q = this.embQ, base = id * dim;
    let s = 0;
    for (let a = 0; a < dim; a++) s += q[base + a] * h[a];
    return this.bOut[id] + s * this.scaleW[id] - logZ;
  }

  predict(context: string[], k: number): Scored[] {
    this.prepare(context);
    return this.topKFrom(this.cur!.logits, this.cur!.logZ, k, undefined, this.nTop);
  }

  *vocabulary(): IterableIterator<string> {
    for (const w of this.vocab) if (w !== "<unk>") yield w;
  }
  size() {
    return { vocab: this.V, bigrams: 0, trigrams: 0 };
  }

  /**
   * Beam-decode multi-word continuations of `seed` in `context`, each returned with
   * `extLogProb` = the joint LOG-probability of the EXTENSION (every word AFTER the seed),
   * from the model's real per-step softmax.
   *
   * This replaces the old greedy "dominance P1/(P1+P2)" rule, which was measured to read a
   * FLAT distribution (top word only ~5% probable, but still 0.57 of the top-two) as
   * "confident" and run on into long, unlikely phrases. Here the caller ranks the bare seed
   * and these candidates by EXPECTED KEYSTROKES SAVED - P(candidate)·(chars it saves) - so
   * the phrase length is the utility argmax (no threshold), a phrase only outranks the short
   * option when it is genuinely predictable, and the beam yields diverse same-first-word
   * branches. Words are cased as the phrase grows (the case of "york" needs "new" stepped).
   * Returns candidates of length ≥ 2 (the bare seed is the caller's responsibility).
   */
  phraseCandidates(
    context: string[],
    rawSeed: string,
    maxWords: number,
    beamWidth: number,
  ): { words: string[]; extLogProb: number }[] {
    if (maxWords < 2 || beamWidth < 1) return [];
    this.prepare(context);
    const baseH = this.cur!.h, baseC = this.cur!.c;
    const seed = rawSeed.toLowerCase();
    const pk = `${this.prefixKeys(this.lcContext(context))[context.length]} ${seed} ${maxWords} ${beamWidth}`;
    const memo = this.phraseCache.get(pk);
    if (memo) return memo;
    const seedSurface = this.renderFromLogits(seed, this.seedCase(baseH, seed));
    // Base state = context + seed.
    this.loadState(baseH, baseC);
    this.stepWord(seed);
    const snap = (): [Float32Array[], Float32Array[]] => [
      this.h.map((a) => a.slice()),
      this.c.map((a) => a.slice()),
    ];
    interface Path { surfaces: string[]; logp: number; h: Float32Array[]; c: Float32Array[] }
    const [h0, c0] = snap();
    let beam: Path[] = [{ surfaces: [], logp: 0, h: h0, c: c0 }];
    const out: { words: string[]; extLogProb: number }[] = [];
    interface Ext { path: Path; word: string; surface: string; logp: number }
    const lastDepth = maxWords - 2; // beyond this we never extend, so never need the stepped state
    for (let depth = 0; depth < maxWords - 1 && beam.length > 0; depth++) {
      // 1. GATHER every candidate extension of every surviving path, with its cased surface and
      //    CUMULATIVE log-prob - but do NOT advance the recurrent net yet. Scoring a candidate
      //    (softmax + case head) is cheap; the recurrent step is not.
      const exts: Ext[] = [];
      for (const path of beam) {
        this.loadState(path.h, path.c);
        this.computeLogits(); // next-word distribution from THIS path's state (state unchanged)
        const cand = this.topKFrom(this.stepLogits!, this.stepLogZ, beamWidth, undefined, this.nPhrase);
        // If the model's MOST likely continuation ends the sentence, the phrase stops here -
        // a suggestion must never run past a sentence boundary. (A lower-ranked "." is just
        // skipped below; only the argmax terminates the path.)
        if (cand.length > 0 && SENTENCE_END.has(cand[0].word)) continue;
        for (const cw of cand) {
          if (SENTENCE_END.has(cw.word) || PUNCT.has(cw.word)) continue; // a phrase never crosses a boundary
          // Case each candidate against the path state (still loaded from computeLogits above).
          this.caseLogitsInto(this.h[this.layers - 1], this.wid.get(cw.word) ?? 0, this.caseScratch);
          exts.push({ path, word: cw.word, surface: this.renderFromLogits(cw.word, this.caseScratch), logp: path.logp + cw.logProb });
        }
      }
      // 2. Prune to the beamWidth best extensions (exact BEFORE stepping - survivors are ranked
      //    purely by cumulative log-prob) and EMIT them as phrases.
      exts.sort((a, b) => b.logp - a.logp);
      const survivors = exts.slice(0, beamWidth);
      for (const e of survivors) out.push({ words: [seedSurface, ...e.path.surfaces, e.surface], extLogProb: e.logp });
      // 3. Advance the recurrent state for the survivors ONLY IF a further depth will extend them.
      //    Skipping the step at the last depth (its state is never read) and stepping only the
      //    survivors - not all beamWidth^2 expansions - are the two savings; the phrases emitted
      //    are byte-for-byte identical to the previous step-everything-then-prune version.
      if (depth >= lastDepth) break;
      beam = [];
      for (const e of survivors) {
        this.loadState(e.path.h, e.path.c);
        this.stepWord(e.word);
        const [hh, cc] = snap();
        beam.push({ surfaces: [...e.path.surfaces, e.surface], logp: e.logp, h: hh, c: cc });
      }
    }
    this.phraseCache.set(pk, out);
    if (this.phraseCache.size > LstmLanguageModel.PHRASE_CACHE_CAP) {
      const oldest = this.phraseCache.keys().next().value;
      if (oldest !== undefined) this.phraseCache.delete(oldest);
    }
    return out;
  }
}
