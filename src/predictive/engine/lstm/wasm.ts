/**
 * Loader for the SIMD kernel (see kernel.ts for why it's inlined as base64).
 *
 * Everything here is best-effort: if WebAssembly or SIMD is missing - an older
 * mobile webview, SIMD needs iOS 16.4+ - loadKernel() returns null and the model
 * silently keeps using its scalar JS path. The kernel is never required.
 */
import { KERNEL_B64 } from "./kernel.ts";

export interface Kernel {
  mem: WebAssembly.Memory;
  /** out[w] = bOut[w] + dot(emb[w*dim..], h) - f32 rows. */
  project_f32(emb: number, bOut: number, h: number, out: number, limit: number, dim: number): void;
  /** out[r] += dot(W[r*cols..], v) - the LSTM recurrent term. */
  matvec_acc_f32(W: number, v: number, out: number, rows: number, cols: number): void;
  /** dst[i] = round(src[i] * inv) as int8. */
  quantise_vec(src: number, dst: number, n: number, inv: number): void;
  /** out[w] = bOut[w] + scaleW[w]*scaleH*dot_int(embQ[w*dim..], hQ) - int8 rows. */
  project_i8(embQ: number, scaleW: number, bOut: number, hQ: number, out: number,
             limit: number, dim: number, scaleH: number): void;
}

const PAGE = 65536;

/** Decode to a plain ArrayBuffer. Returning the ArrayBuffer itself (rather than a
 *  Uint8Array view) keeps it a `BufferSource` under TS's generic typed arrays,
 *  where `Uint8Array<ArrayBufferLike>` is not assignable to one. */
function b64ToWasm(b64: string): ArrayBuffer {
  if (typeof atob === "function") {
    const s = atob(b64);
    const ab = new ArrayBuffer(s.length);
    const a = new Uint8Array(ab);
    for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i);
    return ab;
  }
  // node (tests / build scripts)
  const buf = Buffer.from(b64, "base64");
  const ab = new ArrayBuffer(buf.byteLength);
  new Uint8Array(ab).set(buf);
  return ab;
}

/**
 * Compile the kernel and grow its memory to `bytes`. Returns null if unsupported
 * or if the memory can't be reserved (a big model on a memory-constrained device),
 * in which case the caller must fall back to scalar JS.
 */
export function loadKernel(bytes: number): Kernel | null {
  try {
    if (typeof WebAssembly === "undefined") return null;
    const wasm = b64ToWasm(KERNEL_B64);
    // The module body uses v128 ops, so validate() fails outright when the engine
    // lacks SIMD - this doubles as our feature detection.
    if (!WebAssembly.validate(wasm)) return null;
    const inst = new WebAssembly.Instance(new WebAssembly.Module(wasm), {});
    const k = inst.exports as unknown as Kernel;
    const want = Math.ceil(bytes / PAGE);
    const have = k.mem.buffer.byteLength / PAGE;
    if (want > have) k.mem.grow(want - have);
    if (k.mem.buffer.byteLength < bytes) return null;
    return k;
  } catch {
    return null;
  }
}

/**
 * Bump allocator over the kernel's linear memory. Allocate EVERYTHING up front:
 * mem.grow() detaches the ArrayBuffer and invalidates every existing view, so no
 * allocation may happen once the model has handed out typed-array views.
 */
export class Bump {
  private p = 0;
  private limit: number;
  // NB: no parameter properties - the engine is run through node's strip-only
  // TypeScript mode in tests, which rejects `constructor(private x)`.
  constructor(limit: number) {
    this.limit = limit;
  }
  /** 16-byte aligned so v128 loads stay aligned. */
  alloc(bytes: number): number {
    const o = (this.p + 15) & ~15;
    this.p = o + bytes;
    if (this.p > this.limit) throw new Error("wasm heap exhausted");
    return o;
  }
  get used(): number {
    return this.p;
  }
}
