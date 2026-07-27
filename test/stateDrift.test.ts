/**
 * EXONERATES the LSTM for the "odd suggestion after a while, fixed by reload" drift.
 *
 * Mirrors the engine's real per-keystroke call pattern (predict + per-candidate scoring +
 * empty-context unigram feature + case variants + phrase beam decoding) against the
 * incremental LSTM under heavy ctx-cache pressure and cursor jumps, and asserts the
 * incremental logits are BIT-IDENTICAL to a cold from-scratch model at every step. Because
 * the stored states are full-precision f32, incremental == cold in lockstep - so reloading
 * the plugin would compute the SAME logits at a position, which means the LSTM cannot be what
 * "reload fixes". That pointed at the within-document CacheLanguageModel instead (it was
 * seeded once and only grew); see the reseed-on-edit fix in PredictiveFeature.processDirty.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { LstmLanguageModel } from "../src/predictive/engine/lstm/model.ts";

const RANK = 2;

function buildBin(vocab: string[]): ArrayBuffer {
  const V = vocab.length, dim = 4, hid = 4, layers = 1;
  const enc = new TextEncoder();
  const words = vocab.map((w) => enc.encode(w));
  const nBil = 4 * RANK * dim;
  let n = 4 + 17;
  for (const b of words) n += 2 + b.length;
  n += 4; // surface count (0)
  n += V * dim + V * 4;
  n += (4 * hid * dim + 4 * hid * dim + 4 * hid + 4 * hid) * 4;
  n += V * 4;
  n += (4 * dim + 4 + 4 * dim + V * 4) * 4;
  n += 4 + nBil * 2 * 4;
  const buf = new ArrayBuffer(n);
  const dv = new DataView(buf);
  let o = 0;
  const u32 = (v: number) => { dv.setUint32(o, v, true); o += 4; };
  const u16 = (v: number) => { dv.setUint16(o, v, true); o += 2; };
  const u8 = (v: number) => { dv.setUint8(o, v); o += 1; };
  const f32 = (v: number) => { dv.setFloat32(o, v, true); o += 4; };
  const i8 = (v: number) => { dv.setInt8(o, v); o += 1; };
  u32(0x4c53544d); u32(V); u32(dim); u32(hid); u32(layers); u8(4);
  for (const b of words) { u16(b.length); new Uint8Array(buf, o, b.length).set(b); o += b.length; }
  u32(0);
  const rnd = (() => { let s = 12345; return () => (s = Math.imul(s, 1103515245) + 12345 | 0) / 2 ** 31; })();
  for (let i = 0; i < V; i++) for (let a = 0; a < dim; a++) i8(((i * 37 + a * 11) % 61) - 30);
  for (let i = 0; i < V; i++) f32(0.05);
  for (let i = 0; i < 4 * hid * dim * 2 + 4 * hid * 2; i++) f32(rnd() * 0.8 - 0.4);
  for (let i = 0; i < V; i++) f32(rnd() * 0.2);
  for (let i = 0; i < 4 * dim; i++) f32(0);
  for (let i = 0; i < 4; i++) f32(0);
  for (let i = 0; i < 4 * dim; i++) f32(0);
  for (const _ of vocab) { f32(1); f32(0); f32(0); f32(0); }
  u32(RANK);
  for (let i = 0; i < nBil * 2; i++) f32(0);
  assert.equal(o, n);
  return buf;
}

const VOCAB = ["<unk>", "the", "quick", "brown", "fox", ".", "a", "b", "c", "d", "cat", "dog"];
const logits = (m: LstmLanguageModel, ctx: string[]) => m.predict(ctx, 5).map((s) => `${s.word}:${s.logProb.toFixed(6)}`).join(",");

test("DRIFT: incremental logits stay correct under real interleaved engine usage", () => {
  const m = LstmLanguageModel.fromBuffer(buildBin(VOCAB));
  const cold = LstmLanguageModel.fromBuffer(buildBin(VOCAB));

  // A long "document" with sentence boundaries, so the LSTM crosses many waypoints.
  const words: string[] = [];
  for (let i = 0; i < 400; i++) words.push(VOCAB[1 + (i % (VOCAB.length - 1))]);

  for (let i = 1; i <= words.length; i++) {
    const ctx = words.slice(0, i);
    const seed = VOCAB[1 + (i % (VOCAB.length - 1))];

    // Real per-keystroke pattern: next-word predict, per-candidate scoring incl. the
    // empty-context unigram feature, case variants, and phrase beam decoding.
    m.predict(ctx, 5);
    for (const w of ["the", "fox", "cat", "dog"]) { m.logProb(w, ctx); m.logProb(w, []); }
    m.caseVariants(seed, ctx);
    m.phraseCandidates(ctx, seed, 4, 3);

    // Blow the 8-slot ctx cache with many distinct short contexts, so extending to the
    // next word can no longer use the just-cached parent and must walk from a waypoint.
    for (let j = 1; j <= 12; j++) m.logProb("the", words.slice(Math.max(0, i - j), i));

    const got = logits(m, ctx);
    const want = logits(cold, ctx);
    assert.equal(got, want, `step ${i}: incremental state diverged from cold`);
  }
});

test("DRIFT: cursor jumps backward and forward stay correct after a long session", () => {
  const m = LstmLanguageModel.fromBuffer(buildBin(VOCAB));
  const cold = LstmLanguageModel.fromBuffer(buildBin(VOCAB));
  const words: string[] = [];
  for (let i = 0; i < 500; i++) words.push(VOCAB[1 + (i % (VOCAB.length - 1))]);

  // Type the whole thing with phrase decoding interleaved (builds & pressures waypoints).
  for (let i = 1; i <= words.length; i++) {
    m.predict(words.slice(0, i), 5);
    m.phraseCandidates(words.slice(0, i), words[i - 1], 4, 3);
  }
  // Now jump around, as clicks/edits do.
  for (const pos of [500, 37, 480, 5, 300, 128, 129, 64, 450, 200, 201, 1, 499]) {
    const ctx = words.slice(0, pos);
    assert.equal(logits(m, ctx), logits(cold, ctx), `jump to ${pos} diverged`);
  }
});
