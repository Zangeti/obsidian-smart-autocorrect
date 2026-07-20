/**
 * The fmt-4 loader, the case head (incl. the low-rank bilinear term), and the
 * waypoint/incremental state machinery.
 *
 * Driven by a hand-built fmt-4 binary rather than a trained model: the behaviour
 * under test is the LOADER and the state rules, and a synthetic file lets us assert
 * exact case decisions by setting the case logits directly. (Whether a TRAINED model
 * gets "iPhone" right is a training question, measured separately on the GPU box.)
 *
 * The layout here must mirror export_bin() in build_model/train_word_lstm_cased.py.
 * If this file needs editing to make a new .bin load, the two have drifted.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { LstmLanguageModel } from "../src/predictive/engine/lstm/model.ts";

const CASE_LOWER = 0, CASE_TITLE = 1, CASE_UPPER = 2, CASE_OTHER = 3;
const RANK = 2; // fixture bilinear rank; the trainer default is 8

interface Spec {
  vocab: string[];
  surface?: Record<string, string>;
  /** word -> case logits [lower, title, upper, other] */
  caseBias?: Record<string, number[]>;
  /** [N_CASE*RANK*dim] each; default all-zero so the bilinear term is inert and the
   *  caseBias-driven assertions stay exact. */
  caseA?: Float32Array;
  caseB?: Float32Array;
}

/** Build a minimal but STRUCTURALLY EXACT fmt-4 binary. dim === hid, so no
 *  projection block – matching the trainer, which omits weight_hr when hid == dim. */
function buildBin(spec: Spec): ArrayBuffer {
  const { vocab } = spec;
  const V = vocab.length, dim = 4, hid = 4, layers = 1;
  const enc = new TextEncoder();
  const words = vocab.map((w) => enc.encode(w));
  const surf = Object.entries(spec.surface ?? {}).map(
    ([lo, s]) => [vocab.indexOf(lo), enc.encode(s)] as const,
  );
  const nBil = 4 * RANK * dim; // per bilinear tensor

  let n = 4 + 17;
  for (const b of words) n += 2 + b.length;
  n += 4;
  for (const [, b] of surf) n += 4 + 2 + b.length;
  n += V * dim + V * 4; // embQ + scales
  n += (4 * hid * dim + 4 * hid * dim + 4 * hid + 4 * hid) * 4; // one layer
  n += V * 4; // bOut
  n += (4 * dim + 4 + 4 * dim + V * 4) * 4; // case head additive terms
  n += 4 + nBil * 2 * 4; // rank + caseA + caseB

  const buf = new ArrayBuffer(n);
  const dv = new DataView(buf);
  let o = 0;
  const u32 = (v: number) => { dv.setUint32(o, v, true); o += 4; };
  const u16 = (v: number) => { dv.setUint16(o, v, true); o += 2; };
  const u8 = (v: number) => { dv.setUint8(o, v); o += 1; };
  const f32 = (v: number) => { dv.setFloat32(o, v, true); o += 4; };
  const i8 = (v: number) => { dv.setInt8(o, v); o += 1; };

  u32(0x4c53544d);
  u32(V); u32(dim); u32(hid); u32(layers); u8(4);
  for (const b of words) { u16(b.length); new Uint8Array(buf, o, b.length).set(b); o += b.length; }
  u32(surf.length);
  for (const [id, b] of surf) { u32(id); u16(b.length); new Uint8Array(buf, o, b.length).set(b); o += b.length; }
  // The weights must make the state genuinely PATH-DEPENDENT. Zeroed gates and a
  // uniform embedding would drive every state to zero, and the equivalence tests
  // below would then pass for free while proving nothing. So: a distinct embedding
  // row per word, and small varied gate weights (small enough not to saturate the
  // sigmoids into a constant).
  const rnd = (() => { let s = 12345; return () => (s = Math.imul(s, 1103515245) + 12345 | 0) / 2 ** 31; })();
  for (let i = 0; i < V; i++)
    for (let a = 0; a < dim; a++) i8(((i * 37 + a * 11) % 61) - 30); // per-word rows
  for (let i = 0; i < V; i++) f32(0.05); // per-row scale
  for (let i = 0; i < 4 * hid * dim * 2 + 4 * hid * 2; i++) f32(rnd() * 0.8 - 0.4);
  for (let i = 0; i < V; i++) f32(rnd() * 0.2); // bOut
  for (let i = 0; i < 4 * dim; i++) f32(0); // caseCtxW
  for (let i = 0; i < 4; i++) f32(0); // caseCtxB
  for (let i = 0; i < 4 * dim; i++) f32(0); // caseWordW
  // caseBias [V, 4] – with everything else zeroed this alone decides the class,
  // which is exactly how we pin an expected rendering.
  for (const w of vocab) {
    const row = spec.caseBias?.[w] ?? [1, 0, 0, 0];
    for (let c = 0; c < 4; c++) f32(row[c]);
  }
  // Bilinear term (fmt 4): rank, then caseA and caseB, each [N_CASE, RANK, dim].
  u32(RANK);
  const a = spec.caseA ?? new Float32Array(nBil);
  const bb = spec.caseB ?? new Float32Array(nBil);
  for (let i = 0; i < nBil; i++) f32(a[i]);
  for (let i = 0; i < nBil; i++) f32(bb[i]);
  assert.equal(o, n, "test fixture wrote a different length than it reserved");
  return buf;
}

test("fmt-4 loads: lowercase vocab + irregular table, and rejects older formats", () => {
  const m = LstmLanguageModel.fromBuffer(buildBin({ vocab: ["<unk>", "the", "nasa"] }));
  assert.equal(m.vocabSize, 3);
  assert.equal(m.hasWord("the"), true);

  // A fmt-2 (cased-vocab) file must be refused OUTRIGHT, not silently misread: its
  // vocab is cased and it has no case head, so every byte after the vocab differs.
  const bad = buildBin({ vocab: ["<unk>", "the"] });
  new DataView(bad).setUint8(20, 2); // the fmt byte
  assert.throws(() => LstmLanguageModel.fromBuffer(bad), /factored int8|fmt 4/i);
});

test("the loader consumes the file EXACTLY (a layout drift is caught, not tolerated)", () => {
  const good = buildBin({ vocab: ["<unk>", "the"] });
  // One trailing byte = the exporter and loader disagree. This assertion is the only
  // thing standing between a format change and a silently mis-parsed model.
  const padded = new ArrayBuffer(good.byteLength + 1);
  new Uint8Array(padded).set(new Uint8Array(good));
  assert.throws(() => LstmLanguageModel.fromBuffer(padded), /trailing bytes/i);
});

test("renderCased applies the winning case class", () => {
  const m = LstmLanguageModel.fromBuffer(buildBin({
    vocab: ["<unk>", "the", "paris", "nasa"],
    caseBias: {
      the: [5, 0, 0, 0],   // LOWER
      paris: [0, 5, 0, 0], // TITLE
      nasa: [0, 0, 5, 0],  // UPPER – the case that must never render "Nasa"
    },
  }));
  assert.equal(m.renderCased("the", ["a"]), "the");
  assert.equal(m.renderCased("paris", ["a"]), "Paris");
  assert.equal(m.renderCased("nasa", ["a"]), "NASA");
  // NASA is rebuilt BY RULE from the UPPER tag – it never touches the table.
  assert.equal(m.renderCased("NASA", ["a"]), "NASA"); // input casing is irrelevant
});

test("phraseCandidates beam-extends with real joint log-probs (length>=2, decreasing prob)", () => {
  const m = LstmLanguageModel.fromBuffer(buildBin({ vocab: ["<unk>", "the", "quick", "brown", "fox"] }));
  assert.deepEqual(m.phraseCandidates(["a"], "the", 1, 3), []); // maxWords<2 => no phrases
  const cands = m.phraseCandidates(["a"], "the", 4, 3);
  for (const c of cands) {
    assert.ok(c.words.length >= 2 && c.words.length <= 4, "phrase length within [2, maxWords]");
    assert.equal(c.words[0].toLowerCase(), "the", "phrase starts with the seed");
    assert.ok(c.extLogProb <= 0, "extLogProb is a sum of log-probabilities (<= 0)");
  }
  // The beam explores >1 continuation, so distinct same-first-word branches are possible.
  const texts = new Set(cands.map((c) => c.words.join(" ")));
  assert.ok(texts.size === cands.length, "candidates are distinct");
});

test("caseVariants offers BOTH cases of a homograph, one case of a confident word", () => {
  const m = LstmLanguageModel.fromBuffer(buildBin({
    vocab: ["<unk>", "the", "paris", "polish"],
    caseBias: {
      the: [5, -5, -5, -5],       // confidently LOWER
      paris: [-5, 5, -5, -5],     // confidently TITLE
      polish: [1.0, 1.2, -5, -5], // homograph: LOWER and TITLE within margin
    },
  }));
  // Confident words collapse to a single surface...
  assert.deepEqual(m.caseVariants("the", ["a"]), ["the"]);
  assert.deepEqual(m.caseVariants("paris", ["a"]), ["Paris"]);
  // ...but a homograph the model CAPITALISES comes back as BOTH so the popup shows each.
  assert.deepEqual(m.caseVariants("polish", ["a"]), ["Polish", "polish"]);
});

test("caseVariants is asymmetric: a LOWERCASE prediction never gains a capital alternate", () => {
  const m = LstmLanguageModel.fromBuffer(buildBin({
    vocab: ["<unk>", "may"],
    // LOWER wins but TITLE is close: adding "May" here would be the "the"->"The" bug.
    caseBias: { may: [1.2, 1.0, -5, -5] },
  }));
  assert.deepEqual(m.caseVariants("may", ["we"]), ["may"]);
});

test("casedConfident only re-cases when confident (leaves an ambiguous homograph typed)", () => {
  const m = LstmLanguageModel.fromBuffer(buildBin({
    vocab: ["<unk>", "the", "paris", "polish"],
    caseBias: {
      the: [5, -5, -5, -5],       // LOWER anyway
      paris: [-5, 5, -5, -5],     // confidently TITLE -> still fixed
      polish: [1.0, 1.2, -5, -5], // TITLE barely beats LOWER -> NOT confident
    },
  }));
  assert.equal(m.casedConfident("paris", ["a"]), "Paris"); // confident proper noun: fixed
  assert.equal(m.casedConfident("polish", ["a"]), "polish"); // ambiguous: user's case stands
  assert.equal(m.casedConfident("the", ["a"]), "the");
});

test("CASE_OTHER resolves through the table, and is MASKED when there is no entry", () => {
  const m = LstmLanguageModel.fromBuffer(buildBin({
    vocab: ["<unk>", "iphone", "widget"],
    surface: { iphone: "iPhone" },
    // Both words want OTHER; only one can have it.
    caseBias: { iphone: [0, 1, 0, 5], widget: [0, 1, 0, 5] },
  }));
  assert.equal(m.renderCased("iphone", ["a"]), "iPhone");
  // No stored spelling: OTHER would silently degrade to bare lowercase, so it is
  // masked and the best REGULAR class (TITLE here) wins instead.
  assert.equal(m.renderCased("widget", ["a"]), "Widget");
});

test("the bilinear term is loaded and applied to the case decision", () => {
  // The whole point of fmt 4: a context x word interaction the additive head cannot
  // express. With caseBias fixed, turning the bilinear weights on must change SOME
  // (word, context) rendering – otherwise the loader dropped the section or
  // caseLogitsInto ignores it. (Non-zero weights also exercise the trailing-byte
  // check: a mis-sized read would throw before we get here.)
  const vocab = ["<unk>", "the", "polish", "march", "brown"];
  const caseBias = { // all near-neutral so the bilinear can decide
    the: [0.1, 0, 0, 0], polish: [0.1, 0, 0, 0], march: [0.1, 0, 0, 0], brown: [0.1, 0, 0, 0],
  };
  const dim = 4, nBil = 4 * RANK * dim;
  // Large, class-varying weights so the interaction dominates the tiny bias.
  const A = new Float32Array(nBil), B = new Float32Array(nBil);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < RANK; r++)
      for (let a = 0; a < dim; a++) {
        const i = (c * RANK + r) * dim + a;
        A[i] = (c + 1) * 8 * (a % 2 ? 1 : -1);
        B[i] = 6 * ((c + a) % 2 ? 1 : -1);
      }

  const off = LstmLanguageModel.fromBuffer(buildBin({ vocab, caseBias }));
  const on = LstmLanguageModel.fromBuffer(buildBin({ vocab, caseBias, caseA: A, caseB: B }));

  const contexts = [["the"], ["the", "polish"], ["brown", "march"], ["polish", "brown", "the"]];
  const targets = ["the", "polish", "march", "brown"];
  let differed = false;
  for (const ctx of contexts)
    for (const w of targets)
      if (off.renderCased(w, ctx) !== on.renderCased(w, ctx)) differed = true;
  assert.ok(differed, "bilinear weights changed no decision – the term is not being applied");
});

test("a word outside the vocab is returned untouched", () => {
  const m = LstmLanguageModel.fromBuffer(buildBin({ vocab: ["<unk>", "the"] }));
  assert.equal(m.renderCased("Zzyzx", ["a"]), "Zzyzx");
});

test("the fixture is NOT degenerate (guards every equivalence test below)", () => {
  // If the state did not depend on the words, every assertion in this file about
  // states matching would hold trivially. Prove the fixture discriminates first.
  const vocab = ["<unk>", "a", "b", "c", "d"];
  const m = LstmLanguageModel.fromBuffer(buildBin({ vocab }));
  const p = (ws: string[]) => m.predict(ws, 3).map((s) => s.logProb.toFixed(6)).join(",");
  assert.notEqual(p(["a", "b", "c"]), p(["a", "b", "d"]), "word identity must move the state");
  assert.notEqual(p(["a", "b"]), p(["b", "a"]), "word ORDER must move the state");
  assert.notEqual(p(["a"]), p(["a", "a"]), "context length must move the state");
});

test("WAYPOINTS: an incrementally-extended state equals a from-scratch one", () => {
  // The core correctness property of the no-window design. If these ever diverge,
  // predictions silently depend on HOW you arrived at a cursor rather than on the
  // text – the exact class of bug the old sliding window hid by replaying every time.
  const vocab = ["<unk>", "a", "b", "c", "d"];
  const words = Array.from({ length: 200 }, (_, i) => vocab[1 + (i % 4)]);

  const inc = LstmLanguageModel.fromBuffer(buildBin({ vocab }));
  // Walk it one word at a time, exactly as typing does (each prepare extends the
  // previous state and crosses several waypoint boundaries on the way).
  let last: number[] = [];
  for (let i = 1; i <= words.length; i++) {
    last = inc.predict(words.slice(0, i), 3).map((s) => s.logProb);
  }

  // A cold instance that has never seen a prefix of this text: no cache, no
  // waypoints, so it must build the whole state from scratch.
  const cold = LstmLanguageModel.fromBuffer(buildBin({ vocab }));
  const fresh = cold.predict(words, 3).map((s) => s.logProb);
  assert.deepEqual(last, fresh);
});

test("WAYPOINTS: a cursor JUMP lands on the same state as a straight walk", () => {
  const vocab = ["<unk>", "a", "b", "c", "d"];
  const words = Array.from({ length: 150 }, (_, i) => vocab[1 + (i % 4)]);
  const m = LstmLanguageModel.fromBuffer(buildBin({ vocab }));

  // Type the whole thing (builds waypoints), then jump BACK to word 70 – served by
  // restoring a waypoint and stepping forward a few words.
  for (let i = 1; i <= words.length; i++) m.predict(words.slice(0, i), 3);
  const jumped = m.predict(words.slice(0, 70), 3).map((s) => s.logProb);

  const cold = LstmLanguageModel.fromBuffer(buildBin({ vocab }));
  const direct = cold.predict(words.slice(0, 70), 3).map((s) => s.logProb);
  assert.deepEqual(jumped, direct);
});

test("WAYPOINTS: EDITING earlier text invalidates the state built on it", () => {
  // Waypoints are keyed by a rolling hash of the prefix, so changing word 10 changes
  // every key after it and those waypoints simply stop matching. Nothing to
  // invalidate by hand – this asserts that property holds end to end.
  const vocab = ["<unk>", "a", "b", "c", "d"];
  const original = Array.from({ length: 100 }, () => "a");
  const edited = original.slice();
  edited[10] = "d"; // one word changes, 89 words before the cursor are identical

  const m = LstmLanguageModel.fromBuffer(buildBin({ vocab }));
  for (let i = 1; i <= original.length; i++) m.predict(original.slice(0, i), 3);
  const afterEdit = m.predict(edited, 3).map((s) => s.logProb);

  const cold = LstmLanguageModel.fromBuffer(buildBin({ vocab }));
  const truth = cold.predict(edited, 3).map((s) => s.logProb);
  assert.deepEqual(afterEdit, truth);
});

test("input casing does NOT change the state (lowercase in, cased out)", () => {
  // The root fix for the caps bug: a shouted context is byte-identical to a calm one
  // as far as the model is concerned, so it can never drag suggestions into caps.
  const vocab = ["<unk>", "the", "quick", "brown", "fox"];
  const calm = ["the", "quick", "brown", "fox"];
  const m = LstmLanguageModel.fromBuffer(buildBin({ vocab }));
  const a = m.predict(calm.map((w) => w.toUpperCase()), 3);
  const b = m.predict(calm, 3);
  assert.deepEqual(a, b);
});
