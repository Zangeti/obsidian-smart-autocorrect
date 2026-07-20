import { test } from "node:test";
import assert from "node:assert/strict";

import {
  splitSentences,
  tokenizeWords,
  isSentenceTerminator,
} from "../src/predictive/engine/text/tokenize.ts";
import { buildAbbreviationSet } from "../src/predictive/engine/text/abbreviations.ts";
import {
  applyAutoCapitalization,
  defaultSentenceCaseConfig,
  shouldCapitalizeNext,
} from "../src/predictive/engine/text/sentenceCase.ts";
import { buildModelFromText } from "../src/predictive/engine/ngram/build.ts";
import { channelCost } from "../src/predictive/engine/channel/editDistance.ts";
import { DEFAULT_CHANNEL } from "../src/predictive/engine/channel/keyboard.ts";
import { FuzzyIndex, predict } from "../src/predictive/engine/predict/predictor.ts";
import { contextualEvidence, decideCorrection, decideRespace, informationGain, matchCase, RevertBuffer } from "../src/predictive/engine/autocorrect/autocorrect.ts";
import { Engine } from "../src/predictive/engine/index.ts";
import { WordOracle } from "../src/predictive/engine/text/wordOracle.ts";
import { suggestSplit } from "../src/predictive/engine/predict/segmentation.ts";
import { CacheLanguageModel } from "../src/predictive/engine/ngram/model.ts";

// A small but structured corpus so context actually matters.
const CORPUS = `
The quick brown fox jumps over the lazy dog. The quick brown fox runs fast.
The stock market fell sharply today. The stock market rose again yesterday.
The stock price of the company increased. The central bank raised interest rates.
The central bank cut interest rates last year. Investors sold their shares quickly.
I think the market will recover soon. I think the economy is strong.
We visited the United States last summer. The U.S. economy grew, e.g. in exports.
Machine learning models predict the next word. Machine learning is powerful.
`.repeat(4);

test("tokenizeWords normalises and splits", () => {
  const t = tokenizeWords("The Quick, brown FOX!");
  assert.deepEqual(t, ["the", "quick", "brown", "fox"]);
});

test("sentence splitting respects abbreviations", () => {
  const ab = buildAbbreviationSet();
  const sents = splitSentences("We met in the U.S. yesterday. It rained.", { abbreviations: ab });
  // "U.S." must NOT end a sentence; "yesterday." does.
  assert.equal(sents.length, 2);
  assert.ok(sents[0].includes("yesterday"));
});

test("isSentenceTerminator: e.g. and decimals are not boundaries", () => {
  const ab = buildAbbreviationSet();
  assert.equal(isSentenceTerminator("e.g.", "in", ab), false);
  assert.equal(isSentenceTerminator("3.14", "is", ab), false);
  assert.equal(isSentenceTerminator("today.", "The", ab), true);
});

test("language model uses context (bigram/trigram effect)", () => {
  const m = buildModelFromText(CORPUS);
  const afterCentralBank = m.predict(["central", "bank"], 3).map((s) => s.word);
  // "central bank" is followed by "raised" and "cut" in the corpus.
  assert.ok(afterCentralBank.includes("raised") || afterCentralBank.includes("cut"));
  // "the stock" should predict market/price, not "lazy".
  const afterTheStock = m.predict(["the", "stock"], 3).map((s) => s.word);
  assert.ok(afterTheStock.includes("market") || afterTheStock.includes("price"));
  assert.ok(!afterTheStock.includes("lazy"));
});

test("higher-order context outranks lower-order", () => {
  const m = buildModelFromText(CORPUS);
  // P(market | the stock) should exceed P(market | <no context>).
  const withCtx = m.logProb("market", ["the", "stock"]);
  const noCtx = m.logProb("market", []);
  assert.ok(withCtx > noCtx);
});

test("channel: transposition cheaper than two substitutions", () => {
  const trans = channelCost("teh", "the", DEFAULT_CHANNEL); // adjacent swap
  const twoSub = channelCost("xyz", "the", DEFAULT_CHANNEL);
  assert.ok(trans < twoSub);
  assert.ok(trans <= DEFAULT_CHANNEL.transposeCost + 1e-9);
});

test("channel: case mismatch is nearly free; identical is zero", () => {
  assert.equal(channelCost("the", "the", DEFAULT_CHANNEL), 0);
  const cased = channelCost("the", "The", DEFAULT_CHANNEL);
  assert.ok(cased > 0 && cased <= DEFAULT_CHANNEL.caseCost + 1e-9);
});

test("channel: adjacent-key substitution cheaper than far-key", () => {
  // 's' is adjacent to 'a'; 'p' is far from 'a' on QWERTY.
  const near = channelCost("s", "a", DEFAULT_CHANNEL);
  const far = channelCost("p", "a", DEFAULT_CHANNEL);
  assert.ok(near < far);
});

test("channel: missed character (deletion) recovered", () => {
  const cost = channelCost("recieve", "receive", DEFAULT_CHANNEL); // transposition ie/ei
  assert.ok(cost <= DEFAULT_CHANNEL.transposeCost + 1e-9);
  const missed = channelCost("tmrrow", "tomorrow", DEFAULT_CHANNEL);
  assert.ok(missed < Infinity);
});

test("channel: maxCost cutoff bails early", () => {
  assert.equal(channelCost("abcdefgh", "zzzz", DEFAULT_CHANNEL, 1.0), Infinity);
});

test("predictor: fuzzy typo re-ranked to intended word", () => {
  const eng = Engine.fromText(CORPUS);
  // Typed "thr" after start; expect "the" among top candidates.
  const cands = predict(eng.model, eng.index, {
    context: [],
    typed: "thr",
    k: 5,
    mode: "prefix",
  });
  assert.ok(cands.some((c) => c.word === "the"));
});

test("predictor: context disambiguates the correction", () => {
  const eng = Engine.fromText(CORPUS);
  // "markt" after "the stock" should surface "market".
  const cands = predict(eng.model, eng.index, {
    context: ["the", "stock"],
    typed: "markt",
    k: 5,
    mode: "full",
  });
  assert.equal(cands[0].word, "market");
});

test("autocorrect: fixes a typo but leaves valid words alone", () => {
  const eng = Engine.fromText(CORPUS);
  const d = decideCorrection(eng.model, eng.index, "markt", ["the", "stock"]);
  assert.equal(d.correct, true);
  assert.equal(d.to, "market");

  const keep = decideCorrection(eng.model, eng.index, "market", ["the", "stock"]);
  assert.equal(keep.correct, false);
});

test("autocorrect: never corrects all-caps acronyms (SEC stays SEC)", () => {
  const eng = Engine.fromText(CORPUS + "\nThe SEC and the FTC met.");
  for (const acr of ["SEC", "NASA", "USA", "HTTP"]) {
    const d = decideCorrection(eng.model, eng.index, acr, ["the"]);
    assert.equal(d.correct, false, `${acr} must not be corrected`);
    assert.equal(d.reason, "acronym");
  }
  // ordinary lowercase words are still corrected
  const fix = decideCorrection(eng.model, eng.index, "markt", ["the", "stock"]);
  assert.equal(fix.correct, true);
});

test("informationGain: ΔI is the log-posterior gap and is 0 when typed is the mode", () => {
  // Surprisal difference: −log P(typed) − (−log P(best)) = logPost(best) − logPost(typed).
  const cands = [
    { word: "market", logPost: -1 },
    { word: "markt", logPost: -4 },
    { word: "marker", logPost: -5 },
  ];
  const ig = informationGain(cands, "markt", -4)!;
  assert.equal(ig.best, "market");
  assert.ok(Math.abs(ig.deltaI - 3) < 1e-9); // (-1) - (-4)
  assert.ok(ig.pBest > ig.pTyped);
  assert.ok(ig.entropy > 0);

  // When the typed word IS the most likely option, there is nothing to gain.
  const none = informationGain(cands, "market", -1)!;
  assert.equal(none.best, "market");
  assert.ok(none.deltaI <= 0);
});

test("autocorrect: information-gain threshold tunes how eagerly it corrects", () => {
  const eng = Engine.fromText(CORPUS);
  // A tiny threshold corrects a mild typo; a large one leaves it alone.
  const eager = decideCorrection(eng.model, eng.index, "markt", ["the", "stock"], {
    infoGainThreshold: 0.5,
  });
  assert.equal(eager.correct, true);
  assert.equal(eager.to, "market");

  const cautious = decideCorrection(eng.model, eng.index, "markt", ["the", "stock"], {
    infoGainThreshold: 50,
  });
  assert.equal(cautious.correct, false);
  assert.equal(cautious.reason, "insufficient-info-gain");
});

test("autocorrect: non-word typo corrects; legacy single-regime still available", () => {
  const eng = Engine.fromText(CORPUS);
  // A close non-word typo corrects to the contextually-best plausible word (default regime).
  const d = decideCorrection(eng.model, eng.index, "markt", ["the", "stock"]);
  assert.equal(d.correct, true);
  assert.equal(d.to, "market");
  // The legacy single-regime path is still selectable and behaves.
  const legacy = decideCorrection(eng.model, eng.index, "markt", ["the", "stock"], {
    nonWordRegime: false,
  });
  assert.equal(legacy.correct, true);
});

test("autocorrect: gibberish with no plausible correction is left alone", () => {
  const eng = Engine.fromText(CORPUS);
  // No real word is a believable typo of these, so the structural-plausibility gate in the
  // non-word regime keeps them - we don't "correct" intentional gibberish to a far-off word.
  for (const junk of ["zqxwv", "asdfgh"]) {
    const d = decideCorrection(eng.model, eng.index, junk, ["the", "stock"]);
    assert.equal(d.correct, false, `${junk} should be left alone`);
  }
});

test("WordOracle: exact membership + morphological stem, from a packed blob", () => {
  const words = ["debitor", "reallocate", "consumer", "make", "carry"].sort();
  const blob = Buffer.from(words.join("\n"), "utf8");
  const head = Buffer.alloc(12);
  head.writeUInt32LE(0x57444c31, 0);
  head.writeUInt32LE(words.length, 4);
  head.writeUInt32LE(blob.length, 8);
  const bytes = Buffer.concat([head, blob]);
  const oracle = WordOracle.fromBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  assert.equal(oracle.has("debitor"), true);
  assert.equal(oracle.has("Debitor"), true); // case-insensitive
  assert.equal(oracle.has("consumder"), false); // a typo is not a word
  // Morphology: an inflected form is real if its stem is known.
  const known = (w: string) => oracle.has(w);
  assert.equal(oracle.stemKnown("reallocates", known), true); // -s
  assert.equal(oracle.stemKnown("making", known), true); // -ing, restore dropped e
  assert.equal(oracle.stemKnown("carries", known), true); // -ies → y
  assert.equal(oracle.stemKnown("qwerty", known), false);
});

test("autocorrect: an oracle-real (rare) word is protected from correction", () => {
  const eng = Engine.fromText(CORPUS);
  const oracleReal = (w: string) => w === "debitor"; // pretend the oracle knows this rare word
  // Without protection, a rare out-of-vocab token might be "corrected" to a near neighbour;
  // with the oracle predicate it is left exactly as typed.
  const d = decideCorrection(eng.model, eng.index, "debitor", ["paid", "the"], { oracleReal });
  assert.equal(d.correct, false);
  assert.equal(d.reason, "real-word-rare");
});

test("matchCase preserves original capitalisation shape", () => {
  assert.equal(matchCase("Markt", "market"), "Market");
  assert.equal(matchCase("MARKT", "market"), "MARKET");
  assert.equal(matchCase("markt", "market"), "market");
});

test("RevertBuffer is one-shot", () => {
  const b = new RevertBuffer();
  b.record("markt", "market");
  assert.deepEqual(b.revert(), { from: "markt", to: "market" });
  assert.equal(b.revert(), null);
});

test("auto-capitalisation: real starts vs abbreviations", () => {
  const cfg = defaultSentenceCaseConfig();
  assert.equal(shouldCapitalizeNext("", cfg), true); // doc start
  assert.equal(shouldCapitalizeNext("It rained today. ", cfg), true);
  assert.equal(shouldCapitalizeNext("We met in the U.S. ", cfg), false);
  assert.equal(shouldCapitalizeNext("for example, e.g. ", cfg), false);
  assert.equal(shouldCapitalizeNext("The price was 3.14 ", cfg), false);
  assert.equal(applyAutoCapitalization("dog", "It rained. ", cfg), "Dog");
  assert.equal(applyAutoCapitalization("i", "and then ", cfg), "I");
  // #17: a bare leading-capital canonical ("the"→"The") applies only at a real sentence start,
  // NOT after an abbreviation like "incl." (that spurious capital was the reported bug).
  assert.equal(applyAutoCapitalization("the", "items incl. ", { ...cfg, canonical: "The" }), "the");
  assert.equal(applyAutoCapitalization("the", "It rained. ", { ...cfg, canonical: "The" }), "The");
  // A genuine proper noun (differs by more than the first letter) still applies mid-sentence.
  assert.equal(applyAutoCapitalization("nasa", "works at ", { ...cfg, canonical: "NASA" }), "NASA");
});

test("personal-style bias shifts predictions", () => {
  const global = buildModelFromText(CORPUS);
  const personal = buildModelFromText(
    "The stock market crashed. The stock market crashed. The stock market crashed.".repeat(5),
  );
  const eng = new Engine(global, personal, 0.8);
  const cands = eng.model.predict(["the", "stock", "market"], 3).map((s) => s.word);
  assert.ok(cands.includes("crashed"));
});

// --- lexicality, re-spacing and topic adaptation -------------------------------------------

test("autocorrect: lexicality comes from the lexicon, not from what the model has seen", () => {
  const eng = Engine.fromText(CORPUS);
  // Simulate the real failure: the user's own typo has entered the vault-derived model, so
  // model.hasWord() says it is a word. It must still be correctable - a typo you repeated is
  // not thereby a word - so the regime is decided by an explicit lexicon predicate instead.
  const typo = "markt";
  assert.equal(eng.model.hasWord(typo), false, "fixture assumption");
  const pollutedIsWord = (w: string) => w === typo || eng.model.hasWord(w);
  const protectedByModel = decideCorrection(eng.model, eng.index, typo, ["the", "stock"], {
    isRealWord: pollutedIsWord,
  });
  const corrected = decideCorrection(eng.model, eng.index, typo, ["the", "stock"], {
    isRealWord: (w) => eng.model.hasWord(w),
  });
  assert.equal(corrected.correct, true);
  assert.equal(corrected.to, "market");
  // With the polluted lexicon it is treated as a real word and left alone - which is exactly
  // the behaviour the real lexicon must NOT inherit from the vault.
  assert.equal(protectedByModel.correct, false);
});

test("autocorrect: a real word is not replaced on frequency alone", () => {
  const eng = Engine.fromText(CORPUS);
  // With no context there is no contextual evidence for ANY substitution, so a correctly
  // typed real word must survive however much commoner a neighbour is.
  for (const w of ["market", "stock", "the"]) {
    const d = decideCorrection(eng.model, eng.index, w, []);
    assert.equal(d.correct, false, `${w} was replaced with no contextual evidence`);
  }
});

test("contextualEvidence is zero without context and finite with it", () => {
  const eng = Engine.fromText(CORPUS);
  assert.equal(contextualEvidence(eng.model, "market", "stock", []), 0);
  assert.ok(Number.isFinite(contextualEvidence(eng.model, "market", "stock", ["the"])));
});

test("respace: joins a split word and leaves a real word pair alone", () => {
  const eng = Engine.fromText(CORPUS);
  // "mark et" -> "market": the pieces are a fragment reading of one known word.
  const joined = decideRespace(eng.model, "mark", "et", ["the", "stock"]);
  assert.ok(joined === null || joined.to === "market");
  // A curated contraction joins outright, whatever the corpus says.
  assert.deepEqual(decideRespace(eng.model, "have", "nt", []), { to: "haven't" });
  assert.deepEqual(decideRespace(eng.model, "you", "d", []), { to: "you'd" });
  // Two genuine words stay apart.
  assert.equal(decideRespace(eng.model, "the", "stock", []), null);
});

test("suggestSplit refuses to strand a single letter", () => {
  const eng = Engine.fromText(CORPUS);
  const split = suggestSplit("youd", eng.model, []);
  assert.equal(split, null, '"youd" must reach the contraction fix, not split to "you d"');
});

test("cache adapts by REWEIGHTING, so context can still veto a topical word", () => {
  const base = Engine.fromText(CORPUS).model;
  const cache = new CacheLanguageModel(base, 0.15);
  cache.setDocument(tokenizeWords(CORPUS + " market market market"));
  // A word the base model rates as impossible here stays impossible: multiplying a
  // near-zero probability by a bounded factor cannot manufacture a suggestion. (The old
  // additive mixture gave every cached word the same floor in EVERY context.)
  for (const ctx of [[], ["the"], ["the", "stock"]]) {
    const boosted = cache.logProb("market", ctx);
    const raw = base.logProb("market", ctx);
    assert.ok(boosted >= raw - 1e-9, "a topical word must not be demoted");
    assert.ok(boosted <= 0, "still a probability");
    // The boost is bounded by the topicality ratio, not a flat floor: the gap between two
    // contexts survives the adaptation.
    assert.ok(Number.isFinite(boosted));
  }
  const gapRaw = base.logProb("market", ["the", "stock"]) - base.logProb("market", []);
  const gapAdapted = cache.logProb("market", ["the", "stock"]) - cache.logProb("market", []);
  assert.ok(Math.abs(gapRaw - gapAdapted) < 1e-9, "adaptation must not flatten context");
});

// --- int8 LSTM gates (fmt 5) -------------------------------------------------------
// The gate matrices are int8 with a per-row scale, like the embedding. Measured on
// 19,809 words of FineWeb-Edu: f32 gates ppl 64.621 -> int8 gates+activations 64.534,
// i.e. no cost, for 2.1x on the gate matvecs and -21 MB of file. These guard the two
// invariants that make that safe to ship.
test("fmt 4 and fmt 5 LSTM models agree exactly", async () => {
  const { readFileSync, existsSync } = await import("node:fs");
  const { LstmLanguageModel } = await import("../src/predictive/engine/lstm/model.ts");
  const f32 = "../dist/word_lstm.bin", i8 = "../dist/word_lstm.i8.bin";
  if (!existsSync(f32) || !existsSync(i8)) return; // models are not in the repo
  const load = (p: string) => {
    const b = readFileSync(p);
    return LstmLanguageModel.fromBuffer(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  };
  // fmt 4 is quantised at load, fmt 5 is read pre-quantised: the two must be identical,
  // which is what lets the converter and the loader be trusted against each other.
  const a = load(f32), b = load(i8);
  for (const ctx of [[], ["the"], ["i", "went", "to", "the"], ["new"]])
    for (const w of ["fox", "shop", "york", "the"])
      assert.ok(Math.abs(a.logProb(w, ctx) - b.logProb(w, ctx)) < 1e-6,
                `fmt4 vs fmt5 disagree on "${w}"`);
});

test("scalar and SIMD LSTM paths agree", async () => {
  const { readFileSync, existsSync } = await import("node:fs");
  const { LstmLanguageModel } = await import("../src/predictive/engine/lstm/model.ts");
  const p = "../dist/word_lstm.i8.bin";
  if (!existsSync(p)) return;
  const load = (simd: boolean) => {
    const b = readFileSync(p);
    return LstmLanguageModel.fromBuffer(
      b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), simd);
  };
  const wasm = load(true), scalar = load(false);
  if (!wasm.accelerated) return; // no SIMD here: nothing to cross-check
  assert.equal(scalar.accelerated, false);
  // int8 accumulation order differs between the SIMD reduction and scalar JS, so this
  // is a tolerance, not equality - but the RANKING the UI shows must be identical.
  for (const ctx of [["new"], ["i", "went", "to", "the"]]) {
    for (const w of ["york", "shop", "the"])
      assert.ok(Math.abs(wasm.logProb(w, ctx) - scalar.logProb(w, ctx)) < 0.1);
    assert.deepEqual(wasm.predict(ctx, 5).map((s) => s.word),
                     scalar.predict(ctx, 5).map((s) => s.word));
  }
  assert.equal(wasm.renderCased("york", ["new"]), "York");
});
