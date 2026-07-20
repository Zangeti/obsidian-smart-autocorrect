import { test } from "node:test";
import assert from "node:assert/strict";

import { buildModelFromText } from "../src/predictive/engine/ngram/build.ts";
import { NgramCounts, InMemoryLanguageModel, CacheLanguageModel } from "../src/predictive/engine/ngram/model.ts";
import { GeometryCostModel } from "../src/predictive/engine/channel/costModel.ts";
import { ConfusionModel, emptyConfusionData } from "../src/predictive/engine/channel/confusion.ts";
import { phoneticKey, phoneticCost } from "../src/predictive/engine/channel/phonetic.ts";
import { weightedEdit, prefixCost, IncrementalMatcher, channelCost } from "../src/predictive/engine/channel/editDistance.ts";
import { FuzzyTrie } from "../src/predictive/engine/predict/fuzzyTrie.ts";
import { predict, FuzzyIndex } from "../src/predictive/engine/predict/predictor.ts";
import { Reranker, defaultRerankerWeights } from "../src/predictive/engine/predict/reranker.ts";
import { segment, suggestSplit, RealWordCorrector } from "../src/predictive/engine/predict/segmentation.ts";
import { decideCorrection } from "../src/predictive/engine/autocorrect/autocorrect.ts";
import { Personalization, emptyPersonalization, normalizePersonalization } from "../src/predictive/engine/personalization/state.ts";
import { Engine } from "../src/predictive/engine/index.ts";

const CORPUS = `
The central bank raised interest rates today. The central bank cut interest rates last year.
The stock market rose sharply. The stock market fell sharply yesterday. The stock price increased.
Investors sold their shares quickly. Investors bought more shares this morning.
The company reported strong quarterly earnings. I think the market will recover soon.
There are many houses there. They went to their house. I know where they were.
`.repeat(6);

// ---- #1 fuzzy trie: complete recall incl. first-char errors ----------------

test("#1 FuzzyTrie recovers first-char errors the bucketed index misses", () => {
  const trie = new FuzzyTrie(["the", "market", "there", "their"]);
  const legacy = new FuzzyIndex(["the", "market", "there", "their"]);
  const cm = new GeometryCostModel();
  const trieHas = (typed: string) =>
    trie.neighbours(typed, "full", 4, cm).some((n) => n.word === "the");
  const legacyHas = (typed: string) =>
    legacy.neighbours(typed, "full", 4, cm).some((n) => n.word === "the");

  assert.ok(trieHas("hte"), "trie finds the<-hte (transposed first char)");
  assert.ok(trieHas("he"), "trie finds the<-he (missed first char)");
  // The legacy bucketed index misses these first-char errors (recall hole):
  // it only scans buckets whose first letter is 't' or adjacent to 't'.
  assert.ok(!legacyHas("hte"), "legacy misses hte (first char h not near t)");
  assert.ok(!legacyHas("he"), "legacy misses he");
});

test("#1 FuzzyTrie prefix completion", () => {
  const trie = new FuzzyTrie(["market", "marker", "marketing", "lazy"]);
  const cm = new GeometryCostModel();
  const words = trie.neighbours("mar", "prefix", 3, cm).map((n) => n.word);
  assert.ok(words.includes("market") && words.includes("marker"));
  assert.ok(!words.includes("lazy"));
});

// ---- #2 empirical / adaptive confusion model -------------------------------

test("#2 confusion model adapts to a user's recurring slip", () => {
  const cm = new ConfusionModel(emptyConfusionData());
  const before = cm.sub("m", "n", 0.5); // m->n (not adjacent-ish)
  for (let i = 0; i < 60; i++) cm.learn("machine", "nachine"); // user keeps typing n for m
  const after = cm.sub("m", "n", 0.5);
  assert.ok(after < before, `expected cheaper after learning: ${after} < ${before}`);
});

test("#2 confusion model serialises and merges", () => {
  const a = new ConfusionModel(emptyConfusionData());
  for (let i = 0; i < 10; i++) a.learn("form", "gorm");
  const json = a.toJSON();
  const b = ConfusionModel.fromJSON(JSON.parse(JSON.stringify(json)));
  assert.equal(b.sub("f", "g", 0.5), a.sub("f", "g", 0.5));
  b.mergeFrom(json);
  assert.ok(b.toJSON().observations > json.observations - 1);
});

test("#2 position penalty makes first-char errors dearer", () => {
  const cm = new ConfusionModel(emptyConfusionData());
  const atStart = cm.del("x", 0.0);
  const atEnd = cm.del("x", 1.0);
  assert.ok(atStart > atEnd);
});

// ---- #3 phonetic path ------------------------------------------------------

test("#3 phonetic keys collapse sound-alikes", () => {
  assert.equal(phoneticKey("phone"), phoneticKey("fone"));
  assert.equal(phoneticKey("night"), phoneticKey("nite"));
  assert.equal(phoneticKey("separate"), phoneticKey("seperate"));
  assert.equal(phoneticKey("definitely"), phoneticKey("definately"));
});

test("#3 phoneticCost cheap for homophones, Infinity for unrelated", () => {
  assert.ok(phoneticCost("fone", "phone") < 3);
  assert.equal(phoneticCost("cat", "dog"), Infinity);
});

// ---- #4 segmentation + real-word -------------------------------------------

test("#4 segment splits run-on tokens", () => {
  const m = buildModelFromText("the bank is open. a lot of the money. the bank of the people.".repeat(10));
  assert.deepEqual(segment("thebank", m).words, ["the", "bank"]);
  assert.deepEqual(suggestSplit("alot", m), ["a", "lot"]);
  assert.equal(suggestSplit("bank", m), null); // don't split real words
});

test("#4 real-word correction uses context", () => {
  const m = buildModelFromText(CORPUS);
  const rw = new RealWordCorrector(m.vocabulary());
  // "they went to their house" – after "to" we expect "their" not "there".
  const alt = rw.bestAlternative("there", ["went", "to"], m, 0.5);
  assert.equal(alt, "their");
});

// ---- #5 reranker -----------------------------------------------------------

test("#5 reranker defaults reproduce linear scoring, then learn", () => {
  const beta = 1.0;
  const r = new Reranker(undefined, beta);
  // features: [lmLogProb, negChannel, prefixExact, unigram, caseMatch, phonetic, lenPen, bias]
  const a = [-2, -1, 1, -3, 1, 0, 0, 1];
  const b = [-3, -0.5, 1, -3, 1, 0, 0, 1];
  const linA = -2 - beta * 1; // lm - beta*channel
  assert.ok(Math.abs(r.score(a) - (linA + 0.3 * 1)) < 1e-9); // +caseMatch weight 0.3
  // teach it to prefer b: after updates, b should outrank a
  for (let i = 0; i < 50; i++) r.learn([a, b], 1);
  assert.ok(r.score(b) > r.score(a));
});

// ---- #6 KN continuation + cache model --------------------------------------

test("#6 continuation prob fixes the 'San Francisco' problem", () => {
  // 'francisco' is frequent but only ever follows 'san'; 'the' follows many words.
  const counts = new NgramCounts();
  for (let i = 0; i < 20; i++) counts.addSentence(["san", "francisco"]);
  counts.addSentence(["i", "saw", "the", "cat"]);
  counts.addSentence(["the", "dog", "and", "the", "bird"]);
  counts.addSentence(["we", "left", "the", "town"]);
  const plain = new InMemoryLanguageModel(counts, { unkFloor: 1e-9, wbGuard: 1e-9 });
  const kn = new InMemoryLanguageModel(counts, { unkFloor: 1e-9, wbGuard: 1e-9, useContinuation: true });
  // Raw unigram loves 'francisco'; continuation should prefer 'the'.
  assert.ok(plain.logProb("francisco", []) > plain.logProb("the", []));
  assert.ok(kn.logProb("the", []) > kn.logProb("francisco", []));
});

test("#6 cache boosts recently-used KNOWN words but never surfaces unknown ones", () => {
  const base = buildModelFromText(CORPUS);
  const cache = new CacheLanguageModel(base, 0.4);
  // A word that exists in the base corpus gets boosted by recent use.
  const beforeKnown = cache.logProb("market", ["the"]);
  cache.observe(["market", "market", "market"]);
  assert.ok(cache.logProb("market", ["the"]) > beforeKnown);
  // A word NOT in the base (e.g. a typo you just typed) is never boosted or
  // surfaced, no matter how often it is observed.
  const beforeUnknown = cache.logProb("asdfgh", ["the"]);
  cache.observe(["asdfgh", "asdfgh", "asdfgh", "asdfgh"]);
  assert.equal(cache.logProb("asdfgh", ["the"]), beforeUnknown);
  assert.ok(!cache.predict(["the"], 20).some((s) => s.word === "asdfgh"));
});

// ---- #7 incremental / prefix distance --------------------------------------

test("#7 IncrementalMatcher equals batch weightedEdit", () => {
  const cm = new GeometryCostModel();
  const im = new IncrementalMatcher("market", cm);
  let last = 0;
  for (const ch of "markt") last = im.push(ch);
  assert.ok(Math.abs(last - weightedEdit("markt", "market", cm)) < 1e-9);
});

test("#7 prefixCost <= full cost and rewards true prefixes", () => {
  const cm = new GeometryCostModel();
  assert.ok(prefixCost("mar", "market", cm) <= channelCost("mar", "market"));
  assert.ok(prefixCost("mar", "market", cm) < prefixCost("xyz", "market", cm));
});

// ---- personalization end-to-end -------------------------------------------

test("personalization round-trips, resets, and imports", () => {
  const p = Personalization.empty();
  p.confusion.learn("form", "gorm");
  p.learnList.add("kubernetes");
  p.stats.corrections = 5;
  const json = p.toJSONString();
  const p2 = new Personalization(normalizePersonalization(JSON.parse(json)));
  assert.ok(p2.learnList.has("kubernetes"));
  assert.equal(p2.stats.corrections, 5);
  p2.reset();
  assert.equal(p2.learnList.size, 0);
  assert.equal(p2.stats.corrections, 0);
  // importing a malformed blob is safe
  p2.loadFrom({ garbage: true });
  assert.equal(p2.learnList.size, 0);
});

test("everything composes: Engine with personalization end-to-end", () => {
  const eng = new Engine(buildModelFromText(CORPUS), null, 0);
  const p = Personalization.empty(1.0);
  const cands = predict(eng.model, eng.index, {
    context: ["the", "stock"],
    typed: "markt",
    k: 3,
    mode: "full",
    costModel: p.confusion,
    reranker: p.reranker,
  });
  assert.equal(cands[0].word, "market");
  assert.ok(cands[0].features && cands[0].features.length === 8);
  // reranker can learn from an accept
  const shown = cands.map((c) => c.features!);
  p.reranker.learn(shown, 0);

  const d = decideCorrection(eng.model, eng.index, "markt", ["the", "stock"], {
    costModel: p.confusion,
    reranker: p.reranker,
    realWord: eng.realWord,
    enableSplit: true,
  });
  assert.equal(d.correct, true);
  assert.equal(d.to, "market");
});
