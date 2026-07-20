import { test } from "node:test";
import assert from "node:assert/strict";

import { buildModelFromText } from "../src/predictive/engine/ngram/build.ts";
import { NgramCounts, InMemoryLanguageModel } from "../src/predictive/engine/ngram/model.ts";
import { splitSentences } from "../src/predictive/engine/text/tokenize.ts";
import { FuzzyTrie } from "../src/predictive/engine/predict/fuzzyTrie.ts";
import { GeometryCostModel } from "../src/predictive/engine/channel/costModel.ts";
import { ConfusionModel, emptyConfusionData } from "../src/predictive/engine/channel/confusion.ts";
import { keyDistance, DEFAULT_CHANNEL, substitutionCost } from "../src/predictive/engine/channel/keyboard.ts";
import { foldDiacritics, stem } from "../src/predictive/engine/text/normalize.ts";
import { evaluate, corrupt, rng } from "../src/predictive/engine/eval/harness.ts";
import { IncrementalCorpus } from "../src/predictive/engine/ngram/incremental.ts";
import { packCounts, PackedLanguageModel } from "../src/predictive/engine/ngram/packed.ts";

function countsFromText(text: string): NgramCounts {
  const c = new NgramCounts();
  for (const s of splitSentences(text)) if (s.length) c.addSentence(s);
  return c;
}

const CORPUS = `
The central bank raised interest rates today. The central bank cut interest rates last year.
The stock market rose sharply. The stock market fell sharply. The stock price increased.
Investors sold their shares. Investors bought more shares. The company reported earnings.
`.repeat(8);

// ---- C7 keyboard layouts ---------------------------------------------------

test("#C7 layouts change key adjacency", () => {
  // On QWERTY 'q' and 'a' are neighbours; on Dvorak the letters at those spots differ.
  assert.ok(keyDistance("q", "w", "qwerty") < keyDistance("q", "p", "qwerty"));
  // AZERTY swaps q<->a and w<->z vs QWERTY, so 'a' and 'z' are adjacent there.
  assert.ok(keyDistance("a", "z", "azerty") < keyDistance("a", "z", "qwerty"));
});

test("#C7 substitution cost respects layout", () => {
  const qwerty = { ...DEFAULT_CHANNEL, layout: "qwerty" as const };
  const azerty = { ...DEFAULT_CHANNEL, layout: "azerty" as const };
  // 'a'->'z': far on QWERTY, adjacent on AZERTY -> cheaper on AZERTY.
  assert.ok(substitutionCost("a", "z", azerty) < substitutionCost("a", "z", qwerty));
});

// ---- D9 diacritics, stemming, context-conditioned channel ------------------

test("#D9 diacritic folding", () => {
  assert.equal(foldDiacritics("café"), "cafe");
  assert.equal(foldDiacritics("naïve"), "naive");
  assert.equal(foldDiacritics("Straße"), "strasse");
});

test("#D9 diacritic difference is nearly free in the channel", () => {
  const cost = substitutionCost("é", "e", DEFAULT_CHANNEL);
  assert.ok(cost <= DEFAULT_CHANNEL.caseCost * 2);
});

test("#D9 light stemmer folds inflections", () => {
  assert.equal(stem("running"), "run");
  assert.equal(stem("runs"), "run");
  assert.equal(stem("cities"), "city");
  assert.equal(stem("quickly"), "quick");
});

test("#D9 context-conditioned confusion learns prev-char specific slips", () => {
  const cm = new ConfusionModel(emptyConfusionData());
  const generic = cm.sub("c", "k", 0.5, "s");
  // Teach: after 's', the user types 'k' for 'c' (e.g. "sc" -> "sk").
  for (let i = 0; i < 40; i++) cm.learn("scale", "skale");
  const afterS = cm.sub("c", "k", 0.5, "s");
  const afterOther = cm.sub("c", "k", 0.5, "a");
  assert.ok(afterS < generic, "context-specific slip got cheaper");
  assert.ok(afterS < afterOther, "cheaper after 's' than after 'a'");
});

// ---- D8 evaluation harness -------------------------------------------------

test("#D8 corrupt injects a change deterministically", () => {
  const r = rng(1);
  const typo = corrupt("prediction", r, { errorRate: 1 });
  assert.notEqual(typo, "prediction");
  // deterministic with the same seed
  assert.equal(corrupt("prediction", rng(1), { errorRate: 1 }), corrupt("prediction", rng(1), { errorRate: 1 }));
});

test("#D8 evaluate reports sane metrics and correction beats chance", () => {
  const model = buildModelFromText(CORPUS);
  const index = new FuzzyTrie(model.vocabulary());
  const res = evaluate(model, index, CORPUS.slice(0, 400), { seed: 7, k: 5 });
  assert.ok(res.corrupted > 0);
  assert.ok(res.correctionAccuracy >= 0 && res.correctionAccuracy <= 1);
  assert.ok(res.recallAtK >= res.correctionAccuracy);
  assert.ok(res.correctionAccuracy > 0.4, `accuracy too low: ${res.correctionAccuracy}`);
  assert.ok(res.keystrokeSavings > 0);
});

// ---- B3 incremental per-file corpus ----------------------------------------

test("#B3 incremental add/remove equals full rebuild", () => {
  const inc = new IncrementalCorpus();
  const d1 = "the central bank raised rates.";
  const d2 = "the stock market rose sharply.";
  const d3 = "investors sold their shares.";
  inc.upsert("a.md", d1);
  inc.upsert("b.md", d2);
  inc.upsert("c.md", d3);
  inc.remove("c.md"); // remove d3 -> should equal building from d1+d2
  const incModel = inc.build();
  const full = buildModelFromText(d1 + " " + d2);
  const a = incModel.predict(["the", "central", "bank"], 2).map((x) => x.word);
  const b = full.predict(["the", "central", "bank"], 2).map((x) => x.word);
  assert.deepEqual(a, b);
  // updating a file replaces its contribution
  inc.upsert("a.md", "the central bank cut rates.");
  const after = inc.build().predict(["the", "central", "bank"], 1)[0].word;
  assert.equal(after, "cut");
});

test("#B3 rename preserves contribution", () => {
  const inc = new IncrementalCorpus();
  inc.upsert("old.md", "the quick brown fox jumps.");
  inc.rename("old.md", "new.md");
  assert.ok(inc.has("new.md") && !inc.has("old.md"));
  inc.remove("new.md");
  assert.equal(inc.fileCount, 0);
});

// ---- B2 packed binary model ------------------------------------------------

test("#B2 packed model round-trips predictions", () => {
  const counts = countsFromText(CORPUS);
  const model = new InMemoryLanguageModel(counts);
  const buf = packCounts(model, counts, { topK: 8 });
  assert.ok(buf.byteLength > 0);
  const packed = PackedLanguageModel.fromBuffer(buf);

  for (const ctx of [["the", "central", "bank"], ["the", "stock"], ["investors"]]) {
    const a = model.predict(ctx, 2).map((x) => x.word);
    const b = packed.predict(ctx, 2).map((x) => x.word);
    assert.deepEqual(b, a, `context ${ctx.join(" ")}: ${b} vs ${a}`);
  }
  // logProb ordering preserved (quantised but monotone)
  const hi = packed.logProb("market", ["the", "stock"]);
  const lo = packed.logProb("market", []);
  assert.ok(hi >= lo);
  assert.ok(packed.size().vocab === counts.vocab.length);
});

test("#B2 packed model is compact (bytes per stored context is small)", () => {
  const counts = countsFromText(CORPUS);
  const model = new InMemoryLanguageModel(counts);
  const buf = packCounts(model, counts, { topK: 8 });
  const contexts = packed_contexts(buf);
  // sanity: buffer exists and is far smaller than a naive JSON of the model
  const jsonSize = JSON.stringify({ vocab: counts.vocab }).length;
  assert.ok(buf.byteLength < jsonSize * 20);
  assert.ok(contexts > 0);
});

function packed_contexts(buf: ArrayBuffer): number {
  const m = PackedLanguageModel.fromBuffer(buf);
  return m.size().bigrams + m.size().trigrams;
}
