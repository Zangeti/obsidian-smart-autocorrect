import { test } from "node:test";
import assert from "node:assert/strict";

import { fixDoubleCapital, fixContraction } from "../src/predictive/engine/text/caseFix.ts";
import { applyAutoCapitalization, defaultSentenceCaseConfig } from "../src/predictive/engine/text/sentenceCase.ts";
import { buildModelFromText } from "../src/predictive/engine/ngram/build.ts";
import { FuzzyTrie } from "../src/predictive/engine/predict/fuzzyTrie.ts";
import { decideCorrection } from "../src/predictive/engine/autocorrect/autocorrect.ts";

test("double-capital fix only touches the typo pattern", () => {
  assert.equal(fixDoubleCapital("THe"), "The");
  assert.equal(fixDoubleCapital("TYpe"), "Type");
  assert.equal(fixDoubleCapital("NASA"), "NASA"); // acronym untouched
  assert.equal(fixDoubleCapital("TypeScript"), "TypeScript"); // CamelCase untouched
  assert.equal(fixDoubleCapital("hello"), "hello");
});

test("contraction failsafe fixes unambiguous cases, leaves real words", () => {
  assert.equal(fixContraction("dont"), "don't");
  assert.equal(fixContraction("wont"), "won't");
  assert.equal(fixContraction("im"), "I'm");
  assert.equal(fixContraction("theyre"), "they're");
  // ambiguous real words are NOT in the map
  assert.equal(fixContraction("its"), null); // possessive vs it's
  assert.equal(fixContraction("were"), null); // valid word vs we're
  assert.equal(fixContraction("ill"), null); // valid word vs I'll
  assert.equal(fixContraction("the"), null);
});

test("contraction fix flows through the existing autocorrect decision", () => {
  const model = buildModelFromText("the quick brown fox jumps over the lazy dog.".repeat(4));
  const index = new FuzzyTrie(model.vocabulary());
  const d = decideCorrection(model, index, "dont", ["i"], {});
  assert.equal(d.correct, true);
  assert.equal(d.to, "don't");
  assert.equal(d.reason, "contraction");
});

test("auto-capitalisation composes: double-caps + LSTM canonical + sentence start", () => {
  // `canonical` is resolved per-word by the caller from the cased LSTM (it
  // replaced the old static CaseMap, which capitalised by frequency alone).
  const cfg = { ...defaultSentenceCaseConfig(), canonical: "London" };
  // proper noun, mid-sentence
  assert.equal(applyAutoCapitalization("london", "we met in ", cfg), "London");
  // sentence start capitalises on top of the canonical form
  assert.equal(applyAutoCapitalization("london", "It rained. ", cfg), "London");
  // no canonical advice -> ordinary word untouched mid-sentence
  const plain = { ...defaultSentenceCaseConfig(), canonical: null };
  assert.equal(applyAutoCapitalization("dog", "we saw a ", plain), "dog");
  // ...but a sentence start still capitalises
  assert.equal(applyAutoCapitalization("dog", "It rained. ", plain), "Dog");
  // double capital typo still fixed
  assert.equal(applyAutoCapitalization("THe", "so ", plain), "The");
});
