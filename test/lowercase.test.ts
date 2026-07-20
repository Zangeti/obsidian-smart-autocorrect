/**
 * The LSTM <-> lowercase-pipeline boundary (engine/lstm/lowercase.ts) and the
 * tokeniser that must stay identical to the trainer's regex.
 *
 * The old CasedLstmModel tests died with the class: it marginalised each lowercase
 * word over its cased variants ("paris" = paris + Paris) and picked a casing by
 * comparing their probabilities. There are no variants any more – the vocab is
 * lowercase and casing comes out of the case head – so that whole layer is gone.
 * Case behaviour is now tested against a real trained model in modelCase.test.ts.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { LowercaseModel } from "../src/predictive/engine/lstm/lowercase.ts";
import { tokenizeWordsCased, startsWithTightPunct } from "../src/predictive/engine/text/tokenize.ts";
import { buildModelFromText } from "../src/predictive/engine/ngram/build.ts";

test("the LSTM tokeniser matches the trainer's regex (words + punctuation)", () => {
  assert.deepEqual(
    tokenizeWordsCased("in the markets of the world. The dog barked!"),
    ["in", "the", "markets", "of", "the", "world", ".", "The", "dog", "barked", "!"],
  );
  // smart quotes normalise so "don’t" hits the trained "don't" token
  assert.deepEqual(tokenizeWordsCased("don’t"), ["don't"]);
});

test("LowercaseModel lowercases and keeps only the CURRENT sentence", () => {
  const inner = buildModelFromText("the quick brown fox jumps over the lazy dog.".repeat(4));
  const seen: string[][] = [];
  const spy = {
    logProb(w: string, c: string[]) { seen.push(c); return inner.logProb(w, c); },
    predict: (c: string[], k: number) => inner.predict(c, k),
    hasWord: (w: string) => inner.hasWord(w),
    vocabulary: () => inner.vocabulary(),
    size: () => inner.size(),
  };
  const m = new LowercaseModel(spy);
  m.logProb("Fox", ["I", "went", "Home", ".", "The", "Quick", "Brown"]);
  // Everything after the last TERMINATOR, lowercased, punctuation dropped – exactly
  // the per-sentence word context the n-gram counts were built from; the LSTM's
  // wider cross-sentence view must not leak in here.
  assert.deepEqual(seen[0], ["the", "quick", "brown"]);
  // a comma is not a sentence end: it must not truncate the n-gram's context
  seen.length = 0;
  m.logProb("Fox", ["The", "quick", ",", "brown"]);
  assert.deepEqual(seen[0], ["the", "quick", "brown"]);
});

test("startsWithTightPunct drives the 'no space before punctuation' rule", () => {
  // The model predicts punctuation as ordinary tokens now, so a suggestion can be
  // "," or ". The". Accepting one right after "world " must not give "world ,".
  assert.equal(startsWithTightPunct(","), true);
  assert.equal(startsWithTightPunct(". The"), true);
  assert.equal(startsWithTightPunct("; and"), true);
  assert.equal(startsWithTightPunct("?"), true);
  // ordinary suggestions keep their preceding space
  assert.equal(startsWithTightPunct("the same year"), false);
  assert.equal(startsWithTightPunct("United"), false);
  assert.equal(startsWithTightPunct(""), false);
  // an apostrophe is NOT tight punctuation here: "don't" is one word, and a
  // suggestion never opens with one.
  assert.equal(startsWithTightPunct("'"), false);
});
