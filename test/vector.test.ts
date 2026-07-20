import { test } from "node:test";
import assert from "node:assert/strict";
import { terms, termFreq, documentFrequencies, tfidf, cosineSparse, cosineDense } from "../src/predictive/engine/text/vector.ts";

test("terms drops stop words, short tokens and casing", () => {
  assert.deepEqual(terms("The neural Network is a Model"), ["neural", "network", "model"]);
});

test("tfidf + cosine ranks a topical match above an unrelated chunk", () => {
  const a = termFreq("neural networks learn features via gradient descent");
  const b = termFreq("gradient descent optimises neural network weights");
  const c = termFreq("the recipe needs butter sugar flour and eggs");
  const df = documentFrequencies([a, b, c]);
  const n = 3;
  const va = tfidf(a, df, n), vb = tfidf(b, df, n), vc = tfidf(c, df, n);
  const related = cosineSparse(va, vb);
  const unrelated = cosineSparse(va, vc);
  assert.ok(related > unrelated, `${related} should exceed ${unrelated}`);
  assert.equal(unrelated, 0);
});

test("cosineDense is 1 for identical vectors and 0 for orthogonal", () => {
  assert.equal(cosineDense([1, 2, 3], [1, 2, 3]).toFixed(4), "1.0000");
  assert.equal(cosineDense([1, 0], [0, 1]), 0);
});
