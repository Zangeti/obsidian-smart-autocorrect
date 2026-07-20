import { test } from "node:test";
import assert from "node:assert/strict";
import { isDoubledWord } from "../src/predictive/engine/text/doubledWord.ts";

test("removes accidental function-word doublings", () => {
  for (const w of ["the", "a", "and", "of", "to", "is", "in", "it", "this"])
    assert.equal(isDoubledWord(w, w), true, w);
});

test("is case-insensitive", () => {
  assert.equal(isDoubledWord("The", "the"), true);
  assert.equal(isDoubledWord("THE", "The"), true);
});

test("leaves grammatically valid doublings alone", () => {
  for (const w of ["had", "that", "who", "will", "may", "can", "very", "really", "so", "no"])
    assert.equal(isDoubledWord(w, w), false, w);
});

test("different words are never a doubling", () => {
  assert.equal(isDoubledWord("the", "a"), false);
  assert.equal(isDoubledWord("New", "York"), false);
});
