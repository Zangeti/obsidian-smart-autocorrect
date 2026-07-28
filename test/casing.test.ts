import { test } from "node:test";
import assert from "node:assert/strict";
import { harmonizeProperCase } from "../src/predictive/engine/text/tokenize.ts";

test("harmonizeProperCase repairs a half-cased proper noun", () => {
  assert.equal(harmonizeProperCase("federal Reserve"), "Federal Reserve");
  assert.equal(harmonizeProperCase("middle East"), "Middle East");
  assert.equal(harmonizeProperCase("new York City"), "New York City");
});

test("harmonizeProperCase leaves stopwords and normal phrases alone", () => {
  assert.equal(harmonizeProperCase("the White House"), "the White House"); // article stays lower
  assert.equal(harmonizeProperCase("bank of America"), "bank of America"); // not adjacent to a capital
  assert.equal(harmonizeProperCase("going to the store"), "going to the store"); // all lower
  assert.equal(harmonizeProperCase("Federal Reserve"), "Federal Reserve"); // already consistent
  assert.equal(harmonizeProperCase("hello"), "hello"); // single word
});
