import { test } from "node:test";
import assert from "node:assert/strict";
import { fractionGlyph } from "../src/predictive/engine/text/fractions.ts";

test("fractionGlyph converts fractions that have a glyph", () => {
  assert.deepEqual(fractionGlyph("1/2"), { start: 0, text: "½" });
  assert.deepEqual(fractionGlyph("about 3/4"), { start: "about ".length, text: "¾" });
  assert.equal(fractionGlyph("7/8")!.text, "⅞");
  assert.equal(fractionGlyph("1/10")!.text, "⅒");
});

test("fractionGlyph leaves fractions with no glyph and date-like sequences alone", () => {
  assert.equal(fractionGlyph("3/7"), null); // no precomposed glyph
  assert.equal(fractionGlyph("1/2/2024"), null); // date, not a fraction
  assert.equal(fractionGlyph("10/2/2024"), null);
  assert.equal(fractionGlyph("abc1/2"), null); // glued to a token
});
