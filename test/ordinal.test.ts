import { test } from "node:test";
import assert from "node:assert/strict";
import { ordinalSuffix, fixNumericSuffix } from "../src/predictive/engine/text/ordinal.ts";

test("ordinalSuffix follows English rules", () => {
  assert.equal(ordinalSuffix("1"), "st");
  assert.equal(ordinalSuffix("2"), "nd");
  assert.equal(ordinalSuffix("3"), "rd");
  assert.equal(ordinalSuffix("4"), "th");
  assert.equal(ordinalSuffix("11"), "th");
  assert.equal(ordinalSuffix("12"), "th");
  assert.equal(ordinalSuffix("13"), "th");
  assert.equal(ordinalSuffix("21"), "st");
  assert.equal(ordinalSuffix("22"), "nd");
  assert.equal(ordinalSuffix("103"), "rd");
  assert.equal(ordinalSuffix("111"), "th");
});

test("fixNumericSuffix corrects a wrong ordinal", () => {
  assert.deepEqual(fixNumericSuffix("on the 21th"), { start: "on the ".length, text: "21st" });
  assert.deepEqual(fixNumericSuffix("22th"), { start: 0, text: "22nd" });
  assert.deepEqual(fixNumericSuffix("103th"), { start: 0, text: "103rd" });
  assert.equal(fixNumericSuffix("21TH")!.text, "21ST"); // case preserved
});

test("fixNumericSuffix leaves a correct ordinal or plain decade alone", () => {
  assert.equal(fixNumericSuffix("21st"), null);
  assert.equal(fixNumericSuffix("11th"), null);
  assert.equal(fixNumericSuffix("2nd"), null);
  assert.equal(fixNumericSuffix("1930s"), null);
});

test("fixNumericSuffix drops the apostrophe in a decade", () => {
  assert.deepEqual(fixNumericSuffix("the 1930's"), { start: "the ".length, text: "1930s" });
  assert.deepEqual(fixNumericSuffix("90's"), { start: 0, text: "90s" });
});

test("fixNumericSuffix ignores a number glued to letters", () => {
  assert.equal(fixNumericSuffix("abc21th"), null);
});
