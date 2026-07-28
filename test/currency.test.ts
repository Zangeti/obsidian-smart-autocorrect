import { test } from "node:test";
import assert from "node:assert/strict";

import {
  currencyStyleFor,
  groupThousands,
  parseAmount,
  formatAmount,
  composeCurrency,
  symbolPlacement,
  currencySymbolForWord,
  detectCurrency,
} from "../src/predictive/engine/text/currency.ts";

const COMMA = currencyStyleFor("comma");
const PERIOD = currencyStyleFor("period");
const NONE = currencyStyleFor("none");

test("groupThousands groups digits in threes", () => {
  assert.equal(groupThousands("1000", ","), "1,000");
  assert.equal(groupThousands("1000000", ","), "1,000,000");
  assert.equal(groupThousands("999", ","), "999");
  assert.equal(groupThousands("1000", "."), "1.000");
});

test("parseAmount splits integer and decimal, remembering the separator", () => {
  assert.deepEqual(parseAmount("1000"), { int: "1000", dec: "", decSep: "." });
  assert.deepEqual(parseAmount("1,000"), { int: "1000", dec: "", decSep: "." }); // 3 digits = grouping
  assert.deepEqual(parseAmount("1000.50"), { int: "1000", dec: "50", decSep: "." });
  assert.deepEqual(parseAmount("1000,50"), { int: "1000", dec: "50", decSep: "," });
  assert.equal(parseAmount("abc"), null);
});

test("formatAmount keeps the typed decimal with 'none', converts it with a set separator", () => {
  assert.equal(formatAmount("1000000", COMMA), "1,000,000");
  assert.equal(formatAmount("1000000", PERIOD), "1.000.000");
  assert.equal(formatAmount("1000.50", COMMA), "1,000.50"); // comma thousands -> dot decimal
  assert.equal(formatAmount("1000,50", PERIOD), "1.000,50"); // period thousands -> comma decimal
  assert.equal(formatAmount("1000.50", NONE), "1000.50"); // none -> keep the dot
  assert.equal(formatAmount("1000,50", NONE), "1000,50"); // none -> keep the comma
});

test("symbolPlacement: sign currencies lead, letter currencies trail", () => {
  assert.deepEqual(symbolPlacement("$"), { before: true, space: false });
  assert.deepEqual(symbolPlacement("€"), { before: true, space: false });
  assert.deepEqual(symbolPlacement("kr"), { before: false, space: true });
  assert.deepEqual(symbolPlacement("zł"), { before: false, space: true });
  assert.deepEqual(symbolPlacement("AED"), { before: false, space: true });
});

test("composeCurrency places the symbol on the currency's side", () => {
  assert.equal(composeCurrency("$", "1,000"), "$1,000");
  assert.equal(composeCurrency("€", "1.000"), "€1.000");
  assert.equal(composeCurrency("kr", "1 000".replace(" ", "")), "1000 kr");
  assert.equal(composeCurrency("R$", "1,000"), "R$1,000");
});

test("currencySymbolForWord resolves words and codes", () => {
  assert.equal(currencySymbolForWord("dollars"), "$");
  assert.equal(currencySymbolForWord("USD"), "$");
  assert.equal(currencySymbolForWord("yuan"), "¥");
  assert.equal(currencySymbolForWord("cabbage"), null);
});

test("detectCurrency: word after number becomes the symbol on the right side", () => {
  assert.equal(detectCurrency("I paid 1000 dollars", { format: false, wordToSymbol: true, style: COMMA })!.text, "$1,000");
  assert.equal(detectCurrency("kostet 2500 kronor", { format: false, wordToSymbol: true, style: PERIOD })!.text, "2.500 kr");
  assert.equal(detectCurrency("1000 euros", { format: false, wordToSymbol: true, style: COMMA })!.text, "€1,000");
});

test("detectCurrency: works WITHOUT a space between number and word", () => {
  assert.equal(detectCurrency("1000euro", { format: false, wordToSymbol: true, style: COMMA })!.text, "€1,000");
  assert.equal(detectCurrency("1000yuan", { format: false, wordToSymbol: true, style: COMMA })!.text, "¥1,000");
});

test("detectCurrency: only the number touching the word converts (2003 34 dollars)", () => {
  const r = detectCurrency("in 2003 34 dollars", { format: false, wordToSymbol: true, style: COMMA });
  assert.ok(r);
  assert.equal("in 2003 ".length, r!.start);
  assert.equal(r!.text, "$34"); // -> "in 2003 $34"
});

test("detectCurrency: does not span a space between two numbers (¥1,000 1000 yuan)", () => {
  const r = detectCurrency("¥1,000 1000 yuan", { format: true, wordToSymbol: true, style: COMMA });
  assert.ok(r);
  assert.equal(r!.text, "¥1,000"); // only the trailing "1000 yuan"
  assert.equal("¥1,000 ".length, r!.start); // the earlier ¥1,000 is left intact
});

test("detectCurrency: reflows a symbol amount and moves the symbol to its side", () => {
  assert.deepEqual(detectCurrency("$1000", { format: true, wordToSymbol: false, style: COMMA }), { start: 0, text: "$1,000" });
  // symbol typed AFTER the number is moved to the front for a dollar
  const b = detectCurrency("1000$", { format: true, wordToSymbol: false, style: COMMA });
  assert.equal(b!.text, "$1,000");
});

test("detectCurrency: leaves already-formatted amounts and non-currency numbers alone", () => {
  assert.equal(detectCurrency("$1,000", { format: true, wordToSymbol: true, style: COMMA }), null);
  assert.equal(detectCurrency("in 2024", { format: true, wordToSymbol: true, style: COMMA }), null);
  assert.equal(detectCurrency("1000 apples", { format: true, wordToSymbol: true, style: COMMA }), null);
});

test("detectCurrency: does not slice a number glued to a token", () => {
  assert.equal(detectCurrency("abc1000 dollars", { format: false, wordToSymbol: true, style: COMMA }), null);
});
