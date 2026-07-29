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
  currencyProposal,
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

test("parseAmount uses the style to tell the decimal from a thousands group", () => {
  assert.deepEqual(parseAmount("1000", COMMA), { int: "1000", dec: "", decSep: "." });
  assert.deepEqual(parseAmount("1,000", COMMA), { int: "1000", dec: "", decSep: "." }); // comma = grouping
  assert.deepEqual(parseAmount("1000.50", COMMA), { int: "1000", dec: "50", decSep: "." });
  assert.deepEqual(parseAmount("10.567", COMMA), { int: "10", dec: "567", decSep: "." }); // 3-digit decimal
  assert.deepEqual(parseAmount("1000,50", NONE), { int: "1000", dec: "50", decSep: "," });
  assert.equal(parseAmount("abc", COMMA), null);
});

test("formatAmount pads the decimal to at least two places, keeping more if present", () => {
  assert.equal(formatAmount("1000000", COMMA), "1,000,000");
  assert.equal(formatAmount("1000000", PERIOD), "1.000.000");
  assert.equal(formatAmount("1000.5", COMMA), "1,000.50"); // padded to two
  assert.equal(formatAmount("1000.50", COMMA), "1,000.50");
  assert.equal(formatAmount("10.567", COMMA), "10.567"); // three kept
  assert.equal(formatAmount("1000,5", PERIOD), "1.000,50"); // period thousands -> comma decimal, padded
  assert.equal(formatAmount("1000.5", NONE), "1000.50"); // none -> keep the dot, padded
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

test("negative amounts keep the sign at the front", () => {
  const o = { format: true, wordToSymbol: true, style: COMMA };
  assert.equal(detectCurrency("-1000 dollars", o)!.text, "-$1,000");
  assert.equal(detectCurrency("-$1000", o)!.text, "-$1,000");
  assert.equal(detectCurrency("-50 euros", o)!.text, "-€50");
});

test("euro placement follows the style setting", () => {
  const after = currencyStyleFor("comma", { euroAfter: true });
  assert.equal(detectCurrency("1000 euros", { format: true, wordToSymbol: true, style: after })!.text, "1,000 €");
  assert.equal(detectCurrency("€1000", { format: true, wordToSymbol: true, style: after })!.text, "1,000 €");
  // other currencies are unaffected by the euro setting
  assert.equal(detectCurrency("$1000", { format: true, wordToSymbol: true, style: after })!.text, "$1,000");
});

test("ISO-code mode emits the right code, distinguishing shared signs", () => {
  const code = currencyStyleFor("comma", { useCode: true });
  const o = { format: true, wordToSymbol: true, style: code };
  assert.equal(detectCurrency("1000 dollars", o)!.text, "1,000 USD");
  assert.equal(detectCurrency("1000 CAD", o)!.text, "1,000 CAD"); // shares "$" but coded CAD
  assert.equal(detectCurrency("1000 yuan", o)!.text, "1,000 CNY"); // shares "¥" but coded CNY
  assert.equal(detectCurrency("$5000", o)!.text, "5,000 USD"); // bare "$" defaults to USD
  assert.equal(detectCurrency("-100 pounds", o)!.text, "-100 GBP");
});

test("dong and other letter/after currencies trail the number", () => {
  assert.equal(detectCurrency("1000 dong", { format: false, wordToSymbol: true, style: COMMA })!.text, "1,000 ₫");
  assert.equal(detectCurrency("1000 zloty", { format: false, wordToSymbol: true, style: COMMA })!.text, "1,000 zł");
});

test("currencyProposal fires on a PARTIAL currency word for the popup", () => {
  const opts = { format: true, wordToSymbol: true, style: COMMA };
  const p = currencyProposal("1000 doll", opts);
  assert.ok(p);
  assert.equal(p!.symbol, "$");
  assert.equal(p!.text, "$1,000");
  // a symbol amount still resolves
  assert.equal(currencyProposal("$10000", opts)!.text, "$10,000");
  // a non-currency word does not
  assert.equal(currencyProposal("1000 the", opts), null);
});
