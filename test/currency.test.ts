import { test } from "node:test";
import assert from "node:assert/strict";

import {
  CURRENCY_STYLES,
  currencyStyleFor,
  groupThousands,
  parseAmount,
  formatAmount,
  composeCurrency,
  detectCurrency,
} from "../src/predictive/engine/text/currency.ts";

const EN = CURRENCY_STYLES.english;
const DE = CURRENCY_STYLES.german;

test("groupThousands groups digits in threes", () => {
  assert.equal(groupThousands("1000", ","), "1,000");
  assert.equal(groupThousands("1000000", ","), "1,000,000");
  assert.equal(groupThousands("999", ","), "999");
  assert.equal(groupThousands("1000", "."), "1.000");
});

test("parseAmount splits integer and decimal", () => {
  assert.deepEqual(parseAmount("1000"), { int: "1000", dec: "" });
  assert.deepEqual(parseAmount("1,000"), { int: "1000", dec: "" }); // 3 trailing digits = grouping
  assert.deepEqual(parseAmount("1000.50"), { int: "1000", dec: "50" });
  assert.deepEqual(parseAmount("1000,50"), { int: "1000", dec: "50" }); // German decimal
  assert.equal(parseAmount("abc"), null);
});

test("formatAmount reflows with the style separators", () => {
  assert.equal(formatAmount("1000000", EN), "1,000,000");
  assert.equal(formatAmount("1000000", DE), "1.000.000");
  assert.equal(formatAmount("1000.5", EN), "1,000.5");
  assert.equal(formatAmount("1000,50", DE), "1.000,50");
});

test("composeCurrency places the symbol per style", () => {
  assert.equal(composeCurrency("$", "1,000", EN), "$1,000");
  assert.equal(composeCurrency("€", "1.000", DE), "1.000 €");
});

test("currencyStyleFor maps the separator setting to a style", () => {
  assert.equal(detectCurrency("$1000", { format: true, wordToSymbol: false, style: currencyStyleFor("comma") })!.text, "$1,000");
  assert.equal(detectCurrency("$1000", { format: true, wordToSymbol: false, style: currencyStyleFor("period") })!.text, "1.000 $");
  // "none" leaves the digits ungrouped, so "$1000" is already in its target form → no change.
  assert.equal(detectCurrency("$1000", { format: true, wordToSymbol: false, style: currencyStyleFor("none") }), null);
  // …but the word path still fires with "none": the symbol gets attached, digits ungrouped.
  assert.equal(detectCurrency("1000 dollars", { format: false, wordToSymbol: true, style: currencyStyleFor("none") })!.text, "$1000");
});

test("detectCurrency: word after number becomes the symbol", () => {
  const r = detectCurrency("I paid 1000 dollars", { format: false, wordToSymbol: true, style: EN });
  assert.ok(r);
  assert.equal("I paid ".length, r!.start);
  assert.equal(r!.text, "$1,000");

  const de = detectCurrency("kostet 2500 euro", { format: false, wordToSymbol: true, style: DE });
  assert.ok(de);
  assert.equal(de!.text, "2.500 €");
});

test("detectCurrency: reflows an amount that already has a symbol", () => {
  const a = detectCurrency("$1000", { format: true, wordToSymbol: false, style: EN });
  assert.deepEqual(a, { start: 0, text: "$1,000" });

  const b = detectCurrency("total 1000000$", { format: true, wordToSymbol: false, style: DE });
  assert.ok(b);
  assert.equal(b!.text, "1.000.000 €".replace("€", "$")); // symbol preserved, side per DE
});

test("detectCurrency: leaves already-formatted amounts and non-currency numbers alone", () => {
  assert.equal(detectCurrency("$1,000", { format: true, wordToSymbol: true, style: EN }), null);
  // A bare number is never reformatted (no symbol, no word) - so "2024" or "1000" stay put.
  assert.equal(detectCurrency("in 2024", { format: true, wordToSymbol: true, style: EN }), null);
  assert.equal(detectCurrency("1000 apples", { format: true, wordToSymbol: true, style: EN }), null);
});

test("detectCurrency: recognises the major world currencies (words + ISO codes)", () => {
  const cases: [string, string][] = [
    ["1000 dollars", "$1,000"],
    ["1000 USD", "$1,000"],
    ["1000 euros", "€1,000"],
    ["1000 GBP", "£1,000"],
    ["1000 pounds", "£1,000"],
    ["1000 yen", "¥1,000"],
    ["1000 yuan", "¥1,000"],
    ["1000 won", "₩1,000"],
    ["1000 rupees", "₹1,000"],
    ["1000 rand", "R1,000"],
    ["1000 reais", "R$1,000"],
    ["1000 naira", "₦1,000"],
    ["1000 zloty", "zł1,000"],
    ["1000 baht", "฿1,000"],
    ["1000 shekels", "₪1,000"],
    ["1000 francs", "Fr1,000"],
    ["1000 rubles", "₽1,000"],
    ["1000 lira", "₺1,000"],
    ["1000 BTC", "₿1,000"],
  ];
  for (const [input, want] of cases) {
    const r = detectCurrency(input, { format: false, wordToSymbol: true, style: EN });
    assert.ok(r, `no match for ${input}`);
    assert.equal(r!.text, want, `wrong format for ${input}`);
  }
});

test("detectCurrency: does not slice a number glued to a token", () => {
  // "css3000px" - the 3000 is glued to letters; word path needs a real currency word anyway.
  assert.equal(detectCurrency("abc1000 dollars", { format: false, wordToSymbol: true, style: EN }), null);
});
