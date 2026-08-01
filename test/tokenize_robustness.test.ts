/**
 * The model tokenizers sit on a boundary where text arrives from callers and across the worker
 * RPC. A null/undefined slip must degrade to "no tokens", never throw `undefined.replace(...)` -
 * that exact crash once propagated out of the worker and surfaced as a plugin-activation failure
 * ("Cannot read properties of undefined (reading 'replace')").
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { sanitizeForModel, tokenizeWords, tokenizeWordsCased } from "../src/predictive/engine/text/tokenize.ts";

const BAD: unknown[] = [undefined, null];

test("tokenizers do not throw on null/undefined input", () => {
  for (const bad of BAD) {
    assert.deepEqual(tokenizeWordsCased(bad as string), []);
    assert.deepEqual(tokenizeWords(bad as string), []);
    assert.equal(sanitizeForModel(bad as string), "");
  }
});

test("tokenizers still work on normal input", () => {
  assert.deepEqual(tokenizeWordsCased("Don’t stop"), ["Don't", "stop"]);
  assert.deepEqual(tokenizeWords("The quick fox"), ["the", "quick", "fox"]);
  assert.ok(sanitizeForModel("see `code` here").includes("see"));
});
