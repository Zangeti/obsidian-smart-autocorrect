import { test } from "node:test";
import assert from "node:assert/strict";
import { PROFANITY, isProfane } from "../src/predictive/engine/text/profanity.ts";

test("blocks core profanity, slurs and explicit terms", () => {
  for (const w of ["fuck", "shit", "cunt", "nigger", "faggot", "whore", "cum"])
    assert.equal(isProfane(w), true, w);
});

test("is case-insensitive", () => {
  assert.equal(isProfane("FUCK"), true);
  assert.equal(isProfane("Shit"), true);
});

test("does NOT block benign words (Scunthorpe-safe)", () => {
  // curated away at build time: substrings, clinical, drugs, religion, negatives, mild
  for (const w of [
    "assess", "class", "cockpit", "analysis", "scunthorpe", "matsushita",
    "penis", "breast", "cannabis", "christ", "allah", "abuse", "bully",
    "blackmail", "crap", "damn", "the", "apple", "london",
  ])
    assert.equal(isProfane(w), false, w);
});

test("personal-dictionary allow-set un-blocks a word", () => {
  const allow = new Set(["cock"]);
  assert.equal(isProfane("cock"), true);
  assert.equal(isProfane("cock", allow), false);
  // allow-set doesn't leak to other words
  assert.equal(isProfane("fuck", allow), true);
});

test("blocklist is a reasonable, non-trivial size", () => {
  assert.ok(PROFANITY.size > 80 && PROFANITY.size < 400, `size=${PROFANITY.size}`);
});
