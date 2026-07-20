/**
 * The "tidy double spaces" rule, extracted as a pure function so it can be tested
 * without an Obsidian editor. PredictiveSuggest.eatDoubleSpace mirrors this exactly;
 * the logic lives here because the edge cases are what matter and they are all about
 * text, not about the editor.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { doubleSpaceStart } from "../src/predictive/engine/text/doubleSpace.ts";

test("collapses exactly one space when a word is completed after a double space", () => {
  //                      0123456789
  assert.equal(doubleSpaceStart("hello  ", 7), 6); // "hello  " -> eat one
  assert.equal(doubleSpaceStart("a b  ", 5), 4);
});

test("leaves a single space alone", () => {
  assert.equal(doubleSpaceStart("hello ", 6), 6);
  assert.equal(doubleSpaceStart("hello", 5), 5);
});

test("leaves runs of THREE OR MORE spaces alone – those are deliberate", () => {
  // ASCII art, manual alignment, code-ish layout. Collapsing one space out of a run
  // of five would silently corrupt formatting the user built on purpose.
  assert.equal(doubleSpaceStart("a   ", 4), 4);
  assert.equal(doubleSpaceStart("a     ", 6), 6);
});

test("never touches indentation at the start of a line", () => {
  // ch < 2 can't have two spaces before it; and a line that is ONLY spaces is
  // indentation, which this rule must not eat into.
  assert.equal(doubleSpaceStart("  ", 2), 2);
  assert.equal(doubleSpaceStart(" ", 1), 1);
  assert.equal(doubleSpaceStart("", 0), 0);
});

test("leaves TABS alone – a tab before a word is indentation, not a typo", () => {
  assert.equal(doubleSpaceStart("\t\t", 2), 2);
  assert.equal(doubleSpaceStart("a\t ", 3), 3);
  assert.equal(doubleSpaceStart("a \t", 3), 3);
});

test("only looks at the two characters before the cursor", () => {
  // A double space EARLIER in the line is none of this rule's business – it fires
  // where the word is being completed, not as a line-wide reformatter.
  assert.equal(doubleSpaceStart("a  b c ", 7), 7);
});
