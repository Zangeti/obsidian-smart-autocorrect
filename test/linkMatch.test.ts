import { test } from "node:test";
import assert from "node:assert/strict";
import { findLinkSpans, protectedRanges } from "../src/predictive/engine/text/linkMatch.ts";

const titles: Record<string, { target: string; display: string }> = {
  "project alpha": { target: "Project Alpha", display: "Project Alpha" },
  "neural network": { target: "Neural Network", display: "Neural Network" },
  obsidian: { target: "Obsidian", display: "Obsidian" },
};
const lookup = (p: string) => titles[p] ?? null;

test("matches an existing multi-word title, preserving typed casing", () => {
  const spans = findLinkSpans("met with Project Alpha today", lookup);
  assert.equal(spans.length, 1);
  assert.equal(spans[0].text, "Project Alpha");
  assert.equal(spans[0].target, "Project Alpha");
});

test("longest-match wins (doesn't stop at a shorter prefix)", () => {
  const spans = findLinkSpans("a neural network model", lookup);
  assert.equal(spans.length, 1);
  assert.equal(spans[0].text, "neural network");
});

test("only proposes links to notes that exist – nothing invented", () => {
  assert.deepEqual(findLinkSpans("some random unrelated words", lookup), []);
});

test("does not match inside existing links, code, or math", () => {
  assert.deepEqual(findLinkSpans("see [[Project Alpha]] here", lookup), []);
  assert.deepEqual(findLinkSpans("run `obsidian` cli", lookup), []);
  assert.deepEqual(findLinkSpans("$obsidian$", lookup), []);
});

test("excludeTarget skips the note's own title", () => {
  const spans = findLinkSpans("Obsidian is great", lookup, { excludeTarget: "Obsidian" });
  assert.deepEqual(spans, []);
});

test("protectedRanges covers code/links/math/tags", () => {
  const t = "`code` [[wl]] $x$ #tag";
  const r = protectedRanges(t);
  assert.ok(r.length >= 4);
});
