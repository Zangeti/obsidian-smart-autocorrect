import { test } from "node:test";
import assert from "node:assert/strict";
import { segmentText } from "../src/predictive/engine/text/segment.ts";

test("a whole bullet list is one block, not one block per bullet", () => {
  const md = [
    "## Fruit",
    "- apples are crisp and sweet",
    "- bananas are soft and rich",
    "- cherries are tart and bright",
  ].join("\n");
  const segs = segmentText(md);
  const lists = segs.filter((s) => s.kind === "list");
  assert.equal(lists.length, 1);
  // The list's fingerprint spans every bullet.
  assert.match(lists[0].text, /apples/);
  assert.match(lists[0].text, /cherries/);
});

test("a list under a heading anchors on the heading line (the title above)", () => {
  const md = ["## Fruit basket", "- apples", "- bananas and pears"].join("\n");
  const headingEnd = md.indexOf("\n"); // end of "## Fruit basket"
  const seg = segmentText(md).find((s) => s.kind === "list");
  assert.ok(seg);
  assert.equal(seg!.anchor, headingEnd);
});

test("a list with a lead-in line anchors on that lead-in", () => {
  const md = ["My favourite fruits to eat:", "- apples", "- bananas and pears"].join("\n");
  const leadEnd = md.indexOf("\n");
  const seg = segmentText(md).find((s) => s.kind === "list");
  assert.equal(seg!.anchor, leadEnd);
});

test("a cold list (no title above) anchors at the end of its first item", () => {
  const md = ["- apples are crisp", "- bananas are soft", "- cherries are tart"].join("\n");
  const firstEnd = md.indexOf("\n");
  const seg = segmentText(md).find((s) => s.kind === "list");
  assert.equal(seg!.anchor, firstEnd);
});

test("a plain-text title above bullets merges into the list block", () => {
  const md = "Shopping list\n- milk and butter\n- eggs and fresh bread";
  const segs = segmentText(md);
  const list = segs.find((s) => s.kind === "list");
  assert.ok(list);
  assert.match(list!.text, /Shopping list/); // the title is part of the block, not dropped
  assert.match(list!.text, /eggs/);
  assert.equal(segs.filter((s) => s.kind === "paragraph").length, 0); // title isn't its own paragraph
  assert.equal(list!.anchor, md.indexOf("\n")); // icon/link at the title line
});

test("title, one blank line, then bullets still merges", () => {
  const md = "Weekly goals\n\n- finish the report\n- book the flights";
  const list = segmentText(md).find((s) => s.kind === "list");
  assert.ok(list);
  assert.match(list!.text, /Weekly goals/);
});

test("a paragraph anchors at its end and fuses its lines", () => {
  const md = "The mitochondria produce ATP.\nThey are the powerhouse of the cell.";
  const segs = segmentText(md);
  assert.equal(segs.length, 1);
  assert.equal(segs[0].kind, "paragraph");
  assert.equal(segs[0].anchor, segs[0].to);
});

test("tolerates CRLF and groups a numbered + task list as one block", () => {
  const md = ["1. first thing to research", "2. second avenue worth exploring", "- [ ] follow up on both"].join("\r\n");
  const lists = segmentText(md).filter((s) => s.kind === "list");
  assert.equal(lists.length, 1);
  assert.match(lists[0].text, /follow up/);
});

test("a Setext heading acts as the title above a following list", () => {
  const md = ["Fruit basket", "===========", "- apples", "- bananas and pears"].join("\n");
  const seg = segmentText(md).find((s) => s.kind === "list");
  assert.ok(seg);
  // Anchor is the Setext title line (end of "Fruit basket"), not a bullet.
  assert.equal(seg!.anchor, md.indexOf("\n"));
});

test("detects a pipe table without a leading pipe", () => {
  const md = ["Fruit | Colour", "--- | ---", "Apple | Red", "Banana | Yellow"].join("\n");
  const segs = segmentText(md);
  assert.equal(segs.length, 1);
  assert.equal(segs[0].kind, "table");
});

test("groups a block-quote / callout as one block", () => {
  const md = ["> [!note] Photosynthesis", "> converts sunlight into chemical energy", "> stored as glucose"].join("\n");
  const segs = segmentText(md);
  assert.equal(segs.length, 1);
  assert.equal(segs[0].kind, "quote");
});

test("skips an indented code block but keeps the prose around it", () => {
  const md = ["Here is an example to study:", "", "    const total = a + b + c;", "", "That computes the sum."].join("\n");
  const kinds = segmentText(md).map((s) => s.kind);
  assert.deepEqual(kinds, ["paragraph", "paragraph"]);
});

test("skips an HTML block", () => {
  const md = ["<div class='note'>", "  <span>ignored markup here</span>", "</div>", "", "Real prose about photosynthesis."].join("\n");
  const segs = segmentText(md);
  assert.equal(segs.length, 1);
  assert.equal(segs[0].kind, "paragraph");
});

test("groups a nested list into a single block", () => {
  const md = ["- fruit basket contents", "    - apples and pears", "        - green and red apples", "- vegetables too"].join("\n");
  const lists = segmentText(md).filter((s) => s.kind === "list");
  assert.equal(lists.length, 1);
  assert.match(lists[0].text, /vegetables/);
});

test("ignores links, math and comments already in a block", () => {
  const md = "Studying $E=mc^2$ and %%a hidden note%% about relativity and spacetime.";
  const segs = segmentText(md);
  assert.equal(segs.length, 1);
  assert.equal(segs[0].kind, "paragraph");
});

test("headings, paragraphs and lists are separated; code is skipped", () => {
  const md = [
    "# Notes",
    "",
    "Neural networks learn representations from data.",
    "",
    "- gradient descent optimises the loss",
    "- convolutional layers extract features",
    "",
    "```",
    "const x = compute(value);",
    "```",
  ].join("\n");
  const kinds = segmentText(md).map((s) => s.kind);
  assert.deepEqual(kinds, ["paragraph", "list"]);
});
