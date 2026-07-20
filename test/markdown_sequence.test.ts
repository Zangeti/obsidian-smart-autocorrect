import { test } from "node:test";
import assert from "node:assert/strict";

import { classifyMarkdownContext } from "../src/predictive/engine/text/markdownContext.ts";
import { beamSearchPhrases } from "../src/predictive/engine/predict/sequence.ts";
import { buildModelFromText } from "../src/predictive/engine/ngram/build.ts";
import { CacheLanguageModel } from "../src/predictive/engine/ngram/model.ts";

// ---- markdown / LaTeX awareness -------------------------------------------

const suppressed = (t: string) => classifyMarkdownContext(t).suppressPrediction;
const zone = (t: string) => classifyMarkdownContext(t).zone;

test("markdown: normal prose is allowed", () => {
  assert.equal(suppressed("The quick brown fox jumps ove"), false);
  assert.equal(zone("The quick brown "), "text");
});

test("markdown: inline code is suppressed", () => {
  assert.equal(zone("Use the `git comm"), "inlineCode");
  assert.equal(suppressed("Use the `git comm"), true);
  // closed inline code -> back to text
  assert.equal(zone("Use `git` to comm"), "text");
});

test("markdown: fenced code block is suppressed across lines", () => {
  const t = "text\n```python\ndef foo():\n    retur";
  assert.equal(zone(t), "codeBlock");
  // after closing fence -> text
  const t2 = "```py\nx=1\n```\nnow som";
  assert.equal(zone(t2), "text");
});

test("markdown: LaTeX math inline and block are suppressed", () => {
  assert.equal(zone("The energy $E = mc"), "mathInline");
  assert.equal(zone("$$\n\\int_0^1 x\\,d"), "mathBlock");
  assert.equal(suppressed("so \\alpha and \\bet"), false); // backslash in prose is fine
  assert.equal(zone("closed $x$ then mor"), "text");
});

test("markdown: wikilinks, embeds and urls are suppressed", () => {
  assert.equal(zone("see [[My Not"), "wikilink");
  assert.equal(zone("embed ![[Imag"), "wikilink");
  assert.equal(zone("a [link](http://exa"), "linkUrl");
  assert.equal(zone("visit https://exampl"), "linkUrl");
  assert.equal(zone("done [[Note]] and mor"), "text");
});

test("markdown: tags suppressed but headings allowed", () => {
  assert.equal(zone("this is #proj"), "tag");
  assert.equal(zone("# My Head"), "text"); // heading text should predict
});

test("markdown: frontmatter is suppressed", () => {
  assert.equal(zone("---\ntitle: My Not"), "frontmatter");
  assert.equal(zone("---\ntitle: x\n---\n\nBody tex"), "text");
});

test("markdown: obsidian comments suppressed", () => {
  assert.equal(zone("visible %%hidden comm"), "comment");
  assert.equal(zone("%%c%% visible mor"), "text");
});

// ---- unified beam search (single + multi-word) ----------------------------

const PROSE = `
The central bank is expected to raise interest rates. The central bank is expected to cut rates.
The stock market is expected to rise. Machine learning models are trained on large datasets.
Machine learning models are trained on data. I would like to thank you for your time.
`.repeat(8);

test("beam maxWords=1 gives single next words (unified with prediction)", () => {
  const m = buildModelFromText(PROSE);
  const phrases = beamSearchPhrases(m, ["the", "central", "bank"], { maxWords: 1 });
  assert.ok(phrases.length > 0);
  assert.ok(phrases.every((p) => p.words.length === 1));
  assert.ok(phrases.slice(0, 3).some((p) => p.words[0] === "is"));
});

test("beam maxWords>1 produces coherent multi-word phrases", () => {
  const m = buildModelFromText(PROSE);
  const phrases = beamSearchPhrases(m, ["the", "central", "bank"], {
    maxWords: 6,
    beamWidth: 4,
  });
  const top = phrases[0].words.join(" ");
  // best continuation should be the frequent "is expected to ..." phrase.
  assert.ok(top.startsWith("is expected to"), `got: ${top}`);
  assert.ok(phrases[0].words.length >= 3);
});

test("beam is topic-aware via the within-document cache", () => {
  const base = buildModelFromText(PROSE);
  const cache = new CacheLanguageModel(base, 0.35);
  // The note is about machine learning -> bias continuations that way.
  cache.setDocument("machine learning models are trained on".split(" "));
  const phrases = beamSearchPhrases(cache, ["models", "are", "trained"], { maxWords: 4 });
  assert.ok(phrases[0].words.join(" ").includes("on"));
});

test("length normalization lets a good long phrase beat a mediocre short one", () => {
  const m = buildModelFromText(PROSE);
  const phrases = beamSearchPhrases(m, ["i", "would", "like"], { maxWords: 6, lengthPenalty: 0.6 });
  assert.ok(phrases[0].words.join(" ").startsWith("to thank you"));
});
