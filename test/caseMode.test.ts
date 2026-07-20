/**
 * Suggestion casing + CAPS detection. Detection is a pure function of the text, so
 * every edit (including deletion) simply re-evaluates.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { applySuggestionCase, upperFromText, suggestionCase } from "../src/predictive/engine/text/caseMode.ts";
import { defaultSentenceCaseConfig, shouldCapitalizeNext } from "../src/predictive/engine/text/sentenceCase.ts";
import { classifyMarkdownContext } from "../src/predictive/engine/text/markdownContext.ts";
import { sanitizeForModel, tokenizeWordsCased } from "../src/predictive/engine/text/tokenize.ts";

const cfg = defaultSentenceCaseConfig([]);

test("sentence starts are capitalised, mid-sentence is left alone", () => {
  assert.equal(suggestionCase("", cfg), "title"); // start of document
  assert.equal(suggestionCase("Hello there. ", cfg), "title"); // after a full stop
  assert.equal(suggestionCase("the quick brown ", cfg), "none"); // mid-sentence
  assert.equal(suggestionCase("approx. ", defaultSentenceCaseConfig(["approx."])), "none");
});

test("upper overrides position, including at a sentence start", () => {
  assert.equal(suggestionCase("the quick brown ", cfg, { upper: true }), "upper");
  assert.equal(suggestionCase("", cfg, { upper: true }), "upper");
});

test("TWO capitals in a row turn caps on; one is not enough", () => {
  assert.equal(upperFromText("TH"), true);
  assert.equal(upperFromText("T"), false); // just a sentence start or proper noun
  assert.equal(upperFromText("I"), false);
  assert.equal(upperFromText("A"), false);
  assert.equal(upperFromText("The"), false);
  assert.equal(upperFromText(""), false);
});

test("caps CARRY across a word boundary with nothing typed yet", () => {
  // The point: after finishing a shouted word the NEXT word's suggestions are already
  // caps – no need to type two more capitals to re-earn it.
  assert.equal(upperFromText("THE QUICK "), true);
  assert.equal(upperFromText("THE QUICK BROWN FOX "), true);
  // A single capital after a shout keeps it (no lowercase evidence yet).
  assert.equal(upperFromText("THE QUICK B"), true);
  // Punctuation and digits separate words but are not evidence.
  assert.equal(upperFromText("STOP, YOU "), true);
  assert.equal(upperFromText("WAIT 42 SECONDS "), true);
});

test("a LONE all-caps word (acronym) does NOT turn caps on", () => {
  // "NASA" in ordinary prose is an abbreviation, not shouting – the following
  // suggestion must stay in normal case, not be uppercased.
  assert.equal(upperFromText("My favourite agency is NASA "), false);
  assert.equal(upperFromText("we use HTTP "), false);
  assert.equal(upperFromText("the USA "), false);
  // A single shouted word alone is likewise not yet a caps run (needs a second word or a
  // second capital in the word being typed).
  assert.equal(upperFromText("STOP "), false);
  // But an acronym INSIDE a genuine caps run still rides the run until a lowercase char.
  assert.equal(upperFromText("I LOVE NASA "), true);
  assert.equal(upperFromText("GO NASA TODAY "), true);
  // ...and the run ends the instant a real lowercase letter appears.
  assert.equal(upperFromText("I LOVE NASA t"), false);
});

test("ONE lowercase character drops caps immediately", () => {
  assert.equal(upperFromText("THE QUICK f"), false);
  assert.equal(upperFromText("THE QUICK Br"), false);
  assert.equal(upperFromText("THe"), false);
});

test("DELETING the lowercase char restores caps – no stale state", () => {
  // This is why detection reads the text instead of tracking a mode: the sequence
  // below is just three independent evaluations, so undo/backspace cannot desync it.
  assert.equal(upperFromText("THE QUICK Br"), false); // typed a lowercase 'r'
  assert.equal(upperFromText("THE QUICK B"), true); // deleted it -> caps again
  assert.equal(upperFromText("THE QUICK "), true); // deleted the 'B' too -> still caps
});

test("ordinary prose never triggers caps", () => {
  assert.equal(upperFromText("the quick brown fox "), false);
  assert.equal(upperFromText("I saw a dog. "), false);
  assert.equal(upperFromText("Hello there, World "), false);
});

test("applySuggestionCase capitalises only the FIRST word of a phrase", () => {
  assert.equal(applySuggestionCase("the same year", "title"), "The same year");
  assert.equal(applySuggestionCase("the same year", "upper"), "THE SAME YEAR");
  assert.equal(applySuggestionCase(", and", "title"), ", And");
  assert.equal(applySuggestionCase("", "upper"), "");
});

test("bare URLs and emails suppress suggestions", () => {
  const sup = (s: string) => classifyMarkdownContext(s).suppressPrediction;
  assert.equal(sup("see https://example.com/some-Pa"), true);
  assert.equal(sup("mail me at titus.z@imperial.ac"), true);
  assert.equal(sup("see https://example.com and then wri"), false);
});

test("a line holding only a list/quote marker capitalises its first word", () => {
  // "- overnight" should Title just like a sentence start (matches Obsidian/Word).
  assert.equal(suggestionCase("- ", cfg), "title");
  assert.equal(suggestionCase("* ", cfg), "title");
  assert.equal(suggestionCase("+ ", cfg), "title");
  assert.equal(suggestionCase("  - ", cfg), "title"); // indented
  assert.equal(suggestionCase("1. ", cfg), "title");
  assert.equal(suggestionCase("2) ", cfg), "title");
  assert.equal(suggestionCase("- [ ] ", cfg), "title"); // task item
  assert.equal(suggestionCase("> ", cfg), "title"); // blockquote
  assert.equal(suggestionCase("> - ", cfg), "title"); // nested
  // But once there is content on the item, mid-item words are left alone.
  assert.equal(suggestionCase("- the quick ", cfg), "none");
  assert.equal(shouldCapitalizeNext("- ", cfg), true);
  assert.equal(shouldCapitalizeNext("- some words ", cfg), false);
});

test("dotted abbreviations don't capitalise the following word", () => {
  // The whole abbreviation is the last token here (its letters are typed as one run).
  assert.equal(shouldCapitalizeNext("w.r.t. ", cfg), false);
  assert.equal(shouldCapitalizeNext("e.g. ", cfg), false);
  assert.equal(shouldCapitalizeNext("a.k.a. ", cfg), false);
  assert.equal(shouldCapitalizeNext("etc. ", cfg), false);
  // A genuine sentence end still capitalises.
  assert.equal(shouldCapitalizeNext("done. ", cfg), true);
});

test("sanitizeForModel strips code/math/link/URL machinery, keeps prose", () => {
  const clean = (s: string) => tokenizeWordsCased(sanitizeForModel(s));
  assert.deepEqual(clean("run `foo_bar()` now"), ["run", "now"]);
  assert.deepEqual(clean("energy $E = mc^2$ is"), ["energy", "is"]);
  assert.deepEqual(clean("see [the docs](https://x.com/y) for"), ["see", "the", "docs", "for"]);
  assert.deepEqual(clean("a [[Target Page|display text]] here"), ["a", "display", "text", "here"]);
  assert.deepEqual(clean("visit https://example.com/path today"), ["visit", "today"]);
  // Decoration around real words is dropped by the tokeniser, leaving the words.
  assert.deepEqual(clean("this is **very** _important_"), ["this", "is", "very", "important"]);
  // An unclosed trailing code/math span (being typed) is dropped from its opener.
  assert.deepEqual(clean("the value is `foo"), ["the", "value", "is"]);
});
