/**
 * Broad, behaviour-level accuracy scenarios for the autocorrect decision.
 *
 * The point of this file is NOT to lock in the current thresholds - it is to pin the
 * INTUITIVE behaviour a phone-style autocorrect should have across many situations, so a
 * change that improves one case and quietly regresses others is caught. Every case is
 * expressed the way a user would describe it ("an obvious typo of a common word should be
 * fixed"; "a word I actually meant should be left alone"), and driven through the real
 * decision with the SAME configuration the plugin ships (split/join, real-word homophone
 * peer, phonetic + keyboard channels) - no hardcoded word lists, no per-word special casing.
 *
 * The model here is built from a few paragraphs, so it is far weaker than the shipped 57MB
 * LSTM; every intended word therefore has to appear in the corpus below (the corpus IS the
 * model's whole vocabulary). That is a property of the test, not a limit of the engine.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { Engine } from "../src/predictive/engine/index.ts";
import { decideCorrection } from "../src/predictive/engine/autocorrect/autocorrect.ts";
import { RealWordCorrector } from "../src/predictive/engine/predict/segmentation.ts";

const PROSE = `
I think we should meet at the office tomorrow morning to discuss the project.
The government announced a new policy that will affect the whole country and its people.
She received the letter yesterday and separated the pages into two neat piles.
They went to their house because there was a party there that evening.
The weather was beautiful and everyone enjoyed the afternoon in the garden.
He definitely wanted to finish the report before the meeting on Friday.
Please remember to bring your notebook and a pen to the lecture as well.
We are going to the beach if the weather is good this weekend.
The teacher explained the difficult problem again for the whole class.
Business was slow, so the company decided to lower its prices for a while.
The children were tired after the long walk through the forest.
It is easy to lose a small key, so keep it somewhere safe and do not lose it.
The restaurant serves excellent food and the service is always friendly.
I would like to buy a new computer because my old one is too slow.
The stock market rose sharply today after the central bank cut interest rates.
Investors bought more shares this morning and sold them again in the afternoon.
The doctor said that exercise and a good diet are important for your health.
We watched a wonderful film last night and then talked about it for hours.
A lot of people believe that reading is the best way to learn new things.
The manager thanked everyone for their hard work during a very busy year.
I believe you are right and their friend is weird but very committed to it.
We should have gone across the bridge as well, not in front of the station.
You are welcome to come with us if you want to, it would not be a problem.
We must receive and separate the commitment papers that occurred across the office.
`.repeat(8);

const eng = Engine.fromText(PROSE);
const realWord = new RealWordCorrector(eng.model.vocabulary());

// The plugin's shipped configuration (see EngineCore.correctionDecision): split/join on, the
// real-word homophone peer wired in, keyboard + phonetic channels at full strength, and the
// curated lexicality test answered from the model vocab (mirroring isRealWord).
const cfg = {
  infoGainThreshold: 2.5,
  beta: 1.5,
  fuzzyStrength: 1.0,
  phoneticStrength: 1.0,
  enableSplit: true,
  realWord,
  isRealWord: (w: string) => eng.model.hasWord(w),
};

function decide(typed: string, context: string[]) {
  return decideCorrection(eng.model, eng.index, typed, context, cfg);
}

test("obvious non-word typos of common words are corrected aggressively", () => {
  // Transpositions, doubled/dropped letters, adjacent-key slips and the classic i-before-e
  // misspellings - the bread and butter of "fix words that aren't in the dictionary".
  const cases: [string, string, string[]][] = [
    ["teh", "the", ["at"]],
    ["adn", "and", ["bread"]],
    ["hte", "the", ["at"]],
    ["recieve", "receive", ["should"]],
    ["seperate", "separate", ["and"]],
    ["definately", "definitely", ["he"]],
    ["tommorow", "tomorrow", ["office"]],
    ["goverment", "government", ["the"]],
    ["becuase", "because", ["slow"]],
    ["accross", "across", ["gone"]],
    ["freind", "friend", ["their"]],
    ["beleive", "believe", ["i"]],
    ["wierd", "weird", ["is"]],
    ["thier", "their", ["for"]],
  ];
  const missed: string[] = [];
  for (const [typed, want, ctx] of cases) {
    const d = decide(typed, ctx);
    if (!(d.correct && d.to.toLowerCase() === want)) missed.push(`${typed}->${d.correct ? d.to : "(kept)"} [${d.reason}]`);
  }
  assert.equal(missed.length, 0, `uncorrected typos: ${missed.join(", ")}`);
});

test("missing-apostrophe contractions are restored", () => {
  const cases: [string, string, string[]][] = [
    ["dont", "don't", ["i"]],
    ["cant", "can't", ["i"]],
    ["wouldnt", "wouldn't", ["it"]],
    ["youre", "you're", ["if"]],
  ];
  const missed: string[] = [];
  for (const [typed, want, ctx] of cases) {
    const d = decide(typed, ctx);
    if (!(d.correct && d.to.toLowerCase() === want)) missed.push(`${typed}->${d.correct ? d.to : "(kept)"}`);
  }
  assert.equal(missed.length, 0, `contractions not restored: ${missed.join(", ")}`);
});

test("run-together words are re-spaced (split/join)", () => {
  const cases: [string, string, string[]][] = [
    ["alot", "a lot", ["is"]],
    ["infront", "in front", ["not"]],
    ["aswell", "as well", ["us"]],
  ];
  const missed: string[] = [];
  for (const [typed, want, ctx] of cases) {
    const d = decide(typed, ctx);
    if (!(d.correct && d.to.toLowerCase() === want)) missed.push(`${typed}->${d.correct ? d.to : "(kept)"}`);
  }
  assert.equal(missed.length, 0, `not re-spaced: ${missed.join(", ")}`);
});

test("correctly-typed common words are never touched", () => {
  const words: [string, string[]][] = [
    ["the", ["at"]], ["government", ["the"]], ["received", ["she"]],
    ["beautiful", ["was"]], ["computer", ["new"]], ["restaurant", ["the"]],
    ["important", ["are"]], ["everyone", ["and"]], ["through", ["walk"]],
    ["commitment", ["the"]], ["separate", ["and"]], ["receive", ["must"]],
  ];
  const harmed: string[] = [];
  for (const [w, ctx] of words) {
    const d = decide(w, ctx);
    if (d.correct) harmed.push(`${w}->${d.to} (${d.reason})`);
  }
  assert.equal(harmed.length, 0, `corrupted real words: ${harmed.join(", ")}`);
});

test("correctly-typed homophones are never corrupted on frequency alone", () => {
  // The decision must never break a correctly-typed homophone just because its twin is
  // commoner - that requires genuine contextual evidence, which these positions don't give.
  const pairs: [string, string[]][] = [
    ["their", ["for"]], ["there", ["party"]], ["your", ["bring"]],
    ["its", ["lower"]], ["lose", ["to"]], ["believe", ["i"]],
    ["friend", ["their"]], ["weird", ["is"]], ["across", ["gone"]],
    ["want", ["you"]], ["would", ["it"]], ["welcome", ["are"]],
  ];
  const harmed: string[] = [];
  for (const [w, ctx] of pairs) {
    const d = decide(w, ctx);
    if (d.correct) harmed.push(`${w}->${d.to}`);
  }
  assert.equal(harmed.length, 0, `homophones corrupted: ${harmed.join(", ")}`);
});

test("acronyms and their plurals are left alone", () => {
  for (const w of ["SEC", "NASA", "HTTP", "CEO", "CMOs", "PhDs", "URLs", "IDs"]) {
    const d = decide(w, ["the"]);
    assert.equal(d.correct, false, `${w} should not be corrected, got ${d.to}`);
  }
});

test("capitalised out-of-vocabulary tokens (proper nouns) are kept", () => {
  for (const w of ["Nakamoto", "Zurich", "Dnipro", "Kubernetes", "Anthropic"]) {
    const d = decide(w, ["in"]);
    assert.equal(d.correct, false, `proper noun ${w} should be kept, got ${d.to}`);
  }
});

test("deliberate gibberish is not 'corrected' to an unrelated word", () => {
  for (const w of ["asdfgh", "qwerty", "zxcvbnm", "blargh"]) {
    const d = decide(w, ["the"]);
    assert.equal(d.correct, false, `${w} should be left alone, got ${d.to} (${d.reason})`);
  }
});

test("short intentional words are not over-corrected", () => {
  for (const [w, ctx] of [["to", ["go"]], ["is", ["it"]], ["of", ["lot"]], ["we", ["and"]]] as [string, string[]][]) {
    const d = decide(w, ctx);
    assert.equal(d.correct, false, `${w} should be kept, got ${d.to}`);
  }
});
