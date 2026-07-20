import { test } from "node:test";
import assert from "node:assert/strict";

import { buildCountsFromText } from "../src/predictive/engine/ngram/build.ts";
import { InMemoryLanguageModel, NgramCounts } from "../src/predictive/engine/ngram/model.ts";

// A coherent, grammatical corpus (varied sentences, repeated enough for support).
const CORPUS = (
  `
I would like to thank you for your help with this project.
Please let me know if you have any questions about the schedule.
The meeting is scheduled for next week on Tuesday afternoon.
The central bank is expected to raise interest rates again.
Machine learning models are trained on large datasets of text.
He is going to the store to buy some milk and bread.
She was very happy with the result of the experiment.
We should consider the impact of these changes carefully.
Thank you for taking the time to review the document.
The report shows that sales increased during the last quarter.
`
).repeat(12);

function model(): InMemoryLanguageModel {
  const counts: NgramCounts = buildCountsFromText(CORPUS);
  return new InMemoryLanguageModel(counts);
}

// Replicates PredictiveEngineController.getSuggestions (next-word case):
// single-word list + a phrase only when it is a confident run.
function confidentPhrase(m: InMemoryLanguageModel, context: string[], maxWords: number, minProb: number): string[] | null {
  const words: string[] = [];
  let ctx = context.slice();
  for (let i = 0; i < maxWords; i++) {
    const top = m.predict(ctx, 1)[0];
    if (!top || Math.exp(top.logProb) < minProb) break;
    words.push(top.word);
    ctx = [...ctx, top.word];
  }
  return words.length >= 2 ? words : null;
}

function suggest(m: InMemoryLanguageModel, context: string[], k = 5, minProb = 0.4) {
  const singles = m.predict(context, k).map((s) => s.word);
  const phrase = confidentPhrase(m, context, 8, minProb);
  return { singles, phrase, top: phrase ? phrase.join(" ") : singles[0] };
}

// The set of words that ACTUALLY followed `context` in the corpus – a
// suggestion drawn from this set is grammatical by construction.
function observedContinuations(context: string[]): Set<string> {
  const toks = CORPUS.toLowerCase().match(/[a-z']+/g) ?? [];
  const out = new Set<string>();
  const n = context.length;
  for (let i = n; i < toks.length; i++) {
    let match = true;
    for (let j = 0; j < n; j++) if (toks[i - n + j] !== context[j]) { match = false; break; }
    if (match) out.add(toks[i]);
  }
  return out;
}

test("QUALITY: top single suggestion is a grammatical (observed) continuation", () => {
  const m = model();
  const contexts = [
    ["i", "would", "like"],
    ["please", "let", "me"],
    ["he", "is", "going"],
    ["thank", "you"],
    ["the", "central", "bank"],
    ["interest"],
    ["machine", "learning"],
    ["going", "to", "the"],
  ];
  for (const ctx of contexts) {
    const { singles } = suggest(m, ctx);
    const observed = observedContinuations(ctx);
    // The #1 suggestion must be something that genuinely follows this context.
    assert.ok(observed.has(singles[0]), `after "${ctx.join(" ")}" top="${singles[0]}" not a real continuation`);
  }
});

test("QUALITY: balance – confident context yields a phrase, ambiguous does not", () => {
  const m = model();
  // Strongly predictable runs -> multi-word phrase.
  assert.ok(suggest(m, ["i", "would", "like"]).phrase?.join(" ").startsWith("to thank you"));
  assert.ok(suggest(m, ["the", "central", "bank"]).phrase?.join(" ").startsWith("is expected to"));
  // Ambiguous single-word context -> NO phrase (many possible nouns after "the").
  assert.equal(suggest(m, ["the"]).phrase, null);
  assert.equal(suggest(m, ["to"]).phrase, null);
});

test("QUALITY: every word inside a surfaced phrase clears the confidence bar", () => {
  const m = model();
  const minProb = 0.4;
  for (const ctx of [["i", "would", "like"], ["the", "central", "bank"], ["thank", "you"]]) {
    const phrase = confidentPhrase(m, ctx, 8, minProb);
    if (!phrase) continue;
    // Walk it and confirm each step was >= minProb (i.e. grammatical & expected).
    let c = ctx.slice();
    for (const w of phrase) {
      const top = m.predict(c, 1)[0];
      assert.ok(Math.exp(top.logProb) >= minProb, `weak word "${w}" after "${c.join(" ")}"`);
      assert.equal(top.word, w);
      c = [...c, w];
    }
  }
});

test("QUALITY: raising the bar shrinks/removes phrases; lowering it grows them", () => {
  const m = model();
  const strict = confidentPhrase(m, ["i", "would", "like"], 8, 0.85);
  const loose = confidentPhrase(m, ["i", "would", "like"], 8, 0.3);
  assert.ok((loose?.length ?? 0) >= (strict?.length ?? 0));
});
