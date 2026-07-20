import { test } from "node:test";
import assert from "node:assert/strict";

import { buildModelFromText } from "../src/predictive/engine/ngram/build.ts";
import { serializeCounts, deserializeCounts, modelFromSerialized } from "../src/predictive/engine/ngram/serialize.ts";
import { NgramCounts } from "../src/predictive/engine/ngram/model.ts";
import { buildSerializedCountsFromDocs } from "../src/predictive/engine/ngram/workerKernel.ts";
import { splitSentences } from "../src/predictive/engine/text/tokenize.ts";

// Rebuild NgramCounts from text to serialize (build.ts hides counts, so mirror it).
function countsFromText(text: string): NgramCounts {
  const c = new NgramCounts();
  for (const s of splitSentences(text)) if (s.length) c.addSentence(s);
  return c;
}

test("serialize/deserialize round-trips counts and predictions", () => {
  const text = "the central bank raised rates. the central bank cut rates. the stock market rose.".repeat(6);
  const counts = countsFromText(text);
  const s = serializeCounts(counts);
  // structured-clone simulation
  const s2 = JSON.parse(JSON.stringify(s));
  const model = modelFromSerialized(s2);
  const direct = buildModelFromText(text);
  const a = model.predict(["the", "central", "bank"], 2).map((x) => x.word);
  const b = direct.predict(["the", "central", "bank"], 2).map((x) => x.word);
  assert.deepEqual(a, b);
  // deserializeCounts reconstructs the same vocab size
  assert.equal(deserializeCounts(s2).vocab.length, counts.vocab.length);
});

test("worker kernel produces a usable model (matches builder behaviour)", () => {
  const docs = [
    "The central bank raised interest rates. The central bank cut interest rates.",
    "The stock market rose sharply. The stock market fell.",
  ];
  const s = buildSerializedCountsFromDocs(docs);
  const model = modelFromSerialized(JSON.parse(JSON.stringify(s)));
  const preds = model.predict(["the", "central", "bank"], 3).map((x) => x.word);
  assert.ok(preds.includes("raised") || preds.includes("cut"));
  const stock = model.predict(["the", "stock"], 3).map((x) => x.word);
  assert.ok(stock.includes("market"));
});

test("worker kernel is self-contained (stringifiable, no external refs)", () => {
  // .toString() must be runnable in a Worker: reconstruct and call it.
  const src = buildSerializedCountsFromDocs.toString();
  const fn = new Function(`return (${src})`)() as typeof buildSerializedCountsFromDocs;
  const s = fn(["the cat sat. the cat ran."]);
  assert.ok(s.vocab.includes("cat"));
  assert.ok(s.tri.length > 0);
});
