import { test } from "node:test";
import assert from "node:assert/strict";
import { pathExcluded, parseExcludeList } from "../src/predictive/engine/text/exclude.ts";

test("folder prefix excludes everything beneath it", () => {
  assert.equal(pathExcluded("Templates/daily.md", ["Templates"]), true);
  assert.equal(pathExcluded("Templates/sub/x.md", ["Templates"]), true);
  assert.equal(pathExcluded("Templates", ["Templates"]), true);
  // not a partial-name match
  assert.equal(pathExcluded("Templates-archive/x.md", ["Templates"]), false);
});

test("exact file path", () => {
  assert.equal(pathExcluded("Journal/2026.md", ["Journal/2026.md"]), true);
  assert.equal(pathExcluded("Journal/2027.md", ["Journal/2026.md"]), false);
});

test("globs: * within a segment, ** across segments", () => {
  assert.equal(pathExcluded("Drawing.excalidraw.md", ["*.excalidraw.md"]), true);
  assert.equal(pathExcluded("a/b/secret/notes.md", ["**/secret/**"]), true);
  assert.equal(pathExcluded("secret/notes.md", ["**/secret/**"]), true);
  assert.equal(pathExcluded("Journal/x.md", ["Journal/*"]), true);
  assert.equal(pathExcluded("Journal/sub/x.md", ["Journal/*"]), false); // * doesn't cross /
});

test("case-insensitive and leading-slash tolerant", () => {
  assert.equal(pathExcluded("/templates/x.md", ["Templates"]), true);
});

test("no patterns → never excluded", () => {
  assert.equal(pathExcluded("anything.md", []), false);
});

test("parseExcludeList splits on newline and comma, trims blanks", () => {
  assert.deepEqual(parseExcludeList("Templates\n Journal , *.excalidraw.md\n\n"), [
    "Templates",
    "Journal",
    "*.excalidraw.md",
  ]);
});
