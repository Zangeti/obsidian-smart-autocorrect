// Vendor engine/src -> upstream/src/predictive/engine, stripping ".ts" from
// relative import/export specifiers (esbuild uses extensionless resolution).
import { readdirSync, statSync, readFileSync, writeFileSync, mkdirSync, rmSync } from "node:fs";
import { join, dirname, relative } from "node:path";
import { fileURLToPath } from "node:url";

// Paths relative to this script (upstream/vendor-engine.mjs).
const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = join(HERE, "..", "engine", "src");
const DST = join(HERE, "src", "predictive", "engine");
// Tests come along too, so a clone of this repo can run them. Their imports point at
// engine/src, which lands under src/predictive/engine here.
const TEST_SRC = join(HERE, "..", "engine", "test");
const TEST_DST = join(HERE, "test");

function walk(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) out.push(...walk(p));
    else if (p.endsWith(".ts")) out.push(p);
  }
  return out;
}

// Keep ".ts" on relative specifiers. esbuild and tsgo both resolve them (with
// allowImportingTsExtensions), and keeping them is what lets the vendored engine
// ALSO be imported by `node --experimental-strip-types` in test/, which requires
// fully-specified ESM paths. One copy of the source, usable by build and tests.
function transform(code) {
  return code;
}

rmSync(DST, { recursive: true, force: true });
for (const file of walk(SRC)) {
  const rel = relative(SRC, file);
  const dst = join(DST, rel);
  mkdirSync(dirname(dst), { recursive: true });
  writeFileSync(dst, transform(readFileSync(file, "utf8")));
}
rmSync(TEST_DST, { recursive: true, force: true });
let tests = 0;
try {
  for (const file of walk(TEST_SRC)) {
    const dst = join(TEST_DST, relative(TEST_SRC, file));
    mkdirSync(dirname(dst), { recursive: true });
    // "../src/x.ts" (engine/test -> engine/src) becomes the vendored path.
    writeFileSync(dst, readFileSync(file, "utf8").replace(/(["'])\.\.\/src\//g, "$1../src/predictive/engine/"));
    tests++;
  }
} catch {
  // engine/test only exists in the development monorepo; a plain clone keeps the
  // already-vendored copy.
}
console.log("vendored", walk(SRC).length, "files ->", DST, "+", tests, "tests");
