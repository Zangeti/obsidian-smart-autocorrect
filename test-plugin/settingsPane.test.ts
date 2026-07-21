/**
 * Guards the settings pane description that BOTH render paths are built from.
 *
 * Obsidian 1.13 renders the pane from getSettingDefinitions() and never calls display(); older
 * versions only call display(). Since the pane is described once and rendered two ways, the
 * thing worth testing is the description: that every row still has a name and a control, that
 * the two paths agree, and that a control still reads and writes the settings object.
 *
 * `obsidian` is a types-only package with no runtime, so the module under test is bundled with
 * esbuild against a stub that records what each row does.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { build } from "esbuild";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

/** A stand-in for `Setting` that records the calls a row makes onto it. */
const STUB = `
export class Setting {
  constructor(el) { this.el = el; this.name = ""; this.desc = ""; this.controls = []; this.settingEl = el; if (el && el.__rows) el.__rows.push(this); }
  setName(v) { this.name = v; return this; }
  setDesc(v) { this.desc = v; return this; }
  setHeading() { this.heading = true; return this; }
  setClass() { return this; }
  add(kind, cb) { const c = component(kind, this); this.controls.push(c); cb(c); return this; }
  addToggle(cb) { return this.add("toggle", cb); }
  addSlider(cb) { return this.add("slider", cb); }
  addText(cb) { return this.add("text", cb); }
  addTextArea(cb) { return this.add("textarea", cb); }
  addDropdown(cb) { return this.add("dropdown", cb); }
  addButton(cb) { return this.add("button", cb); }
  addExtraButton(cb) { return this.add("extra", cb); }
}
function component(kind) {
  const c = { kind, value: undefined, handler: undefined };
  c.setValue = (v) => { c.value = v; return c; };
  c.getValue = () => c.value;
  c.onChange = (h) => { c.handler = h; return c; };
  c.onClick = (h) => { c.handler = h; return c; };
  c.setLimits = () => c; c.setDynamicTooltip = () => c; c.setPlaceholder = () => c;
  c.addOptions = (o) => { c.options = o; return c; };
  c.setButtonText = (t) => { c.text = t; return c; };
  c.setCta = () => c; c.setWarning = () => c; c.setDisabled = () => c; c.setTooltip = () => c;
  return c;
}
`;

const ENTRY = `
export { buildPredictiveSettingGroups, DEFAULT_PREDICTIVE_SETTINGS } from "../src/predictive/PredictiveSettings.ts";
export { toSettingDefinitions, renderPaneGroups } from "../src/predictive/settingsPane.ts";
export { Setting } from "obsidian";
`;

async function loadPane() {
  const dir = mkdtempSync(join(tmpdir(), "sa-settings-"));
  const stub = join(dir, "obsidian-stub.mjs");
  writeFileSync(stub, STUB);
  const out = join(dir, "pane.mjs");
  await build({
    stdin: { contents: ENTRY, resolveDir: import.meta.dirname, loader: "ts" },
    bundle: true,
    format: "esm",
    platform: "node",
    outfile: out,
    logLevel: "silent",
    plugins: [
      {
        name: "obsidian-stub",
        setup(b) {
          b.onResolve({ filter: /^obsidian$/ }, () => ({ path: stub }));
        },
      },
    ],
  });
  return (await import(pathToFileURL(out).href)) as {
    buildPredictiveSettingGroups: (...a: unknown[]) => PaneGroup[];
    DEFAULT_PREDICTIVE_SETTINGS: Record<string, unknown>;
    toSettingDefinitions: (g: PaneGroup[]) => Definition[];
    Setting: new (el: unknown) => StubSetting;
  };
}

interface PaneGroup {
  heading?: string;
  items: { kind: string; row?: { name: string; desc: string; apply: (s: unknown, w: boolean) => void } }[];
}
interface Definition {
  type: string;
  heading?: string;
  items: { name: string; desc?: string; render: (s: unknown) => void }[];
}

test("every settings row has a name, and names are unique", async () => {
  const pane = await loadPane();
  const settings = { ...pane.DEFAULT_PREDICTIVE_SETTINGS };
  const groups = pane.buildPredictiveSettingGroups(settings, () => {});
  const names = groups.flatMap((g) => g.items.filter((i) => i.kind === "row").map((i) => i.row!.name));

  assert.ok(names.length >= 30, `expected the full pane, got ${names.length} rows`);
  for (const n of names) assert.notEqual(n.trim(), "", "a row was left without a name");
  assert.equal(new Set(names).size, names.length, "two rows share a name, so search cannot tell them apart");
});

test("the declarative definitions carry the same rows as the pane groups", async () => {
  const pane = await loadPane();
  const settings = { ...pane.DEFAULT_PREDICTIVE_SETTINGS };
  const groups = pane.buildPredictiveSettingGroups(settings, () => {});
  const defs = pane.toSettingDefinitions(groups);

  const groupNames = groups.flatMap((g) => g.items.filter((i) => i.kind === "row").map((i) => i.row!.name));
  const defNames = defs.flatMap((d) => d.items.map((i) => i.name)).filter((n) => n !== "");
  for (const n of groupNames) assert.ok(defNames.includes(n), `"${n}" is missing from the definitions`);
  assert.ok(
    defs.every((d) => d.type === "group"),
    "every top-level definition should be a group",
  );
});

test("a row's control still reads and writes the settings object", async () => {
  const pane = await loadPane();
  const { Setting } = pane;
  const settings = { ...pane.DEFAULT_PREDICTIVE_SETTINGS, enablePredictions: true } as Record<string, unknown>;
  let saved = 0;
  const groups = pane.buildPredictiveSettingGroups(settings, () => {
    saved++;
  });

  const row = groups
    .flatMap((g) => g.items)
    .find((i) => i.kind === "row" && i.row!.name.startsWith("Predictive text"))?.row;
  assert.ok(row, "the predictive-text row disappeared from the pane");

  const s = new Setting(null);
  row.apply(s, true);
  assert.equal(s.name.startsWith("Predictive text"), true);
  assert.equal(s.controls.length, 1, "the row lost its control");
  assert.equal(s.controls[0].value, true, "the toggle did not read the current setting");

  s.controls[0].handler!(false);
  assert.equal(settings.enablePredictions, false, "the toggle did not write the setting back");
  assert.equal(saved, 1, "changing a setting no longer persists it");
});

interface StubSetting {
  name: string;
  controls: { value: unknown; handler?: (v: unknown) => void }[];
}
