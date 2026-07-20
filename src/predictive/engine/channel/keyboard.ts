/**
 * Keyboard error model. Per-operation costs (nats) for the weighted edit
 * distance. Substitution uses an empirical confusion table when available,
 * otherwise a physical key-distance prior - now for a SELECTABLE layout
 * (QWERTY / QWERTZ / AZERTY / Dvorak), so non-US typists get accurate geometry.
 */
import { foldDiacritics } from "../text/normalize.ts";

export type KeyboardLayoutName = "qwerty" | "qwertz" | "azerty" | "dvorak";

const LAYOUTS: Record<KeyboardLayoutName, string[]> = {
  qwerty: ["qwertyuiop", "asdfghjkl", "zxcvbnm"],
  qwertz: ["qwertzuiop", "asdfghjkl", "yxcvbnm"],
  azerty: ["azertyuiop", "qsdfghjklm", "wxcvbn"],
  dvorak: ["pyfgcrl", "aoeuidhtns", "qjkxbmwvz"],
};

const ROW_OFFSET = [0, 0.25, 0.75]; // horizontal stagger per row

const layoutCache = new Map<KeyboardLayoutName, Map<string, { x: number; y: number }>>();

function keyPos(layout: KeyboardLayoutName): Map<string, { x: number; y: number }> {
  let m = layoutCache.get(layout);
  if (m) return m;
  m = new Map();
  const rows = LAYOUTS[layout] ?? LAYOUTS.qwerty;
  rows.forEach((row, r) => {
    for (let c = 0; c < row.length; c++) m!.set(row[c], { x: c + ROW_OFFSET[r], y: r });
  });
  layoutCache.set(layout, m);
  return m;
}

export function keyDistance(a: string, b: string, layout: KeyboardLayoutName = "qwerty"): number {
  const pos = keyPos(layout);
  const pa = pos.get(a);
  const pb = pos.get(b);
  if (!pa || !pb) return 3; // unknown key: far
  return Math.hypot(pa.x - pb.x, pa.y - pb.y);
}

export interface ChannelConfig {
  sigma: number;
  deleteCost: number;
  insertCost: number;
  transposeCost: number;
  caseCost: number;
  /** physical keyboard layout for the geometric substitution prior. */
  layout: KeyboardLayoutName;
  /** empirical substitution costs keyed "x>y" (nats), overrides geometry. */
  subCost?: Map<string, number>;
}

export const DEFAULT_CHANNEL: ChannelConfig = {
  sigma: 1.0,
  deleteCost: 2.3,
  insertCost: 2.3,
  transposeCost: 1.5,
  caseCost: 0.05,
  layout: "qwerty",
};

export function substitutionCost(intended: string, typed: string, cfg: ChannelConfig): number {
  if (intended === typed) return 0;
  const li = intended.toLowerCase();
  const lt = typed.toLowerCase();
  if (li === lt) return cfg.caseCost; // same letter, different case
  // Diacritic difference only (é/e, ñ/n) is nearly free.
  if (foldDiacritics(li) === foldDiacritics(lt)) return cfg.caseCost * 1.5;

  const emp = cfg.subCost?.get(li + ">" + lt);
  if (emp !== undefined) return emp;

  const d = keyDistance(li, lt, cfg.layout);
  return (d / cfg.sigma) * (d / cfg.sigma) + 0.5;
}
