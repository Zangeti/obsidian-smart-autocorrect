/**
 * Fraction glyphs: turn a plain "1/2" into "½" when a single Unicode fraction exists. Pure and
 * unit-tested; the plugin runs it on a word boundary, gated by a setting. Only the fractions Unicode
 * actually has a precomposed glyph for are converted, and a date-like "1/2/2024" is never touched.
 */

const FRACTIONS: Record<string, string> = {
  "1/2": "½", "1/3": "⅓", "2/3": "⅔", "1/4": "¼", "3/4": "¾",
  "1/5": "⅕", "2/5": "⅖", "3/5": "⅗", "4/5": "⅘",
  "1/6": "⅙", "5/6": "⅚", "1/7": "⅐", "1/8": "⅛", "3/8": "⅜", "5/8": "⅝", "7/8": "⅞",
  "1/9": "⅑", "1/10": "⅒",
};

/**
 * If `before` ends in a simple "N/M" fraction that has a precomposed glyph, return the slice to
 * replace (from `start` to the end) and the glyph. Returns null otherwise. The fraction must not be
 * glued to another digit or slash (so "1/2/2024" and "10/2" tails are left alone) and must sit on a
 * word boundary.
 */
export function fractionGlyph(before: string): { start: number; text: string } | null {
  const m = before.match(/(?:^|[^\w/])(\d{1,2}\/\d{1,2})$/);
  if (!m) return null;
  const glyph = FRACTIONS[m[1]];
  if (!glyph) return null;
  return { start: before.length - m[1].length, text: glyph };
}
