/**
 * Numeric-suffix tidying for the autocorrect boundary: fix a wrong ordinal ending ("21th" -> "21st")
 * and drop the stray apostrophe in a decade ("1930's" -> "1930s"). A CORRECT ordinal or a plain
 * decade is left exactly as written. Pure and unit-tested.
 */

/** The correct ordinal suffix for a number, from English rules (11-13 are the "th" exceptions). */
export function ordinalSuffix(numeric: string): "st" | "nd" | "rd" | "th" {
  const lastTwo = parseInt(numeric.slice(-2), 10);
  if (lastTwo >= 11 && lastTwo <= 13) return "th";
  switch (numeric.slice(-1)) {
    case "1": return "st";
    case "2": return "nd";
    case "3": return "rd";
    default: return "th";
  }
}

/**
 * If `before` ends in a number with the WRONG ordinal suffix, or a decade written with an
 * apostrophe, return the slice to replace (from `start` to the end) and its corrected text. Returns
 * null when the ending is already correct or the number is glued to a letter.
 */
export function fixNumericSuffix(before: string): { start: number; text: string } | null {
  const ord = before.match(/(\d+)(st|nd|rd|th)$/i);
  if (ord) {
    const correct = ordinalSuffix(ord[1]);
    if (ord[2].toLowerCase() === correct) return null; // already right
    const start = before.length - ord[0].length;
    if (!boundaryOk(before, start)) return null;
    const cased = ord[2] === ord[2].toUpperCase() ? correct.toUpperCase() : correct;
    return { start, text: ord[1] + cased };
  }
  const decade = before.match(/(\d{2,4})'s$/);
  if (decade) {
    const start = before.length - decade[0].length;
    if (!boundaryOk(before, start)) return null;
    return { start, text: decade[1] + "s" };
  }
  return null;
}

/** The char before the expression must not be a letter or digit (so we don't slice mid-token). */
function boundaryOk(before: string, start: number): boolean {
  if (start <= 0) return true;
  return !/[A-Za-z0-9]/.test(before[start - 1]);
}
