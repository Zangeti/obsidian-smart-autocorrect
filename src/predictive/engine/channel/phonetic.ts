/**
 * Phonetic channel path (#3). A compact Metaphone-style key collapses words to
 * how they sound, so cognitive/spelling errors that geometry can't explain
 * ("fone"->"phone", "nite"->"night", "seperate"->"separate", "definately"->
 * "definitely") become cheap. The predictor takes min(channelCost, phoneticCost).
 */
import { structuralEdit } from "./editDistance.ts";

const VOWELS = new Set(["A", "E", "I", "O", "U"]);

/** Reduced phonetic key. Not full Double Metaphone, but effective and cheap. */
export function phoneticKey(word: string): string {
  let w = word.toUpperCase().replace(/[^A-Z]/g, "");
  if (!w) return "";

  // Leading silent clusters: drop the first (silent) letter.
  w = w.replace(/^(KN|GN|PN|WR|AE|PS)/, (m) => m.slice(1));
  if (w.startsWith("WH")) w = "W" + w.slice(2);
  if (w.startsWith("X")) w = "S" + w.slice(1);

  // Digraph rewrites (order matters).
  w = w
    .replace(/PH/g, "F")
    .replace(/GH/g, "")
    .replace(/CK/g, "K")
    .replace(/SCH/g, "SK")
    .replace(/SH/g, "X")
    .replace(/TH/g, "T")
    .replace(/CH/g, "X")
    .replace(/MB$/g, "M");

  const first = w[0];
  let out = first; // keep the first character verbatim (even if a vowel)
  for (let i = 1; i < w.length; i++) {
    const c = w[i];
    const next = w[i + 1] ?? "";
    if (VOWELS.has(c)) continue; // vowels dropped after the first char
    let mapped = c;
    switch (c) {
      case "C":
        mapped = "EIY".includes(next) ? "S" : "K";
        break;
      case "Q":
        mapped = "K";
        break;
      case "V":
        mapped = "F";
        break;
      case "Z":
        mapped = "S";
        break;
      case "G":
        mapped = "EIY".includes(next) ? "J" : "K";
        break;
      case "D":
        mapped = "T";
        break;
      case "X":
        mapped = "KS";
        break;
      case "W":
      case "Y":
      case "H":
        mapped = VOWELS.has(next) ? c : ""; // only kept before a vowel
        break;
      default:
        mapped = c;
    }
    out += mapped;
  }
  // collapse consecutive duplicates.
  return out.replace(/(.)\1+/g, "$1");
}

/**
 * Phonetic cost (nats) of `typed` vs `intended` = an estimate of −log P(typed |
 * intended) along the "sounds alike" channel. Defined only when the two words
 * genuinely sound the same - identical phonetic key - otherwise Infinity, so the
 * keyboard-geometry channel handles it instead. (A dropped/added consonant sound
 * changes the key, e.g. "worf"→WRF vs "were"→WR, so those are NOT sound-alikes:
 * that's the difference between a real homophone and a keyboard slip.)
 *
 * When the words do sound alike, the cost is PROPORTIONAL to how much the
 * spelling was actually distorted (structural edit distance), at a per-edit rate
 * cheaper than keyboard geometry - sounding alike discounts each edit, it doesn't
 * make the whole word free. So "definately"→"definitely" (1 edit) is cheap, while
 * a same-sounding but badly-mangled "peoppe"→"pop" (3 edits) is dearer than the
 * closer "people" and loses the posterior on its own - no edit-count cutoff.
 */
export function phoneticCost(typed: string, intended: string, perEdit = 0.9): number {
  const a = phoneticKey(typed);
  const b = phoneticKey(intended);
  if (!a || !b || a !== b) return Infinity; // must sound the same
  return perEdit * structuralEdit(typed, intended);
}
