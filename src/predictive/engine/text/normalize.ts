/**
 * Diacritic folding and light morphological stemming (part of #D9).
 *
 *  - foldDiacritics: "café" -> "cafe", "naïve" -> "naive", "Straße" -> "strasse".
 *    Used so accented spellings match their ASCII forms cheaply.
 *  - stem: a conservative suffix stripper (plural / -ing / -ed / -ly) with a
 *    doubled-consonant fix so "running"/"runs" -> "run", letting inflected forms
 *    share statistics and match each other.
 */

const SPECIAL: Record<string, string> = {
  ß: "ss",
  æ: "ae",
  œ: "oe",
  ø: "o",
  đ: "d",
  ł: "l",
  þ: "th",
};

export function foldDiacritics(s: string): string {
  let out = s.toLowerCase();
  out = out.replace(/[ßæœøđłþ]/g, (c) => SPECIAL[c] ?? c);
  // Decompose and strip combining marks.
  return out.normalize("NFD").replace(/[̀-ͯ]/g, "");
}

function undouble(w: string): string {
  // "runn" -> "run", "stopp" -> "stop"
  if (w.length >= 3 && w[w.length - 1] === w[w.length - 2] && !"aeiou".includes(w[w.length - 1])) {
    return w.slice(0, -1);
  }
  return w;
}

export function stem(word: string): string {
  let w = foldDiacritics(word);
  if (w.length <= 3) return w;
  if (w.endsWith("ies") && w.length > 4) return w.slice(0, -3) + "y";
  if (w.endsWith("sses")) return w.slice(0, -2);
  if (w.endsWith("ing") && w.length > 5) return undouble(w.slice(0, -3));
  if (w.endsWith("edly")) return undouble(w.slice(0, -4));
  if (w.endsWith("ed") && w.length > 4) return undouble(w.slice(0, -2));
  if (w.endsWith("ly") && w.length > 4) return w.slice(0, -2);
  if (w.endsWith("es") && w.length > 4) return w.slice(0, -2);
  if (w.endsWith("s") && !w.endsWith("ss") && w.length > 3) return w.slice(0, -1);
  return w;
}
