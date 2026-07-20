/**
 * Non-breaking prefixes / abbreviations: a period after one of these tokens is
 * (usually) NOT a sentence boundary. Used by sentence splitting and by
 * auto-capitalisation so we don't capitalise after "vs.", "U.S.", "e.g." etc.
 *
 * Stored lower-cased and without the trailing period.
 */
export const DEFAULT_ABBREVIATIONS: string[] = [
  // Latin / editorial
  "e.g", "i.e", "etc", "vs", "cf", "al", "ca", "viz", "nb", "vol", "ed", "eds",
  "pp", "p", "fig", "figs", "no", "nos", "op", "cit", "ibid",
  // Titles
  "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "rev", "hon", "gen", "col",
  "capt", "lt", "sgt", "gov", "sen", "rep", "pres",
  // Time / calendar
  "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov",
  "dec", "mon", "tue", "wed", "thu", "fri", "sat", "sun",
  // Measures / business
  "inc", "ltd", "co", "corp", "dept", "est", "approx", "min", "max", "avg",
  "misc", "dept", "univ", "assn", "bros",
  // Common acronym-with-dots handled separately (U.S., U.K., U.N., a.m., p.m.)
  "u.s", "u.k", "u.n", "a.m", "p.m", "e.u", "d.c",
  // Multi-part abbreviations that are NOT sentence ends - so the word AFTER them is
  // not wrongly capitalised ("w.r.t. the plan" keeps "the" lowercase).
  "w.r.t", "wrt", "a.k.a", "aka", "resp", "et.al", "e.t.c", "esp", "incl", "excl",
  "u.s.a", "a.d", "b.c", "b.c.e", "c.e", "ph.d", "b.a", "m.a", "b.sc", "m.sc",
];

export function buildAbbreviationSet(extra: string[] = []): Set<string> {
  const s = new Set<string>();
  for (const a of DEFAULT_ABBREVIATIONS) s.add(a.toLowerCase());
  for (const a of extra) s.add(a.toLowerCase().replace(/\.$/, ""));
  return s;
}
