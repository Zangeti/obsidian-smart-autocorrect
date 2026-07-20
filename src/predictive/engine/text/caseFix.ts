/**
 * Deterministic casing/contraction fixes that complement the statistical
 * autocorrect (they don't replace it):
 *
 *  - fixDoubleCapital: "THe" -> "The" (shift held a beat too long), leaving
 *    acronyms (NASA) and CamelCase (TypeScript) alone.
 *  - fixContraction:   "dont" -> "don't" - a curated failsafe for the common
 *    apostrophe cases, so they're guaranteed corrected AND never learned into
 *    the personal recommendation data. Only unambiguous non-words are listed.
 *
 * Proper-noun casing USED to live here as a static CaseMap. It was removed: the
 * LSTM's case head decides capitalisation in context (see engine/lstm/model.ts
 * renderCased), which a frequency map cannot do - it would capitalise "polish" in
 * "i will polish it" just because "Polish" is common.
 */

/** "THe" -> "The". Only the exact two-leading-capitals typo pattern. */
export function fixDoubleCapital(word: string): string {
  if (/^[A-Z][A-Z][a-z]+$/.test(word)) {
    return word[0] + word[1].toLowerCase() + word.slice(2);
  }
  return word;
}

/** Curated, unambiguous contractions (each key is NOT a valid word on its own). */
export const CONTRACTIONS: Record<string, string> = {
  dont: "don't", cant: "can't", wont: "won't", isnt: "isn't", arent: "aren't",
  wasnt: "wasn't", werent: "weren't", havent: "haven't", hasnt: "hasn't",
  hadnt: "hadn't", doesnt: "doesn't", didnt: "didn't", couldnt: "couldn't",
  wouldnt: "wouldn't", shouldnt: "shouldn't", mustnt: "mustn't", neednt: "needn't",
  im: "I'm", ive: "I've", youre: "you're", youve: "you've", youll: "you'll",
  theyre: "they're", theyve: "they've", theyll: "they'll", weve: "we've",
  wouldve: "would've", couldve: "could've", shouldve: "should've",
  thats: "that's", whats: "what's", hes: "he's", shes: "she's", whos: "who's",
  wheres: "where's", theres: "there's", heres: "here's",
  // "would"-elisions. Each key is a non-word (unlike "wed"/"hed"/"shed"/"id", which are real
  // words and so must never be rewritten), which is what makes them safe to fix outright.
  youd: "you'd", theyd: "they'd", whod: "who'd", itd: "it'd", thered: "there'd",
  wholl: "who'll", itll: "it'll",
};

/** Returns the corrected contraction for `word`, or null if none applies. */
export function fixContraction(word: string): string | null {
  const fixed = CONTRACTIONS[word.toLowerCase()];
  return fixed && fixed.toLowerCase() !== word.toLowerCase() ? fixed : null;
}

