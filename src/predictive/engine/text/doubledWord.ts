/**
 * Accidental doubled-word detection ("the the" -> "the").
 *
 * Only a curated set of function words is ever removed, because many doublings are
 * grammatically valid and must be left alone: "had had" (past perfect), "that that"
 * ("I know that that is true"), "who who", "will will" (a name). The set below holds
 * articles, prepositions, conjunctions, the copula, and pronouns/demonstratives that
 * have no valid doubled use - so removal is safe and precise rather than a blunt
 * "collapse any repeat".
 */
export const NEVER_DOUBLED: ReadonlySet<string> = new Set([
  // articles
  "the", "a", "an",
  // conjunctions
  "and", "or", "nor", "but", "than", "then",
  // prepositions
  "of", "to", "in", "on", "at", "for", "with", "from", "into", "onto", "as", "by",
  // copula / auxiliaries with no valid doubling
  "is", "are", "was", "were", "be", "am",
  // pronouns / demonstratives
  "it", "its", "we", "they", "this", "these", "those", "i",
]);

/**
 * True when `token` is an accidental duplicate of the word immediately before it and
 * safe to remove. Case-insensitive; both words must match and be in NEVER_DOUBLED.
 */
export function isDoubledWord(prevWord: string, token: string): boolean {
  if (!prevWord || !token) return false;
  const a = prevWord.toLowerCase();
  const b = token.toLowerCase();
  return a === b && NEVER_DOUBLED.has(b);
}
