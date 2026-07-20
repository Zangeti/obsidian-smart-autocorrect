/**
 * How suggestions should be CASED before the user ever sees them.
 *
 * Case is a property of the POSITION (and of what the user is physically doing) -
 * never of whichever channel produced the candidate. Deciding it here fixes two real
 * bugs: suggestions used to be shown in whatever case the model returned and only got
 * capitalised by autocorrect AFTER the word was committed (so an empty document
 * offered "the", you accepted "the", and it silently became "The"), and a sentence
 * start could offer "The", "there", "Their" at once because the LSTM, n-gram and
 * fuzzy channels disagreed. One decision, applied to the whole list.
 */
import { shouldCapitalizeNext, capitalizeFirst, type SentenceCaseConfig } from "./sentenceCase.ts";

/** The transform to apply to a suggestion. */
export type SuggestionCase = "none" | "title" | "upper";

export interface SuggestionCaseOptions {
  /** The user is writing in CAPS here (see upperFromText). Overrides position. */
  upper?: boolean;
}

/**
 * Is the user writing in CAPS at this point in the text?
 *
 * A pure function of the text before the cursor - deliberately NOT a stateful "mode"
 * and not a Shift listener. That is what makes it rigorous: deleting a character
 * simply re-evaluates the shorter text, so "THe" (off) going back to "TH" turns caps
 * on again with no bookkeeping to get stale, and a mode can never survive an edit that
 * removed the evidence for it. It also sees CAPS LOCK, which no Shift listener can.
 *
 * Caps mode is on only when the user is ACTIVELY writing caps, evidenced by EITHER:
 *   - 2+ capitals in the word adjacent to the cursor (they are shouting right now), OR
 *   - a RUN of 2+ consecutive all-caps words (sustained caps carried across a boundary,
 *     so after "THE QUICK " the next suggestion is already caps with nothing typed yet).
 * A single all-caps token in otherwise-lowercase text is an ACRONYM ("...is NASA "), not
 * shouting, and must NOT flip the following suggestions to caps - that is what produced
 * an uppercased next-word suggestion after a plain acronym. An acronym that sits INSIDE a
 * caps run still counts (it is part of the run), so the run continues until the first
 * actual lowercase letter, which always turns caps off immediately.
 */
export function upperFromText(textBeforeCursor: string): boolean {
  const text = textBeforeCursor;
  const isLetter = (c: string) => (c >= "A" && c <= "Z") || (c >= "a" && c <= "z");
  let i = text.length - 1;

  // Read the trailing word (maximal letter run) ending at `i`, skipping separators
  // before it; advances `i` past it. `lower`/`caps` count each kind of letter.
  const readWord = (): { caps: number; lower: number; saw: boolean } => {
    while (i >= 0 && !isLetter(text[i])) i--; // skip spaces/digits/punctuation
    let caps = 0, lower = 0, saw = false;
    while (i >= 0 && isLetter(text[i])) {
      saw = true;
      if (text[i] <= "Z") caps++;
      else lower++;
      i--;
    }
    return { caps, lower, saw };
  };

  // If the last character is a letter the cursor sits mid-word - that word is the one
  // being typed right now.
  const midWord = i >= 0 && isLetter(text[i]);
  const w0 = readWord();
  if (!w0.saw || w0.lower > 0) return false; // no word, or a lowercase letter: caps are over
  if (midWord && w0.caps >= 2) return true;  // 2+ caps in the word being typed: shouting now
  // Count a RUN of consecutive all-caps words. A completed nearest word counts itself; a
  // partially-typed one does not (its <2 caps already failed above) - it only rides on a
  // run the PRECEDING completed words establish, so "THE QUICK B" stays caps while a lone
  // "NASA " (one caps word in lowercase text) does not.
  let capsWords = midWord ? 0 : 1;
  for (;;) {
    const w = readWord();
    if (!w.saw || w.lower > 0) break; // a lowercase word (or no more words): the run ends
    if (++capsWords >= 2) return true;
  }
  return false;
}

/**
 * What case suggestions take here.
 *
 * `upper` is deliberate evidence from the text, so it overrides position: at a sentence
 * start "THE" is both an opener and a shout, and upper is right for both.
 */
export function suggestionCase(
  precedingText: string,
  cfg: SentenceCaseConfig,
  opts: SuggestionCaseOptions = {},
): SuggestionCase {
  if (opts.upper) return "upper";
  // Capitalise real sentence starts, including the start of the document.
  return shouldCapitalizeNext(precedingText, cfg) ? "title" : "none";
}

/**
 * Apply a case decision to suggestion text.
 *
 * "title" capitalises only the FIRST word: a suggestion may be a whole phrase
 * ("the same year"), and a sentence start capitalises the sentence, not every word.
 */
export function applySuggestionCase(text: string, mode: SuggestionCase): string {
  if (!text) return text;
  if (mode === "upper") return text.toUpperCase();
  if (mode === "title") return capitalizeFirst(text);
  return text;
}
