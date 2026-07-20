/**
 * "Tidy double spaces": when a word is completed after a double space, collapse the
 * pair to one.
 *
 * Deliberately narrow. Two spaces before the word you are completing is almost always
 * a slip; three or more is a choice (alignment, ASCII art), and a tab is indentation.
 * So this only ever removes ONE space, only when there are EXACTLY two, and never
 * looks at anything but the two characters before the cursor - it completes a word, it
 * is not a line reformatter.
 */

/**
 * Where the replacement should START so that completing a word at `ch` eats one
 * redundant space - or `ch` itself when there is nothing to tidy.
 *
 * Returning a START POSITION rather than performing an edit is what keeps the whole
 * completion a single undo step: the caller folds this into the range it was already
 * replacing. A separate tidy-up edit would make Ctrl-Z restore the double space while
 * leaving the completion, which reads as the editor fighting the user.
 */
export function doubleSpaceStart(line: string, ch: number): number {
  if (ch < 2) return ch; // nothing (or only indentation) before the cursor
  if (line[ch - 1] !== " " || line[ch - 2] !== " ") return ch;
  if (ch >= 3 && line[ch - 3] === " ") return ch; // 3+ spaces: intentional layout
  // A line that is ONLY spaces up to here is indentation, not a double-space slip.
  if (line.slice(0, ch).trim() === "") return ch;
  return ch - 1;
}
