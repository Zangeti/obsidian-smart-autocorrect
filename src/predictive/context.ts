/**
 * Shared context extraction used by the suggester, autocorrect, and ghost-text.
 *
 * Returns ONE array of the LSTM's own tokens: words plus sentence and clause
 * punctuation, exactly as the trainer tokenised the corpus. Both views the engine
 * needs are recoverable from it:
 *   - the LSTM takes it as-is and lowercases internally (its vocab is lowercase and
 *     casing comes back out of a case head) - the continuous punctuated stream it
 *     was trained on, spanning sentences, so it can SEE where a sentence ends and
 *     only expect a capitalised opener after a real terminator;
 *   - the n-gram (LowercaseModel) drops the punctuation and keeps only the words
 *     after the last terminator - the per-sentence context its counts were built from.
 *
 * The punctuation is what marks the boundaries, so no separate sentence-splitting
 * or marker token is needed here.
 *
 * NOT WINDOWED, and that is load-bearing. There used to be a `.slice(-32)` here. It
 * looked like a harmless bound but it was a performance BUG: an LSTM state cannot
 * forget its oldest word, so once a document passed the window the FIRST context word
 * changed on every keystroke, no cached state was ever a prefix of the next context,
 * and the engine cold-replayed the whole window on every keystroke. Letting the
 * context grow monotonically means the previous context is always a prefix of the next
 * one, which is what lets the model extend its state by ONE step (see prepare() in
 * engine/lstm/model.ts, and the waypoints that make cursor jumps cheap). Full-document
 * context and speed are the same change here, not a trade-off - and full context is
 * also the regime the model is trained in (stateful BPTT carries state across whole
 * passages; only the gradient is truncated).
 */
import { sanitizeForModel, tokenizeWordsCased } from "./engine/index";

export function contextWords(precedingText: string, _extraAbbreviations: string[]): string[] {
  // Strip code/math/link/URL machinery first so the model reads plain prose, like its
  // training data, rather than the letters buried inside `code`, $\latex$ or a URL path.
  return tokenizeWordsCased(sanitizeForModel(precedingText));
}
