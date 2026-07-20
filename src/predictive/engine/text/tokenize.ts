/**
 * Lightweight, allocation-conscious tokenizer + abbreviation-aware sentence
 * boundary detection. Shared by the offline model builder, the runtime context
 * extractor, and the auto-capitalisation module so all three agree on what a
 * "word" and a "sentence start" are.
 */
import { buildAbbreviationSet } from "./abbreviations.ts";

/** Sentinel prepended to a sentence so the first words are ordinary n-grams. */
export const SOS = "<s>";

const WORD_RE = /[A-Za-z][A-Za-z'’.-]*[A-Za-z]|[A-Za-z]/g;

/** Normalise a surface token for model keys: lower-case, strip surrounding punctuation. */
export function normalizeWord(w: string): string {
  return w
    .toLowerCase()
    .replace(/^[^a-z0-9]+/i, "")
    .replace(/[^a-z0-9]+$/i, "");
}

/** Sentence terminators, as the LSTM tokeniser emits them. */
export const SENTENCE_END = new Set([".", "!", "?"]);

/**
 * True when `text` opens with punctuation that takes NO space before it.
 *
 * The model predicts punctuation as ordinary tokens, so a suggestion can legitimately
 * be "," or ". The". But the popup fires right after "word ", so inserting at the
 * cursor would render "world ,". Callers use this to swallow the preceding space.
 */
export function startsWithTightPunct(text: string): boolean {
  return /^[.,!?;:]/.test(text);
}

/**
 * True when `text` ENDS with sentence/clause punctuation.
 *
 * Used to auto-append a trailing space after accepting a suggestion that ends in
 * punctuation (e.g. a phrase like "polish it." or a bare ","), so the caret lands
 * ready to type the next word instead of hard against the mark.
 */
export function endsWithTightPunct(text: string): boolean {
  return /[.,!?;:]$/.test(text);
}

/**
 * The LSTM's tokens: case-preserving words + sentence/clause punctuation.
 *
 * This regex MUST stay identical to WORD_RE in
 * build_model/train_word_lstm_cased.py. The model only predicts well on the token
 * stream it was trained on, and the punctuation is load-bearing: it is the only
 * marker of a sentence boundary. Without it the model sees "... of the world The
 * dog" and learns that a capitalised opener may follow any word at all - which is
 * what produced suggestions like "in the markets of the world The".
 */
const LSTM_TOKEN_RE = /[A-Za-z]+(?:'[A-Za-z]+)?|[.,!?;:]/g;

/**
 * Strip markdown / LaTeX / code / link machinery from text BEFORE it is tokenised for the
 * model, so the context the LSTM reads looks like the plain prose it was trained on.
 *
 * The token regex already drops decoration characters (`*`, `#`, `[`, backticks …), so the
 * job here is the CONTENT that would otherwise leak as spurious words: the letters inside a
 * code span or a `\alpha`, a URL's path, an image's target. Link/wikilink TEXT is kept (it is
 * real prose the reader sees) while the target/URL is dropped. Input is a prefix up to the
 * cursor, so unclosed trailing spans (a code span or math the user is still typing) are
 * dropped from their opener on.
 */
export function sanitizeForModel(text: string): string {
  return (
    text
      // Fenced code blocks, closed then an unclosed trailing fence.
      .replace(/```[\s\S]*?```/g, " ")
      .replace(/```[\s\S]*$/g, " ")
      // Inline code, closed then unclosed trailing.
      .replace(/`[^`\n]*`/g, " ")
      .replace(/`[^`\n]*$/g, " ")
      // Display and inline math, closed then unclosed trailing.
      .replace(/\$\$[\s\S]*?\$\$/g, " ")
      .replace(/\$[^$\n]*\$/g, " ")
      .replace(/\$[^$\n]*$/g, " ")
      // HTML comments and tags.
      .replace(/<!--[\s\S]*?-->/g, " ")
      .replace(/<\/?[A-Za-z][^>\n]*>/g, " ")
      // Images ![alt](url) and links [text](url): keep the human-visible text only.
      .replace(/!\[([^\]]*)\]\([^)\n]*\)/g, " ")
      .replace(/\[([^\]]*)\]\([^)\n]*\)/g, "$1")
      // Wikilinks [[target|alias]] -> alias, [[target]] -> target.
      .replace(/\[\[(?:[^\]|\n]*\|)?([^\]\n]*)\]\]/g, "$1")
      // Bare URLs.
      .replace(/\b(?:https?:\/\/|www\.)\S+/gi, " ")
  );
}

/** Split raw text into the CASE-PRESERVING word+punctuation tokens the LSTM wants. */
export function tokenizeWordsCased(text: string): string[] {
  // Smart quotes -> straight, so "don’t" tokenises as the trained "don't" rather
  // than splitting into "don" + "t".
  return text.replace(/[‘’]/g, "'").match(LSTM_TOKEN_RE) ?? [];
}

/** Split raw text into lower-cased word tokens (no sentence structure). */
export function tokenizeWords(text: string): string[] {
  const out: string[] = [];
  const matches = text.match(WORD_RE);
  if (!matches) return out;
  for (const m of matches) {
    const n = normalizeWord(m);
    if (n) out.push(n);
  }
  return out;
}

export interface SentenceSplitOptions {
  abbreviations?: Set<string>;
}

/**
 * Decide whether the period at the end of `token` terminates a sentence, given
 * the following raw token (or undefined at end of text). Rule-first, Punkt-like:
 * a period is NOT a boundary when the token is a known abbreviation, an
 * initial/acronym (single letters separated by dots), or a decimal number.
 */
export function isSentenceTerminator(
  tokenWithPeriod: string,
  nextRawToken: string | undefined,
  abbreviations: Set<string>,
): boolean {
  if (!/[.!?…]["')\]]?$/.test(tokenWithPeriod)) return false;

  // ! and ? are (almost) always terminators.
  if (/[!?…]["')\]]?$/.test(tokenWithPeriod)) return true;

  const core = tokenWithPeriod.replace(/["')\]]+$/, "");
  const withoutDot = core.replace(/\.$/, "");
  const lower = withoutDot.toLowerCase();

  // Decimal or version like "3.14" or numeric - not a boundary.
  if (/\d$/.test(withoutDot)) return false;

  // Known abbreviation (incl. dotted forms like "u.s", "e.g") - not a boundary.
  if (abbreviations.has(lower)) return false;

  // Acronym / initials: "A.B.C." - letters separated by dots. Ambiguous, so
  // only treat as a boundary when the next token looks like a fresh sentence.
  if (/^([A-Za-z]\.)+[A-Za-z]?$/.test(core)) {
    return nextStartsSentence(nextRawToken);
  }

  return true;
}

function nextStartsSentence(nextRawToken: string | undefined): boolean {
  if (nextRawToken === undefined) return true; // end of text
  // A following capitalised word (that isn't itself an all-caps continuation)
  // strongly suggests a new sentence.
  return /^[A-Z]/.test(nextRawToken);
}

/**
 * Split text into sentences, each returned as an array of normalized word
 * tokens with `SOS` sentinels are NOT included here (the model builder adds
 * them). Preserves the surface tokens needed for boundary decisions.
 */
export function splitSentences(
  text: string,
  opts: SentenceSplitOptions = {},
): string[][] {
  const abbreviations = opts.abbreviations ?? buildAbbreviationSet();
  // Split on whitespace but keep punctuation attached for boundary detection.
  const rawTokens = text.split(/\s+/).filter((t) => t.length > 0);
  const sentences: string[][] = [];
  let current: string[] = [];

  for (let i = 0; i < rawTokens.length; i++) {
    const raw = rawTokens[i];
    const norm = normalizeWord(raw);
    if (norm) current.push(norm);

    if (isSentenceTerminator(raw, rawTokens[i + 1], abbreviations)) {
      if (current.length) sentences.push(current);
      current = [];
    }
  }
  if (current.length) sentences.push(current);
  return sentences;
}
