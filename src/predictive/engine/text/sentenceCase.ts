/**
 * Smart auto-capitalisation. Given the text immediately before a word, decide
 * whether that word begins a new sentence and should be capitalised - while
 * NOT tripping on "vs.", "U.S.", "e.g.", decimals, or mid-word periods.
 */
import { buildAbbreviationSet } from "./abbreviations.ts";
import { isSentenceTerminator } from "./tokenize.ts";
import { fixDoubleCapital } from "./caseFix.ts";

export interface SentenceCaseConfig {
  abbreviations: Set<string>;
  /** also fix standalone lowercase "i" -> "I". */
  fixI: boolean;
  /** fix double capitals ("THe" -> "The"). */
  fixDoubleCaps?: boolean;
  /** Proper-noun casing for THIS word, already resolved by the caller from the
   *  cased LSTM (contextual). Replaces the old static CaseMap, which capitalised
   *  by frequency alone and so could not tell "Polish" from "polish". */
  canonical?: string | null;
}

export function defaultSentenceCaseConfig(extra: string[] = []): SentenceCaseConfig {
  return { abbreviations: buildAbbreviationSet(extra), fixI: true, fixDoubleCaps: true };
}

/**
 * Should the word that follows `precedingText` be capitalised as a sentence
 * start? `precedingText` is everything on the line/paragraph before the current
 * word (the current word itself excluded).
 */
export function shouldCapitalizeNext(
  precedingText: string,
  cfg: SentenceCaseConfig,
): boolean {
  const trimmed = precedingText.replace(/[\s]+$/, "");
  // Start of document / line / paragraph.
  if (trimmed.length === 0) return true;

  // A line that so far holds ONLY a list/quote marker is the start of that line's content,
  // so its first word capitalises like any sentence start ("- overnight" -> "- Overnight"),
  // matching Obsidian and Word. Covers bullets (-, *, +), ordered items (1. / 1)), task
  // checkboxes (- [ ]) and blockquotes (>), possibly indented and nested. The trailing space
  // is kept for the pattern because the marker needs it.
  if (/^\s*(?:(?:[-*+]|\d+[.)])[ \t]+(?:\[[ xX]\][ \t]+)?|>[ \t]?)+$/.test(precedingText)) return true;

  // Grab the last whitespace-delimited raw token (keeps its punctuation).
  const m = trimmed.match(/(\S+)$/);
  if (!m) return true;
  const lastToken = m[1];

  // A bare terminator like "." or "!" following a space.
  if (/^[.!?…]+$/.test(lastToken)) return true;

  return isSentenceTerminator(lastToken, undefined, cfg.abbreviations);
}

/** Capitalise the first alphabetic character of `word`. */
export function capitalizeFirst(word: string): string {
  const i = word.search(/[A-Za-z]/);
  if (i < 0) return word;
  return word.slice(0, i) + word[i].toUpperCase() + word.slice(i + 1);
}

/**
 * Apply auto-capitalisation to a freshly-completed `word` given its preceding
 * text. Returns the possibly-transformed word (or the original).
 */
export function applyAutoCapitalization(
  word: string,
  precedingText: string,
  cfg: SentenceCaseConfig,
): string {
  let out = word;
  if (cfg.fixDoubleCaps) out = fixDoubleCapital(out);
  if (cfg.fixI && /^i$/.test(out)) out = "I";
  const sentenceStart = shouldCapitalizeNext(precedingText, cfg);
  // Is the word immediately preceded by a KNOWN abbreviation + period ("incl.", "etc.", "e.g.")?
  // That is exactly where the cased LSTM over-capitalises - it reads the period as a sentence end.
  const abbrevMatch = /(?:^|\s)([A-Za-z][A-Za-z.]*)\.\s*$/.exec(precedingText);
  const afterAbbreviation =
    !!abbrevMatch && cfg.abbreviations.has(abbrevMatch[1].toLowerCase().replace(/\.$/, ""));
  // Proper-noun / acronym casing (e.g. "london" -> "London", "nasa" -> "NASA"), decided in context
  // by the LSTM. Applied everywhere EXCEPT: a canonical that only adds a leading capital to an
  // otherwise-lowercase word ("the" -> "The"), right after an abbreviation, is the spurious "incl.
  // capitalises the next word" cap - suppress it there. Genuine proper nouns (canonical differs by
  // more than the first letter: NASA, McDonald) still apply even after an abbreviation.
  if (cfg.canonical) {
    const c = cfg.canonical;
    const onlyLeadingCap =
      c.length === out.length &&
      c.slice(1) === out.slice(1) &&
      out.charAt(0) === out.charAt(0).toLowerCase() &&
      c.charAt(0) === out.charAt(0).toUpperCase();
    if (!(onlyLeadingCap && afterAbbreviation)) out = c;
  }
  // A real sentence start still capitalises the first letter on top of that.
  if (sentenceStart) out = capitalizeFirst(out);
  return out;
}
