/**
 * Deterministic link-opportunity matching. This is a RETRIEVAL problem, not a generation
 * one: we only ever propose a link to a note that already exists, by matching spans of the
 * text against known note titles/aliases. Nothing is invented, so there are no hallucinated
 * links - the neural model is not involved here at all.
 *
 * Longest-match, word-boundary, skips anything already inside a link / code / math, and
 * ignores trivially-short or stop-word spans so it stays quiet and precise.
 */

export interface LinkTarget {
  /** The link destination (note basename or path). */
  target: string;
  /** The canonical display title (may differ in casing from the matched text). */
  display: string;
}

export interface LinkSpan extends LinkTarget {
  from: number;
  to: number;
  /** The exact text that was matched (preserves the user's casing). */
  text: string;
}

/** Single words this short or common are never matched on their own (a title of exactly
 *  one of these would be too noisy to underline everywhere). Multi-word titles are exempt. */
const STOP = new Set([
  "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "is", "it", "as",
  "by", "be", "we", "i", "you", "he", "she", "they", "this", "that", "with", "from",
  "was", "are", "but", "not", "have", "has", "had", "will", "can", "all", "one", "so",
  "if", "up", "out", "no", "do", "my", "me", "us",
]);

/** Ranges (in char offsets) that must never be matched: fenced/inline code, math, existing
 *  wikilinks and markdown links, tags, and frontmatter. */
export function protectedRanges(text: string): Array<[number, number]> {
  const ranges: Array<[number, number]> = [];
  const add = (re: RegExp) => {
    re.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(text))) {
      ranges.push([m.index, m.index + m[0].length]);
      if (m[0].length === 0) re.lastIndex++;
    }
  };
  add(/```[\s\S]*?```/g); // fenced code (backticks)
  add(/~~~[\s\S]*?~~~/g); // fenced code (tildes)
  add(/``[^\n]*?``/g); // inline code (double backtick)
  add(/`[^`\n]*`/g); // inline code
  add(/\$\$[\s\S]*?\$\$/g); // block math
  add(/\$[^$\n]*\$/g); // inline math
  add(/%%[\s\S]*?%%/g); // Obsidian comments
  add(/<!--[\s\S]*?-->/g); // HTML comments
  add(/!?\[\[[^\]]*\]\]/g); // existing wikilinks and embeds
  add(/\[[^\]]*\]\([^)]*\)/g); // markdown links
  add(/(^|\s)#[\w/-]+/g); // tags
  add(/https?:\/\/\S+/g); // urls
  // YAML frontmatter block at the very top (tolerating a BOM and CRLF endings), so its
  // fields (date, tags, aliases) are never treated as prose to segment or link.
  const fm = /^\uFEFF?---[ \t]*\r?\n[\s\S]*?\r?\n---[ \t]*(?:\r?\n|$)/.exec(text);
  if (fm) ranges.push([0, fm[0].length]);
  return ranges;
}

function overlaps(from: number, to: number, ranges: Array<[number, number]>): boolean {
  for (const [a, b] of ranges) if (from < b && to > a) return true;
  return false;
}

/**
 * Find link opportunities in `text`.
 *
 * @param lookup  lower-cased phrase -> target, or null if not a known note title/alias.
 * @param opts.maxWords     longest title (in words) to try matching (default 6).
 * @param opts.excludeTarget  a target to never link to (the note's own title).
 * @param opts.protect      pre-computed protected ranges (defaults to protectedRanges(text)).
 */
export function findLinkSpans(
  text: string,
  lookup: (phrase: string) => LinkTarget | null,
  opts: { maxWords?: number; excludeTarget?: string; protect?: Array<[number, number]> } = {},
): LinkSpan[] {
  const maxWords = opts.maxWords ?? 6;
  const protect = opts.protect ?? protectedRanges(text);
  const excl = opts.excludeTarget?.toLowerCase();

  // Word tokens with their offsets.
  const toks: Array<{ s: number; e: number }> = [];
  const wre = /[A-Za-z0-9][A-Za-z0-9'’.-]*/g;
  let wm: RegExpExecArray | null;
  while ((wm = wre.exec(text))) toks.push({ s: wm.index, e: wm.index + wm[0].length });

  const out: LinkSpan[] = [];
  let i = 0;
  while (i < toks.length) {
    let matched = false;
    const hi = Math.min(i + maxWords - 1, toks.length - 1);
    for (let j = hi; j >= i; j--) {
      const from = toks[i].s;
      const to = toks[j].e;
      const phrase = text.slice(from, to);
      const nWords = j - i + 1;
      if (nWords === 1 && (phrase.length < 3 || STOP.has(phrase.toLowerCase()))) continue;
      const hit = lookup(phrase.toLowerCase());
      if (!hit) continue;
      if (excl && hit.target.toLowerCase() === excl) continue;
      if (overlaps(from, to, protect)) continue;
      out.push({ from, to, text: phrase, target: hit.target, display: hit.display });
      i = j + 1; // consume the matched run
      matched = true;
      break;
    }
    if (!matched) i++;
  }
  return out;
}
