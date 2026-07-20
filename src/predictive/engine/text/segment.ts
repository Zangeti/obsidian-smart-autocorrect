/**
 * Split a note into the blocks a reader actually thinks in, the way they appear in
 * Obsidian: a paragraph, a whole list (not one bullet at a time), a block-quote or
 * callout, a table. Headings are not blocks to link on their own; they act as the
 * "title" a following list attaches to.
 *
 * Built to cope with the broad range of ways people format notes:
 *   - headings: ATX (`#`..`######`) and Setext (`===` / `---` underline).
 *   - lists: `-`/`*`/`+`, `1.`/`1)`, task items (`- [ ]`), nested/indented, and loose
 *     lists with blank lines between items - all one block.
 *   - block-quotes and callouts (`>`, `> [!note]`), grouped.
 *   - tables, with or without a leading pipe (a GFM delimiter row is enough).
 *   - CRLF or LF line endings.
 *   - code, math and frontmatter are protected and dropped.
 *
 * `from`/`to` bound the whole block (its topical fingerprint + already-present links);
 * `anchor` is where the icon and the inserted link go:
 *   - paragraph  → end of the paragraph.
 *   - list       → the title above it (heading or lead-in line), else end of first item.
 *   - quote/table→ end of the block's first line.
 */
import { protectedRanges } from "./linkMatch.ts";

export type SegmentKind = "paragraph" | "list" | "quote" | "table";

export interface Segment {
  from: number;
  to: number;
  /** Where the link icon is shown and the link is inserted. */
  anchor: number;
  text: string;
  kind: SegmentKind;
  /** Nearest heading above this block (markers stripped), for `[[Note#Heading]]` links. */
  heading?: string;
}

const HEADING = /^\s{0,3}#{1,6}\s+\S/;
const LIST_ITEM = /^\s*(?:[-*+]|\d+[.)])\s+\S/;
const DEF_ITEM = /^\s{0,3}[:~]\s+\S/; // definition-list line (": definition")
const QUOTE = /^\s{0,3}>/;
const HR = /^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/;
const HTML_LINE = /^\s{0,3}<(?:!|\/?[a-zA-Z])/;
const SETEXT = /^\s{0,3}(?:=+|-{2,})\s*$/; // underline turning the line above into a heading
const TABLE_DELIM = /^\s{0,3}\|?\s*:?-{1,}:?\s*(?:\|\s*:?-{1,}:?\s*)*\|?\s*$/;
const MIN_TERMS = 3;

type LineKind = "blank" | "heading" | "list" | "quote" | "hr" | "html" | "prose";

interface Line {
  start: number;
  end: number; // exclusive of the line break
  kind: LineKind;
  indent: number; // leading whitespace columns (tab = 4)
  hasPipe: boolean;
}

/** Leading-whitespace columns, counting a tab as 4. */
function indentOf(text: string): number {
  let n = 0;
  for (const ch of text) {
    if (ch === " ") n++;
    else if (ch === "\t") n += 4;
    else break;
  }
  return n;
}

function classify(text: string): LineKind {
  if (text.trim().length === 0) return "blank";
  if (HR.test(text)) return "hr";
  if (HEADING.test(text)) return "heading";
  if (LIST_ITEM.test(text) || DEF_ITEM.test(text)) return "list";
  if (QUOTE.test(text)) return "quote";
  if (HTML_LINE.test(text)) return "html";
  return "prose";
}

function wordish(text: string): number {
  return (text.match(/[a-zA-Z][a-zA-Z'-]{2,}/g) ?? []).length;
}

/** True only when the block sits ENTIRELY inside one protected range (a fenced code
 *  block, frontmatter, a lone URL). Inline math/code/comments inside a normal paragraph
 *  are subsets, not containers, so the paragraph is kept and still linkable. */
function containedInProtected(from: number, to: number, prot: Array<[number, number]>): boolean {
  for (const [a, b] of prot) if (a <= from && to <= b) return true;
  return false;
}

export function segmentText(text: string): Segment[] {
  const prot = protectedRanges(text);
  const rawLines = text.split("\n");
  const lines: Line[] = [];
  let offset = 0;
  for (let t of rawLines) {
    const start = offset;
    let len = t.length;
    offset = start + len + 1; // + the "\n" we split on
    if (t.endsWith("\r")) {
      t = t.slice(0, -1); // tolerate CRLF: drop the \r for classification
      len -= 1;
    }
    lines.push({ start, end: start + len, kind: classify(t), indent: indentOf(t), hasPipe: t.includes("|") });
  }

  // Setext pass: a `===`/`---` underline promotes the prose line above it to a heading.
  for (let k = 1; k < lines.length; k++) {
    const under = text.slice(lines[k].start, lines[k].end);
    if (SETEXT.test(under) && lines[k - 1].kind === "prose") {
      lines[k - 1].kind = "heading";
      lines[k].kind = "blank";
    }
  }

  const out: Segment[] = [];
  let currentHeading: string | undefined;
  const push = (from: number, to: number, anchor: number, kind: SegmentKind) => {
    if (containedInProtected(from, to, prot)) return;
    const body = text.slice(from, to).trim();
    if (wordish(body) < MIN_TERMS) return;
    out.push({ from, to, anchor, text: body, kind, heading: currentHeading });
  };
  const cleanHeading = (s: string): string =>
    s.replace(/^\s*#+\s*/, "").replace(/\s+#+\s*$/, "").trim();

  /** End offset of the nearest heading or prose line strictly above line `i`, skipping
   *  blanks - the "title" a list hangs under. Null when the run above is not a title. */
  const titleAbove = (i: number): number | null => {
    for (let j = i - 1; j >= 0; j--) {
      if (lines[j].kind === "blank") continue;
      if (lines[j].kind === "heading" || lines[j].kind === "prose") return lines[j].end;
      return null;
    }
    return null;
  };

  /** Consume a list starting at line `k` (items + nested/indented content + inner blanks). */
  const consumeList = (k: number): { to: number; next: number } => {
    const baseIndent = lines[k].indent;
    let to = lines[k].end;
    let j = k + 1;
    while (j < lines.length) {
      const l = lines[j];
      if (l.kind === "list" || (l.indent > baseIndent && l.kind !== "blank")) {
        to = l.end;
        j++;
      } else if (l.kind === "blank" && j + 1 < lines.length && (lines[j + 1].kind === "list" || lines[j + 1].indent > baseIndent)) {
        j++;
      } else {
        break;
      }
    }
    return { to, next: j };
  };

  const isTableStart = (i: number): boolean =>
    lines[i].hasPipe &&
    (text.slice(lines[i].start, lines[i].end).trimStart().startsWith("|") ||
      (i + 1 < lines.length && TABLE_DELIM.test(text.slice(lines[i + 1].start, lines[i + 1].end))));

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    if (line.kind === "heading") {
      currentHeading = cleanHeading(text.slice(line.start, line.end)) || currentHeading;
      i++;
      continue;
    }
    if (line.kind === "blank" || line.kind === "hr") {
      i++;
      continue;
    }

    // HTML block: skip to the next blank line (never linked or fingerprinted).
    if (line.kind === "html") {
      i++;
      while (i < lines.length && lines[i].kind !== "blank") i++;
      continue;
    }

    // Indented code block: a run indented ≥4 columns that starts fresh (previous line
    // blank / top of note), i.e. NOT the indented continuation of a list. Skipped.
    if (line.indent >= 4 && (i === 0 || lines[i - 1].kind === "blank")) {
      i++;
      while (i < lines.length && (lines[i].indent >= 4 || lines[i].kind === "blank")) {
        // stop if a blank is followed by a non-indented line (block ended)
        if (lines[i].kind === "blank" && (i + 1 >= lines.length || lines[i + 1].indent < 4)) break;
        i++;
      }
      continue;
    }

    if (line.kind === "list") {
      const { to, next } = consumeList(i);
      push(line.start, to, titleAbove(i) ?? lines[i].end, "list");
      i = next;
      continue;
    }

    if (line.kind === "quote") {
      const from = line.start;
      let to = line.end;
      const firstEnd = line.end;
      i++;
      while (i < lines.length && lines[i].kind === "quote") {
        to = lines[i].end;
        i++;
      }
      push(from, to, firstEnd, "quote");
      continue;
    }

    // prose line: could be the start of a table, otherwise a paragraph.
    if (isTableStart(i)) {
      const from = line.start;
      let to = line.end;
      const firstEnd = line.end;
      i++;
      while (i < lines.length && lines[i].hasPipe && lines[i].kind !== "blank") {
        to = lines[i].end;
        i++;
      }
      push(from, to, firstEnd, "table");
      continue;
    }

    const paraStart = line.start;
    const startIdx = i;
    let paraEnd = line.end;
    i++;
    while (i < lines.length && lines[i].kind === "prose" && !isTableStart(i)) {
      paraEnd = lines[i].end;
      i++;
    }
    // Title + list: a LONE prose line directly above a list (over at most one blank) is that
    // list's title. Merge them into one block, anchored at the title, so "Heading / - a / - b"
    // is a single segment whose text and preview both include the heading line.
    let k = i;
    let blanks = 0;
    while (k < lines.length && lines[k].kind === "blank") {
      k++;
      blanks++;
    }
    // Only a TITLE merges with the list, not a full sentence that happens to sit above one.
    // Titles aren't sentences: they don't end in sentence punctuation (a trailing ":" is a
    // classic lead-in and still counts as a title).
    const leadIn = text.slice(paraStart, paraEnd).trim();
    const looksLikeTitle = leadIn.length > 0 && !/[.!?]$/.test(leadIn);
    if (i - startIdx === 1 && blanks <= 1 && looksLikeTitle && k < lines.length && lines[k].kind === "list") {
      const { to, next } = consumeList(k);
      push(paraStart, to, paraEnd, "list");
      i = next;
      continue;
    }
    push(paraStart, paraEnd, paraEnd, "paragraph");
  }

  return out;
}
