/**
 * Markdown / Obsidian / LaTeX context classifier.
 *
 * Prediction and autocorrect must NOT fire inside structural regions where
 * "fixing spelling" would corrupt meaning: fenced & inline code, `$…$` / `$$…$$`
 * math (LaTeX), `[[wikilinks]]`, `[text](urls)`, `#tags`, YAML frontmatter, and
 * `%%Obsidian comments%%`.
 *
 * This is a portable, line-aware heuristic that works from the text before the
 * cursor alone (no CodeMirror internals), so it is unit-testable and also serves
 * as a fallback if the editor's syntax tree is unavailable.
 */
export type MarkdownZone =
  | "text"
  | "frontmatter"
  | "codeBlock"
  | "inlineCode"
  | "mathBlock"
  | "mathInline"
  | "wikilink"
  | "linkUrl"
  | "tag"
  | "comment"
  /** a bare URL or email typed in prose (not markdown link syntax). */
  | "bareUrl";

export interface MarkdownContextResult {
  zone: MarkdownZone;
  /** don't show/accept predictions here. */
  suppressPrediction: boolean;
  /** don't autocorrect here. */
  suppressAutocorrect: boolean;
}

const SUPPRESSED: MarkdownZone[] = [
  "frontmatter",
  "codeBlock",
  "inlineCode",
  "mathBlock",
  "mathInline",
  "wikilink",
  "linkUrl",
  "tag",
  "comment",
  "bareUrl",
];

function result(zone: MarkdownZone): MarkdownContextResult {
  const s = SUPPRESSED.includes(zone);
  return { zone, suppressPrediction: s, suppressAutocorrect: s };
}

/** Count non-overlapping occurrences of `needle` in `s`. */
function count(s: string, needle: string): number {
  let n = 0;
  let i = 0;
  while ((i = s.indexOf(needle, i)) !== -1) {
    n++;
    i += needle.length;
  }
  return n;
}

export function classifyMarkdownContext(textBeforeCursor: string): MarkdownContextResult {
  const text = textBeforeCursor;
  const lines = text.split("\n");
  const currentLine = lines[lines.length - 1];

  // --- block-level state (spans multiple lines) ---

  // YAML frontmatter: a "---" on the very first line opens it; the next "---"
  // (or "...") on its own line closes it.
  if (lines.length >= 1 && lines[0].trim() === "---") {
    let closed = false;
    for (let i = 1; i < lines.length - 1; i++) {
      const t = lines[i].trim();
      if (t === "---" || t === "...") {
        closed = true;
        break;
      }
    }
    // If the current line is still before the closing fence, we're inside it.
    if (!closed) return result("frontmatter");
  }

  // Fenced code block: toggle on lines beginning with ``` or ~~~.
  let inFence = false;
  let fenceMarker = "";
  for (let i = 0; i < lines.length - 1; i++) {
    const t = lines[i].trimStart();
    const m = t.match(/^(```+|~~~+)/);
    if (m) {
      if (!inFence) {
        inFence = true;
        fenceMarker = m[1][0];
      } else if (m[1][0] === fenceMarker) {
        inFence = false;
      }
    }
  }
  if (inFence) return result("codeBlock");
  // The opening fence line itself (```lang) - suppress on that line too.
  if (/^\s*(```+|~~~+)/.test(currentLine)) return result("codeBlock");

  // Math block $$…$$ - odd number of "$$" so far means we're inside one.
  if (count(text, "$$") % 2 === 1) return result("mathBlock");

  // Obsidian comment %%…%% - odd number of "%%" means inside a comment.
  if (count(text, "%%") % 2 === 1) return result("comment");

  // --- inline state (within the current line, before the cursor) ---

  // Inline code: odd number of backticks on the current line.
  if (count(currentLine, "`") % 2 === 1) return result("inlineCode");

  // Inline math $…$ - count single '$' not part of '$$'.
  const singleDollars = (currentLine.match(/(?<!\$)\$(?!\$)/g) ?? []).length;
  if (singleDollars % 2 === 1) return result("mathInline");

  // Wikilink [[… or embed ![[… - open "[[" after the last "]]".
  const lastOpen = currentLine.lastIndexOf("[[");
  const lastClose = currentLine.lastIndexOf("]]");
  if (lastOpen !== -1 && lastOpen > lastClose) return result("wikilink");

  // Markdown link URL: inside "](" … before its ")".
  const lastParenOpen = currentLine.lastIndexOf("](");
  const lastParenClose = currentLine.lastIndexOf(")");
  if (lastParenOpen !== -1 && lastParenOpen > lastParenClose) return result("linkUrl");

  // Bare URL currently being typed.
  const tokenMatch = currentLine.match(/(\S+)$/);
  const token = tokenMatch ? tokenMatch[1] : "";
  if (/^(https?:\/\/|www\.)/.test(token)) return result("linkUrl");

  // Tag #foo (but NOT a heading "# " at line start).
  if (/(^|\s)#[\w/-]*$/.test(currentLine) && !/^#{1,6}\s/.test(currentLine)) {
    return result("tag");
  }

  // A bare URL or email typed in prose. `linkUrl` only covers markdown "](...)"
  // syntax, so "https://example.com/some-Path" or "a.b@corp.com" would otherwise get
  // word suggestions and autocorrect applied to their path/domain segments - which
  // silently corrupts a link the user cannot see being edited.
  // Anchored to the token the cursor sits in: once whitespace follows, prose resumes.
  if (/(?:https?:\/\/|www\.|\S+@)\S*$/i.test(currentLine)) return result("bareUrl");

  return result("text");
}
