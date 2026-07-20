/**
 * "Link selection" (#7): replaces the plain add-link flow for a highlighted selection with OUR
 * recommendations, in the SAME right-docked chooser the link icon uses - list above, section
 * preview below, arrow keys to browse. Reuses the RelatedIndex scoring the ambient icons and
 * the `[[` picker use, so the ranking is consistent everywhere.
 *
 * It is deliberately not a modal. The point of this flow is that the selection stays selected
 * and visible while you pick what to link it to; a modal would dim the note and take the
 * selection away. Choosing replaces the selection with `[[Note#Heading|selection]]`, so the
 * text you highlighted is what you end up reading.
 */
import type { App, Editor } from "obsidian";
import { segmentText } from "./engine/index";
import { LinkChooser } from "./LinkChooser";
import type { RelatedCandidate, RelatedIndex } from "./RelatedIndex";
import type { PredictiveSettings } from "./PredictiveSettings";

/** How many RANKED recommendations to compute for an explicit ask. */
const MAX_CANDIDATES = 12;
/** Total rows in the list once the vault is appended behind the ranked ones. */
const MAX_BROWSE = 50;

export async function openLinkSelection(opts: {
  app: App;
  editor: Editor;
  index: RelatedIndex;
  settings: PredictiveSettings;
  chooser: LinkChooser;
  /** Text to rank against: the selection, else the line/word around the caret. */
  queryText: string;
  /** The selected text, which becomes the link's display text. Empty when nothing is selected. */
  selection: string;
  /** Screen position of the caret, so the list opens next to where the user is working. */
  preferredTop: number;
}): Promise<number> {
  const { app, editor, index, settings, chooser, queryText, selection, preferredTop } = opts;
  const file = app.workspace.getActiveFile();
  const exclude = new Set<string>();
  if (file?.basename) exclude.add(file.basename.toLowerCase());
  // Rank against the whole SECTION the caret sits in, not the highlighted phrase. A
  // selection is usually two or three words - far too little to match a note on, and it
  // would give a different list depending on exactly what you dragged over. The section is
  // also the unit the ambient link icon scores, so the same block offers the same notes
  // however you got here. The selection still decides the link's display text.
  const ranked = sectionQuery(editor) || queryText;
  // Explicit action: no corroboration gate (the user has already chosen to link).
  const candidates = await index.candidatesFor(
    ranked,
    file?.path,
    exclude,
    MAX_CANDIDATES,
    settings.relatedSensitivity,
    false,
  );
  // Never dead-end. Asking to link is an explicit instruction, so the chooser must always
  // offer the whole vault the way `[[` does - ranked notes first, then everything else. A
  // "no related notes found" refusal was the wrong answer to "link this": the user knows
  // which note they want even when nothing scores as topically similar.
  const listed = new Set(candidates.map((c) => c.path));
  const rest = browseAll(app, exclude, listed, file?.path);
  const all = [...candidates, ...rest].slice(0, MAX_BROWSE);
  if (all.length === 0) return 0; // genuinely nothing to link to: a one-note vault

  chooser.open({
    candidates: all,
    preferredTop,
    title: selection ? `Link “${truncate(selection)}” to…` : "Link to…",
    hint: selection
      ? "Your selected text stays as the link's text."
      : "Tip: select text first to link just that phrase.",
    onChoose: (c) => {
      const dest = c.heading ? `${c.display}#${c.heading}` : c.display;
      const alias = selection && selection !== dest ? `|${selection}` : "";
      // replaceSelection targets the selection we kept alive throughout.
      editor.replaceSelection(`[[${dest}${alias}]]`);
      editor.focus();
    },
  });
  // The number of rows OFFERED, not the ranked count: the caller warns the user when this
  // is 0, and the chooser is open now, so reporting the ranked count would warn over a
  // perfectly usable list.
  return all.length;
}

/**
 * The block the caret is in, phrased exactly as the ambient link icon phrases it
 * (`heading. text`), so both surfaces rank against identical input. Empty when the
 * caret falls outside every segment, in which case the caller keeps its own text.
 */
function sectionQuery(editor: Editor): string {
  const offset = editor.posToOffset(editor.getCursor("from"));
  for (const seg of segmentText(editor.getValue())) {
    if (offset >= seg.from && offset <= seg.to)
      return seg.heading ? `${seg.heading}. ${seg.text}` : seg.text;
  }
  return "";
}

/**
 * The rest of the vault, as plain browse rows behind the ranked ones: most recently
 * modified first, since the note you want to link to is usually one you touched lately.
 * `score: 0` marks them as unranked, which is honest - they are here to be findable,
 * not because anything measured them as related.
 */
function browseAll(
  app: App,
  exclude: Set<string>,
  already: Set<string>,
  currentPath: string | undefined,
): RelatedCandidate[] {
  return app.vault
    .getMarkdownFiles()
    .filter((f) => f.path !== currentPath && !already.has(f.path)
                && !exclude.has(f.basename.toLowerCase()))
    .sort((a, b) => (b.stat?.mtime ?? 0) - (a.stat?.mtime ?? 0))
    .slice(0, MAX_BROWSE)
    .map((f) => ({
      target: f.basename,
      display: f.basename,
      path: f.path,
      score: 0,
      semantic: 0,
      keyword: 0,
      snippet: f.parent?.path && f.parent.path !== "/" ? f.parent.path : "",
      sectionText: "",
    }));
}

function truncate(s: string): string {
  return s.length <= 32 ? s : `${s.slice(0, 31)}…`;
}
