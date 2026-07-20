/**
 * Optional guard: stop Tab from indenting a list item when the caret is in the MIDDLE of the
 * item's text.
 *
 * Why: Tab is the accept key, but the suggestion popup closes a beat before some Tab presses
 * land. With the popup gone the Tab falls through to the editor, and Obsidian indents the
 * whole bullet - so a missed "Tab to accept" silently shifts your list item right. Word (and
 * most editors) only indent a list item when the caret sits at the very start of its content,
 * right after the marker; anywhere else Tab does not restructure the list. This extension
 * reproduces that: with the setting on, a Tab whose caret is past the marker is swallowed
 * (no accidental indent), while a Tab at the start of the item still indents as normal.
 *
 * Registered at HIGH precedence so it is consulted before Obsidian's own list-indent keymap,
 * and only when the popup is closed (an open popup consumes Tab for accept before CodeMirror
 * ever sees it).
 */
import { Prec, StateField, type Transaction } from "@codemirror/state";
import { keymap, type EditorView } from "@codemirror/view";
import type { Extension } from "@codemirror/state";
import type { PredictiveSettings } from "./PredictiveSettings";

/** Leading list/quote marker: indentation + bullet or ordered number + optional checkbox. */
const LIST_MARKER = /^(\s*(?:>[ \t]?)*)(?:[-*+]|\d+[.)])[ \t]+(?:\[[ xX]\][ \t]+)?/;

/**
 * True when a transaction is an INDENT/OUTDENT edit - it only touches the leading whitespace
 * of a line (a Tab that shifts a bullet right, or Shift-Tab that shifts it left), not the
 * words. Tab-to-indent is structural, so it must not be read as "the user typed something,
 * suggest the next word": that is what made a recommendation pop up right after indenting.
 *
 * The test: the edit changes only whitespace, and what it inserts lands in the line's leading
 * indentation (from line-start to the edit is all whitespace). This excludes a normal space
 * typed after a word ("word " → its leading run is not all whitespace) and Enter (inserts a
 * newline, which isn't [ \t]).
 */
export function isIndentEdit(tr: Transaction): boolean {
  if (!tr.docChanged) return false;
  let any = false;
  let indentOnly = true;
  tr.changes.iterChanges((_fromA, _toA, fromB, toB, inserted) => {
    any = true;
    if (!/^[ \t]*$/.test(inserted.toString())) indentOnly = false; // inserted non-whitespace
    else {
      const line = tr.newDoc.lineAt(fromB);
      if (!/^[ \t]*$/.test(tr.newDoc.sliceString(line.from, toB))) indentOnly = false; // not leading
    }
  });
  return any && indentOnly;
}

/** Tracks whether the most recent document change was an indent/outdent, so the popup
 *  suggester (which sees no transaction in onTrigger) can skip a suggestion right after one. */
export const indentEditField = StateField.define<boolean>({
  create: () => false,
  update(value, tr) {
    if (tr.docChanged) return isIndentEdit(tr);
    return value; // a bare cursor move keeps the last verdict
  },
});

export function tabIndentGuard(getSettings: () => PredictiveSettings): Extension {
  const handleTab = (view: EditorView): boolean => {
    if (!getSettings().tabIndentAtBulletStartOnly) return false; // feature off: normal Tab
    const { state } = view;
    const sel = state.selection.main;
    if (!sel.empty) return false; // a selection: leave block-indent alone
    const line = state.doc.lineAt(sel.head);
    const m = line.text.match(LIST_MARKER);
    if (!m) return false; // not a list item: Tab keeps its usual behaviour
    const contentStart = line.from + m[0].length;
    if (sel.head <= contentStart) return false; // right after the marker: allow the indent
    return true; // mid-item: swallow Tab so a missed accept can't indent the bullet
  };
  return Prec.high(keymap.of([{ key: "Tab", run: handleTab }]));
}
