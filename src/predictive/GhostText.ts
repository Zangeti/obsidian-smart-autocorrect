/**
 * Inline "ghost text" prediction (#C5). Shows the single most likely completion
 * as dimmed text ahead of the cursor (like a phone / Copilot); press Tab to
 * accept. Implemented as a CodeMirror 6 editor extension so it composes with
 * Obsidian's editor. Markdown-aware and suppressed when disabled in settings.
 *
 * ASYNC BY DESIGN: the engine runs in a Web Worker, so a keystroke cannot wait for
 * a prediction. update() only FIRES a request; when the answer arrives we dispatch
 * a StateEffect carrying the ghost and a StateField turns it into a decoration.
 * Each result is stamped with the cursor it was computed for and dropped if the
 * editor has moved on - otherwise a slow reply could paint a ghost for text you
 * have already changed.
 *
 * Coexists with the popup: the popup's Tab handler (its own scope) takes
 * precedence when it is open, so ghost-Tab only fires when no popup is showing.
 */
import { Prec, StateEffect, StateField, type Transaction } from "@codemirror/state";
import {
  Decoration,
  EditorView,
  keymap,
  ViewPlugin,
  type ViewUpdate,
  WidgetType,
} from "@codemirror/view";
import {
  applySuggestionCase,
  classifyMarkdownContext,
  defaultSentenceCaseConfig,
  upperFromText,
  startsWithTightPunct,
  suggestionCase,
} from "./engine/index";
import { contextWords } from "./context";
import { isIndentEdit } from "./tabIndent";
import type { PredictiveEngineController } from "./PredictiveEngineController";
import type { PredictiveSettings } from "./PredictiveSettings";

class GhostWidget extends WidgetType {
  readonly text: string;
  constructor(text: string) {
    super();
    this.text = text;
  }
  eq(other: GhostWidget): boolean {
    return other.text === this.text;
  }
  toDOM(): HTMLElement {
    return createSpan({ cls: "predictive-ghost", text: this.text });
  }
  ignoreEvent(): boolean {
    return true;
  }
}

/** The ghost to show, and the cursor position it belongs to. */
interface Ghost {
  text: string;
  pos: number;
}

const setGhost = StateEffect.define<Ghost | null>();

/**
 * When the user types characters that continue the current ghost, advance it in place
 * instead of clearing and waiting for the worker. This is what makes it feel instant:
 * typing "auto" while the ghost reads "matically" shrinks it to "matically"→"…" with no
 * round-trip. Returns the shortened ghost, or null when the edit isn't a clean type-along.
 */
function advanceGhost(g: Ghost | null, tr: Transaction): Ghost | null {
  if (!g) return null;
  let result: Ghost | null | undefined;
  let n = 0;
  tr.changes.iterChanges((fromA, toA, _fromB, _toB, inserted) => {
    if (++n > 1) { result = null; return; }
    const ins = inserted.toString();
    // A single pure insertion at the ghost anchor whose text is the ghost's next chars.
    if (fromA === toA && fromA === g.pos && ins.length > 0 && g.text.toLowerCase().startsWith(ins.toLowerCase())) {
      const rest = g.text.slice(ins.length);
      result = rest ? { text: rest, pos: g.pos + ins.length } : null;
    } else {
      result = null;
    }
  });
  return result ?? null;
}

const ghostField = StateField.define<Ghost | null>({
  create: () => null,
  update(value, tr) {
    // Typing along the ghost advances it instantly; any other edit, or a bare cursor
    // move, invalidates it (its anchor is stale). Clearing here rather than when the
    // async reply lands stops a wrong ghost from lingering for even one frame.
    if (tr.docChanged) value = advanceGhost(value, tr);
    else if (tr.selection) value = null;
    for (const e of tr.effects) if (e.is(setGhost)) value = e.value;
    return value;
  },
  provide: (f) =>
    EditorView.decorations.from(f, (g) =>
      g
        ? Decoration.set([
            Decoration.widget({ widget: new GhostWidget(g.text), side: 1 }).range(g.pos),
          ])
        : Decoration.none,
    ),
});

/** What the cursor implies we should ask for, or null if no ghost applies. */
function ghostQuery(
  view: EditorView,
  settings: PredictiveSettings,
): { ctx: string[]; query: string; pos: number; before: string } | null {
  if (!settings.pluginEnabled || !settings.ghostText) return null;
  const sel = view.state.selection.main;
  if (!sel.empty) return null;
  const pos = sel.head;
  const before = view.state.doc.sliceString(0, pos);
  // Only at end of line (nothing but whitespace to the right).
  const after = view.state.doc.sliceString(pos, Math.min(view.state.doc.length, pos + 1));
  if (after && !/\s/.test(after)) return null;
  if (settings.markdownAware && classifyMarkdownContext(before).suppressPrediction) return null;

  const line = before.slice(before.lastIndexOf("\n") + 1);
  const wm = line.match(/([A-Za-z][A-Za-z'-]*)$/);
  const query = wm ? wm[1] : "";
  if (query.length === 0 && !/\w\s$/.test(line)) return null;
  if (query.length > 0 && query.length < settings.minChars) return null;
  return { ctx: contextWords(before, settings.extraAbbreviations), query, pos, before };
}

/** The part of `top` still to be typed, or "" if it isn't a clean completion. */
function suffixFor(top: string, query: string): string {
  if (!top) return "";
  if (!query) return top;
  // Only a straightforward completion makes sense as ghost text.
  if (top.toLowerCase().startsWith(query.toLowerCase()) && top.length > query.length)
    return top.slice(query.length);
  return "";
}

export function ghostTextExtension(
  engine: PredictiveEngineController,
  getSettings: () => PredictiveSettings,
  /** True when the active file is in an excluded folder - no ghost there. */
  isExcluded: () => boolean = () => false,
  /** Called when a ghost is accepted, with the inserted text and keystrokes saved. */
  onAccept: (text: string, saved: number) => void = (t) => engine.recordAccept(t, t.length),
) {
  const plugin = ViewPlugin.fromClass(
    class {
      private seq = 0;
      private view: EditorView;
      constructor(view: EditorView) {
        this.view = view;
        this.request();
      }
      update(u: ViewUpdate) {
        // Only ask the model when the TEXT changed. Cursor moves and focus changes
        // must not trigger a request - arrow keys, clicks and pane switches would
        // otherwise spin the engine while idle.
        if (!u.docChanged) return;
        // Indenting/outdenting a bullet is a structural edit, not typing - don't treat it as
        // a cue to suggest the next word. Without this, Tab-to-indent makes a ghost pop up.
        if (u.transactions.some((t) => isIndentEdit(t))) return;
        // If the ghost just advanced optimistically and still sits at the cursor, we
        // already show a valid completion - skip the worker call entirely. This is the
        // big performance win: typing along a suggestion costs zero predictions.
        const g = u.state.field(ghostField, false);
        if (g && g.pos === u.state.selection.main.head && g.text.length > 0) return;
        this.request();
      }
      request() {
        if (!engine.ready || isExcluded()) return;
        const q = ghostQuery(this.view, getSettings());
        if (!q) return;
        const id = ++this.seq;
        // Phrase-free fast path: the ghost only ever shows one word, so skip the
        // multi-word beam search the popup does. Much faster, same filtering.
        void engine
          .getSuggestions(q.ctx, q.query, 1, false)
          .then((items) => {
            // Stale guard: a newer request superseded this one, or the cursor
            // moved since we asked - either way the answer is no longer about the
            // text on screen.
            if (id !== this.seq) return;
            if (this.view.state.selection.main.head !== q.pos) return;
            const s = getSettings();
            // Case the ghost exactly as the popup cases its list: both answer the same
            // question about the same position, so showing "the" inline while the popup
            // offers "The" would be incoherent. Cased BEFORE suffixFor so the prefix
            // still lines up with what the user actually typed.
            const top = applySuggestionCase(
              items[0]?.insert ?? "",
              suggestionCase(q.before, defaultSentenceCaseConfig(s.extraAbbreviations), {
                upper: upperFromText(q.before + q.query),
              }),
            );
            const text = suffixFor(top, q.query);
            if (!text) return;
            this.view.dispatch({ effects: setGhost.of({ text, pos: q.pos }) });
          })
          .catch(() => {
            /* a failed prediction just means no ghost */
          });
      }
    },
  );

  // Tab accepts the ghost. Highest precedence so it beats Obsidian's Tab-indent and the
  // list-indent guard - but ONLY when a ghost is actually showing: with no ghost it returns
  // false and Tab keeps its normal behaviour. (The popup is disabled whenever ghost text is
  // on, so there is no popup-Tab to compete with here.)
  const acceptKeymap = Prec.highest(
    keymap.of([
      {
        key: "Tab",
        run: (view: EditorView): boolean => {
          const g = view.state.field(ghostField, false);
          if (!g) return false;
          const pos = view.state.selection.main.head;
          if (pos !== g.pos) return false;
          // The model predicts punctuation as ordinary tokens, so a ghost can open
          // with one. English puts no space before "," or "." - and ghost text only
          // appears after "word ", so inserting at the cursor would give "world ,".
          // Replace from the space instead of after it.
          let from = pos;
          if (startsWithTightPunct(g.text) && pos > 0 &&
              view.state.doc.sliceString(pos - 1, pos) === " ") from = pos - 1;
          view.dispatch({
            changes: { from, to: pos, insert: g.text },
            selection: { anchor: from + g.text.length },
            userEvent: "input.complete",
          });
          // The whole ghost suffix is text the user didn't type - that's the saving.
          onAccept(g.text, g.text.length);
          return true;
        },
      },
    ]),
  );

  const theme = EditorView.baseTheme({
    ".predictive-ghost": { opacity: "0.4", fontStyle: "italic" },
  });

  return [ghostField, plugin, acceptKeymap, theme];
}
