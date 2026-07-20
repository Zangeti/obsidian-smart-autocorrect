/**
 * Ambient link suggestion (#link). Scans the visible text for spans that match an existing
 * note title/alias (see LinkIndex) and marks them with a subtle dotted underline - a quiet
 * "there's a link opportunity here", never a popup that interrupts typing.
 *
 *  - Hover a marked span → Obsidian's native page preview shows the target note (a window
 *    onto where you'd be linking).
 *  - Click a marked span → a small menu: insert the [[link]], or dismiss the suggestion.
 *
 * Nothing is ever inserted automatically; the neural model is not involved (matching is
 * deterministic retrieval against notes that actually exist).
 */
import { Menu } from "obsidian";
import type { App } from "obsidian";
import { StateEffect, StateField } from "@codemirror/state";
import { Decoration, EditorView, ViewPlugin, type DecorationSet, type ViewUpdate } from "@codemirror/view";
import { RangeSetBuilder } from "@codemirror/state";
import type { LinkIndex } from "./LinkIndex";
import type { PredictiveSettings } from "./PredictiveSettings";

/** Dispatched to force a redecorate (e.g. after a dismiss) without a doc edit. */
const refreshLinks = StateEffect.define<null>();

const MARK = Decoration.mark({ class: "smart-autocorrect-linkable" });

export function linkSuggestExtension(
  app: App,
  index: LinkIndex,
  getSettings: () => PredictiveSettings,
  isExcluded: () => boolean,
  /** Titles (lower-cased) the user dismissed this session - never re-underlined. */
  dismissed: Set<string>,
) {
  const spanAt = (view: EditorView, node: Node) => {
    const el =
      node.instanceOf(HTMLElement)
        ? node.closest(".smart-autocorrect-linkable")
        : node.parentElement?.closest(".smart-autocorrect-linkable") ?? null;
    if (!el) return null;
    const phrase = el.textContent ?? "";
    const hit = index.lookup(phrase.toLowerCase());
    if (!hit) return null;
    const from = view.posAtDOM(el, 0);
    return { el: el as HTMLElement, phrase, hit, from, to: from + phrase.length };
  };

  const view = ViewPlugin.fromClass(
    class {
      decorations: DecorationSet;
      constructor(v: EditorView) {
        this.decorations = this.build(v);
      }
      update(u: ViewUpdate) {
        if (u.docChanged || u.viewportChanged || u.transactions.some((t) => t.effects.some((e) => e.is(refreshLinks))))
          this.decorations = this.build(u.view);
      }
      build(v: EditorView): DecorationSet {
        const s = getSettings();
        if (!s.suggestLinks || isExcluded()) return Decoration.none;
        const exclude = app.workspace.getActiveFile()?.basename;
        // Collect spans across all visible ranges, then add in document order (RangeSetBuilder
        // requires ascending positions).
        const marks: Array<{ from: number; to: number }> = [];
        for (const { from, to } of v.visibleRanges) {
          const text = v.state.doc.sliceString(from, to);
          for (const sp of index.findLinks(text, exclude)) {
            if (dismissed.has(sp.target.toLowerCase())) continue;
            marks.push({ from: from + sp.from, to: from + sp.to });
          }
        }
        marks.sort((a, b) => a.from - b.from);
        const b = new RangeSetBuilder<Decoration>();
        for (const m of marks) b.add(m.from, m.to, MARK);
        return b.finish();
      }
    },
    { decorations: (v) => v.decorations },
  );

  const handlers = EditorView.domEventHandlers({
    mousedown(evt, v) {
      const info = spanAt(v, evt.target as Node);
      if (!info) return false;
      evt.preventDefault();
      const menu = new Menu();
      menu.addItem((i) =>
        i
          .setTitle(`Link to [[${info.hit.display}]]`)
          .setIcon("link")
          .onClick(() => {
            const insert =
              info.phrase === info.hit.display
                ? `[[${info.hit.display}]]`
                : `[[${info.hit.target}|${info.phrase}]]`;
            v.dispatch({ changes: { from: info.from, to: info.to, insert } });
          }),
      );
      menu.addItem((i) =>
        i
          .setTitle("Dismiss this suggestion")
          .setIcon("x")
          .onClick(() => {
            dismissed.add(info.hit.target.toLowerCase());
            v.dispatch({ effects: refreshLinks.of(null) });
          }),
      );
      menu.showAtMouseEvent(evt);
      return true;
    },
    mouseover(evt, v) {
      const info = spanAt(v, evt.target as Node);
      if (!info) return false;
      // Native page preview (requires the core Page Preview plugin) - the "window onto
      // where you'd be linking".
      app.workspace.trigger("hover-link", {
        event: evt,
        source: "smart-autocorrect",
        hoverParent: v.dom,
        targetEl: info.el,
        linktext: info.hit.target,
      });
      return false;
    },
  });

  // Kept as its own field so `refreshLinks` is a recognised effect even in tests.
  const effectField = StateField.define<number>({
    create: () => 0,
    update: (val, tr) => (tr.effects.some((e) => e.is(refreshLinks)) ? val + 1 : val),
  });

  const theme = EditorView.baseTheme({
    ".smart-autocorrect-linkable": {
      textDecoration: "underline",
      textDecorationStyle: "dotted",
      textDecorationColor: "var(--text-accent)",
      textUnderlineOffset: "3px",
      cursor: "pointer",
    },
  });

  return [view, handlers, effectField, theme];
}
