/**
 * Related-link suggestion UI. Analyses the note block by block (paragraph, whole list,
 * callout, table) and, only where another note's SECTION is a close topical match, drops a
 * small link icon at a readable anchor for that block. Clicking the icon opens a chooser of
 * the feasible targets - each pointing at the specific section it matched - to insert a
 * [[Note#Heading]] link, or dismiss.
 *
 * Restraint, by request:
 *   - a block must have at least `minLinkWords` words to be eligible (no icons on a stray
 *     half-sentence);
 *   - the block you're currently typing in never shows an icon (the cursor's block is
 *     skipped), so suggestions only appear once you've moved on;
 *   - accepting a link removes that block's suggestion immediately.
 *
 * Matching runs off the render path (debounced, async) and is pushed back via a StateEffect
 * so typing is never blocked.
 */
import { setIcon } from "obsidian";
import type { App } from "obsidian";
import { StateEffect, StateField } from "@codemirror/state";
import { Decoration, EditorView, ViewPlugin, WidgetType, type DecorationSet, type ViewUpdate } from "@codemirror/view";
import { RangeSetBuilder } from "@codemirror/state";
import { segmentText } from "./engine/index";
import { LinkChooser } from "./LinkChooser";
import type { RelatedIndex, RelatedCandidate } from "./RelatedIndex";
import type { PredictiveSettings } from "./PredictiveSettings";

interface SegSuggestion {
  anchor: number; // where the icon sits and a link is inserted
  from: number;
  to: number;
  key: string;
  candidates: RelatedCandidate[];
  /** The phrase within the segment that anchors the link - underlined when the setting is on.
   *  Chosen as the words the segment and its semantically-matched note share, so the underline
   *  sits on the text the link is actually about. Absent when there is no clean anchor. */
  span?: { from: number; to: number };
}

/** Common words never worth anchoring a link on. */
const ANCHOR_STOP = new Set([
  "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "is", "it", "as", "by", "be",
  "we", "you", "he", "she", "they", "this", "that", "with", "from", "was", "are", "but", "not",
  "have", "has", "had", "will", "can", "all", "one", "so", "if", "up", "out", "no", "do", "my",
  "me", "us", "our", "their", "its", "his", "her", "them", "then", "than", "there", "here", "who",
  "which", "what", "when", "where", "how", "why", "some", "any", "each", "very", "more", "most",
]);

/**
 * Pick the phrase in a segment that best anchors a link to `cand` - the LONGEST contiguous run of
 * content words the segment shares with the matched note (its title, heading and matched section).
 * The MATCH is semantic (the note was found by embedding similarity); this only decides WHERE to
 * put the underline. Returns absolute document offsets, or null when there is no substantial shared
 * phrase (in which case the end-of-segment icon still covers it).
 */
function pickAnchorSpan(segText: string, segFrom: number, cand: RelatedCandidate): { from: number; to: number } | null {
  const anchorWords = new Set<string>();
  const collect = (t: string) => {
    for (const w of t.toLowerCase().match(/[a-z][a-z'-]{2,}/g) ?? []) if (!ANCHOR_STOP.has(w)) anchorWords.add(w);
  };
  collect(cand.display);
  if (cand.heading) collect(cand.heading);
  collect(cand.sectionText || cand.snippet || "");
  if (anchorWords.size === 0) return null;

  const toks: Array<{ s: number; e: number; w: string }> = [];
  const re = /[A-Za-z][A-Za-z'-]*/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(segText))) toks.push({ s: m.index, e: m.index + m[0].length, w: m[0].toLowerCase() });

  let best: { from: number; to: number } | null = null;
  let bestLen = 0;
  let i = 0;
  while (i < toks.length) {
    if (!anchorWords.has(toks[i].w)) { i++; continue; }
    let j = i;
    while (j + 1 < toks.length && anchorWords.has(toks[j + 1].w)) j++;
    const run = toks.slice(i, j + 1);
    const words = run.length;
    const substantial = run.some((t) => t.w.length >= 4 && !ANCHOR_STOP.has(t.w));
    if ((words >= 2 || substantial) && words > bestLen) {
      best = { from: segFrom + run[0].s, to: segFrom + run[run.length - 1].e };
      bestLen = words;
    }
    i = j + 1;
  }
  return best;
}

const setRelated = StateEffect.define<SegSuggestion[]>();
/** Dispatched by the "Refresh link suggestions" command to force a re-scan (after the
 *  shared dismissal set has been cleared), so previously-dismissed blocks reappear. */
export const forceRescan = StateEffect.define<null>();

/**
 * How long to wait after a document change before re-scanning for link suggestions. Matching a
 * block runs an embedding comparison per candidate note, so it must not chase every keystroke.
 */
const IDLE_DELAY = 900;
/** Next tick: for changes whose whole point is that the answer just changed (see removesLink). */
const IMMEDIATE = 0;

/**
 * Did this update DELETE link markup?
 *
 * A block showing no icon because it already contains a link becomes eligible again the instant
 * that link is removed - so waiting out the idle delay is exactly wrong there: the user deleted
 * the link and then watched a second pass before the suggestion came back. Deleting a link is a
 * rare, discrete act (unlike typing), so recomputing immediately costs nothing in practice.
 */
function removesLink(u: ViewUpdate): boolean {
  let removed = false;
  u.changes.iterChanges((fromA, toA, _fromB, _toB, inserted) => {
    if (removed) return;
    const before = u.startState.doc.sliceString(fromA, toA);
    if (!/\[\[|\]\]|\]\(/.test(before)) return;
    if (!/\[\[|\]\]|\]\(/.test(inserted.toString())) removed = true;
  });
  return removed;
}

function segKey(text: string): string {
  return text.trim().toLowerCase().slice(0, 80);
}
function wordCount(text: string): number {
  return (text.match(/[a-zA-Z][a-zA-Z'-]*/g) ?? []).length;
}
/** True if the block already contains a wikilink or a markdown link. */
function hasLink(text: string): boolean {
  return /\[\[[^\]\n]+\]\]/.test(text) || /\[[^\]\n]+\]\([^)\n]+\)/.test(text);
}
/** The wikilink text for a candidate, e.g. `[[Note#Heading]]` or `[[Note|alias]]`. */
function linkText(c: RelatedCandidate, alias?: string): string {
  const dest = c.heading ? `${c.display}#${c.heading}` : c.display;
  return alias && alias !== dest ? `[[${dest}|${alias}]]` : `[[${dest}]]`;
}

class LinkIconWidget extends WidgetType {
  constructor(
    private seg: SegSuggestion,
    private open: (seg: SegSuggestion, iconEl: HTMLElement, view: EditorView) => void,
  ) {
    super();
  }
  eq(other: LinkIconWidget): boolean {
    // Anchor MUST be part of identity: when the doc shifts and the field re-maps the
    // anchor, a widget that compared equal would be reused with its old anchor, and a
    // no-selection insert would land at the stale position. Comparing the anchor forces a
    // replacement so the click always inserts where the icon currently is.
    return (
      other.seg.key === this.seg.key &&
      other.seg.anchor === this.seg.anchor &&
      other.seg.candidates[0]?.target === this.seg.candidates[0]?.target
    );
  }
  toDOM(view: EditorView): HTMLElement {
    const el = createSpan({ cls: "smart-autocorrect-link-icon" });
    el.setAttribute("aria-label", `${this.seg.candidates.length} related note${this.seg.candidates.length === 1 ? "" : "s"}`);
    setIcon(el, "link");
    el.onmousedown = (e) => {
      e.preventDefault();
      e.stopPropagation();
      this.open(this.seg, el, view);
    };
    return el;
  }
  ignoreEvent(): boolean {
    return false;
  }
}

/** The icon's chooser: the shared LinkChooser, opened for one segment's candidates. */
class RelatedPopover {
  private chooser: LinkChooser;

  constructor(
    private onInsert: (seg: SegSuggestion, c: RelatedCandidate, view: EditorView) => void,
    private onDismiss: (seg: SegSuggestion, view: EditorView) => void,
    /** Render markdown into an element (lists, bold, etc.), so the preview looks like the note. */
    renderMarkdown: (md: string, el: HTMLElement, sourcePath: string) => void,
  ) {
    this.chooser = new LinkChooser(renderMarkdown);
  }

  close(): void {
    this.chooser.close();
  }

  open(seg: SegSuggestion, iconEl: HTMLElement, view: EditorView): void {
    const hasSelection = !view.state.selection.main.empty;
    this.chooser.open({
      candidates: seg.candidates,
      // Line the list up with the icon that opened it, so the connection is obvious; the dock
      // moves it up if the preview would otherwise have no room.
      preferredTop: iconEl.getBoundingClientRect().top,
      title: "Link this section to…",
      hint: hasSelection
        ? "Your selected text will become the link."
        : "Tip: select text first to link just that phrase.",
      onChoose: (c) => this.onInsert(seg, c, view),
      onDismiss: () => this.onDismiss(seg, view),
    });
  }
}

export function relatedLinksExtension(
  app: App,
  index: RelatedIndex,
  getSettings: () => PredictiveSettings,
  isExcluded: () => boolean,
  dismissed: Set<string>,
  renderMarkdown: (md: string, el: HTMLElement, sourcePath: string) => void,
) {
  const popover = new RelatedPopover(
    (seg, c, view) => {
      const sel = view.state.selection.main;
      if (!sel.empty) {
        // Highlight-to-link: turn the selected text into the link.
        const alias = view.state.doc.sliceString(sel.from, sel.to);
        view.dispatch({ changes: { from: sel.from, to: sel.to, insert: linkText(c, alias) } });
      } else {
        const before = view.state.doc.sliceString(Math.max(0, seg.anchor - 1), seg.anchor);
        const pad = before && before !== " " && before !== "\n" ? " " : "";
        view.dispatch({ changes: { from: seg.anchor, insert: `${pad}${linkText(c)}` } });
      }
      // Remove the icon now. We do NOT permanently dismiss: the inserted [[link]] means the
      // next recompute excludes this target (it's already linked), so the icon stays away -
      // but if the link is later deleted or undone, recompute brings the suggestion back.
      view.dispatch({ effects: setRelated.of(currentFor(view).filter((s) => s.key !== seg.key)) });
    },
    (seg, view) => {
      dismissed.add(seg.key);
      view.dispatch({ effects: setRelated.of(currentFor(view).filter((s) => s.key !== seg.key)) });
    },
    renderMarkdown,
  );
  const openPopover = (seg: SegSuggestion, iconEl: HTMLElement, view: EditorView) => popover.open(seg, iconEl, view);

  const LINK_MARK = Decoration.mark({ class: "smart-autocorrect-link-underline" });
  const field = StateField.define<{ segs: SegSuggestion[]; deco: DecorationSet }>({
    create: () => ({ segs: [], deco: Decoration.none }),
    update(value, tr) {
      let segs = value.segs;
      for (const e of tr.effects) if (e.is(setRelated)) segs = e.value;
      if (segs === value.segs && !tr.docChanged) return value;
      const mapped = tr.docChanged
        ? segs.map((s) => ({
            ...s,
            anchor: tr.changes.mapPos(s.anchor),
            from: tr.changes.mapPos(s.from),
            to: tr.changes.mapPos(s.to),
            span: s.span ? { from: tr.changes.mapPos(s.span.from), to: tr.changes.mapPos(s.span.to) } : undefined,
          }))
        : segs;
      const cfg = getSettings();
      // The icon and the underline are independent: the end-of-segment icon follows `suggestLinks`,
      // the inline underline follows `underlineLinks`. Both are added in ascending position order.
      const entries: Array<{ from: number; to: number; order: number; deco: Decoration }> = [];
      let last = -1;
      for (const s of [...mapped].sort((a, c) => a.anchor - c.anchor)) {
        if (s.anchor <= last) continue;
        last = s.anchor;
        if (cfg.suggestLinks)
          entries.push({ from: s.anchor, to: s.anchor, order: 1, deco: Decoration.widget({ widget: new LinkIconWidget(s, openPopover), side: 1 }) });
        if (cfg.underlineLinks && s.span && s.span.from < s.span.to)
          entries.push({ from: s.span.from, to: s.span.to, order: 0, deco: LINK_MARK });
      }
      entries.sort((a, c) => a.from - c.from || a.to - c.to || a.order - c.order);
      const b = new RangeSetBuilder<Decoration>();
      for (const e of entries) b.add(e.from, e.to, e.deco);
      return { segs: mapped, deco: b.finish() };
    },
    provide: (f) => EditorView.decorations.from(f, (v) => v.deco),
  });

  const currentFor = (view: EditorView) => view.state.field(field).segs;

  const runner = ViewPlugin.fromClass(
    class {
      private timer = 0;
      private token = 0;
      constructor(view: EditorView) {
        this.schedule(view, IDLE_DELAY);
      }
      update(u: ViewUpdate) {
        const forced = u.transactions.some((t) => t.effects.some((e) => e.is(forceRescan)));
        // NOT on selectionSet: selecting/unselecting text doesn't change content, so it must not
        // drop and re-add icons (that flicker was removing the icon just as you selected a phrase
        // to link with it, #5). Content edits re-evaluate the "block you're typing in" skip.
        if (u.docChanged || u.viewportChanged || forced)
          this.schedule(u.view, forced || removesLink(u) ? IMMEDIATE : IDLE_DELAY);
      }
      destroy() {
        window.clearTimeout(this.timer);
        popover.close();
      }
      private schedule(view: EditorView, delay: number) {
        window.clearTimeout(this.timer);
        this.timer = window.setTimeout(() => void this.recompute(view), delay);
      }
      private async recompute(view: EditorView) {
        const s = getSettings();
        const mine = ++this.token;
        // The same semantic pass feeds both the end-of-segment icons (suggestLinks) and the inline
        // underline (underlineLinks), so run whenever EITHER is on; the field then shows each part
        // per its own setting.
        if (!s.pluginEnabled || (!s.suggestLinks && !s.underlineLinks) || isExcluded()) {
          if (currentFor(view).length) view.dispatch({ effects: setRelated.of([]) });
          return;
        }
        const minWords = Math.max(3, s.minLinkWords);
        const doc = view.state.doc.toString();
        const cursor = view.state.selection.main.head;
        const excludePath = app.workspace.getActiveFile()?.path;
        const excludeSelf = app.workspace.getActiveFile()?.basename?.toLowerCase();
        const out: SegSuggestion[] = [];
        for (const seg of segmentText(doc)) {
          const key = segKey(seg.text);
          if (dismissed.has(key)) continue;
          // Precondition: never show the icon on a block that already contains a link. This
          // is what makes it come back automatically - delete the link and the block qualifies
          // again on the next scan.
          if (hasLink(seg.text)) continue;
          if (wordCount(seg.text) < minWords) continue; // too short to link
          if (cursor >= seg.from && cursor <= seg.to) continue; // the block you're typing in
          const exclude = new Set<string>();
          if (excludeSelf) exclude.add(excludeSelf);
          const queryText = seg.heading ? `${seg.heading}. ${seg.text}` : seg.text;
          // Ambient icons demand a STRONG connection (two corroborating signals) - the
          // explicit [[ picker stays looser, since there the user has already chosen to link.
          const cands = await index.candidatesFor(queryText, excludePath, exclude, 5, s.relatedSensitivity, true);
          if (this.token !== mine) return;
          if (cands.length > 0) {
            const span = pickAnchorSpan(seg.text, seg.from, cands[0]) ?? undefined;
            out.push({ anchor: seg.anchor, from: seg.from, to: seg.to, key, candidates: cands, span });
          }
        }
        if (this.token !== mine) return;
        view.dispatch({ effects: setRelated.of(out) });
      }
    },
  );

  const theme = EditorView.baseTheme({
    ".smart-autocorrect-link-icon": {
      display: "inline-flex",
      alignItems: "center",
      verticalAlign: "middle",
      marginLeft: "4px",
      cursor: "pointer",
      opacity: "0.45",
      color: "var(--text-accent)",
    },
    ".smart-autocorrect-link-icon:hover": { opacity: "1" },
    ".smart-autocorrect-link-icon svg": { width: "0.85em", height: "0.85em" },
    ".smart-autocorrect-link-underline": {
      textDecoration: "underline",
      textDecorationStyle: "dotted",
      textDecorationColor: "color-mix(in srgb, var(--text-accent) 35%, transparent)",
      textUnderlineOffset: "3px",
      cursor: "pointer",
    },
    ".smart-autocorrect-link-underline:hover": { textDecorationColor: "var(--text-accent)" },
  });

  // Click / hover on the underlined phrase: same chooser as the icon, but the phrase itself becomes
  // the link. We select the span first, so the chooser's highlight-to-link path applies it.
  const segAt = (view: EditorView, evt: MouseEvent): SegSuggestion | null => {
    const pos = view.posAtCoords({ x: evt.clientX, y: evt.clientY });
    if (pos == null) return null;
    return currentFor(view).find((s) => s.span && pos >= s.span.from && pos <= s.span.to) ?? null;
  };
  const underlineHandlers = EditorView.domEventHandlers({
    mousedown(evt, view) {
      const el = (evt.target as HTMLElement)?.closest?.(".smart-autocorrect-link-underline") as HTMLElement | null;
      if (!el) return false;
      const seg = segAt(view, evt);
      if (!seg?.span) return false;
      evt.preventDefault();
      view.dispatch({ selection: { anchor: seg.span.from, head: seg.span.to } });
      popover.open(seg, el, view);
      return true;
    },
    mouseover(evt, view) {
      const el = (evt.target as HTMLElement)?.closest?.(".smart-autocorrect-link-underline") as HTMLElement | null;
      if (!el) return false;
      const seg = segAt(view, evt);
      const target = seg?.candidates[0]?.target;
      if (!target) return false;
      app.workspace.trigger("hover-link", { event: evt, source: "smart-autocorrect", hoverParent: view.dom, targetEl: el, linktext: target });
      return false;
    },
  });

  return [field, runner, underlineHandlers, theme];
}
