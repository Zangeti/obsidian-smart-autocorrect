/**
 * Replacement for Obsidian's built-in `[[` link picker - and for every other route that
 * inserts an internal link (the "Add internal link" command, which also types `[[`, so the
 * text trigger below catches it). It shows the SAME experience as the link-icon chooser:
 *   - our topical RECOMMENDATIONS first (each links straight to `[[Note#Heading]]`), with a
 *     detailed section preview in a separate floating window on hover;
 *   - then the COMPLETE list of your notes, greyed, so every note still exists and is
 *     selectable - exactly like Obsidian's default `[[` view, just with ours on top.
 *
 * Registered ahead of the core picker (see PredictiveFeature) so it wins the `[[` trigger
 * when enabled; when the setting is off, onTrigger returns null and the built-in picker runs.
 */
import { EditorSuggest, TFile, type App, type Editor, type EditorPosition, type EditorSuggestContext, type EditorSuggestTriggerInfo } from "obsidian";
import type { LinkIndex } from "./LinkIndex";
import type { RelatedIndex, RelatedCandidate } from "./RelatedIndex";
import type { PredictiveSettings } from "./PredictiveSettings";
import { SectionPreview } from "./SectionPreview";
import { renderLinkCard } from "./linkCard";
import { dockSuggestionMenu } from "./linkDock";

interface LinkItem {
  target: string;
  display: string;
  path?: string;
  heading?: string;
  snippet?: string;
  sectionText?: string; // matched section, for the hover preview (recommendations only)
  scorePct?: number; // match strength for the badge (recommendations only)
  related: boolean;
}

/** `[[` followed by the (not-yet-closed) link text. Matches whether `[[` was typed or
 *  inserted by the "Add internal link" command. */
const TRIGGER = /\[\[([^\]\n|]*)$/;
/** Upper bound on the greyed "all notes" list, so a huge vault can't render thousands of rows
 *  at once; typing narrows it via the title search. */
const ALL_NOTES_CAP = 500;

export class LinkPicker extends EditorSuggest<LinkItem> {
  private index: LinkIndex;
  private related: RelatedIndex;
  private getSettings: () => PredictiveSettings;
  private preview: SectionPreview;
  /** Lazily-read note bodies for the hover preview of a plain (non-recommended) note. */
  private bodyCache = new Map<string, string>();
  /** Items in render order, indexed by the `data-sa-index` stamped on each row. */
  private rendered: LinkItem[] = [];
  private selectionObserver: MutationObserver | null = null;
  private observedEl: HTMLElement | null = null;

  constructor(
    app: App,
    index: LinkIndex,
    related: RelatedIndex,
    getSettings: () => PredictiveSettings,
    renderMarkdown: (md: string, el: HTMLElement, sourcePath: string) => void,
  ) {
    super(app);
    this.index = index;
    this.related = related;
    this.getSettings = getSettings;
    this.preview = new SectionPreview(renderMarkdown);
  }

  onTrigger(cursor: EditorPosition, editor: Editor): EditorSuggestTriggerInfo | null {
    const s = this.getSettings();
    if (!s.pluginEnabled || !s.replaceLinkMenu) return null;
    const before = editor.getLine(cursor.line).slice(0, cursor.ch);
    const m = TRIGGER.exec(before);
    if (!m) return null;
    const query = m[1];
    return { start: { line: cursor.line, ch: cursor.ch - query.length }, end: cursor, query };
  }

  async getSuggestions(context: EditorSuggestContext): Promise<LinkItem[]> {
    this.rendered = []; // rows are about to be re-rendered; indices restart
    const q = context.query.trim();
    const file = context.file;
    const exclude = new Set<string>();
    if (file?.basename) exclude.add(file.basename.toLowerCase());

    // Our recommendations: sections of other notes that topically match what you're writing.
    const para = paragraphAround(context.editor, context.start.line);
    const sens = Math.min(5, this.getSettings().relatedSensitivity + 1); // explicit ask: loosen a touch
    const related: RelatedCandidate[] = para
      ? await this.related.candidatesFor(para, file?.path, exclude, 8, sens)
      : [];
    const recByTarget = new Map<string, RelatedCandidate>();
    for (const c of related) if (!recByTarget.has(c.target.toLowerCase())) recByTarget.set(c.target.toLowerCase(), c);

    const items: LinkItem[] = [];
    const seen = new Set<string>();
    const pushOnce = (it: LinkItem) => {
      const key = `${it.target.toLowerCase()}#${it.heading ?? ""}`;
      if (seen.has(key)) return;
      seen.add(key);
      items.push(it);
    };
    const asItem = (c: RelatedCandidate): LinkItem => ({
      target: c.target, display: c.display, path: c.path, heading: c.heading,
      snippet: c.snippet, sectionText: c.sectionText, scorePct: Math.round(c.score * 100), related: true,
    });

    if (q.length > 0) {
      // Typed a name: our recommendations that also match the name first (normal), then EVERY
      // other title match, greyed - the complete list, so any note is reachable by typing.
      const matches = this.index.searchTitles(q, 50);
      for (const t of matches) {
        const rec = recByTarget.get(t.target.toLowerCase());
        if (rec) pushOnce(asItem(rec));
      }
      for (const t of matches) pushOnce({ target: t.target, display: t.display, related: false });
      return items;
    }

    // Empty query: recommendations first (normal), then the COMPLETE list of notes, greyed,
    // most-recent first (capped so a giant vault stays responsive; typing filters the rest).
    for (const c of related) pushOnce(asItem(c));
    const recent = this.app.vault.getMarkdownFiles().sort((a, b) => b.stat.mtime - a.stat.mtime);
    for (const f of recent) {
      if (items.length >= ALL_NOTES_CAP) break;
      if (f === file) continue;
      pushOnce({ target: f.basename, display: f.basename, path: f.path, related: false });
    }
    return items;
  }

  renderSuggestion(item: LinkItem, el: HTMLElement): void {
    // Remember which item each row shows, so the selection watcher can map the row Obsidian
    // highlights back to its item without re-deriving it from the DOM text.
    const index = this.rendered.push(item) - 1;
    el.dataset.saIndex = String(index);
    // Dock the menu itself the first time a row is drawn for this batch. Obsidian positions
    // this element after it fills it, so the placement is repeated on the next frame to land
    // last; both writes are idempotent, so the menu never visibly moves.
    if (index === 0) {
      const dock = () => {
        const menu = el.closest<HTMLElement>(".suggestion-container");
        if (menu) dockSuggestionMenu(menu);
      };
      dock();
      // Repeated on the next frame because Obsidian positions the menu after filling it, and
      // because the row is not guaranteed to be in the document yet on this call. Both writes
      // are idempotent, so the menu never visibly moves.
      window.requestAnimationFrame(dock);
    }
    // The SAME card layout as the link-icon chooser, so the two menus look identical; the
    // "other" notes (offered for completeness) render muted.
    renderLinkCard(el, {
      name: item.display,
      heading: item.heading,
      snippet: item.snippet,
      scorePct: item.related ? item.scorePct : undefined,
      muted: !item.related,
    });
    // Hovering a row opens the detailed section in the SAME separate preview window the
    // link-icon chooser uses (recommendations show their matched section; a plain note shows
    // the note's opening, read on demand).
    el.addEventListener("mouseenter", () => this.showPreview(item, el));
  }

  /** Anchor the preview on the whole suggestion menu (not the hovered row), so it sits below
   *  the list as one docked column and doesn't jump row-to-row. Falls back to the row if the
   *  container can't be found. */
  private menuRect(rowEl: HTMLElement): DOMRect {
    const menu = rowEl.closest<HTMLElement>(".suggestion-container");
    return (menu ?? rowEl).getBoundingClientRect();
  }

  private showPreview(item: LinkItem, rowEl: HTMLElement): void {
    this.watchSelection(rowEl);
    const title = item.heading ? `${item.display} › ${item.heading}` : item.display;
    const key = `${item.target}#${item.heading ?? ""}`;
    const rect = this.menuRect(rowEl);
    if (item.sectionText) {
      this.preview.show(rect, title, item.sectionText, item.path ?? "", key);
      return;
    }
    const path = item.path ?? this.app.metadataCache.getFirstLinkpathDest(item.target, "")?.path;
    const cached = path ? this.bodyCache.get(path) : undefined;
    if (cached !== undefined) {
      this.preview.show(rect, title, cached, path!, key);
      return;
    }
    const f = path ? this.app.vault.getAbstractFileByPath(path) : null;
    if (f instanceof TFile) {
      void this.app.vault.cachedRead(f).then((txt) => {
        const body = stripFrontmatter(txt).slice(0, 600);
        this.bodyCache.set(f.path, body);
        this.preview.show(this.menuRect(rowEl), title, body, f.path, key);
      });
    } else {
      this.preview.show(rect, title, "", "", key); // nothing to read: just the title
    }
  }

  /**
   * Preview whatever row is SELECTED, not just whatever the mouse is over.
   *
   * EditorSuggest owns its keyboard navigation and offers no hook for "the selection moved",
   * so we watch the class it puts on the chosen row. This is what makes arrowing through the
   * list show each section, which is how the list is actually used - the mouse is the
   * exception, not the rule. One observer for the life of the menu; re-pointed if Obsidian
   * rebuilds the container.
   */
  private watchSelection(rowEl: HTMLElement): void {
    const container = rowEl.closest(".suggestion") ?? rowEl.parentElement;
    if (!container || container === this.observedEl) return;
    this.selectionObserver?.disconnect();
    this.observedEl = container as HTMLElement;
    this.selectionObserver = new MutationObserver(() => {
      const sel = this.observedEl?.querySelector(".is-selected") as HTMLElement | null;
      if (!sel) return;
      const i = Number(sel.dataset.saIndex);
      const item = Number.isInteger(i) ? this.rendered[i] : undefined;
      if (item) this.showPreview(item, sel);
    });
    this.selectionObserver.observe(container, {
      subtree: true,
      attributes: true,
      attributeFilter: ["class"],
    });
  }

  close(): void {
    this.preview.hide();
    this.selectionObserver?.disconnect();
    this.selectionObserver = null;
    this.observedEl = null;
    this.rendered = [];
    super.close();
  }

  selectSuggestion(item: LinkItem): void {
    this.preview.hide();
    const ctx = this.context;
    if (!ctx) return;
    const editor = ctx.editor;
    const line = editor.getLine(ctx.end.line);
    // Replace the `[[` too (start-2) and swallow an auto-inserted `]]` if present.
    const from: EditorPosition = { line: ctx.start.line, ch: Math.max(0, ctx.start.ch - 2) };
    let endCh = ctx.end.ch;
    if (line.slice(endCh, endCh + 2) === "]]") endCh += 2;
    const insert = item.heading ? `[[${item.target}#${item.heading}]]` : `[[${item.target}]]`;
    editor.replaceRange(insert, from, { line: ctx.end.line, ch: endCh });
    editor.setCursor({ line: from.line, ch: from.ch + insert.length });
  }
}

/** The paragraph (blank-line-delimited) containing `line`, as plain text. */
function paragraphAround(editor: Editor, line: number): string {
  const last = editor.lineCount() - 1;
  let a = line, b = line;
  while (a > 0 && editor.getLine(a - 1).trim() !== "") a--;
  while (b < last && editor.getLine(b + 1).trim() !== "") b++;
  const parts: string[] = [];
  for (let i = a; i <= b; i++) parts.push(editor.getLine(i));
  return parts.join(" ").replace(/\[\[[^\]]*$/, "").trim();
}

/** Drop a leading YAML frontmatter block so the preview shows prose, not tags/date. */
function stripFrontmatter(text: string): string {
  return text.replace(/^\uFEFF?---\r?\n[\s\S]*?\r?\n---[ \t]*\r?\n?/, "").trim();
}
