/**
 * The link chooser: the list half of the right-docked link column, plus its section preview.
 *
 * One implementation shared by every route that offers our recommendations - the ambient link
 * icon and "link this selection" use it directly, and the `[[` picker is styled to match it
 * (it must stay an Obsidian EditorSuggest to keep the native typing behaviour, so it cannot
 * literally be this class). Layout and behaviour therefore only exist in one place.
 *
 * Deliberately NOT a Modal. The selection flow has to keep the editor's selection visible and
 * intact while you choose - a modal would dim the note and drop the very selection you are
 * about to link. A plain floating element leaves the editor untouched, so we take the keys we
 * need in the capture phase and let everything else through.
 */
import { renderLinkCard } from "./linkCard";
import { SectionPreview } from "./SectionPreview";
import { placeList } from "./linkDock";
import type { RelatedCandidate } from "./RelatedIndex";

export interface ChooserOptions {
  candidates: RelatedCandidate[];
  /** Where the user acted (the icon, or the caret). The list is placed near this, within the
   *  dock's constraints. */
  preferredTop: number;
  title: string;
  hint: string;
  onChoose: (c: RelatedCandidate) => void;
  /** Optional "Dismiss" button; omitted when there is nothing to dismiss (the selection flow). */
  onDismiss?: () => void;
}

export class LinkChooser {
  private el: HTMLElement | null = null;
  private preview: SectionPreview;
  private rows: HTMLElement[] = [];
  private opts: ChooserOptions | null = null;
  private active = -1;

  private onDoc = (e: MouseEvent) => {
    if (this.el && !this.el.contains(e.target as Node)) this.close();
  };
  private onKey = (e: KeyboardEvent) => {
    if (!this.el || !this.opts) return;
    // Arrow keys move through the candidates AND update the preview, so the section you are
    // about to link is visible without ever touching the mouse.
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      e.stopPropagation();
      const n = this.opts.candidates.length;
      if (n > 0) this.select((this.active + (e.key === "ArrowDown" ? 1 : n - 1)) % n);
      return;
    }
    if (e.key === "Enter" && this.active >= 0) {
      e.preventDefault();
      e.stopPropagation();
      const c = this.opts.candidates[this.active];
      this.close();
      this.opts.onChoose(c);
      return;
    }
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      this.close();
    }
  };

  constructor(private renderMarkdown: (md: string, el: HTMLElement, sourcePath: string) => void) {
    this.preview = new SectionPreview(renderMarkdown);
  }

  get isOpen(): boolean {
    return this.el !== null;
  }

  close(): void {
    this.preview.hide();
    this.rows = [];
    this.opts = null;
    this.active = -1;
    if (!this.el) return;
    this.el.remove();
    this.el = null;
    document.removeEventListener("mousedown", this.onDoc, true);
    document.removeEventListener("keydown", this.onKey, true);
  }

  /** Highlight candidate `i` and show its section in the preview window below the list. */
  private select(i: number): void {
    if (!this.opts || !this.el) return;
    const c = this.opts.candidates[i];
    if (!c) return;
    this.active = i;
    this.rows.forEach((r, j) => r.toggleClass("is-selected", j === i));
    this.rows[i]?.scrollIntoView({ block: "nearest" });
    const title = c.heading ? `${c.display} › ${c.heading}` : c.display;
    this.preview.show(
      this.el.getBoundingClientRect(),
      title,
      c.sectionText || c.snippet,
      c.path,
      `${c.target}#${c.heading ?? ""}`,
    );
  }

  open(opts: ChooserOptions): void {
    this.close();
    this.opts = opts;
    const panel = createDiv({ cls: "smart-autocorrect-related-popover" });

    const head = panel.createDiv({ cls: "sa-rel-head" });
    head.createSpan({ cls: "sa-rel-title", text: opts.title });
    if (opts.onDismiss) {
      const dismiss = head.createEl("button", { cls: "sa-rel-dismiss", text: "Dismiss" });
      dismiss.onclick = () => {
        const cb = opts.onDismiss!;
        this.close();
        cb();
      };
    }

    const list = panel.createDiv({ cls: "sa-rel-list" });
    this.rows = opts.candidates.map((c, i) => {
      const card = list.createDiv();
      renderLinkCard(card, {
        name: c.display,
        heading: c.heading,
        snippet: c.snippet,
        scorePct: Math.round(c.score * 100),
      });
      card.addEventListener("mouseenter", () => this.select(i));
      card.onclick = () => {
        this.close();
        opts.onChoose(c);
      };
      return card;
    });

    panel.createDiv({ cls: "sa-rel-hint", text: opts.hint });

    document.body.appendChild(panel);
    this.el = panel;
    placeList(panel, opts.preferredTop);
    if (opts.candidates.length > 0) this.select(0);
    window.setTimeout(() => {
      document.addEventListener("mousedown", this.onDoc, true);
      document.addEventListener("keydown", this.onKey, true);
    }, 0);
  }
}
