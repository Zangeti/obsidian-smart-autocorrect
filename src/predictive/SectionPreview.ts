/**
 * A single, detached floating window that previews the section a link would point at,
 * rendered as markdown. It sits directly BELOW whatever list is open - the link-icon chooser,
 * the `[[` picker, or "link this selection" - as the lower half of one right-docked column.
 * Reused by all three so they look and behave identically; see ./linkDock for the geometry
 * and the reasoning behind it.
 */
import { placePreview } from "./linkDock";

type RenderMd = (md: string, el: HTMLElement, sourcePath: string) => void;

export class SectionPreview {
  private el: HTMLElement | null = null;
  private titleEl: HTMLElement | null = null;
  private bodyEl: HTMLElement | null = null;
  private activeKey = "";

  constructor(private renderMarkdown: RenderMd) {}

  /**
   * Show the section below `listRect` (the rectangle of the whole list window - not of the
   * hovered row, so the preview does not chase the cursor up and down). `key` dedupes repeat
   * hovers of the same section so we don't re-render on every mousemove.
   */
  show(
    listRect: DOMRect,
    title: string,
    sectionText: string,
    sourcePath: string,
    key: string,
  ): void {
    if (!this.el) {
      this.el = document.createElement("div");
      this.el.className = "smart-autocorrect-related-preview";
      this.titleEl = this.el.createDiv({ cls: "sa-rel-preview-title" });
      this.bodyEl = this.el.createDiv({ cls: "sa-rel-preview-body markdown-rendered" });
      document.body.appendChild(this.el);
    }
    if (this.activeKey !== key) {
      this.activeKey = key;
      this.titleEl!.setText(title);
      this.bodyEl!.empty();
      this.renderMarkdown(sectionText || title, this.bodyEl!, sourcePath);
      // A new section starts at its top, however far the previous one was scrolled.
      this.el.scrollTop = 0;
    }
    placePreview(this.el, listRect);
  }

  hide(): void {
    this.activeKey = "";
    if (this.el) {
      this.el.remove();
      this.el = null;
      this.titleEl = null;
      this.bodyEl = null;
    }
  }
}
