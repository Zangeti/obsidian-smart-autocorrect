/**
 * Shared geometry for the link UI.
 *
 * Every route that offers links - the ambient link icon, the `[[` picker, and "link this
 * selection" - shows the same two windows: a compact LIST of candidates with a SECTION PREVIEW
 * directly beneath it. This module owns where those two windows go, so the three routes cannot
 * drift apart and there is exactly one place to change the layout.
 *
 * The rules, and why:
 *
 *  - Both windows share ONE width and ONE right edge, pinned to the right of the viewport. A
 *    column hugging the edge covers the least prose; two windows of different widths read as
 *    two unrelated popups rather than one thing.
 *  - The list sits ABOVE the preview. It is the part you navigate, so it belongs at a fixed,
 *    predictable place; the preview is the part that changes as you move.
 *  - The list's position NEVER depends on the preview. That is what stops the whole UI jumping
 *    when you arrow onto a longer section - the previous layout placed the preview relative to
 *    the list and then flipped it above when it did not fit, which moved the list under the
 *    cursor. Instead the list is placed once, with room reserved beneath it, and the preview
 *    takes whatever space is left and SCROLLS when it needs more.
 */

/** Gap from the viewport edges, and between the two windows. Pixels. */
export const DOCK_MARGIN = 12;
export const DOCK_GAP = 6;

/**
 * The shared width of both windows, read from the `--sa-link-dock-width` custom property so
 * the value lives in the stylesheet with the rest of the look, and the `[[` picker (whose
 * element Obsidian owns and which we can only reach through CSS) is guaranteed the same number.
 */
export function dockWidth(): number {
  const raw = getComputedStyle(document.body).getPropertyValue("--sa-link-dock-width");
  const n = Number.parseFloat(raw);
  const width = Number.isFinite(n) && n > 0 ? n : 340;
  return Math.min(width, window.innerWidth - 2 * DOCK_MARGIN);
}

/** Left coordinate of the docked column, for a given width. */
export function dockLeft(width: number): number {
  return Math.max(DOCK_MARGIN, window.innerWidth - width - DOCK_MARGIN);
}

/**
 * Place the LIST window. `preferredTop` is where the caller would like it (beside the icon, at
 * the cursor); it is honoured unless that would leave the preview no room.
 *
 * Room is reserved below the list equal to the list's own height: an even split between "what
 * I am choosing from" and "what I am looking at", derived from the list rather than from a
 * chosen number, and - crucially - independent of the preview's content, so nothing moves as
 * you arrow through sections of different sizes.
 */
export function placeList(el: HTMLElement, preferredTop: number): void {
  const width = dockWidth();
  el.style.width = `${width}px`;
  el.style.left = `${dockLeft(width)}px`;
  const listH = el.offsetHeight;
  const latest = window.innerHeight - DOCK_MARGIN - listH - DOCK_GAP - listH;
  const top = Math.max(DOCK_MARGIN, Math.min(preferredTop, Math.max(DOCK_MARGIN, latest)));
  el.style.top = `${top}px`;
}

/**
 * Dock the `[[` picker's menu, whose element Obsidian creates AND positions itself.
 *
 * The geometry is written inline rather than from the stylesheet for two reasons: Obsidian
 * sets `left` and `top` inline, which no stylesheet rule can override without `!important`;
 * and reaching the element from CSS at all meant matching it with `:has()`, making every
 * suggester in the app pay for a selector that only ever applies to this one. The caller
 * already holds the element, so it can simply be measured and placed like our own windows.
 */
export function dockSuggestionMenu(el: HTMLElement): void {
  el.classList.add("sa-link-dock");
  const width = dockWidth();
  el.style.width = `${width}px`;
  el.style.left = `${dockLeft(width)}px`;
}

/**
 * Place the PREVIEW window directly beneath `listRect`, filling the space left to the bottom of
 * the viewport. It never moves the list and never flips above it: when the section is taller
 * than the space available the window scrolls internally instead, which keeps the one thing the
 * user is steering - the list - perfectly still.
 */
export function placePreview(el: HTMLElement, listRect: DOMRect): void {
  const width = dockWidth();
  el.style.width = `${width}px`;
  el.style.left = `${dockLeft(width)}px`;
  const top = listRect.bottom + DOCK_GAP;
  el.style.top = `${top}px`;
  el.style.maxHeight = `${Math.max(0, window.innerHeight - DOCK_MARGIN - top)}px`;
}
