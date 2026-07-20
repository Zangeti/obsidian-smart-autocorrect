/**
 * The one card layout used by BOTH the link-icon chooser and the `[[` picker, so the two
 * menus are visually identical. A card is: the note (and matched heading), an optional
 * one-line snippet, and an optional match-strength badge on the right. "Other" notes (offered
 * for completeness but not recommended) render muted.
 */
export interface LinkCard {
  name: string;
  heading?: string;
  snippet?: string;
  /** Match strength 0..100 to show as a badge (recommendations only). */
  scorePct?: number;
  /** Dim the card: a viable but not-recommended note. */
  muted?: boolean;
}

export function renderLinkCard(el: HTMLElement, c: LinkCard): void {
  el.addClass("sa-rel-card");
  if (c.muted) el.addClass("sa-rel-card-muted");
  const main = el.createDiv({ cls: "sa-rel-card-main" });
  main.createDiv({ cls: "sa-rel-name", text: c.heading ? `${c.name}  ›  ${c.heading}` : c.name });
  if (c.snippet) main.createDiv({ cls: "sa-rel-snippet", text: c.snippet });
  const meta = el.createDiv({ cls: "sa-rel-meta" });
  if (c.scorePct != null) meta.createSpan({ cls: "sa-rel-score", text: `${c.scorePct}%` });
}
