/**
 * An in-memory index of the vault's note titles, aliases, and tags - the retrieval side
 * of link/tag suggestion. Built from Obsidian's metadata cache (no file reads), and kept
 * fresh on vault changes. The neural model is NOT involved: a link is only ever proposed
 * to a note that actually exists, so suggestions can't be hallucinated.
 */
import type { App, TFile } from "obsidian";
import { findLinkSpans, termFreq, type LinkSpan, type LinkTarget } from "./engine/index";

export class LinkIndex {
  private app: App;
  /** lower-cased title/alias -> link target (first note to claim a phrase wins). */
  private phrase = new Map<string, LinkTarget>();
  /** tag name (no #) -> how many notes use it. */
  tags = new Map<string, number>();

  constructor(app: App) {
    this.app = app;
  }

  rebuild(): void {
    this.phrase.clear();
    this.tags.clear();
    for (const f of this.app.vault.getMarkdownFiles()) this.add(f);
  }

  private add(f: TFile): void {
    const base = f.basename;
    if (base.length >= 3 && !/^\d+([.\-/]\d+)*$/.test(base)) {
      const key = base.toLowerCase();
      if (!this.phrase.has(key)) this.phrase.set(key, { target: base, display: base });
    }
    const cache = this.app.metadataCache.getFileCache(f);
    // Aliases (frontmatter `aliases:` or `alias:`), string or list.
    const fm: Record<string, unknown> | undefined = cache?.frontmatter;
    let aliases = fm?.aliases ?? fm?.alias;
    if (typeof aliases === "string") aliases = [aliases];
    if (Array.isArray(aliases))
      for (const a of aliases)
        if (typeof a === "string" && a.length >= 3) {
          const key = a.toLowerCase();
          if (!this.phrase.has(key)) this.phrase.set(key, { target: base, display: a });
        }
    // Tags: inline (#tag) and frontmatter.
    for (const t of cache?.tags ?? []) this.bumpTag(t.tag);
    const fmTags = fm?.tags;
    const list =
      typeof fmTags === "string" ? fmTags.split(/[\s,]+/) : Array.isArray(fmTags) ? fmTags : [];
    for (const t of list) if (typeof t === "string" && t) this.bumpTag(t);
  }

  private bumpTag(raw: string): void {
    const name = raw.replace(/^#/, "").toLowerCase();
    if (name) this.tags.set(name, (this.tags.get(name) ?? 0) + 1);
  }

  /** Full rebuild on any change - the metadata cache is in-memory, so re-indexing a few
   *  thousand notes is a few milliseconds; called debounced from PredictiveFeature. */
  refresh(): void {
    this.rebuild();
  }

  lookup = (phrase: string): LinkTarget | null => this.phrase.get(phrase) ?? null;

  /** Note titles/aliases matching `query` (substring, case-insensitive), prefix matches
   *  first. One entry per note. Powers the [[ picker's name search. */
  searchTitles(query: string, limit: number): LinkTarget[] {
    const q = query.toLowerCase();
    const seen = new Set<string>();
    const out: LinkTarget[] = [];
    for (const [key, t] of this.phrase) {
      if (q && !key.includes(q)) continue;
      if (seen.has(t.target)) continue;
      seen.add(t.target);
      out.push(t);
    }
    out.sort((a, b) => {
      const ap = a.display.toLowerCase().startsWith(q) ? 0 : 1;
      const bp = b.display.toLowerCase().startsWith(q) ? 0 : 1;
      return ap !== bp ? ap - bp : a.display.localeCompare(b.display);
    });
    return out.slice(0, limit);
  }

  findLinks(text: string, excludeTarget?: string): LinkSpan[] {
    if (this.phrase.size === 0) return [];
    return findLinkSpans(text, this.lookup, { excludeTarget });
  }

  /**
   * Existing vault tags relevant to `text`, best first. A tag is relevant when its name
   * (or its final path segment, so `#area/biology` matches on "biology") appears in the
   * note. Ranked by how strongly the note is *about* it: how often the term occurs here,
   * times how established the tag is across the vault (log of its note count). This keeps
   * tags consistent with the ones you already use rather than inventing new spellings.
   */
  suggestTags(text: string, alreadyApplied: Set<string>): { tag: string; count: number }[] {
    const tf = termFreq(text);
    const counts = new Map<string, number>();
    // termFreq drops hyphens, so also index bare word tokens for multi-word tag leaves.
    for (const w of text.toLowerCase().match(/[a-z0-9][a-z0-9-]{2,}/g) ?? [])
      counts.set(w, (counts.get(w) ?? 0) + 1);

    const scored: { tag: string; count: number; score: number }[] = [];
    for (const [tag, noteCount] of this.tags) {
      if (alreadyApplied.has(tag)) continue;
      const leaf = tag.split("/").pop() ?? tag;
      const hits = (tf.get(tag) ?? 0) + (tf.get(leaf) ?? 0) + (counts.get(tag) ?? 0) + (counts.get(leaf) ?? 0);
      if (hits === 0) continue;
      scored.push({ tag, count: noteCount, score: hits * Math.log(1 + noteCount) });
    }
    return scored
      .sort((a, b) => b.score - a.score)
      .slice(0, 8)
      .map(({ tag, count }) => ({ tag, count }));
  }
}
