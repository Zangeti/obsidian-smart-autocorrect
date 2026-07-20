/**
 * Tag autocomplete, the Obsidian-native way: as soon as you type `#`, a suggester offers
 * tags relevant to THIS note. It ranks by how much the note is about the term and biases
 * toward niche, descriptive words over generic filler (via the model's word-rarity signal),
 * so you get `#photosynthesis`, not `#people`. Existing vault tags come first (keeping your
 * taxonomy consistent); a few new tags drawn from the note's own distinctive terms follow.
 *
 * This is the same idea as the related-LINK chooser, in the surface that suits tags: links
 * live inline in the prose where a concept is mentioned, tags are typed with `#`. One
 * consistent, native feel for both.
 */
import { EditorSuggest, type App, type Editor, type EditorPosition, type EditorSuggestContext, type EditorSuggestTriggerInfo, type TFile } from "obsidian";
import { termFreq } from "./engine/index";
import type { LinkIndex } from "./LinkIndex";
import type { PredictiveEngineController } from "./PredictiveEngineController";
import type { PredictiveSettings } from "./PredictiveSettings";

interface TagItem {
  tag: string;
  count: number; // notes using it (0 for a brand-new tag)
  isNew: boolean;
  score: number;
}

/** `#` then optional tag characters, where the `#` starts a word (not mid-word, not `##`). */
const TRIGGER = /(^|\s)#([\p{L}\p{N}/_-]*)$/u;

/**
 * A tag Obsidian will actually accept: letters, digits, `_`, `-`, `/` only, and at least one
 * character that is not a digit (a purely numeric tag is not a tag).
 *
 * This has to be checked because our NEW-tag candidates come from the note's own words, and a
 * word is not a tag - "shouldn't" tokenises perfectly well but the apostrophe terminates the
 * tag, so accepting it would have written `#shouldn` followed by stray text. Existing vault
 * tags are valid by construction, but they are filtered through the same predicate so there is
 * one definition of "a tag we may offer" rather than two.
 */
const TAG_PATTERN = /^[\p{L}\p{N}_/-]+$/u;
function isValidTag(tag: string): boolean {
  return TAG_PATTERN.test(tag) && /[^\p{N}/_-]/u.test(tag);
}

export class TagSuggest extends EditorSuggest<TagItem> {
  private index: LinkIndex;
  private engine: PredictiveEngineController;
  private getSettings: () => PredictiveSettings;
  private onAccept: (word: string, saved: number) => void;
  private rarityCache = new Map<string, number>();

  constructor(
    app: App,
    index: LinkIndex,
    engine: PredictiveEngineController,
    getSettings: () => PredictiveSettings,
    onAccept: (word: string, saved: number) => void,
  ) {
    super(app);
    this.index = index;
    this.engine = engine;
    this.getSettings = getSettings;
    this.onAccept = onAccept;
  }

  onTrigger(cursor: EditorPosition, editor: Editor): EditorSuggestTriggerInfo | null {
    const st = this.getSettings();
    if (!st.pluginEnabled || !st.suggestTagsOnHash) return null;
    const before = editor.getLine(cursor.line).slice(0, cursor.ch);
    const m = TRIGGER.exec(before);
    if (!m) return null;
    const query = m[2];
    const hashCh = cursor.ch - query.length - 1;
    // Note: a heading is "# " (hash + space), which never matches TRIGGER, so a bare `#`
    // here is always a tag in the making - fine to suggest even at the start of a line.
    return { start: { line: cursor.line, ch: hashCh }, end: cursor, query };
  }

  async getSuggestions(context: EditorSuggestContext): Promise<TagItem[]> {
    const q = context.query.toLowerCase();
    const text = context.editor.getValue();
    const tf = termFreq(text);
    const applied = appliedTags(this.app, context.file);

    // Gather candidate underlying words so we fetch all rarities in one worker call.
    const wordsNeeded = new Set<string>();
    const existing: { tag: string; count: number; leaf: string }[] = [];
    for (const [tag, count] of this.index.tags) {
      if (applied.has(tag)) continue;
      if (!isValidTag(tag)) continue;
      const leaf = tag.split("/").pop() ?? tag;
      if (q && !tag.includes(q) && !leaf.includes(q)) continue;
      existing.push({ tag, count, leaf });
      wordsNeeded.add(leaf);
    }
    const fresh: string[] = [];
    for (const [w] of tf) {
      if (w.length < 4 || this.index.tags.has(w) || applied.has(w)) continue;
      if (!isValidTag(w)) continue; // a word of the note is not automatically a legal tag
      if (q && !w.includes(q)) continue;
      fresh.push(w);
      wordsNeeded.add(w);
    }
    await this.ensureRarities([...wordsNeeded]);
    const rarity = (w: string) => this.rarityCache.get(w) ?? 0.5;

    const items: TagItem[] = [];
    for (const { tag, count, leaf } of existing) {
      const hits = (tf.get(tag) ?? 0) + (tf.get(leaf) ?? 0);
      const match = q ? (tag.startsWith(q) || leaf.startsWith(q) ? 2 : 1) : 1;
      // Relevance to the note × descriptiveness × how established the tag is.
      const score = match * (0.5 + hits) * rarity(leaf) * Math.log(2 + count);
      items.push({ tag, count, isNew: false, score });
    }
    for (const w of fresh) {
      const r = rarity(w);
      if (r < 0.4) continue; // skip generic words as brand-new tags ("people", "thing")
      const score = (tf.get(w) ?? 0) * r * (q && w.startsWith(q) ? 1.4 : 1) * 0.8; // slightly below existing
      items.push({ tag: w, count: 0, isNew: true, score });
    }
    return items.sort((a, b) => b.score - a.score).slice(0, 8);
  }

  renderSuggestion(item: TagItem, el: HTMLElement): void {
    el.addClass("smart-autocorrect-tag-suggestion");
    el.createSpan({ cls: "sa-tag-name", text: `#${item.tag}` });
    el.createSpan({ cls: "sa-tag-meta", text: item.isNew ? "new tag" : `used in ${item.count} note${item.count === 1 ? "" : "s"}` });
  }

  selectSuggestion(item: TagItem): void {
    const ctx = this.context;
    if (!ctx) return;
    ctx.editor.replaceRange(`#${item.tag} `, ctx.start, ctx.end);
    const ch = ctx.start.ch + item.tag.length + 2;
    ctx.editor.setCursor({ line: ctx.start.line, ch });
    // Characters the user didn't have to type = the tag letters beyond what was typed.
    const saved = Math.max(0, item.tag.length - ctx.query.length);
    if (saved > 0) this.onAccept(`#${item.tag}`, saved);
  }

  private async ensureRarities(words: string[]): Promise<void> {
    const missing = words.filter((w) => !this.rarityCache.has(w));
    if (missing.length === 0) return;
    try {
      const r = await this.engine.rarities(missing);
      missing.forEach((w, i) => this.rarityCache.set(w, r[i] ?? 0.5));
    } catch {
      for (const w of missing) this.rarityCache.set(w, 0.5);
    }
  }
}

/** Tags already on the note (inline + frontmatter), lower-cased, so we don't re-suggest. */
function appliedTags(app: App, file: TFile | null): Set<string> {
  const out = new Set<string>();
  if (!file) return out;
  const cache = app.metadataCache.getFileCache(file);
  for (const t of cache?.tags ?? []) out.add(t.tag.replace(/^#/, "").toLowerCase());
  const frontmatter: Record<string, unknown> | undefined = cache?.frontmatter;
  const fm = frontmatter?.tags;
  const list = Array.isArray(fm) ? fm : typeof fm === "string" ? fm.split(/[,\s]+/) : [];
  for (const t of list) if (typeof t === "string" && t) out.add(t.replace(/^#/, "").toLowerCase());
  return out;
}
