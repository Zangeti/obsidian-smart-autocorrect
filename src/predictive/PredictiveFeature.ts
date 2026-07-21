/**
 * Single entry point wiring the predictive/autocorrect feature into the plugin
 * lifecycle: engine + personalisation store + suggester + autocorrect + settings.
 * The fork's `main.ts` constructs this in onload and calls `enable()`.
 */
import { Component, debounce, MarkdownRenderer, MarkdownView, Menu, Notice, Plugin, TFile, type SettingDefinitionItem } from "obsidian";
import type { Editor, MarkdownFileInfo } from "obsidian";
import type { EditorView } from "@codemirror/view";
import { pathExcluded, termFreq } from "./engine/index";
import { PredictiveEngineController } from "./PredictiveEngineController";
import { LinkIndex } from "./LinkIndex";
import { linkSuggestExtension } from "./linkSuggest";
import { RelatedIndex } from "./RelatedIndex";
import { relatedLinksExtension, forceRescan } from "./relatedLinks";
import { TagSuggest } from "./TagSuggest";
import { LinkPicker } from "./LinkPicker";
import { openLinkSelection } from "./LinkSelectionModal";
import { ensureAssets, missingAssets } from "./ModelAssets";
import { LinkChooser } from "./LinkChooser";
import { PersonalizationStore } from "./PersonalizationStore";
import { PredictiveSuggest } from "./PredictiveSuggest";
import { AutocorrectController } from "./AutocorrectController";
import { ghostTextExtension } from "./GhostText";
import { tabIndentGuard, indentEditField } from "./tabIndent";
import { EngagementStore, type EngagementState } from "./EngagementStore";
import { StatsModal, type StatsSnapshot } from "./StatsModal";
import { ConfirmModal } from "./ConfirmModal";
import {
  DEFAULT_PREDICTIVE_SETTINGS,
  buildPredictiveSettingGroups,
  type AccelerationState,
  type PredictiveSettings,
} from "./PredictiveSettings";
import { renderPaneGroups, toSettingDefinitions, type PaneGroup } from "./settingsPane";

const DEV_CORPUS_FILE = "predictive-corpus.txt";
const PACKED_GLOBAL_FILE = "predictive-global.bin";
const LSTM_FILE = "word_lstm.bin";
const WORDLIST_FILE = "wordlist.bin";

export class PredictiveFeature {
  private plugin: Plugin;
  settings: PredictiveSettings;
  store: PersonalizationStore;
  engine: PredictiveEngineController;
  private suggest: PredictiveSuggest | null = null;
  private autocorrect: AutocorrectController | null = null;
  private enabled = false;
  private dirty = new Set<string>();
  private flushDirty: () => void;

  /** Vault title/alias/tag index powering tag suggestions + exact-title underlining. */
  private linkIndex: LinkIndex;
  /** Semantic + keyword index powering related-note link suggestions. */
  private relatedIndex: RelatedIndex;
  /** Titles the user dismissed this session (kept out of link underlining). */
  private dismissedLinks = new Set<string>();
  /** Chunk keys the user dismissed this session (kept out of related-link icons). */
  private dismissedRelated = new Set<string>();
  private refreshLinkIndex: () => void;
  private refreshRelated: () => void;
  /** Files whose frontmatter tags may need re-syncing to their inline #tags. */
  private tagDirty = new Set<string>();
  private flushTagSync: () => void;
  private reconcileDict: () => void;

  /** Gamification: keystrokes-saved total, daily streak, milestones. */
  engagement: EngagementStore;
  /** Set by main.ts so an engagement change (per accept) persists to data.json. */
  onEngagementChange?: () => void;
  /** Set by main.ts so a settings change made at runtime (e.g. undo adding a word to the
   *  dictionary) persists to data.json. */
  onPersistSettings?: () => void;
  private statusBar: HTMLElement | null = null;
  /** The stats dashboard while it is open, so live tallies can be pushed into it. */
  private statsModal: StatsModal | null = null;
  /** The shared right-docked link chooser, used by the "link selection" command. Created
   *  lazily on first use so the markdown renderer is bound to a live plugin instance. */
  private linkChooserInstance: LinkChooser | null = null;
  /** Cached answers the settings pane can only get asynchronously; see AccelerationState. */
  private readonly accelState: AccelerationState = {};
  /** Markdown-render scopes, keyed by the element rendered into (see renderMarkdown). */
  private readonly renderScopes = new Map<HTMLElement, Component>();
  private saveEngagement: () => void;

  constructor(
    plugin: Plugin,
    settings: Partial<PredictiveSettings> | undefined,
    engagement?: Partial<EngagementState>,
  ) {
    this.plugin = plugin;
    this.settings = { ...DEFAULT_PREDICTIVE_SETTINGS, ...(settings ?? {}) };
    this.store = new PersonalizationStore(plugin, this.settings.beta);
    this.engine = new PredictiveEngineController(plugin.app, this.settings, this.store);
    this.engagement = new EngagementStore(engagement);
    this.linkIndex = new LinkIndex(plugin.app);
    this.relatedIndex = new RelatedIndex(plugin.app, this.engine);
    this.flushDirty = debounce(() => void this.processDirty(), 3000, false);
    this.saveEngagement = debounce(() => this.onEngagementChange?.(), 4000, false);
    this.refreshLinkIndex = debounce(() => this.linkIndex.refresh(), 3000, false);
    this.refreshRelated = debounce(() => void this.relatedIndex.rebuild(), 8000, false);
    // No real delay (#6): we react as soon as Obsidian's metadataCache reports the change, which
    // is ALREADY coalesced (it doesn't fire per keystroke - only once the parse settles after you
    // pause), so there is no half-typed tag to guard against and no reason to add our own wait.
    // The 0ms debounce is a next-tick coalesce only, merging the content change and our own
    // frontmatter write into a single reconcile pass. The reconcile is idempotent, so it can't loop.
    this.flushTagSync = debounce(() => void this.processTagSync(), 0, false);
    // No delay of its own: pruning is now a lookup per dictionary word against the corpus's
    // document-frequency table (see reconcileDictionary), not a vault re-read, so it simply runs
    // once the corpus has absorbed the edit. The only wait left is the corpus's own flush, which
    // is there because re-counting a file costs real work - not because pruning does.
    this.reconcileDict = debounce(() => void this.reconcileDictionary(), 0, false);
  }

  /** Reconcile each dirty file's frontmatter tags: with the #tags in its body. */
  private async processTagSync(): Promise<void> {
    if (!this.settings.syncFrontmatterTags) return;
    const paths = [...this.tagDirty];
    this.tagDirty.clear();
    for (const p of paths) {
      const f = this.plugin.app.vault.getAbstractFileByPath(p);
      if (f instanceof TFile) await this.syncFrontmatterTags(f);
    }
  }

  /** Make the note's frontmatter tags: mirror the inline #tags used in its body. Adds tags
   *  that newly appear inline and removes ones whose last inline use was deleted. Idempotent,
   *  so the write it triggers doesn't loop. */
  private async syncFrontmatterTags(file: TFile): Promise<void> {
    const cache = this.plugin.app.metadataCache.getFileCache(file);
    const inline = new Set<string>();
    for (const t of cache?.tags ?? []) inline.add(t.tag.replace(/^#/, ""));
    const current = frontmatterTags(cache?.frontmatter);
    const currentSet = new Set(current);
    const same = current.length === inline.size && current.every((t) => inline.has(t));
    if (same) return;
    const added = [...inline].filter((t) => !currentSet.has(t)).sort();
    const removed = current.filter((t) => !inline.has(t)).sort();
    await this.plugin.app.fileManager.processFrontMatter(file, (fm: Record<string, unknown>) => {
      if (inline.size === 0) delete fm.tags;
      else fm.tags = [...inline].sort();
    });
    const parts: string[] = [];
    if (added.length) parts.push(`added ${added.map((t) => `#${t}`).join(", ")}`);
    if (removed.length) parts.push(`removed ${removed.map((t) => `#${t}`).join(", ")}`);
    if (parts.length) new Notice(`Frontmatter tags: ${parts.join("; ")}`);
  }

  /**
   * Render markdown into `el` under a component scoped to that element, NOT to the plugin.
   * The plugin lives for the whole session, so anything a rendered note registers against it
   * (embeds, callouts, mermaid) would never be released. One scope per target element,
   * unloaded when that element is rendered into again, detached, or the feature shuts down.
   */
  private renderMarkdown = (md: string, el: HTMLElement, sourcePath: string): void => {
    // The preview windows are created and destroyed as they open and close, so retire any
    // scope whose element has gone: without this the map would grow with every hover.
    for (const [target, scope] of this.renderScopes) {
      if (target === el || !target.isConnected) {
        scope.unload();
        this.renderScopes.delete(target);
      }
    }
    const scope = new Component();
    scope.load();
    this.renderScopes.set(el, scope);
    void MarkdownRenderer.render(this.plugin.app, md, el, sourcePath, scope);
  };

  /** The shared link chooser, built on first use. */
  private get linkChooser(): LinkChooser {
    if (!this.linkChooserInstance) this.linkChooserInstance = new LinkChooser(this.renderMarkdown);
    return this.linkChooserInstance;
  }

  engagementState(): EngagementState {
    return this.engagement.toState();
  }

  /** One accept (popup OR ghost text): learn from it, tally keystrokes saved, celebrate
   *  milestones, refresh the status bar, and schedule a persist. */
  private onAccepted(word: string, saved: number): void {
    this.engine.recordAccept(word, saved);
    for (const m of this.engagement.record(saved))
      new Notice(`🎉 ${m.toLocaleString()} keystrokes saved with Smart Autocorrect!`, 6000);
    this.renderStatus();
    this.saveEngagement();
  }

  private renderStatus(): void {
    if (this.statusBar) this.statusBar.setText(this.engagement.statusText());
    // The dashboard, if open, is showing the numbers we just changed - refresh it too.
    this.statsModal?.refresh();
  }

  /** One read-only snapshot of every gamification / learning number, for the dashboard. */
  private statsSnapshot(): StatsSnapshot {
    const st = this.store.personalization.stats;
    const e = this.engagement;
    return {
      keystrokesSaved: e.total,
      minutesSaved: e.minutesSaved,
      todaySaved: e.today,
      streak: e.streak,
      bestStreak: e.bestStreak,
      corrections: st.corrections,
      accepts: st.accepts,
      reverts: st.reverts,
      learnedWords: this.settings.userDictionary.length,
      nextMilestone: e.nextMilestone,
      allMilestones: e.allMilestones,
    };
  }

  private openStats(): void {
    this.statsModal = new StatsModal(
      this.plugin.app,
      () => this.statsSnapshot(),
      () => {
        this.statsModal = null;
      },
    );
    this.statsModal.open();
  }

  /** Confirm (loudly) then zero every statistic. Keeps learned adaptation and your
   *  dictionary; only the tallies go to zero. Persisted to the vault immediately. */
  private confirmResetStats(): void {
    new ConfirmModal(this.plugin.app, {
      title: "Reset all statistics?",
      body: "This clears your keystrokes saved, streak, milestones, and every tally. It can't be undone. Your personal dictionary and learned adaptation are kept.",
      confirmText: "Reset statistics",
      onConfirm: () => void this.resetStats(),
    }).open();
  }

  private async resetStats(): Promise<void> {
    this.engagement.reset();
    await this.store.resetStats();
    this.renderStatus();
    this.onPersistSettings?.();
    new Notice("Statistics reset");
  }

  /** Confirm (loudly) then wipe everything the plugin has learned about your writing:
   *  the adaptive keyboard, learned ranking, cache and the don't-correct list. Settings and
   *  your personal dictionary are untouched. Same red confirmation as resetting statistics. */
  private confirmResetPersonalization(): void {
    new ConfirmModal(this.plugin.app, {
      title: "Reset personalization?",
      body: "This erases everything the plugin has learned about how you write - the adaptive keyboard, learned ranking, and the words it has learned to leave alone. It can't be undone. Your settings and personal dictionary are kept.",
      confirmText: "Reset personalization",
      onConfirm: () => void this.store.reset(),
    }).open();
  }

  /** Reset every SETTING to its default, keeping the personal dictionary and learned
   *  personalization. Mutates the live settings object in place so held references stay valid. */
  private confirmResetSettings(save: () => Promise<void>, redraw: () => void): void {
    new ConfirmModal(this.plugin.app, {
      title: "Reset all settings?",
      body: "This puts every option in this menu back to its default. Your personal dictionary and everything the plugin has learned about how you write are kept.",
      confirmText: "Reset settings",
      danger: false, // reversible-ish and non-destructive to your data: no red
      onConfirm: () => void this.resetSettings(save, redraw),
    }).open();
  }

  private async resetSettings(save: () => Promise<void>, redraw: () => void): Promise<void> {
    const fresh = JSON.parse(JSON.stringify(DEFAULT_PREDICTIVE_SETTINGS)) as PredictiveSettings;
    // Preserve the user's own words and custom abbreviations - those are personal, not options.
    fresh.userDictionary = this.settings.userDictionary;
    fresh.extraAbbreviations = this.settings.extraAbbreviations;
    Object.assign(this.settings, fresh);
    this.onSettingsChanged();
    await save();
    this.onPersistSettings?.();
    redraw();
    new Notice("Settings reset to defaults");
  }

  /** Factory reset: settings, personalization, statistics AND the personal dictionary - the lot. */
  private confirmFactoryReset(save: () => Promise<void>, redraw: () => void): void {
    new ConfirmModal(this.plugin.app, {
      title: "Factory reset?",
      body: "Erases EVERYTHING this plugin stores: all settings, your personal dictionary, every statistic, and everything it has learned about how you write. It can't be undone.",
      confirmText: "Factory reset",
      onConfirm: () => void this.factoryReset(save, redraw),
    }).open();
  }

  private async factoryReset(save: () => Promise<void>, redraw: () => void): Promise<void> {
    Object.assign(this.settings, JSON.parse(JSON.stringify(DEFAULT_PREDICTIVE_SETTINGS)) as PredictiveSettings);
    this.engagement.reset();
    await this.store.reset(); // wipes learned personalization + its stats, persists
    this.onSettingsChanged();
    this.renderStatus();
    await save();
    this.onPersistSettings?.();
    redraw();
    new Notice("Factory reset complete");
  }

  /** Undo of a correction adds the word to the personal dictionary (the don't-touch list),
   *  if that setting is on. Case-sensitive, deduplicated, persisted, and pushed to the engine.
   *  Skips words that are already KNOWN (in the 120k LM vocab or already personal) - adding
   *  those is redundant, which is why reverting a correction of a common word like "too" must
   *  NOT learn it (#11/#14). An oracle-only word (real but outside the 120k) is still addable,
   *  so it can be featured in suggestions. */
  private async addToDictionary(word: string): Promise<void> {
    if (!this.settings.undoAddsToDictionary) return;
    const w = word.trim();
    if (!w || /\s/.test(w) || this.settings.userDictionary.includes(w)) return; // ignore merge-reverts (contain a space)
    try {
      if (await this.engine.isKnownWord(w)) return; // already featured → nothing to add
    } catch {
      /* engine unavailable: fall through and add (best effort) */
    }
    this.settings.userDictionary = [...this.settings.userDictionary, w];
    this.onSettingsChanged();
    this.onPersistSettings?.();
    new Notice(`Added “${w}” to your personal dictionary`);
  }

  /** Explicit add (right-click / #13): unlike the undo path this is NOT gated on the undo
   *  setting, and an oracle-only real word IS addable (so it can be featured in suggestions).
   *  Gives feedback when the word is already known, so the action never silently no-ops. */
  private async addWordExplicit(word: string): Promise<void> {
    const w = word.trim();
    if (!w) return;
    if (this.settings.userDictionary.includes(w)) {
      new Notice(`“${w}” is already in your personal dictionary`);
      return;
    }
    try {
      if (await this.engine.isKnownWord(w)) {
        new Notice(`“${w}” is already recognised – no need to add it`);
        return;
      }
    } catch {
      /* engine unavailable: add best-effort */
    }
    this.settings.userDictionary = [...this.settings.userDictionary, w];
    this.onSettingsChanged();
    this.onPersistSettings?.();
    new Notice(`Added “${w}” to your personal dictionary`);
  }

  /** Reverting a sentence-initial capitalisation after "word." teaches that "word" is an
   *  abbreviation, so we stop capitalising after it (#10, e.g. "etc.", "incl."). Gated on the
   *  setting; deduplicated against the built-in + user abbreviation lists. */
  private learnAbbreviation(abbrev: string): void {
    if (!this.settings.learnAbbreviationsOnRevert) return;
    this.addAbbreviation(abbrev);
  }

  /** Add an abbreviation so we stop treating its full stop as a sentence end. Shared by the
   *  undo-driven path (#10) and the explicit right-click one; deduplicated, persisted. */
  private addAbbreviation(abbrev: string): void {
    const a = abbrev.trim().toLowerCase().replace(/\.$/, "");
    if (!a || this.settings.extraAbbreviations.some((x) => x.toLowerCase().replace(/\.$/, "") === a))
      return;
    this.settings.extraAbbreviations = [...this.settings.extraAbbreviations, a];
    this.onSettingsChanged();
    this.onPersistSettings?.();
    new Notice(`Won't capitalise after “${a}.” anymore`);
  }

  /** Undo of the above: treat the full stop as a sentence end again. */
  private forgetAbbreviation(abbrev: string): void {
    const a = abbrev.trim().toLowerCase().replace(/\.$/, "");
    const next = this.settings.extraAbbreviations.filter(
      (x) => x.toLowerCase().replace(/\.$/, "") !== a,
    );
    if (next.length === this.settings.extraAbbreviations.length) return;
    this.settings.extraAbbreviations = next;
    this.onSettingsChanged();
    this.onPersistSettings?.();
    new Notice(`Will capitalise after “${a}.” again`);
  }

  /**
   * Drop personal-dictionary words that no longer appear ANYWHERE in the vault (#8): if you
   * delete the notes that used a word, the plugin forgets it too.
   *
   * This asks the engine's incremental corpus how many DOCUMENTS still contain each word, which
   * it already tracks as a side effect of the per-file counting it does anyway. The previous
   * implementation re-read and re-tokenised every note in the vault, which is why it had to hide
   * behind a 12-second debounce and why a removal took so long to show up. Now the answer is a
   * map lookup per dictionary word, so it can run the moment the corpus has taken the edit in.
   *
   * A word the corpus cannot answer for (personal learning switched off, corpus not built yet)
   * yields null, and nothing is pruned - the cost of a wrong answer here is a silently deleted
   * dictionary entry, so no answer must never be read as "gone".
   */
  private async reconcileDictionary(): Promise<void> {
    if (!this.settings.pruneDictionaryFromVault || this.settings.userDictionary.length === 0) return;
    const dict = this.settings.userDictionary;
    let freq: number[] | null;
    try {
      freq = await this.engine.documentFrequencies(dict);
    } catch {
      return;
    }
    if (!freq || freq.length !== dict.length) return;
    const keep = dict.filter((_, i) => freq[i] > 0);
    const removed = dict.filter((_, i) => freq[i] === 0);
    if (removed.length === 0) return;
    this.settings.userDictionary = keep;
    this.onSettingsChanged();
    this.onPersistSettings?.();
    new Notice(
      removed.length === 1
        ? `Removed “${removed[0]}” from your personal dictionary (no longer in the vault)`
        : `Removed ${removed.length} words from your personal dictionary (no longer in the vault)`,
    );
  }

  /** Remove a word from the personal dictionary, with a confirming notice (#13). */
  private removeFromDictionary(word: string): void {
    const w = word.trim();
    if (!this.settings.userDictionary.includes(w)) return;
    this.settings.userDictionary = this.settings.userDictionary.filter((x) => x !== w);
    this.onSettingsChanged();
    this.onPersistSettings?.();
    new Notice(`Removed “${w}” from your personal dictionary`);
  }

  /** Add `tag` to the note's YAML frontmatter `tags:` list, the canonical place to
   *  classify a whole note. Uses Obsidian's frontmatter API so existing tags, formatting
   *  and other fields are preserved and never corrupted. */
  private async addTagToFrontmatter(file: TFile, tag: string): Promise<void> {
    const clean = tag.replace(/^#/, "").trim();
    if (!clean) return;
    await this.plugin.app.fileManager.processFrontMatter(file, (fm: Record<string, unknown>) => {
      const cur = fm.tags;
      const list = Array.isArray(cur)
        ? cur.map((t) => String(t))
        : typeof cur === "string" && cur.trim()
          ? cur.split(/[,\s]+/).filter(Boolean)
          : [];
      const norm = list.map((t) => t.replace(/^#/, ""));
      if (!norm.some((t) => t.toLowerCase() === clean.toLowerCase())) norm.push(clean);
      fm.tags = norm;
    });
    new Notice(`Added #${clean} to this note's tags`);
  }

  private async processDirty(): Promise<void> {
    const paths = [...this.dirty];
    this.dirty.clear();
    for (const p of paths) {
      const f = this.plugin.app.vault.getAbstractFileByPath(p);
      if (f instanceof TFile) await this.engine.onFileModified(f);
    }
    // The corpus now reflects these edits, so its document-frequency table is the current
    // answer to "does this dictionary word still exist anywhere?" - prune right away (#8).
    if (paths.length) this.reconcileDict();
  }

  async enable(): Promise<void> {
    await this.store.load();
    this.engine.updateSettings(this.settings);
    await this.loadModels();
    await this.engine.rebuildPersonal();
    this.seedActiveDocument();
    // Offer the one-time model download AFTER the plugin is already usable, so a slow
    // or declined download never delays startup. Asked once; `modelPromptShown` makes
    // "not now" stick, and settings has a button to change your mind.
    void this.maybeOfferAssets();

    if (this.enabled) return; // idempotent: don't double-register listeners
    this.enabled = true;

    // Live keystrokes-saved / streak indicator in the status bar.
    this.statusBar = this.plugin.addStatusBarItem();
    this.statusBar.addClass("mod-clickable");
    this.statusBar.setAttribute("aria-label", "Writing stats (click to open)");
    this.plugin.registerDomEvent(this.statusBar, "click", () => this.openStats());
    this.renderStatus();

    this.suggest = new PredictiveSuggest(
      this.plugin.app,
      this.engine,
      this.settings,
      (word, saved) => this.onAccepted(word, saved),
    );
    this.plugin.registerEditorSuggest(this.suggest);

    // Obsidian-native tag autocomplete on `#`, note-aware and niche-biased.
    const tagSuggest = new TagSuggest(this.plugin.app, this.linkIndex, this.engine, () => this.settings, (word, saved) => this.onAccepted(word, saved));
    this.plugin.registerEditorSuggest(tagSuggest);
    // Run BEFORE Obsidian's built-in `#` completion, which otherwise consumes the trigger
    // (so our new-tag suggestions never appear on a note with no existing tags). Moving our
    // suggester to the front of the manager's list makes it win the `#` trigger. Guarded:
    // it's an internal field, so a version change just falls back to core behaviour.
    // Our own `[[` picker, ranked by topical relevance (see LinkSuggest).
    const linkSuggest = new LinkPicker(
      this.plugin.app,
      this.linkIndex,
      this.relatedIndex,
      () => this.settings,
      this.renderMarkdown,
    );
    this.plugin.registerEditorSuggest(linkSuggest);
    // Both need to run BEFORE Obsidian's built-in `#` and `[[` completions, which otherwise
    // consume the trigger. Moving ours to the front of the manager's list makes them win.
    // Guarded: it's an internal field, so a version change just falls back to core behaviour.
    try {
      const arr = (this.plugin.app.workspace as unknown as { editorSuggest?: { suggests?: unknown[] } }).editorSuggest?.suggests;
      if (Array.isArray(arr)) {
        for (const s of [tagSuggest, linkSuggest]) {
          const i = arr.indexOf(s);
          if (i > 0) {
            arr.splice(i, 1);
            arr.unshift(s);
          }
        }
      }
    } catch {
      /* internal API moved; core completions still work */
    }

    this.autocorrect = new AutocorrectController(this.plugin, this.engine, this.settings);
    this.autocorrect.onRevert = (word) => void this.addToDictionary(word);
    this.autocorrect.onLearnAbbreviation = (abbrev) => this.learnAbbreviation(abbrev);
    this.autocorrect.register();

    // Inline ghost-text prediction (#C5) - reads live settings each render.
    this.plugin.registerEditorExtension(
      ghostTextExtension(
        this.engine,
        () => this.settings,
        () => this.activeFileExcluded(),
        (word, saved) => this.onAccepted(word, saved),
      ),
    );
    // Word-style Tab indent: only indent a bullet from its start, so a missed Tab-accept
    // doesn't shove the list item right (reads live settings each keypress).
    this.plugin.registerEditorExtension(tabIndentGuard(() => this.settings));
    // Always-on: tracks whether the last edit was a bullet indent, so the popup can skip a
    // suggestion right after one (the ghost path checks the transaction directly).
    this.plugin.registerEditorExtension(indentEditField);

    // Ambient link suggestion: underline text matching an existing note title/alias.
    this.linkIndex.rebuild();
    this.plugin.registerEditorExtension(
      linkSuggestExtension(
        this.plugin.app,
        this.linkIndex,
        () => this.settings,
        () => this.activeFileExcluded(),
        this.dismissedLinks,
      ),
    );
    // Let hovering a suggested link (in the chooser) show Obsidian's native page/section
    // preview without holding a modifier key.
    const ws = this.plugin.app.workspace as unknown as {
      registerHoverLinkSource?: (id: string, info: { display: string; defaultMod: boolean }) => void;
    };
    ws.registerHoverLinkSource?.("smart-autocorrect", { display: "Smart Autocorrect", defaultMod: false });

    // Related-note link suggestions (semantic + keyword), per paragraph/bullet.
    this.plugin.registerEditorExtension(
      relatedLinksExtension(
        this.plugin.app,
        this.relatedIndex,
        () => this.settings,
        () => this.activeFileExcluded(),
        this.dismissedRelated,
        this.renderMarkdown,
      ),
    );

    // Right-click a word (or selection) to add/remove it from the personal dictionary (#13),
    // or to link a selection to a related note (#7).
    this.plugin.registerEvent(
      this.plugin.app.workspace.on("editor-menu", (menu, editor) => {
        if (!this.settings.pluginEnabled) return;
        // Linking a SELECTION is offered before the word-level items, and has to sit ahead
        // of the `!word` guard below: a selection is usually a phrase, which
        // `selectedOrCursorWord` cannot represent. Obsidian's built-in link action only
        // wraps the text in [[ ]] and leaves you to type the destination blind; this opens
        // the ranked chooser with the selection intact and applies the pick as
        // [[dest|selection]]. Same widget as the link icon and the `[[` suggester.
        const linkSel = editor.getSelection().trim();
        if (linkSel) {
          const openChooser = () => {
            void openLinkSelection({
              app: this.plugin.app,
              editor,
              index: this.relatedIndex,
              settings: this.settings,
              chooser: this.linkChooser,
              queryText: linkSel,
              selection: linkSel,
              preferredTop: caretTop(editor),
            }).then((n) => {
              if (n === 0) new Notice("No related notes found for that selection");
            });
          };
          // TAKE OVER the built-in item rather than adding a second one: two near-identical
          // "link" entries is exactly the confusion this feature is meant to remove.
          if (!replaceLinkMenuItem(menu, openChooser)) {
            menu.addItem((item) =>
              item.setTitle("Add link to a related note").setIcon("link").onClick(openChooser),
            );
          }
        }
        const word = selectedOrCursorWord(editor);
        if (!word) return;
        // An abbreviation's problem is never its spelling - it is that the full stop reads as
        // the end of a sentence and capitalises the next word. So offer the fix that actually
        // applies to it, rather than a dictionary entry that would change nothing.
        if (isAbbreviationToken(word)) {
          const bare = word.toLowerCase().replace(/\.$/, "");
          const known = this.settings.extraAbbreviations.some(
            (x) => x.toLowerCase().replace(/\.$/, "") === bare,
          );
          menu.addItem((item) =>
            item
              .setTitle(
                known
                  ? `Capitalise after “${word}” again`
                  : `Don't capitalise after “${word}” (abbreviation)`,
              )
              .setIcon(known ? "book-minus" : "book-plus")
              .onClick(() => (known ? this.forgetAbbreviation(bare) : this.addAbbreviation(bare))),
          );
          return;
        }
        if (this.settings.userDictionary.includes(word)) {
          menu.addItem((item) =>
            item
              .setTitle(`Remove “${word}” from personal dictionary`)
              .setIcon("book-minus")
              .onClick(() => this.removeFromDictionary(word)),
          );
        } else {
          menu.addItem((item) =>
            item
              .setTitle(`Add “${word}” to personal dictionary`)
              .setIcon("book-plus")
              .onClick(() => void this.addWordExplicit(word)),
          );
        }
      }),
    );

    // Keep the title/alias/tag index fresh as the vault changes.
    this.plugin.registerEvent(this.plugin.app.metadataCache.on("resolved", () => this.linkIndex.refresh()));
    this.plugin.registerEvent(
      this.plugin.app.metadataCache.on("changed", (f) => {
        this.refreshLinkIndex();
        if (this.settings.syncFrontmatterTags && f instanceof TFile && f.extension === "md") {
          this.tagDirty.add(f.path);
          this.flushTagSync();
        }
      }),
    );

    // Incremental per-file corpus maintenance (#B3).
    const isMd = (f: unknown): f is TFile => f instanceof TFile && f.extension === "md";
    this.plugin.registerEvent(
      this.plugin.app.vault.on("modify", (f) => {
        if (isMd(f)) {
          this.dirty.add(f.path);
          this.flushDirty(); // pruning follows the corpus update, in processDirty (#8)
          this.refreshRelated();
        }
      }),
    );
    this.plugin.registerEvent(
      this.plugin.app.vault.on("create", (f) => {
        if (isMd(f)) {
          this.dirty.add(f.path);
          this.flushDirty();
          this.refreshRelated();
        }
      }),
    );
    this.plugin.registerEvent(
      this.plugin.app.vault.on("delete", (f) => {
        if (isMd(f)) {
          void this.engine.onFileDeleted(f.path);
          this.refreshRelated();
          this.reconcileDict(); // a delete may have removed the last use of a dictionary word (#8)
        }
      }),
    );
    this.plugin.registerEvent(
      this.plugin.app.vault.on("rename", (f, oldPath) => {
        if (isMd(f)) {
          void this.engine.onFileRenamed(oldPath, f.path);
          this.refreshRelated();
        }
      }),
    );
    this.plugin.registerEvent(
      this.plugin.app.workspace.on("active-leaf-change", () => this.seedActiveDocument()),
    );

    this.registerCommands();
  }

  private registerCommands(): void {
    this.plugin.addCommand({
      id: "show-writing-stats",
      name: "Show writing stats",
      callback: () => this.openStats(),
    });

    // Replace the plain "add link" flow for a selection with our recommendations (#7). Bind this
    // to your add-link hotkey if you want it to take over that muscle memory.
    this.plugin.addCommand({
      id: "link-selection",
      name: "Link selection to a related note",
      editorCallback: (editor: Editor) => {
        if (!this.settings.pluginEnabled) return;
        const selection = editor.getSelection().trim();
        const word = selection || selectedOrCursorWord(editor) || "";
        const line = editor.getLine(editor.getCursor().line);
        const queryText = selection || line || word;
        void openLinkSelection({
          app: this.plugin.app,
          editor,
          index: this.relatedIndex,
          settings: this.settings,
          chooser: this.linkChooser,
          queryText,
          selection,
          preferredTop: caretTop(editor),
        }).then((n) => {
          if (n === 0) new Notice("No related notes found for that selection");
        });
      },
    });

    this.plugin.addCommand({
      id: "refresh-link-suggestions",
      name: "Refresh link suggestions (bring back dismissed)",
      editorCallback: (editor: Editor) => {
        this.dismissedRelated.clear();
        const cm = (editor as unknown as { cm?: EditorView }).cm;
        cm?.dispatch({ effects: forceRescan.of(null) });
        new Notice("Link suggestions refreshed");
      },
    });

    this.plugin.addCommand({
      id: "suggest-links",
      name: "Suggest links in this note",
      editorCallback: async (editor: Editor) => {
        const text = editor.getValue();
        const file = this.plugin.app.workspace.getActiveFile();
        const exclude = new Set<string>();
        if (file?.basename) exclude.add(file.basename.toLowerCase());
        // Threshold matches the icons' sensitivity so the list and the icons agree.
        const th = [0.5, 0.42, 0.34, 0.27, 0.2][Math.min(4, Math.max(0, this.settings.relatedSensitivity - 1))];
        const cands = await this.relatedIndex.candidatesFor(text, file?.path, exclude, 12, th);
        if (cands.length === 0) {
          new Notice("No related notes found for this note yet. Keep writing, or raise the sensitivity in settings.");
          return;
        }
        const menu = new Menu();
        for (const c of cands)
          menu.addItem((i) =>
            i
              .setTitle(`[[${c.display}]]  ·  ${Math.round(c.score * 100)}% related`)
              .setIcon("link")
              .onClick(() => editor.replaceSelection(`[[${c.display}]]`)),
          );
        const cm = (editor as unknown as { cm?: EditorView }).cm;
        const rect = cm?.coordsAtPos(cm.state.selection.main.head);
        if (rect) menu.showAtPosition({ x: rect.left, y: rect.bottom });
        else menu.showAtPosition({ x: 200, y: 200 });
      },
    });

    this.plugin.addCommand({
      id: "suggest-tags",
      name: "Suggest tags for this note",
      editorCallback: (editor: Editor, ctx: MarkdownFileInfo) => {
        const text = editor.getValue();
        const applied = new Set<string>();
        const cache = ctx.file ? this.plugin.app.metadataCache.getFileCache(ctx.file) : null;
        for (const t of cache?.tags ?? []) applied.add(t.tag.replace(/^#/, "").toLowerCase());
        for (const t of frontmatterTags(cache?.frontmatter)) applied.add(t.toLowerCase());

        // Existing tags, ranked by relevance to this note (keeps your tagging consistent).
        const existing = this.linkIndex.suggestTags(text, applied);
        // A couple of brand-new tags from the note's own distinctive, repeated terms.
        const known = this.linkIndex.tags;
        const fresh = [...termFreq(text).entries()]
          .filter(([w, c]) => c >= 2 && w.length >= 4 && !known.has(w) && !applied.has(w))
          .sort((a, b) => b[1] - a[1])
          .slice(0, 3)
          .map(([w]) => w);

        if (existing.length === 0 && fresh.length === 0) {
          new Notice("No tag suggestions yet - write a little more, or add some tags elsewhere in the vault.");
          return;
        }
        const menu = new Menu();
        const apply = (tag: string) => {
          if (ctx.file) void this.addTagToFrontmatter(ctx.file, tag);
          else editor.replaceSelection(`#${tag} `);
        };
        for (const c of existing)
          menu.addItem((i) =>
            i.setTitle(`#${c.tag}  ·  used in ${c.count} note${c.count === 1 ? "" : "s"}`).setIcon("tag").onClick(() => apply(c.tag)),
          );
        if (existing.length && fresh.length) menu.addSeparator();
        for (const w of fresh)
          menu.addItem((i) => i.setTitle(`#${w}  ·  new tag`).setIcon("plus").onClick(() => apply(w)));

        const cm = (editor as unknown as { cm?: EditorView }).cm;
        const rect = cm?.coordsAtPos(cm.state.selection.main.head);
        if (rect) menu.showAtPosition({ x: rect.left, y: rect.bottom });
        else menu.showAtPosition({ x: 200, y: 200 });
      },
    });
    this.plugin.addCommand({
      id: "predictive-rebuild-personal",
      name: "Predictive: rebuild personal (vault) model",
      callback: async () => {
        await this.engine.rebuildPersonal();
        new Notice("Rebuilt personal prediction model");
      },
    });
    this.plugin.addCommand({
      id: "predictive-pack-global",
      name: "Predictive: pack global model to binary",
      callback: async () => {
        const buf = await this.engine.packGlobal();
        if (!buf) {
          new Notice("No text-built global model to pack");
          return;
        }
        const dir = this.plugin.manifest.dir ?? ".";
        await this.plugin.app.vault.adapter.writeBinary(`${dir}/${PACKED_GLOBAL_FILE}`, buf);
        new Notice(`Packed global model → ${PACKED_GLOBAL_FILE} (${(buf.byteLength / 1024) | 0} KB)`);
      },
    });
    this.plugin.addCommand({
      id: "predictive-run-evaluation",
      name: "Predictive: evaluate on current note",
      callback: async () => {
        const view = this.plugin.app.workspace.getActiveViewOfType(MarkdownView);
        const text = view?.editor.getValue() ?? "";
        const res = await this.engine.runEvaluation(text);
        if (!res) {
          new Notice("Model not ready");
          return;
        }
        new Notice(
          `Eval: correction ${(res.correctionAccuracy * 100) | 0}% · recall@k ${(res.recallAtK * 100) | 0}% · keystrokes saved ${(res.keystrokeSavings * 100) | 0}% (${res.corrupted} typos)`,
          8000,
        );
      },
    });
  }

  dispose(): void {
    for (const scope of this.renderScopes.values()) scope.unload();
    this.renderScopes.clear();
    this.engine.dispose();
  }

  private seedActiveDocument(): void {
    const view = this.plugin.app.workspace.getActiveViewOfType(MarkdownView);
    if (view) this.engine.setActiveDocument(view.editor.getValue());
  }

  /** True when the currently-active file is in an excluded folder (ghost text / any
   *  surface that can't get the file path directly asks here). */
  activeFileExcluded(): boolean {
    const f = this.plugin.app.workspace.getActiveFile();
    return f ? pathExcluded(f.path, this.settings.excludedFolders) : false;
  }

  /** (Re)read every model asset from the plugin folder. Each is optional. */
  private async loadModels(): Promise<void> {
    await this.loadGlobalModel();
    await this.loadLstmModel();
    await this.loadWordlist();
  }

  /**
   * Offer the first-run model download, at most once unless the user asks again.
   * Returns true if new assets landed (models are then reloaded in place, so the
   * plugin goes from degraded to full without a restart).
   */
  async installAssets(force = false): Promise<boolean> {
    const got = await ensureAssets(this.plugin, force);
    if (!got) return false;
    await this.loadModels();
    await this.engine.rebuildPersonal();
    this.seedActiveDocument();
    this.renderStatus();
    return true;
  }

  private async maybeOfferAssets(): Promise<void> {
    if (this.settings.modelPromptShown) return;
    if ((await missingAssets(this.plugin)).length === 0) return;
    this.settings.modelPromptShown = true;
    this.onPersistSettings?.();
    await this.installAssets();
  }

  private async loadGlobalModel(): Promise<void> {
    const dir = this.plugin.manifest.dir ?? ".";
    const adapter = this.plugin.app.vault.adapter;
    try {
      // Prefer a prebuilt packed binary (#B2), else fall back to a text corpus.
      const packed = `${dir}/${PACKED_GLOBAL_FILE}`;
      if (await adapter.exists(packed)) {
        await this.engine.loadGlobalPacked(await adapter.readBinary(packed));
        return;
      }
      const corpus = `${dir}/${DEV_CORPUS_FILE}`;
      if (await adapter.exists(corpus)) {
        await this.engine.loadGlobal(await adapter.read(corpus));
        return;
      }
    } catch {
      /* personal-only */
    }
    await this.engine.loadGlobal(null);
  }

  /** Load the real-word oracle (wordlist.bin) from the plugin folder, if present. Absent
   *  file → the oracle stays empty, so rare-word protection is simply off (graceful). */
  private async loadWordlist(): Promise<void> {
    const dir = this.plugin.manifest.dir ?? ".";
    const adapter = this.plugin.app.vault.adapter;
    try {
      const path = `${dir}/${WORDLIST_FILE}`;
      if (await adapter.exists(path)) {
        await this.engine.loadWordlist(await adapter.readBinary(path));
      }
    } catch (e) {
      console.warn("[predictive] word oracle not loaded", e);
    }
  }

  /** Load the word-level LSTM next-word model from the plugin folder, if present.
   *  Absent file → the engine stays n-gram-only (graceful fallback). */
  private async loadLstmModel(): Promise<void> {
    const dir = this.plugin.manifest.dir ?? ".";
    const adapter = this.plugin.app.vault.adapter;
    try {
      const path = `${dir}/${LSTM_FILE}`;
      if (await adapter.exists(path)) {
        await this.engine.loadLstm(await adapter.readBinary(path));
      }
    } catch (e) {
      console.warn("[predictive] LSTM model not loaded", e);
    }
  }

  private rebuildPersonalDebounced = debounce(() => void this.engine.rebuildPersonal(), 4000, false);

  onSettingsChanged(): void {
    this.engine.updateSettings(this.settings);
    this.suggest?.updateSettings(this.settings);
    this.autocorrect?.updateSettings(this.settings);
    if (this.settings.personalBias) this.rebuildPersonalDebounced();
  }

  /**
   * Describe the settings pane once. Obsidian 1.13+ renders it from these groups (via the
   * setting tab's getSettingDefinitions), older versions draw the same groups imperatively -
   * see settingsPane.ts. `refresh` re-renders whichever surface is showing it.
   */
  settingGroups(save: () => Promise<void>, refresh: () => void): PaneGroup[] {
    return buildPredictiveSettingGroups(
      this.settings,
      async () => {
        this.onSettingsChanged();
        await save();
      },
      {
        onReset: () => this.confirmResetPersonalization(),
        onExport: (p) => this.store.exportTo(p),
        onImport: (p, merge) => this.store.importFrom(p, merge),
        getStats: () => {
          const s = this.store.personalization;
          const e = this.engagement;
          return {
            ...s.stats,
            charsSaved: e.total,
            streak: e.streak,
            bestStreak: e.bestStreak,
            minutesSaved: e.minutesSaved,
            learnListSize: s.learnList.size,
          };
        },
        onOpenStats: () => this.openStats(),
        onResetStats: () => this.confirmResetStats(),
        onResetSettings: () => this.confirmResetSettings(save, refresh),
        onFactoryReset: () => this.confirmFactoryReset(save, refresh),
      },
      refresh,
      {
        status: () => this.engine.status(),
        // Re-read word_lstm.bin and rebuild the model so the WASM-SIMD toggle applies
        // immediately (the kernel is chosen at load time). Cheap in practice - toggled
        // once, if ever - and it is the honest way to make the change take effect now.
        reload: () => this.loadLstmModel(),
        missingAssets: async () => (await missingAssets(this.plugin)).length,
        installAssets: () => this.installAssets(true),
      },
      this.accelState,
    );
  }

  /** Obsidian 1.13+ rendering path: the same description, as setting definitions. */
  settingDefinitions(save: () => Promise<void>, refresh: () => void): SettingDefinitionItem[] {
    return toSettingDefinitions(this.settingGroups(save, refresh));
  }

  /** Pre-1.13 rendering path, driven from the same description. */
  renderSettings(containerEl: HTMLElement, save: () => Promise<void>): void {
    const redraw = () => {
      containerEl.empty();
      this.renderSettings(containerEl, save);
    };
    renderPaneGroups(containerEl, this.settingGroups(save, redraw));
  }
}

/** Tag names declared in a note's frontmatter (`tags:` as a list or a string). */
/**
 * A single word, INCLUDING a dotted abbreviation ("e.g.", "z.b.", "w.r.t.").
 *
 * Interior and trailing dots are part of the token here. They were previously excluded, which
 * meant right-clicking an abbreviation - the one kind of token whose capitalisation behaviour
 * you most often want to change - offered no menu item at all.
 */
const WORD_OR_ABBREV = /^[A-Za-z][A-Za-z'-]*(?:\.[A-Za-z'-]+)*\.?$/;

/** The selected text if it's a single word, else the word under the cursor - for the
 *  right-click "add to dictionary" menu. Returns null if there is no plain word there. */
function selectedOrCursorWord(editor: Editor): string | null {
  const sel = editor.getSelection().trim();
  if (sel) return WORD_OR_ABBREV.test(sel) ? sel : null;
  const cur = editor.getCursor();
  const line = editor.getLine(cur.line);
  const isW = (c: string) => /[A-Za-z'.-]/.test(c);
  let s = cur.ch;
  let e = cur.ch;
  while (s > 0 && isW(line[s - 1])) s--;
  while (e < line.length && isW(line[e])) e++;
  const w = line.slice(s, e).replace(/^['.-]+/, "").replace(/^['-]+|['-]+$/g, "");
  return WORD_OR_ABBREV.test(w) ? w : null;
}

/**
 * Point the context menu's existing "link" entry at `run` instead of its own handler,
 * returning true if one was found.
 *
 * Obsidian's built-in action wraps the selection in [[ ]] and leaves you to type the
 * destination blind, which is strictly worse than our ranked chooser - so we take the
 * entry over rather than adding a second, near-identical one next to it.
 *
 * There is NO public API for editing another provider's menu item, so this reads
 * `Menu.items`, which is internal. It is written to fail SOFT: if that field is missing
 * or its shape changes, we return false and the caller adds its own item instead. The
 * only public thing we rely on is `MenuItem.onClick`, which replaces the callback.
 * Matching is on the title, deliberately narrow (an exact set of known labels) so we
 * cannot hijack an unrelated entry like "Copy link" or another plugin's linker.
 */
const NATIVE_LINK_TITLES = new Set(["add link", "insert link", "link to note", "add internal link"]);

function replaceLinkMenuItem(menu: Menu, run: () => void): boolean {
  const items = (menu as unknown as { items?: unknown[] }).items;
  if (!Array.isArray(items)) return false;
  for (const raw of items) {
    const item = raw as { titleEl?: HTMLElement; onClick?: (cb: () => void) => unknown };
    const title = item?.titleEl?.textContent?.trim().toLowerCase();
    if (!title || !NATIVE_LINK_TITLES.has(title)) continue;
    if (typeof item.onClick !== "function") return false;
    item.onClick(run);
    return true;
  }
  return false;
}

/** Screen y of the caret, so a chooser opens level with where the user is working. Falls back
 *  to the top of the viewport when the editor can't report a coordinate. */
function caretTop(editor: Editor): number {
  const cm = (editor as unknown as { cm?: EditorView }).cm;
  const head = cm?.state.selection.main.head;
  const rect = cm && head !== undefined ? cm.coordsAtPos(head) : null;
  return rect ? rect.top : 0;
}

/** Does this token read as an abbreviation rather than a plain word? Dots are the whole
 *  signal: "z.b.", "e.g.", "etc." - which is what makes the capitaliser treat what follows
 *  as a new sentence. */
function isAbbreviationToken(token: string): boolean {
  return token.includes(".");
}

function frontmatterTags(fm: unknown): string[] {
  const t = (fm as Record<string, unknown> | undefined)?.tags;
  if (Array.isArray(t)) return t.map((x) => String(x).replace(/^#/, ""));
  if (typeof t === "string") return t.split(/[,\s]+/).map((x) => x.replace(/^#/, "")).filter(Boolean);
  return [];
}
