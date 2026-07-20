/**
 * Obsidian-side facade over the prediction engine.
 *
 * The engine itself (models, inference, personalisation, learning) lives in
 * EngineCore, which runs inside a Web Worker via InferenceClient - prediction used
 * to run synchronously on the main thread inside CodeMirror's update cycle, which
 * is what made typing stutter. Everything here is therefore ASYNC.
 *
 * This class keeps only what genuinely needs the main thread: vault reads (the
 * Obsidian API is not available to workers), persistence of the learned state, and
 * the case map, which autocorrect applies synchronously while typing.
 */
import type { App, TFile } from "obsidian";
import { emptyPersonalization } from "./engine/index";
import type { CorrectionDecision, EvalResult } from "./engine/index";
import { InferenceClient } from "./InferenceClient";
import type { SuggestItem } from "./EngineCore";
import type { PredictiveSettings } from "./PredictiveSettings";
import type { PersonalizationStore } from "./PersonalizationStore";

export type { SuggestItem };

export class PredictiveEngineController {
  private app: App;
  private settings: PredictiveSettings;
  private store: PersonalizationStore;
  private client: InferenceClient;
  private isReady = false;

  constructor(app: App, settings: PredictiveSettings, store: PersonalizationStore) {
    this.app = app;
    this.settings = settings;
    this.store = store;
    this.client = new InferenceClient(settings, this.personalizationForEngine());
  }

  /** True once a model is loaded. Kept as a cheap sync flag because GhostText and
   *  the suggester check it on every keystroke. */
  get ready(): boolean {
    return this.isReady;
  }

  /** True when prediction is genuinely off the main thread. */
  get threaded(): boolean {
    return this.client.threaded;
  }

  /** Full engine status for the settings pane: ready, SIMD acceleration, model loaded. */
  status(): Promise<{ ready: boolean; accelerated: boolean; lstmLoaded: boolean }> {
    return this.client.status();
  }

  /** Semantic topic fingerprint for a passage, for related-link matching. Null when no
   *  neural model is loaded. */
  embed(text: string): Promise<number[] | null> {
    return this.client.embed(text);
  }

  /** Dimensionality of `embed` vectors, 0 when no neural model is loaded. */
  embedDim(): Promise<number> {
    return this.client.embedDim();
  }

  /** Per-word descriptiveness scores for niche tag bias. */
  rarities(words: string[]): Promise<number[]> {
    return this.client.rarities(words);
  }

  private async refreshStatus(): Promise<void> {
    try {
      this.isReady = (await this.client.status()).ready;
    } catch {
      this.isReady = false;
    }
  }

  async loadLstm(buffer: ArrayBuffer): Promise<void> {
    await this.client.loadLstm(buffer);
    await this.refreshStatus();
  }

  /** Load the real-word oracle asset (wordlist.bin). Safe to skip if the file is absent. */
  async loadWordlist(buffer: ArrayBuffer): Promise<void> {
    await this.client.loadWordlist(buffer);
  }

  /** Is the word already featured (in the 120k LM vocab or the user's personal additions)? */
  /** How many distinct vault notes contain each word; null when the corpus cannot say. */
  documentFrequencies(words: string[]): Promise<number[] | null> {
    return this.client.documentFrequencies(words);
  }

  isKnownWord(word: string): Promise<boolean> {
    return this.client.isKnownWord(word);
  }

  async loadGlobal(corpusText: string | null): Promise<void> {
    await this.client.loadGlobalText(corpusText);
    await this.refreshStatus();
  }

  async loadGlobalPacked(buffer: ArrayBuffer): Promise<void> {
    await this.client.loadGlobalPacked(buffer);
    await this.refreshStatus();
  }

  /** Contextual proper-noun casing, decided by the cased LSTM. */
  caseFor(word: string, context: string[]): Promise<string | null> {
    return this.client.caseFor(word, context);
  }

  packGlobal(): Promise<ArrayBuffer | null> {
    return this.client.packGlobal();
  }

  /** Full (re)build of the personal corpus. Vault reads happen here (main thread);
   *  the counting + case-mapping happen in the worker. */
  async rebuildPersonal(): Promise<void> {
    if (!this.settings.personalBias) {
      await this.client.rebuildPersonal(null);
      await this.refreshStatus();
      return;
    }
    const entries: { path: string; text: string }[] = [];
    for (const f of this.app.vault.getMarkdownFiles()) {
      try {
        entries.push({ path: f.path, text: await this.app.vault.cachedRead(f) });
      } catch {
        /* skip */
      }
    }
    await this.client.rebuildPersonal(entries);
    await this.refreshStatus();
  }

  async onFileModified(file: TFile): Promise<void> {
    if (!this.settings.personalBias) return;
    try {
      const text = await this.app.vault.cachedRead(file);
      // false => the worker has no corpus yet, so a full rebuild is required.
      if (!(await this.client.setFile(file.path, text))) await this.rebuildPersonal();
    } catch {
      /* ignore unreadable file */
    }
  }

  async onFileDeleted(path: string): Promise<void> {
    await this.client.removeFile(path);
  }

  async onFileRenamed(oldPath: string, newPath: string): Promise<void> {
    await this.client.renameFile(oldPath, newPath);
  }

  runEvaluation(text: string): Promise<EvalResult | null> {
    return this.client.runEvaluation(text);
  }

  setActiveDocument(text: string): void {
    void this.client.setActiveDocument(text);
  }

  updateSettings(settings: PredictiveSettings): void {
    const wasPersonalized = this.settings.personalizationEnabled;
    this.settings = settings;
    this.store.updateBeta(settings.beta);
    void this.client.updateSettings(settings);
    // Toggling personalization must take effect NOW, not at next restart: the engine
    // holds the learned state, so it has to be swapped for a blank one (or restored
    // from the store) the moment the switch moves.
    if (wasPersonalized !== settings.personalizationEnabled) void this.pushPersonalization();
  }

  getSuggestions(context: string[], typed: string, k: number, includePhrases = true): Promise<SuggestItem[]> {
    return this.client.getSuggestions(context, typed, k, includePhrases);
  }

  decide(typed: string, context: string[]): Promise<CorrectionDecision> {
    return this.client.decide(typed, context);
  }

  /** Should an accidental space between `prev` and `cur` be removed? Returns the joined word. */
  mergeDecision(prev: string, cur: string, context: string[]): Promise<string | null> {
    return this.client.mergeDecision(prev, cur, context);
  }

  // --- learning hooks ----------------------------------------------------
  // The worker owns the learned state (inference reads it every keystroke, so it
  // would be wasteful to ship it back and forth). We pull a snapshot only when
  // it's time to persist.

  /**
   * The state the ENGINE should run on.
   *
   * When personalization is off this is a FRESH state, not the stored one - the switch
   * has to stop the engine APPLYING what it learned, not merely stop it learning more.
   * The stored data is untouched on disk, so turning the switch back on restores it.
   */
  private personalizationForEngine() {
    return this.settings.personalizationEnabled
      ? this.store.personalization.toState()
      : emptyPersonalization(this.settings.beta);
  }

  recordAccept(insert: string, saved = 0): void {
    if (!this.settings.personalizationEnabled) return;
    void this.client.recordAccept(insert, saved).then(() => this.syncPersonalization());
  }

  recordCorrection(from: string, to: string): void {
    if (!this.settings.personalizationEnabled) return;
    void this.client.recordCorrection(from, to).then(() => this.syncPersonalization());
  }

  recordRevert(original: string): void {
    if (!this.settings.personalizationEnabled) return;
    void this.client.recordRevert(original).then(() => this.syncPersonalization());
  }

  /** Copy the worker's learned state into the store and schedule a debounced save. */
  private async syncPersonalization(): Promise<void> {
    if (!this.settings.personalizationEnabled) return; // never write back a blank state
    try {
      const state = await this.client.personalizationState();
      this.store.replace(state);
      this.store.touch();
    } catch {
      /* a failed sync must never break typing */
    }
  }

  /** Push externally-changed learned state (load / import / reset) into the engine. */
  async pushPersonalization(): Promise<void> {
    await this.client.setPersonalization(this.personalizationForEngine());
  }

  dispose(): void {
    this.client.dispose();
  }
}
