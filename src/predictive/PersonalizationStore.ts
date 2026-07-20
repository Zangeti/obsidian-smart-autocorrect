/**
 * Persists the user's learned personalisation.
 *
 * Design choice (Obsidian-idiomatic, no vault clutter): the live state lives in
 * the plugin's own data folder - `<vault>/.obsidian/plugins/<id>/personalization.json`.
 * That path is hidden from the note tree, is carried by Obsidian Sync / any
 * vault copy (so it transfers between machines automatically), and sits next to
 * the plugin's normal `data.json`. For sharing, `exportTo`/`importFrom` read and
 * write a plain JSON file at any vault-relative path the user picks.
 */
import type { Plugin } from "obsidian";
import { debounce, normalizePath, Notice } from "obsidian";
import {
  Personalization,
  normalizePersonalization,
  type PersonalizationState,
} from "./engine/index";

const FILE = "personalization.json";

export class PersonalizationStore {
  private plugin: Plugin;
  private beta: number;
  personalization: Personalization;
  private saveDebounced: () => void;

  constructor(plugin: Plugin, beta: number) {
    this.plugin = plugin;
    this.beta = beta;
    this.personalization = Personalization.empty(beta);
    this.saveDebounced = debounce(() => void this.save(), 3000, false);
  }

  private path(): string {
    const dir = this.plugin.manifest.dir ?? ".";
    return normalizePath(`${dir}/${FILE}`);
  }

  async load(): Promise<void> {
    try {
      const adapter = this.plugin.app.vault.adapter;
      const p = this.path();
      if (await adapter.exists(p)) {
        const raw = JSON.parse(await adapter.read(p)) as PersonalizationState;
        this.personalization = new Personalization(
          normalizePersonalization(raw, this.beta),
          this.beta,
        );
        return;
      }
    } catch (e) {
      console.warn("[predictive] failed to load personalization", e);
    }
    this.personalization = Personalization.empty(this.beta);
  }

  async save(): Promise<void> {
    try {
      await this.plugin.app.vault.adapter.write(
        this.path(),
        this.personalization.toJSONString(),
      );
    } catch (e) {
      console.warn("[predictive] failed to save personalization", e);
    }
  }

  /** Adopt state learned elsewhere. The engine runs in a worker and owns the live
   *  learned state, so after a learning hook fires we mirror its snapshot back
   *  here - this store remains the thing that writes the file. */
  replace(state: PersonalizationState): void {
    this.personalization = new Personalization(normalizePersonalization(state, this.beta), this.beta);
  }

  /** Call after any learning update; batches writes. */
  touch(): void {
    this.saveDebounced();
  }

  async reset(): Promise<void> {
    this.personalization.reset();
    await this.save();
    new Notice("Personalization reset");
  }

  /** Zero the tallied statistics only (keeps learned adaptation). */
  async resetStats(): Promise<void> {
    this.personalization.resetStats();
    await this.save();
  }

  async exportTo(vaultRelativePath: string): Promise<void> {
    const p = normalizePath(vaultRelativePath);
    await this.plugin.app.vault.adapter.write(
      p,
      this.personalization.toJSONString(true),
    );
    new Notice(`Exported personalization → ${p}`);
  }

  async importFrom(vaultRelativePath: string, merge: boolean): Promise<void> {
    const p = normalizePath(vaultRelativePath);
    const adapter = this.plugin.app.vault.adapter;
    if (!(await adapter.exists(p))) {
      new Notice(`No file at ${p}`);
      return;
    }
    const raw: unknown = JSON.parse(await adapter.read(p));
    if (merge) this.personalization.mergeFrom(raw);
    else this.personalization.loadFrom(raw);
    await this.save();
    new Notice(`Imported personalization ${merge ? "(merged)" : ""} from ${p}`);
  }

  updateBeta(beta: number): void {
    this.beta = beta;
  }
}
