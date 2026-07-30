/**
 * The personal-dictionary manager: a proper window listing every pinned word, the ones you added
 * yourself above the ones learned automatically (each alphabetical), with a remove button per word
 * and an add field at the bottom. Mutates the live settings object and calls `onChange` to persist.
 */
import { Modal, Setting } from "obsidian";
import type { App } from "obsidian";
import type { PredictiveSettings } from "./PredictiveSettings";

export class DictionaryModal extends Modal {
  constructor(
    app: App,
    private settings: PredictiveSettings,
    private onChange: () => void,
  ) {
    super(app);
  }

  onOpen(): void {
    this.render();
  }

  private render(): void {
    const el = this.contentEl;
    el.empty();
    el.addClass("smart-autocorrect-dict-modal");
    el.createEl("h2", { text: "Personal dictionary" });
    el.createEl("p", {
      cls: "setting-item-description",
      text:
        "Words that are always correct as written, so they're never autocorrected or re-cased. " +
        "Case-sensitive. Words you added yourself are listed first; the rest were learned when you " +
        "undid a correction.",
    });

    const userAdded = new Set(this.settings.userDictionaryUserAdded ?? []);
    const byName = (a: string, b: string) => a.localeCompare(b, undefined, { sensitivity: "base" });
    const added = this.settings.userDictionary.filter((w) => userAdded.has(w)).sort(byName);
    const learned = this.settings.userDictionary.filter((w) => !userAdded.has(w)).sort(byName);

    const remove = (w: string) => {
      this.settings.userDictionary = this.settings.userDictionary.filter((x) => x !== w);
      this.settings.userDictionaryUserAdded = (this.settings.userDictionaryUserAdded ?? []).filter((x) => x !== w);
      this.onChange();
      this.render();
    };
    const add = (raw: string) => {
      const w = raw.trim();
      if (!w || this.settings.userDictionary.includes(w)) return;
      this.settings.userDictionary = [...this.settings.userDictionary, w];
      this.settings.userDictionaryUserAdded = [...(this.settings.userDictionaryUserAdded ?? []), w];
      this.onChange();
      this.render();
    };

    const section = (title: string, words: string[]) => {
      if (words.length === 0) return;
      el.createEl("div", { cls: "smart-autocorrect-dict-heading", text: `${title} (${words.length})` });
      const list = el.createDiv({ cls: "smart-autocorrect-dict-rows" });
      for (const w of words) {
        const row = list.createDiv({ cls: "smart-autocorrect-dict-row" });
        row.createSpan({ text: w, cls: "smart-autocorrect-dict-word" });
        row.createEl("button", { text: "Remove", cls: "smart-autocorrect-dict-remove-btn" }).onclick = () => remove(w);
      }
    };
    section("Added by you", added);
    section("Learned automatically", learned);
    if (added.length === 0 && learned.length === 0)
      el.createEl("p", { cls: "setting-item-description", text: "No words yet — add one below, or right-click a word in a note." });

    let pending = "";
    new Setting(el)
      .setName("Add a word")
      .addText((t) => {
        t.setPlaceholder("word");
        t.onChange((v) => (pending = v));
        t.inputEl.addEventListener("keydown", (e) => {
          if (e.key === "Enter") { e.preventDefault(); add(t.getValue()); }
        });
      })
      .addButton((b) => b.setButtonText("Add").setCta().onClick(() => add(pending)));
  }

  onClose(): void {
    this.contentEl.empty();
  }
}
