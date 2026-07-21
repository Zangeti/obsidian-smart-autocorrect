/**
 * "Getting started": a five-step tour shown once, after the first-run model download.
 *
 * Deliberately short. The plugin has a lot of surface, but only a few things a new user MUST
 * know or they will not understand what is happening to their text: Tab accepts, space
 * corrects, undo both restores and teaches, and links are offered rather than inserted. Every
 * other feature is discoverable from the settings tab and does not need explaining up front.
 * A longer tour is a tour people click through without reading, which teaches nothing.
 *
 * It ends on the stats counter rather than another feature: finishing on what the plugin has
 * done FOR you is a better last impression than one more thing to learn, and it points at the
 * status bar, which is the one piece of the UI that is always visible.
 *
 * Each step is one line of text and one picture, because that is what gets read.
 */
import { Modal, Setting } from "obsidian";
import type { App } from "obsidian";
import { TUTORIAL_IMAGES } from "./tutorialImages";

export interface TutorialStep {
  title: string;
  /** One sentence. If it needs two, it is two steps or it is not a need-to-know. */
  body: string;
  /** Keys the step is about, drawn as keycaps under the text. */
  keys?: string[];
  /** Key into TUTORIAL_IMAGES. The step renders without it if no picture is bundled. */
  image?: keyof typeof TUTORIAL_IMAGES;
}

export const TUTORIAL_STEPS: TutorialStep[] = [
  {
    title: "Press Tab to accept",
    body:
      "As you type, a list of likely next words appears. Press Tab to take the highlighted one, " +
      "or just keep typing to ignore it.",
    keys: ["Tab"],
    image: "suggest",
  },
  {
    title: "Typos fix themselves",
    body:
      "Finish a word with a space or punctuation and an obvious misspelling is corrected, the " +
      "way a phone keyboard does. Capital letters are handled for you too.",
    keys: ["Space"],
    image: "autocorrect",
  },
  {
    title: "Undo is how you teach it",
    body:
      "Not the word you wanted? Undo puts your original straight back, and the plugin remembers " +
      "not to make that correction again.",
    keys: ["Ctrl", "Z"],
    image: "undo",
  },
  {
    title: "Link notes as you write",
    body:
      "Experimental: text matching a note you already wrote is underlined. Hover to preview, " +
      "click to link. Nothing is inserted on its own, and you can turn it off in settings.",
    image: "links",
  },
  {
    title: "See what you have saved",
    body:
      "The counter in the status bar tracks the keystrokes this has saved you. Click it any " +
      "time for your streak, time saved, and what the plugin has learned.",
    image: "stats",
  },
];

export class TutorialModal extends Modal {
  private step = 0;
  private onDone?: () => void;

  constructor(app: App, onDone?: () => void) {
    super(app);
    this.onDone = onDone;
  }

  onOpen(): void {
    this.modalEl.addClass("smart-autocorrect-tutorial-modal");
    this.render();
    // Arrow keys page through, which is what anyone tries first in a stepped dialog.
    this.scope.register([], "ArrowRight", () => this.go(1));
    this.scope.register([], "ArrowLeft", () => this.go(-1));
  }

  private go(delta: number): void {
    const next = this.step + delta;
    if (next < 0) return;
    if (next >= TUTORIAL_STEPS.length) {
      this.close();
      return;
    }
    this.step = next;
    this.render();
  }

  private render(): void {
    const s = TUTORIAL_STEPS[this.step];
    const root = this.contentEl;
    root.empty();
    root.addClass("smart-autocorrect-tutorial");

    root.createEl("h2", { text: s.title, cls: "sa-tut-title" });
    root.createEl("p", { text: s.body, cls: "sa-tut-body" });

    if (s.keys?.length) {
      const keys = root.createDiv({ cls: "sa-tut-keys" });
      s.keys.forEach((k, i) => {
        if (i > 0) keys.createSpan({ cls: "sa-tut-plus", text: "+" });
        keys.createEl("kbd", { text: k });
      });
    }

    const src = s.image ? TUTORIAL_IMAGES[s.image] : undefined;
    if (src) {
      const img = root.createEl("img", { cls: "sa-tut-img" });
      img.src = src;
      img.alt = s.title;
    }

    // Dots first, so "where am I" is answered above the buttons the eye lands on.
    const dots = root.createDiv({ cls: "sa-tut-dots" });
    TUTORIAL_STEPS.forEach((_, i) => {
      dots.createSpan({ cls: i === this.step ? "sa-tut-dot is-active" : "sa-tut-dot" });
    });

    const last = this.step === TUTORIAL_STEPS.length - 1;
    const nav = new Setting(root);
    nav.settingEl.addClass("sa-tut-nav");
    if (this.step > 0) nav.addButton((b) => b.setButtonText("Back").onClick(() => this.go(-1)));
    else nav.addButton((b) => b.setButtonText("Skip").onClick(() => this.close()));
    nav.addButton((b) =>
      b
        .setButtonText(last ? "Start writing" : "Next")
        .setCta()
        .onClick(() => this.go(1)),
    );
  }

  onClose(): void {
    this.contentEl.empty();
    // Closing at any point counts as done: a tour you have to sit through is worse than no
    // tour, and it can always be reopened from the settings tab.
    this.onDone?.();
  }
}
