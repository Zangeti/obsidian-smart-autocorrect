/** A blunt, red confirmation dialog for destructive actions (e.g. resetting statistics). */
import { Modal } from "obsidian";
import type { App } from "obsidian";

interface ConfirmOptions {
  title: string;
  body: string;
  confirmText: string;
  onConfirm: () => void;
  /** Style as a destructive action (red border + red confirm button). Default true.
   *  Set false for a reversible confirm (e.g. resetting options) - a plain confirm dialog. */
  danger?: boolean;
}

export class ConfirmModal extends Modal {
  private opts: ConfirmOptions;

  constructor(app: App, opts: ConfirmOptions) {
    super(app);
    this.opts = opts;
  }

  onOpen(): void {
    const { contentEl } = this;
    const danger = this.opts.danger !== false; // default: destructive styling
    contentEl.empty();
    contentEl.addClass("smart-autocorrect-confirm");
    if (danger) contentEl.addClass("sa-confirm-red");
    contentEl.createEl("h2", { cls: "sa-confirm-title", text: this.opts.title });
    contentEl.createEl("p", { cls: "sa-confirm-body", text: this.opts.body });
    const row = contentEl.createDiv({ cls: "sa-confirm-actions" });
    const cancel = row.createEl("button", { text: "Cancel" });
    cancel.onclick = () => this.close();
    const confirm = row.createEl("button", {
      cls: danger ? "mod-warning sa-confirm-danger" : "mod-cta",
      text: this.opts.confirmText,
    });
    confirm.onclick = () => {
      this.opts.onConfirm();
      this.close();
    };
  }

  onClose(): void {
    this.contentEl.empty();
  }
}
