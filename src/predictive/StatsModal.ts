/**
 * "Your writing stats" dashboard: a proper window onto the gamification numbers that
 * otherwise only show as a small status-bar tally. Opened by clicking the status bar
 * or via the "Show writing stats" command. Read-only; all values come from a snapshot
 * so the modal has no dependency on the engine internals.
 */
import { Modal } from "obsidian";
import type { App } from "obsidian";
import { BMC_QR_DATA_URI } from "./bmcQr";

const BMC_URL = "https://buymeacoffee.com/zangeti";

export interface StatsSnapshot {
  keystrokesSaved: number;
  minutesSaved: number;
  todaySaved: number;
  streak: number;
  bestStreak: number;
  corrections: number;
  accepts: number;
  reverts: number;
  learnedWords: number;
  nextMilestone: number | null;
  allMilestones: number[];
}

function timeText(mins: number): string {
  if (mins >= 60) return `${(mins / 60).toFixed(1)} hours`;
  return `${Math.round(mins)} min`;
}

export class StatsModal extends Modal {
  private getSnapshot: () => StatsSnapshot;
  private onClosed?: () => void;

  constructor(app: App, getSnapshot: () => StatsSnapshot, onClosed?: () => void) {
    super(app);
    this.getSnapshot = getSnapshot;
    this.onClosed = onClosed;
  }

  onOpen(): void {
    this.render();
  }

  /**
   * Re-draw from a fresh snapshot. Called whenever the tallies change, so the numbers you are
   * looking at are the current ones - the dashboard used to render once on open and then sit
   * there stale while you carried on typing behind it. Driven by the accept event rather than
   * a timer, so it costs nothing when nothing is happening and never lags behind.
   */
  refresh(): void {
    if (this.contentEl.isShown()) this.render();
  }

  private render(): void {
    const s = this.getSnapshot();
    const root = this.contentEl;
    root.empty();
    root.addClass("smart-autocorrect-stats");
    root.createEl("h2", { text: "Your writing stats" });

    // Headline cards: the three numbers people care about.
    const cards = root.createDiv({ cls: "sa-stat-cards" });
    const card = (value: string, label: string) => {
      const c = cards.createDiv({ cls: "sa-stat-card" });
      c.createDiv({ cls: "sa-stat-value", text: value });
      c.createDiv({ cls: "sa-stat-label", text: label });
    };
    card(s.keystrokesSaved.toLocaleString(), "keystrokes saved");
    card(timeText(s.minutesSaved), "typing time saved");
    card(`${s.streak} day${s.streak === 1 ? "" : "s"}`, "current streak");

    // Secondary line: today + best streak.
    const sub = root.createEl("p", { cls: "sa-stat-sub" });
    sub.setText(
      `${s.todaySaved.toLocaleString()} saved today · longest streak ${s.bestStreak} day${s.bestStreak === 1 ? "" : "s"}`,
    );

    // Progress to the next milestone.
    if (s.nextMilestone !== null) {
      const prev = [...s.allMilestones].reverse().find((m) => m < s.nextMilestone!) ?? 0;
      const span = s.nextMilestone - prev;
      const done = Math.min(1, Math.max(0, (s.keystrokesSaved - prev) / span));
      const wrap = root.createDiv({ cls: "sa-progress-wrap" });
      wrap.createDiv({
        cls: "sa-progress-label",
        text: `Next up: ${s.nextMilestone.toLocaleString()} keystrokes saved`,
      });
      const bar = wrap.createDiv({ cls: "sa-progress-bar" });
      const fill = bar.createDiv({ cls: "sa-progress-fill" });
      fill.style.width = `${Math.round(done * 100)}%`;
      wrap.createDiv({
        cls: "sa-progress-remaining",
        text: `${(s.nextMilestone - s.keystrokesSaved).toLocaleString()} to go`,
      });
    } else {
      root.createEl("p", {
        cls: "sa-stat-sub",
        text: "🏆 You've hit every milestone. Typing legend.",
      });
    }

    // Key stats.
    root.createEl("h3", { text: "Key stats" });
    const rows = root.createDiv({ cls: "sa-stat-rows" });
    const row = (label: string, value: number) => {
      const r = rows.createDiv({ cls: "sa-stat-row" });
      r.createSpan({ cls: "sa-stat-row-label", text: label });
      r.createSpan({ cls: "sa-stat-row-value", text: value.toLocaleString() });
    };
    row("Suggestions accepted", s.accepts);
    row("Typos fixed", s.corrections);
    row("Corrections you undid", s.reverts);
    row("Words in your personal dictionary", s.learnedWords);

    // Support / Buy me a coffee.
    const support = root.createDiv({ cls: "smart-autocorrect-support sa-stat-support" });
    const p = support.createEl("p", { cls: "setting-item-description" });
    p.appendText("Enjoying the plugin? You can ");
    const link = p.createEl("a", { text: "buy me a coffee ☕", href: BMC_URL });
    link.setAttribute("target", "_blank");
    link.setAttribute("rel", "noopener");
    p.appendText(".");
    const qr = support.createEl("img", { cls: "smart-autocorrect-qr" });
    qr.src = BMC_QR_DATA_URI;
    qr.alt = "Buy Me a Coffee QR code";
    qr.width = 110;
    qr.height = 110;
  }

  onClose(): void {
    this.contentEl.empty();
    this.onClosed?.();
  }
}
