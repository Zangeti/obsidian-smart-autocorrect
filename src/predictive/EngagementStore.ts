/**
 * Gamification / engagement state: how many keystrokes the plugin has saved you, a
 * daily streak, your best streak, and milestone celebrations. Lives on the main thread
 * (the accept path already runs here and knows the saving) and is persisted in the
 * plugin's data.json alongside settings.
 *
 * "Keystrokes saved" is the sum of characters the user did NOT have to type because a
 * suggestion/ghost completion was accepted. Time saved is a rough estimate at an average
 * typing speed.
 */

/** Average typing speed used to turn saved characters into saved minutes (~40 wpm). */
const CHARS_PER_MINUTE = 200;

/** Thresholds that fire a one-time celebration toast. */
const MILESTONES = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000];

export interface EngagementState {
  totalSaved: number;
  todaySaved: number;
  lastActiveDay: string; // YYYY-MM-DD, local
  streak: number;
  bestStreak: number;
  milestones: number[]; // thresholds already celebrated
}

function emptyEngagement(): EngagementState {
  return { totalSaved: 0, todaySaved: 0, lastActiveDay: "", streak: 0, bestStreak: 0, milestones: [] };
}

/** Local calendar day as YYYY-MM-DD (so streaks roll over at the user's midnight). */
function localDay(d = new Date()): string {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

/** Whole days from day-string `a` to day-string `b` (b - a). */
function dayDiff(a: string, b: string): number {
  const [ay, am, ad] = a.split("-").map(Number);
  const [by, bm, bd] = b.split("-").map(Number);
  const ta = Date.UTC(ay, am - 1, ad);
  const tb = Date.UTC(by, bm - 1, bd);
  return Math.round((tb - ta) / 86_400_000);
}

export class EngagementStore {
  private s: EngagementState;

  constructor(state?: Partial<EngagementState>) {
    this.s = { ...emptyEngagement(), ...(state ?? {}) };
  }

  get total(): number { return this.s.totalSaved; }
  get today(): number { return this.s.todaySaved; }
  get streak(): number { return this.s.streak; }
  get bestStreak(): number { return this.s.bestStreak; }
  get minutesSaved(): number { return this.s.totalSaved / CHARS_PER_MINUTE; }

  /** Milestone thresholds already reached, ascending. */
  get achievedMilestones(): number[] { return MILESTONES.filter((m) => this.s.totalSaved >= m); }
  /** The next milestone to aim for, or null once the last one is passed. */
  get nextMilestone(): number | null { return MILESTONES.find((m) => this.s.totalSaved < m) ?? null; }
  /** All milestone thresholds (reached or not), for rendering a progress ladder. */
  get allMilestones(): number[] { return [...MILESTONES]; }

  toState(): EngagementState { return { ...this.s, milestones: [...this.s.milestones] }; }

  /**
   * Record `saved` keystrokes from one accept. Handles the day rollover (advancing or
   * resetting the streak) and returns any milestone thresholds newly crossed so the
   * caller can celebrate them.
   */
  record(saved: number): number[] {
    if (saved <= 0) return [];
    const day = localDay();
    if (day !== this.s.lastActiveDay) {
      const gap = this.s.lastActiveDay ? dayDiff(this.s.lastActiveDay, day) : Infinity;
      this.s.streak = gap === 1 ? this.s.streak + 1 : 1;
      this.s.todaySaved = 0;
      this.s.lastActiveDay = day;
    }
    this.s.todaySaved += saved;
    this.s.totalSaved += saved;
    if (this.s.streak > this.s.bestStreak) this.s.bestStreak = this.s.streak;

    const crossed: number[] = [];
    for (const m of MILESTONES) {
      if (this.s.totalSaved >= m && !this.s.milestones.includes(m)) {
        this.s.milestones.push(m);
        crossed.push(m);
      }
    }
    return crossed;
  }

  reset(): void {
    this.s = emptyEngagement();
  }

  /** Compact status-bar label, e.g. "⌨️ 1,240 saved · 🔥 6d". */
  statusText(): string {
    const n = this.s.totalSaved.toLocaleString();
    const streak = this.s.streak > 1 ? ` · 🔥 ${this.s.streak}d` : "";
    return `⌨️ ${n} saved${streak}`;
  }

  /** Human "time saved" string, e.g. "≈ 3.2 hrs" or "≈ 14 min". */
  timeSavedText(): string {
    const mins = this.minutesSaved;
    if (mins >= 60) return `≈ ${(mins / 60).toFixed(1)} hrs`;
    return `≈ ${Math.round(mins)} min`;
  }
}
