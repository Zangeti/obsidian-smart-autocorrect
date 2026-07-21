/**
 * One description of the settings pane, rendered two ways.
 *
 * Obsidian 1.13 renders a settings tab from `getSettingDefinitions()` and, when that returns
 * anything, never calls `display()` at all. Older versions only have `display()`. Writing the
 * pane twice would guarantee the two drift apart, so the pane is described ONCE, as groups of
 * rows, and each path renders that same description:
 *
 *   - 1.13+  -> toSettingDefinitions(), the declarative API, so settings are searchable.
 *   - older  -> renderPaneGroups(), which builds the same DOM display() always built.
 *
 * The rows do not describe their controls declaratively. Each one carries the SAME callback
 * that used to sit in `new Setting(el).addToggle(...)`, replayed onto whichever `Setting`
 * object the active path hands it. That is the point: the control code, and every side effect
 * hanging off its onChange, is literally unchanged, so the two paths cannot behave differently.
 */
import { Setting } from "obsidian";
import type {
  ButtonComponent,
  DropdownComponent,
  ExtraButtonComponent,
  SettingDefinitionItem,
  SliderComponent,
  TextAreaComponent,
  TextComponent,
  ToggleComponent,
} from "obsidian";

/**
 * Records the fluent `Setting` calls a row makes instead of performing them, so the row can be
 * described before there is an element to draw it into. Mirrors the subset of `Setting` the
 * pane actually uses; anything else should be added here rather than worked around.
 */
export class RowBuilder {
  name = "";
  desc = "";
  private calls: ((s: Setting) => void)[] = [];

  setName(v: string): this {
    this.name = v;
    return this;
  }
  setDesc(v: string): this {
    this.desc = v;
    return this;
  }
  setClass(v: string): this {
    this.calls.push((s) => {
      s.setClass(v);
    });
    return this;
  }
  addToggle(cb: (c: ToggleComponent) => unknown): this {
    this.calls.push((s) => {
      s.addToggle((c) => void cb(c));
    });
    return this;
  }
  addSlider(cb: (c: SliderComponent) => unknown): this {
    this.calls.push((s) => {
      s.addSlider((c) => void cb(c));
    });
    return this;
  }
  addText(cb: (c: TextComponent) => unknown): this {
    this.calls.push((s) => {
      s.addText((c) => void cb(c));
    });
    return this;
  }
  addTextArea(cb: (c: TextAreaComponent) => unknown): this {
    this.calls.push((s) => {
      s.addTextArea((c) => void cb(c));
    });
    return this;
  }
  addDropdown(cb: (c: DropdownComponent) => unknown): this {
    this.calls.push((s) => {
      s.addDropdown((c) => void cb(c));
    });
    return this;
  }
  addButton(cb: (c: ButtonComponent) => unknown): this {
    this.calls.push((s) => {
      s.addButton((c) => void cb(c));
    });
    return this;
  }
  addExtraButton(cb: (c: ExtraButtonComponent) => unknown): this {
    this.calls.push((s) => {
      s.addExtraButton((c) => void cb(c));
    });
    return this;
  }

  /**
   * Replay onto a real row. `withText` is false on the declarative path, where Obsidian has
   * already written the name and description from the definition itself.
   */
  apply(s: Setting, withText: boolean): void {
    if (withText) {
      s.setName(this.name);
      if (this.desc) s.setDesc(this.desc);
    }
    for (const c of this.calls) c(s);
  }
}

/** A normal setting row: a name, a description, and controls. */
export interface PaneRow {
  kind: "row";
  row: RowBuilder;
  /** Extra words that should find this row in settings search. */
  aliases?: string[];
}

/** A paragraph of explanation between rows, with no control of its own. */
export interface PaneNote {
  kind: "note";
  text: string;
}

/** A block that builds its own DOM (the stats line, the support block). */
export interface PaneCustom {
  kind: "custom";
  /** Named so it can still be found in settings search; not shown for notes. */
  name: string;
  build: (el: HTMLElement) => void;
}

export type PaneItem = PaneRow | PaneNote | PaneCustom;

export interface PaneGroup {
  heading?: string;
  items: PaneItem[];
}

/**
 * Collects groups as the pane is described. `group()` starts a section, and the `row`/`note`/
 * `custom` helpers append to whichever section is open, so the describing code reads top to
 * bottom exactly like the imperative version it replaces.
 */
export class PaneBuilder {
  readonly groups: PaneGroup[] = [];

  constructor() {
    this.groups.push({ items: [] }); // anything before the first heading
  }

  group(heading?: string): void {
    this.groups.push({ heading, items: [] });
  }

  private get current(): PaneGroup {
    return this.groups[this.groups.length - 1];
  }

  row(aliases?: string[]): RowBuilder {
    const row = new RowBuilder();
    this.current.items.push({ kind: "row", row, aliases });
    return row;
  }

  note(text: string): void {
    this.current.items.push({ kind: "note", text });
  }

  custom(name: string, build: (el: HTMLElement) => void): void {
    this.current.items.push({ kind: "custom", name, build });
  }
}

/** Pre-1.13 path: build the DOM into `containerEl`, as display() always did. */
export function renderPaneGroups(containerEl: HTMLElement, groups: PaneGroup[]): void {
  for (const g of groups) {
    if (g.items.length === 0) continue;
    if (g.heading) new Setting(containerEl).setName(g.heading).setHeading();
    for (const item of g.items) {
      if (item.kind === "row") item.row.apply(new Setting(containerEl), true);
      else if (item.kind === "note")
        containerEl.createEl("p", { text: item.text, cls: "setting-item-description" });
      else item.build(containerEl);
    }
  }
}

/**
 * 1.13+ path: the same groups as definitions. Every row is a `render` definition rather than a
 * declared control type, so the control code stays byte-for-byte what the older path runs; the
 * name and description are lifted out as real definition fields, which is what makes them
 * searchable.
 */
export function toSettingDefinitions(groups: PaneGroup[]): SettingDefinitionItem[] {
  return groups
    .filter((g) => g.items.length > 0)
    .map((g) => ({
      type: "group" as const,
      heading: g.heading,
      items: g.items.map((item) => {
        if (item.kind === "row") {
          return {
            name: item.row.name,
            desc: item.row.desc || undefined,
            aliases: item.aliases,
            render: (s: Setting) => item.row.apply(s, false),
          };
        }
        if (item.kind === "note") {
          // A description with no name: the same explanatory paragraph, in a row of its own.
          // Excluded from search because a paragraph is not something you can go and change.
          return { name: "", desc: item.text, searchable: false, render: () => {} };
        }
        return {
          name: item.name,
          render: (s: Setting) => {
            s.settingEl.addClass("smart-autocorrect-custom-row");
            item.build(s.settingEl);
          },
        };
      }),
    }));
}
