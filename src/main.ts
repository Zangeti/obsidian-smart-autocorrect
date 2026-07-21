/**
 * Smart Autocorrect - entry point.
 *
 * On-device predictive text + autocorrect for Obsidian, powered by a local neural
 * language model (see ./predictive). The plugin is intentionally tiny: it loads
 * settings, wires the predictive feature, and exposes its settings tab. All behaviour
 * lives in ./predictive (Obsidian glue) and its vendored ./predictive/engine
 * (unit-tested core). Fully offline - nothing leaves the vault.
 */
import { App, Plugin, PluginSettingTab, type SettingDefinitionItem } from "obsidian";
import { PredictiveFeature } from "./predictive/PredictiveFeature";
import type { PredictiveSettings } from "./predictive/PredictiveSettings";
import type { EngagementState } from "./predictive/EngagementStore";

interface PluginData {
  predictive?: Partial<PredictiveSettings>;
  engagement?: Partial<EngagementState>;
}

export default class SmartAutocorrectPlugin extends Plugin {
  predictive!: PredictiveFeature;

  async onload(): Promise<void> {
    const data = ((await this.loadData()) as PluginData | null) ?? {};
    this.predictive = new PredictiveFeature(this, data.predictive, data.engagement);
    this.predictive.onEngagementChange = () => void this.persist();
    this.predictive.onPersistSettings = () => void this.persist();
    await this.predictive.enable();
    this.addSettingTab(new SmartSettingTab(this.app, this));
  }

  onunload(): void {
    this.predictive?.dispose();
  }

  /** Persist the predictive settings + engagement stats (personalisation saves itself
   *  separately, in its own file). */
  async persist(): Promise<void> {
    await this.saveData({
      predictive: this.predictive.settings,
      engagement: this.predictive.engagementState(),
    });
  }
}

/**
 * The pane is described once, in PredictiveFeature.settingGroups, and rendered by whichever
 * path this version of Obsidian supports:
 *
 *   - 1.13+: getSettingDefinitions() returns the description, Obsidian renders it, and the
 *     settings are searchable. display() is NOT called when this returns anything.
 *   - older: display() draws the same description imperatively.
 *
 * Both run the identical control callbacks, so they cannot behave differently.
 */
class SmartSettingTab extends PluginSettingTab {
  private plugin: SmartAutocorrectPlugin;

  constructor(app: App, plugin: SmartAutocorrectPlugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  getSettingDefinitions(): SettingDefinitionItem[] {
    return this.plugin.predictive.settingDefinitions(
      () => this.plugin.persist(),
      // A setting that changes which OTHER settings apply asks for this: update() re-reads the
      // description and re-renders, the declarative equivalent of redrawing.
      //
      // Called optionally, and typed structurally, because update() only exists on Obsidian
      // 1.13+ while this plugin supports 1.11.4. That is not a hypothetical gap: on an older
      // version the base class has no such method. It is also unreachable there, since this
      // callback is only ever invoked by the definitions renderer, which older versions do
      // not have - so the optional call is the honest expression of "newer versions only".
      () => {
        const tab = this as { update?: () => void };
        tab.update?.();
      },
    );
  }

  display(): void {
    this.containerEl.empty();
    this.plugin.predictive.renderSettings(this.containerEl, () => this.plugin.persist());
  }
}
