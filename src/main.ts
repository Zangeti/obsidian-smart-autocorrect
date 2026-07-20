/**
 * Smart Autocorrect - entry point.
 *
 * On-device predictive text + autocorrect for Obsidian, powered by a local neural
 * language model (see ./predictive). The plugin is intentionally tiny: it loads
 * settings, wires the predictive feature, and exposes its settings tab. All behaviour
 * lives in ./predictive (Obsidian glue) and its vendored ./predictive/engine
 * (unit-tested core). Fully offline - nothing leaves the vault.
 */
import { App, Plugin, PluginSettingTab } from "obsidian";
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

class SmartSettingTab extends PluginSettingTab {
  private plugin: SmartAutocorrectPlugin;

  constructor(app: App, plugin: SmartAutocorrectPlugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display(): void {
    this.containerEl.empty();
    this.plugin.predictive.renderSettings(this.containerEl, () => this.plugin.persist());
  }
}
