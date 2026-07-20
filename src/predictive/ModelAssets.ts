/**
 * First-run download of the language-model assets.
 *
 * Obsidian's community installer only ever fetches main.js, manifest.json and
 * styles.css, so the model files cannot ride along with the plugin. They are
 * published as attachments on the GitHub release for this exact version and
 * fetched once, into the plugin's own folder, after the user agrees.
 *
 * Rules this follows, because it is the only network access the plugin makes:
 *   - Nothing is downloaded until the user explicitly agrees. Decline is durable.
 *   - It is a plain GET of a public release asset. NOTHING is uploaded, and no
 *     identifier, telemetry or vault content is attached to the request.
 *   - Every file is size- and SHA-256-checked before it is written, so a truncated
 *     or substituted download cannot become a model the engine will load.
 *   - It never blocks startup: the plugin runs (degraded) with any subset present.
 */
import { App, Modal, Notice, Plugin, requestUrl, Setting } from "obsidian";

/** One downloadable asset, pinned by size and digest. */
export interface AssetSpec {
  file: string;
  bytes: number;
  sha256: string;
  /** What the user loses if this one is missing - shown in the consent dialog. */
  purpose: string;
}

/**
 * `RELEASE_TAG` names the release that CARRIES the model files, which is not the same as
 * the current plugin version: the models are large and change rarely, so they are published
 * once and every plugin release that can read them points at that same tag. Bump it only
 * when a new model is published (the .bin format is versioned - see engine/src/lstm/model.ts),
 * which keeps the pin doing its real job: an older plugin can never pull a newer,
 * incompatible model. It also keeps ordinary plugin releases to the three files Obsidian
 * actually installs.
 */
export const RELEASE_TAG = "1.0.0";
export const ASSET_BASE =
  `https://github.com/Zangeti/obsidian-smart-autocorrect/releases/download/${RELEASE_TAG}`;

export const MODEL_ASSETS: AssetSpec[] = [
  {
    file: "word_lstm.bin",
    bytes: 57665197,
    sha256: "be91e9d4f59786e5623bea9dbf908b1d7508b3445225d6ad03a922b01a8ae468",
    purpose: "next-word prediction, phrase completion and capitalisation",
  },
  {
    file: "predictive-global.bin",
    bytes: 26274133,
    sha256: "803270d341771ab87eac3d382b3540b1f027a309b036b937cd6226e5c9697c69",
    purpose: "word-frequency model used for autocorrect scoring",
  },
  {
    file: "wordlist.bin",
    bytes: 2804013,
    sha256: "598418c15388fb3f67acbb59161a409b42bb4af72ef40f5c8c42440451ab1c6e",
    purpose: "known-word list that stops real words being 'corrected'",
  },
];

const MB = 1024 * 1024;
export const totalMegabytes = (assets: AssetSpec[]): number =>
  Math.round(assets.reduce((n, a) => n + a.bytes, 0) / MB);

async function sha256Hex(buf: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/** Which of `MODEL_ASSETS` are not yet in the plugin folder. */
export async function missingAssets(plugin: Plugin): Promise<AssetSpec[]> {
  const dir = plugin.manifest.dir ?? ".";
  const adapter = plugin.app.vault.adapter;
  const out: AssetSpec[] = [];
  for (const a of MODEL_ASSETS) {
    if (!(await adapter.exists(`${dir}/${a.file}`))) out.push(a);
  }
  return out;
}

/**
 * Fetch `assets` into the plugin folder. Returns the ones that landed.
 *
 * A file is written only after its digest matches, so a failed verification
 * leaves the previous state untouched rather than a half-model on disk.
 */
export async function downloadAssets(
  plugin: Plugin,
  assets: AssetSpec[],
  onProgress?: (msg: string) => void,
): Promise<AssetSpec[]> {
  const dir = plugin.manifest.dir ?? ".";
  const adapter = plugin.app.vault.adapter;
  const done: AssetSpec[] = [];
  for (let i = 0; i < assets.length; i++) {
    const a = assets[i];
    const label = `${a.file} (${Math.round(a.bytes / MB)} MB, ${i + 1}/${assets.length})`;
    onProgress?.(`Downloading ${label}…`);
    try {
      const res = await requestUrl({ url: `${ASSET_BASE}/${a.file}`, method: "GET" });
      const buf = res.arrayBuffer;
      if (buf.byteLength !== a.bytes) {
        throw new Error(`expected ${a.bytes} bytes, got ${buf.byteLength}`);
      }
      // Skip the digest check only when the release was published without one.
      if (!a.sha256.startsWith("__")) {
        const got = await sha256Hex(buf);
        if (got !== a.sha256) throw new Error(`checksum mismatch (${got.slice(0, 12)}…)`);
      }
      await adapter.writeBinary(`${dir}/${a.file}`, buf);
      done.push(a);
    } catch (e) {
      console.error(`[smart-autocorrect] could not download ${a.file}`, e);
      onProgress?.(`Could not download ${a.file}: ${(e as Error).message}`);
      return done;
    }
  }
  return done;
}

/**
 * Ask before downloading. Resolves true if the user agreed.
 *
 * The dialog states the exact size, the exact source, and that nothing is sent -
 * a user who installed an offline plugin deserves to be told plainly the one time
 * it wants the network.
 */
export class AssetConsentModal extends Modal {
  private assets: AssetSpec[];
  private resolve: (ok: boolean) => void;
  private answered = false;

  constructor(app: App, assets: AssetSpec[], resolve: (ok: boolean) => void) {
    super(app);
    this.assets = assets;
    this.resolve = resolve;
  }

  onOpen(): void {
    const { contentEl } = this;
    contentEl.createEl("h2", { text: "Download the language model" });
    contentEl.createEl("p", {
      text:
        `Smart Autocorrect needs its language model (${totalMegabytes(this.assets)} MB). Obsidian's ` +
        `installer can't carry files this large, so it is downloaded once from GitHub.`,
    });
    const list = contentEl.createEl("ul");
    for (const a of this.assets) {
      list.createEl("li", { text: `${a.file} - ${a.purpose}` });
    }
    contentEl.createEl("p", {
      text:
        "This is the only network request the plugin makes. Nothing is uploaded, and each " +
        "file is checksum-verified before use.",
    });
    contentEl.createEl("p", {
      cls: "mod-warning",
      text:
        "You can decline and still use the plugin - it will learn from your vault alone, but " +
        "prediction and autocorrect will be much weaker. You can download later from settings.",
    });
    new Setting(contentEl)
      .addButton((b) =>
        b
          .setButtonText("Not now")
          .onClick(() => this.finish(false)),
      )
      .addButton((b) =>
        b
          .setButtonText(`Download ${totalMegabytes(this.assets)} MB`)
          .setCta()
          .onClick(() => this.finish(true)),
      );
  }

  private finish(ok: boolean): void {
    this.answered = true;
    this.resolve(ok);
    this.close();
  }

  onClose(): void {
    this.contentEl.empty();
    // Dismissing the dialog is a "no", not an unanswered question.
    if (!this.answered) this.resolve(false);
  }
}

export function askForAssets(app: App, assets: AssetSpec[]): Promise<boolean> {
  return new Promise((resolve) => new AssetConsentModal(app, assets, resolve).open());
}

/**
 * Full first-run flow: work out what is missing, ask, fetch, report.
 * Returns true if anything new was written (so the caller can reload the models).
 */
export async function ensureAssets(plugin: Plugin, force = false): Promise<boolean> {
  const missing = await missingAssets(plugin);
  if (missing.length === 0) {
    if (force) new Notice("Smart Autocorrect: language model is already installed.");
    return false;
  }
  if (!(await askForAssets(plugin.app, missing))) return false;

  const notice = new Notice("Smart Autocorrect: starting download…", 0);
  const got = await downloadAssets(plugin, missing, (m) => notice.setMessage(m));
  notice.hide();
  if (got.length === missing.length) {
    new Notice("Smart Autocorrect: language model installed.", 6000);
    return true;
  }
  new Notice(
    `Smart Autocorrect: downloaded ${got.length} of ${missing.length} files. ` +
      `Retry from the plugin settings.`,
    9000,
  );
  return got.length > 0;
}
