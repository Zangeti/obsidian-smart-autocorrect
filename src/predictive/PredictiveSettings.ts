/**
 * Settings for the predictive/autocorrect layer, plus a renderer that adds a
 * section to the plugin's settings tab. Every feature is an independent toggle,
 * matching the design.
 */
import { PaneBuilder, renderPaneGroups } from "./settingsPane";
import type { PaneGroup } from "./settingsPane";
import { parseExcludeList } from "./engine/index";
import type { KeyboardLayoutName } from "./engine/index";
import { BMC_QR_DATA_URI } from "./bmcQr";

const BMC_URL = "https://buymeacoffee.com/zangeti";

/**
 * Keys offered for accepting a suggestion.
 *
 * Tab is the default and Enter is deliberately NOT also bound: Obsidian's own
 * EditorSuggest binds Enter for free, so both used to accept, and an Enter that
 * sometimes writes a newline and sometimes accepts a suggestion is a coin flip the
 * user cannot see. Exactly one key accepts, and it is the one named here - whichever
 * is chosen, the others do their normal editor job (see PredictiveSuggest.bindKeys).
 */
export const ACCEPT_KEYS = ["Tab", "Enter", "ArrowRight", "End"] as const;
export type AcceptKey = (typeof ACCEPT_KEYS)[number];

export interface PredictiveSettings {
  /** Master switch for the WHOLE plugin: when off, nothing runs - no autocorrect, no
   *  predictions/ghost, no link or tag suggestions. Leaves the plugin installed and its
   *  settings/personalization intact, so you can turn everything back on in one click. */
  pluginEnabled: boolean;

  /** master switch for context-aware predictions in the popup. */
  enablePredictions: boolean;

  /** learn from and bias toward the user's own vault. */
  personalBias: boolean;
  /** strength of the vault bias, 0..1 (mixture weight alpha). */
  alpha: number;
  /** blend of the neural (LSTM) next-word prior vs the word-frequency model,
   *  0..1: 0 = n-gram only, 1 = LSTM only. Only applies when an LSTM is loaded. */
  lstmWeight: number;

  /** Whether the one-time "download the language model" prompt has been shown. The
   *  prompt is offered once; declining is durable, and the settings pane has a button
   *  for changing your mind. */
  modelPromptShown: boolean;

  /** channel "trust" weight: high => trust typed chars, low => trust context. */
  beta: number;
  /** spread (in keys) of the geometric keyboard-substitution prior. */
  channelSigma: number;
  /** physical keyboard layout for the geometric typo prior. */
  keyboardLayout: KeyboardLayoutName;
  /** max channel cost (nats) for a fuzzy neighbour to be considered. */
  maxEditCost: number;

  /** mobile-style replace-on-space. */
  autocorrectOnSpace: boolean;
  /**
   * Information-gain (nats) a correction must clear: the typed word's Shannon
   * surprisal must exceed the chosen word's by at least this much. The single
   * autocorrect-strength control. Low = correct eagerly; high = only when the
   * typed word is very unlikely vs. the chosen one.
   */
  infoGainThreshold: number;
  /** capitalise real sentence starts. */
  autoCapitalize: boolean;
  /** remove an accidental doubled function word on space ("the the" -> "the"). Only a
   *  curated set of never-validly-doubled words, so "had had"/"that that" are safe. */
  removeDoubledWords: boolean;
  /** extra abbreviations that must not trigger capitalisation. */
  extraAbbreviations: string[];

  /**
   * User dictionary: words that are correct as written and must never be
   * autocorrected or re-cased.
   *
   * CASE-SENSITIVE by design, and that is the point: it is the escape hatch for
   * spellings the model cannot represent. The model factors casing into
   * lower/Title/UPPER + a learned table of irregular forms, so anything it has not
   * seen ("kubeCTL", "myVar", "NixOS") can be pinned here exactly as written.
   * "GmbH" and "gmbh" are therefore different entries.
   */
  userDictionary: string[];
  /** also offer dictionary words as completions, not just protect them. */
  suggestUserDictionary: boolean;
  /** when you undo a correction, add that word to the personal dictionary so it is never
   *  corrected again (the dictionary IS the "don't-touch" list). */
  undoAddsToDictionary: boolean;
  /** when you undo a sentence-initial capitalisation that followed "word.", learn "word" as an
   *  abbreviation so we stop capitalising after it (e.g. "etc.", "incl."). */
  learnAbbreviationsOnRevert: boolean;
  /** drop a personal-dictionary word once it no longer appears anywhere in the vault, so deleting
   *  the notes that used it also forgets it (the dictionary tracks words you actually write). */
  pruneDictionaryFromVault: boolean;

  /**
   * Filter profanity, slurs, and explicit/NSFW words out of what the plugin OFFERS -
   * they are never suggested and never used as an autocorrect target. On by default.
   * This never touches what the user types: a blocked word typed deliberately is left
   * exactly as written and is never "corrected" away.
   */
  filterProfanity: boolean;


  // --- matching-quality features (#1-#7) ---
  /** keyboard-geometry matching strength (higher = more keyboard-typo tolerant; 0 = off). */
  fuzzyStrength: number;
  /** phonetic (sound-alike) matching strength (higher = more sound-alike tolerant; 0 = off). */
  phoneticStrength: number;
  /** adaptive keyboard confusion model that learns the user's slips. */
  adaptiveKeyboard: boolean;
  /** learned reranker that adapts candidate order to accepts. */
  learnedRanking: boolean;
  /** real-word correction ("form"->"from" in context). */
  realWordCorrection: boolean;
  /** split/join correction ("alot"->"a lot"). */
  splitCorrection: boolean;
  /** Kneser-Ney continuation probabilities. */
  useContinuation: boolean;
  /** within-document cache weight (0 disables). */
  cacheGamma: number;
  /** vault-relative path used for export/import of personalization. NOTE this is NOT
   *  where personalization lives - the live store is always
   *  <vault>/.obsidian/plugins/<id>/personalization.json. This path is only touched by
   *  the Export/Import buttons. */
  personalizationSharePath: string;
  /** Master switch for personalization: learning AND applying what was learned.
   *  Off = the engine behaves identically for everyone, and nothing new is recorded. */
  personalizationEnabled: boolean;

  /** Key that accepts the highlighted suggestion. */
  acceptKey: AcceptKey;
  /** Collapse a double space before a word as you complete it. */
  collapseDoubleSpace: boolean;
  /** Run the neural model on the WASM-SIMD kernel (default). Off forces scalar JS. */
  wasmSimd: boolean;

  /** max number of suggestions shown in the popup (e.g. 3). */
  maxSuggestions: number;

  /** Only let Tab indent a list item when the caret is at the start of the item's content
   *  (right after the bullet), like Word - so a Tab-accept that misses because the popup
   *  closed can't accidentally indent the bullet. */
  tabIndentAtBulletStartOnly: boolean;

  // --- markdown / performance ---
  /** skip prediction/autocorrect inside code, math, links, tags, frontmatter. */
  markdownAware: boolean;
  /** run the WHOLE engine (prediction, autocorrect, model building) in a Web
   *  Worker. Off = run it inline on the main thread, which will stutter typing. */
  offMainThread: boolean;
  /** show the top prediction as dimmed inline "ghost text" (Tab to accept). */
  ghostText: boolean;

  // --- triggers / scope ---
  /** Folders or files (glob patterns allowed) where predictions AND autocorrect never
   *  run - e.g. "Templates", "Journal/*", "*.excalidraw.md". */
  excludedFolders: string[];
  /** Minimum characters typed into the CURRENT word before completions appear. 1 = as
   *  soon as you start a word; higher = only after a longer prefix (less noise).
   *  Next-word prediction right after a space is unaffected. */
  minChars: number;

  /** Suggest related notes to link, per paragraph/bullet, blending the neural model's
   *  topic fingerprint with keyword overlap. A link icon appears only where a close match
   *  exists; clicking it offers the feasible targets. Never inserts anything on its own. */
  suggestLinks: boolean;
  /** How eager related-link suggestions are, 1 (only very close matches) to 5 (looser). */
  relatedSensitivity: number;
  /** Minimum words in a block before it can get a link icon (keeps stray half-sentences
   *  and lines you're mid-typing free of suggestions). */
  minLinkWords: number;
  /** Suggest relevant tags as you type `#` (Obsidian-native tag autocomplete). */
  suggestTagsOnHash: boolean;
  /** Keep the note's frontmatter `tags:` in sync with the #tags used in its body: add a
   *  tag when it first appears inline, remove it when its last inline use is deleted. */
  syncFrontmatterTags: boolean;
  /** Replace Obsidian's built-in `[[` link picker with this plugin's, which ranks notes by
   *  topical relevance to what you're writing and can link to a specific section. */
  replaceLinkMenu: boolean;
}

export interface PersonalizationHandlers {
  onReset: () => void | Promise<void>;
  onExport: (path: string) => void | Promise<void>;
  onImport: (path: string, merge: boolean) => void | Promise<void>;
  getStats: () => { corrections: number; accepts: number; reverts: number; charsSaved: number; streak: number; bestStreak: number; minutesSaved: number; learnListSize: number };
  /** Open the full "writing stats" dashboard. */
  onOpenStats: () => void;
  /** Reset all statistics (behind a confirmation). */
  onResetStats: () => void;
  /** Reset every SETTING to its default, keeping your personal dictionary and learned
   *  personalization (behind a confirmation). */
  onResetSettings: () => void;
  /** Factory reset: settings, personalization, statistics and the personal dictionary - the
   *  lot (behind a loud red confirmation). */
  onFactoryReset: () => void;
}

export interface AccelerationHandlers {
  /** Live engine status, so the pane reports what is ACTUALLY happening rather than
   *  what the toggle merely requests. */
  status: () => Promise<{ ready: boolean; accelerated: boolean; lstmLoaded: boolean }>;
  /** Re-load the neural model so a change to the WASM-SIMD toggle takes effect now. */
  reload: () => Promise<void>;
  /** How many model files are still missing, so the pane can offer to fetch them. */
  missingAssets: () => Promise<number>;
  /** Run the download (with its consent dialog). Resolves true if anything landed. */
  installAssets: () => Promise<boolean>;
}

export const DEFAULT_PREDICTIVE_SETTINGS: PredictiveSettings = {
  pluginEnabled: true,
  enablePredictions: true,
  personalBias: true,
  alpha: 0.15,
  lstmWeight: 0.6,
  // 1.5 (trust the typed letters a bit more than pure 1.0) measured best on a real-model
  // A/B: same typo recovery as 1.0 but fewer harmful mis-corrections (a far context-driven
  // word replacing a closer typo fix, e.g. "th"→"to"); ≥2 starts losing recovery.
  beta: 1.5,
  modelPromptShown: false,
  channelSigma: 1.0,
  keyboardLayout: "qwerty",
  maxEditCost: 4.0,
  autocorrectOnSpace: true,
  infoGainThreshold: 2.5,
  autoCapitalize: true,
  removeDoubledWords: true,
  extraAbbreviations: [],
  userDictionary: [],
  suggestUserDictionary: true,
  undoAddsToDictionary: true,
  learnAbbreviationsOnRevert: true,
  pruneDictionaryFromVault: true,
  filterProfanity: true,
  fuzzyStrength: 1.0,
  phoneticStrength: 1.0,
  adaptiveKeyboard: true,
  learnedRanking: true,
  realWordCorrection: true,
  splitCorrection: true,
  useContinuation: true,
  cacheGamma: 0.15,
  personalizationSharePath: "predictive-personalization.json",
  personalizationEnabled: true,
  acceptKey: "Tab",
  collapseDoubleSpace: true,
  wasmSimd: true,
  maxSuggestions: 3,
  tabIndentAtBulletStartOnly: false,
  markdownAware: true,
  offMainThread: true,
  ghostText: false,
  excludedFolders: [],
  minChars: 1,
  suggestLinks: true,
  relatedSensitivity: 3,
  minLinkWords: 12,
  suggestTagsOnHash: true,
  syncFrontmatterTags: true,
  replaceLinkMenu: true,
};

/**
 * Answers the settings pane needs but can only get asynchronously (is the model installed, is
 * SIMD actually running). The pane is described synchronously and re-described on every render,
 * so the answers are cached HERE, outside any one render, and fetched only when missing. A
 * fetch that resolves calls `redraw`; because the result is cached, that redraw does not fetch
 * again, so there is no loop. Invalidate a field to ask again.
 */
export interface AccelerationState {
  status?: string;
  missing?: number;
  pending?: boolean;
}

/** Describe the whole pane. Called fresh on every render, so plain `if`s are enough to make a
 *  setting conditional: the description is rebuilt from current state each time. */
export function buildPredictiveSettingGroups(
  settings: PredictiveSettings,
  onChange: () => void | Promise<void>,
  personalization?: PersonalizationHandlers,
  /** Re-render the whole pane. Needed by settings whose DESCRIPTION or visibility
   *  depends on another setting, so the pane cannot go stale under the user. */
  redraw?: () => void,
  /** Lets the WASM-SIMD toggle report the real acceleration state and apply changes
   *  immediately. Absent (e.g. in tests) hides the status line. */
  acceleration?: AccelerationHandlers,
  accelState: AccelerationState = {},
): PaneGroup[] {
  const b = new PaneBuilder();
  const row = () => b.row();
  const commit = () => void onChange();
  const bag = settings as unknown as Record<string, boolean>;
  const toggle = (name: string, desc: string, key: keyof PredictiveSettings) =>
    row()
      .setName(name)
      .setDesc(desc)
      .addToggle((t) =>
        t.setValue(bag[key as string]).onChange((v) => {
          bag[key as string] = v;
          commit();
        }),
      );

  b.group("Smart predictions & autocorrect");
  b.note(
    "Context-aware next-word prediction, phone-style autocorrect, and multi-word phrases. Accept with Tab. Learns from your vault as you write.",
  );

  // --- master switch + resets, right at the top ---------------------------
  row()
    .setName("Enable Smart Autocorrect")
    .setDesc("Master switch. Off = nothing runs (no autocorrect, predictions, ghost text, or link/tag suggestions), but the plugin stays installed and your settings and learned personalization are kept.")
    .addToggle((t) =>
      t.setValue(settings.pluginEnabled).onChange((v) => {
        settings.pluginEnabled = v;
        commit();
        redraw?.();
      }),
    );

  if (personalization) {
    row()
      .setName("Reset settings")
      .setDesc("Put every option in this menu back to its default. Your personal dictionary and everything the plugin has learned about how you write are kept. Asks you to confirm first.")
      .addButton((b) => b.setButtonText("Reset settings").onClick(() => personalization.onResetSettings()));
    row()
      .setName("Factory reset")
      .setDesc("Wipe everything this plugin stores - settings, personalization, statistics and your personal dictionary. Can't be undone.")
      .addButton((b) => b.setButtonText("Factory reset").setWarning().onClick(() => personalization.onFactoryReset()));
  }

  if (!settings.pluginEnabled) {
    b.note("Smart Autocorrect is turned off. Turn the master switch back on to change the options below.");
    return b.groups; // nothing else is active, so don't show a wall of dead options
  }

  b.group("Predictions & autocorrect");

  row()
    .setName("Predictive text (suggest the next word)")
    .setDesc("Shows likely next words from the recent context. The neural model reads up to about 24 words back; the word-frequency model uses the last one or two. Turn off to disable all suggestions.")
    .addToggle((t) =>
      t.setValue(settings.enablePredictions).onChange((v) => {
        settings.enablePredictions = v;
        commit();
      }),
    );

  row()
    .setName("Autocorrect typos when you press space")
    .setDesc("Automatically fixes a misspelt word on space/punctuation (mobile-keyboard style). Undo with Ctrl/Cmd-Z; that also teaches it to leave that word alone.")
    .addToggle((t) =>
      t.setValue(settings.autocorrectOnSpace).onChange((v) => {
        settings.autocorrectOnSpace = v;
        commit();
      }),
    );

  row()
    .setName("Autocorrect strength (information gain)")
    .setDesc(
      "How surprising the typed word must be, versus the best alternative, before it's replaced (Shannon information gain, in nats). The single control for how readily it corrects: lower = corrects even mildly-off words; higher = only fixes words that are very unlikely in context.",
    )
    .addSlider((s) =>
      s
        .setLimits(0.5, 8, 0.5)
        .setValue(settings.infoGainThreshold)
        .setDynamicTooltip()
        .onChange((v) => {
          settings.infoGainThreshold = v;
          commit();
        }),
    );

  row()
    .setName("Remove accidental doubled words")
    .setDesc(
      'Delete a repeated function word as you type ("the the" → "the"). Only words that are ' +
        'never validly doubled are touched, so "had had" and "that that" are left alone.',
    )
    .addToggle((t) =>
      t.setValue(settings.removeDoubledWords).onChange((v) => {
        settings.removeDoubledWords = v;
        commit();
      }),
    );

  row()
    .setName("Auto-capitalise sentences & proper nouns")
    .setDesc('Capitalises real sentence starts (safe with "U.S.", "e.g.", decimals), fixes "THe"→"The", and learns proper-noun casing ("london"→"London").')
    .addToggle((t) =>
      t.setValue(settings.autoCapitalize).onChange((v) => {
        settings.autoCapitalize = v;
        commit();
      }),
    );

  row()
    .setName("Learn my writing style from this vault")
    .setDesc("Biases predictions toward the words and phrasing you actually use in your notes.")
    .addToggle((t) =>
      t.setValue(settings.personalBias).onChange((v) => {
        settings.personalBias = v;
        commit();
      }),
    );

  row()
    .setName("Vault influence")
    .setDesc("How much your own notes outweigh the general dictionary. Higher = more personalised, lower = more generic.")
    .addSlider((s) =>
      s
        .setLimits(0, 1, 0.05)
        .setValue(settings.alpha)
        .setDynamicTooltip()
        .onChange((v) => {
          settings.alpha = v;
          commit();
        }),
    );

  row()
    .setName("Neural vs. word-frequency blend")
    .setDesc("How much the neural (LSTM) next-word model influences suggestions and corrections, vs the word-frequency model. 0 = word-frequency only; 1 = neural only. Only applies when a neural model is installed.")
    .addSlider((s) =>
      s
        .setLimits(0, 1, 0.05)
        .setValue(settings.lstmWeight)
        .setDynamicTooltip()
        .onChange((v) => {
          settings.lstmWeight = v;
          commit();
        }),
    );

  row()
    .setName("Trust typing vs. context (β)")
    .setDesc("When a typo is ambiguous: higher trusts the exact letters you typed, lower trusts what the sentence expects.")
    .addSlider((s) =>
      s
        .setLimits(0.2, 3, 0.1)
        .setValue(settings.beta)
        .setDynamicTooltip()
        .onChange((v) => {
          settings.beta = v;
          commit();
        }),
    );

  row()
    .setName("Words that aren't sentence ends")
    .setDesc('Comma-separated abbreviations that should NOT trigger capitalisation after their period, e.g. "approx., dept.".')
    .addTextArea((t) =>
      t
        .setValue(settings.extraAbbreviations.join(", "))
        .onChange((v) => {
          settings.extraAbbreviations = v
            .split(",")
            .map((s) => s.trim())
            .filter((s) => s.length > 0);
          commit();
        }),
    );

  row()
    .setName("Personal dictionary")
    .setDesc(
      'Comma-separated words that are always correct as written, so they are never autocorrected or re-cased. ' +
      'Case-sensitive, so odd spellings the model cannot guess ("NixOS", "kubeCTL", "myVar") are pinned exactly as you type them. ' +
      '"GmbH" and "gmbh" are separate entries.',
    )
    .addTextArea((t) =>
      t
        .setValue(settings.userDictionary.join(", "))
        .onChange((v) => {
          // Case and inner spaces preserved; only the separators are trimmed.
          settings.userDictionary = v
            .split(",")
            .map((s) => s.trim())
            .filter((s) => s.length > 0);
          commit();
        }),
    );

  row()
    .setName("Suggest personal dictionary words")
    .setDesc("Offer your dictionary words as completions too, not just protect them from autocorrect.")
    .addToggle((t) =>
      t.setValue(settings.suggestUserDictionary).onChange((v) => {
        settings.suggestUserDictionary = v;
        commit();
      }),
    );

  row()
    .setName("Undo adds the word to your dictionary")
    .setDesc(
      "When you undo an autocorrection (Ctrl/Cmd-Z), add that word to your personal dictionary " +
        "above so it's never corrected again. Your dictionary is the plugin's don't-touch list.",
    )
    .addToggle((t) =>
      t.setValue(settings.undoAddsToDictionary).onChange((v) => {
        settings.undoAddsToDictionary = v;
        commit();
      }),
    );

  row()
    .setName("Learn abbreviations when you undo a capital")
    .setDesc(
      "When you undo a capital letter that was added after an abbreviation (e.g. undoing the " +
        "capital in “etc. Then” back to “then”), remember that word so the plugin stops " +
        "capitalising after it.",
    )
    .addToggle((t) =>
      t.setValue(settings.learnAbbreviationsOnRevert).onChange((v) => {
        settings.learnAbbreviationsOnRevert = v;
        commit();
      }),
    );

  row()
    .setName("Forget dictionary words removed from the vault")
    .setDesc(
      "When a word in your personal dictionary no longer appears in any note, drop it " +
        "automatically. Keeps the dictionary to words you actually write.",
    )
    .addToggle((t) =>
      t.setValue(settings.pruneDictionaryFromVault).onChange((v) => {
        settings.pruneDictionaryFromVault = v;
        commit();
      }),
    );

  row()
    .setName("Filter profanity & NSFW words")
    .setDesc(
      "Never suggest or autocorrect to swear words, slurs, or explicit terms. " +
        "This only affects what the plugin offers. Anything you type yourself is left exactly " +
        "as written and never corrected away. To un-block a specific word, add it to your " +
        "personal dictionary above; dictionary words are never filtered.",
    )
    .addToggle((t) =>
      t.setValue(settings.filterProfanity).onChange((v) => {
        settings.filterProfanity = v;
        commit();
      }),
    );

  // --- matching quality ---------------------------------------------------
  b.group("Accuracy boosters");
  b.note("Each makes corrections smarter for a different kind of mistake. All recommended on.");
  row()
    .setName("Keyboard-typo strength")
    .setDesc('How much nearby-key slips are trusted as typos ("teh" → "the"). Higher = more keyboard corrections; 0 = ignore keyboard geometry entirely.')
    .addSlider((s) =>
      s
        .setLimits(0, 3, 0.1)
        .setValue(settings.fuzzyStrength)
        .setDynamicTooltip()
        .onChange((v) => {
          settings.fuzzyStrength = v;
          commit();
        }),
    );
  row()
    .setName("Sound-alike strength")
    .setDesc('How much sound-alike spellings are trusted, independent of keyboard distance ("fone" → "phone", "definately" → "definitely"). Higher = more phonetic corrections; 0 = off.')
    .addSlider((s) =>
      s
        .setLimits(0, 3, 0.1)
        .setValue(settings.phoneticStrength)
        .setDynamicTooltip()
        .onChange((v) => {
          settings.phoneticStrength = v;
          commit();
        }),
    );
  toggle("Fix the wrong real word", 'Catches valid words used incorrectly in context ("form" → "from", "their" → "there").', "realWordCorrection");
  toggle("Fix missing spaces", 'Splits run-together words ("alot" → "a lot", "thebank" → "the bank").', "splitCorrection");
  toggle("Smarter context ranking", "Ranks words by how many contexts they appear in, not just raw frequency. Reins in over-eager rare words.", "useContinuation");
  toggle("Learn my keyboard slips", "Adapts the typo model to the specific key mistakes you personally make over time.", "adaptiveKeyboard");
  toggle("Learn from my choices", "Reorders suggestions based on which ones you actually pick.", "learnedRanking");
  row()
    .setName("Remember words used in this note")
    .setDesc("Boosts words you've already used in the current note (topic awareness). 0 disables.")
    .addSlider((s) =>
      s
        .setLimits(0, 0.5, 0.05)
        .setValue(settings.cacheGamma)
        .setDynamicTooltip()
        .onChange((v) => {
          settings.cacheGamma = v;
          commit();
        }),
    );

  row()
    .setName("Suggestions shown")
    .setDesc("How many suggestions appear in the popup at once.")
    .addSlider((s) =>
      s
        .setLimits(1, 8, 1)
        .setValue(settings.maxSuggestions)
        .setDynamicTooltip()
        .onChange((v) => {
          settings.maxSuggestions = v;
          commit();
        }),
    );

  row()
    .setName("Start suggesting after")
    .setDesc(
      "How many letters of a word you must type before completions appear. 1 = as soon as " +
        "you start a word; higher cuts noise on very short prefixes. Next-word prediction " +
        "after a space is unaffected.",
    )
    .addSlider((s) =>
      s
        .setLimits(1, 5, 1)
        .setValue(settings.minChars)
        .setDynamicTooltip()
        .onChange((v) => {
          settings.minChars = v;
          commit();
        }),
    );

  row()
    .setName("Accept suggestion with")
    .setDesc(
      "The key that inserts the highlighted suggestion. Only this key accepts; the " +
        "others keep their normal behaviour, so if you pick Tab then Enter still starts " +
        "a new line.",
    )
    .addDropdown((d) => {
      for (const k of ACCEPT_KEYS) d.addOption(k, k === "ArrowRight" ? "Right arrow" : k);
      d.setValue(settings.acceptKey).onChange((v) => {
        settings.acceptKey = v as AcceptKey;
        commit();
      });
    });

  row()
    .setName("Tab indents bullets only from the start")
    .setDesc(
      "Like Word: Tab only indents a list item when the cursor is right after the bullet. " +
        "Mid-item, Tab does nothing, so a Tab meant to accept a suggestion can't shove the " +
        "bullet right when the popup has already closed.",
    )
    .addToggle((t) =>
      t.setValue(settings.tabIndentAtBulletStartOnly).onChange((v) => {
        settings.tabIndentAtBulletStartOnly = v;
        commit();
      }),
    );

  row()
    .setName("Tidy double spaces")
    .setDesc(
      "When you complete a word that has a double space before it, collapse it to one.",
    )
    .addToggle((t) =>
      t.setValue(settings.collapseDoubleSpace).onChange((v) => {
        settings.collapseDoubleSpace = v;
        commit();
      }),
    );

  // --- markdown / performance --------------------------------------------
  b.group("Links & tags");
  // One dropdown drives the two linking features so you can pick exactly what you want:
  //  - "tooltips": the ambient link icons beside a block (suggestLinks)
  //  - "menu": our [[ picker replacing Obsidian's (replaceLinkMenu)
  const linkMode = (): "both" | "menu" | "tooltips" | "off" =>
    settings.suggestLinks && settings.replaceLinkMenu
      ? "both"
      : settings.suggestLinks
        ? "tooltips"
        : settings.replaceLinkMenu
          ? "menu"
          : "off";
  row()
    .setName("Linking assistance")
    .setDesc(
      "Choose which linking help you want. “Automatic tooltips” drops a small link icon beside a " +
        "block when another note's section is a close topical match – click it to insert a " +
        "[[link]] to that section. “Enhanced [[ menu” replaces Obsidian's [[ picker with one that " +
        "ranks notes by how relevant they are to what you're writing (other notes still appear, " +
        "greyed). Pick either, both, or turn linking off.",
    )
    .addDropdown((d) =>
      d
        .addOptions({
          both: "Menu + automatic tooltips",
          menu: "Enhanced [[ menu only",
          tooltips: "Automatic tooltips only",
          off: "Off",
        })
        .setValue(linkMode())
        .onChange((v) => {
          settings.suggestLinks = v === "both" || v === "tooltips";
          settings.replaceLinkMenu = v === "both" || v === "menu";
          commit();
          redraw?.(); // the sensitivity/length sliders below only matter for tooltips
        }),
    );
  if (settings.suggestLinks) {
    row()
      .setName("Related-link sensitivity")
      .setDesc(
        "How eager the link icons are. 1 shows an icon only for a very close topical match; " +
          "5 is looser. Thresholds are calibrated from your vault's own similarity distribution, " +
          "not fixed guesses. If you see too many icons, lower it.",
      )
      .addSlider((s) =>
        s
          .setLimits(1, 5, 1)
          .setValue(settings.relatedSensitivity)
          .setDynamicTooltip()
          .onChange((v) => {
            settings.relatedSensitivity = v;
            commit();
          }),
      );
    row()
      .setName("Minimum block length for a link")
      .setDesc(
        "A paragraph or list must have at least this many words before it can show a link icon. " +
          "Higher keeps short lines (and whatever you're mid-typing) icon-free.",
      )
      .addSlider((s) =>
        s
          .setLimits(5, 40, 1)
          .setValue(settings.minLinkWords)
          .setDynamicTooltip()
          .onChange((v) => {
            settings.minLinkWords = v;
            commit();
          }),
      );
  }
  toggle(
    "Suggest tags on #",
    "As you type #, suggest tags relevant to this note, biased toward niche, descriptive words " +
      "rather than generic ones. Existing vault tags come first; a few new tags from the note's " +
      "own distinctive terms follow.",
    "suggestTagsOnHash",
  );
  toggle(
    "Mirror #tags into frontmatter",
    "Keep the note's frontmatter tags: list matching the #tags you actually use in the body: a " +
      "tag is added when it first appears inline, and removed once its last inline use is gone, " +
      "with a notice each time. On by default. Note: frontmatter tags that never appear in the " +
      "body get removed, so turn this off if you tag notes only in frontmatter.",
    "syncFrontmatterTags",
  );
  b.note(
    "Links go inline where a concept is mentioned; tags are typed with #. For a full list at " +
      "once, run \"Suggest links in this note\" or \"Suggest tags for this note\" (Ctrl/Cmd-P).",
  );

  b.group("Where it works & performance");
  toggle(
    "Don't touch code, math, links & tags",
    "Never predict or autocorrect inside code blocks, LaTeX math, [[wikilinks]], URLs, #tags, or frontmatter, so it can't corrupt them.",
    "markdownAware",
  );
  row()
    .setName("Excluded folders & files")
    .setDesc(
      "Files where predictions and autocorrect never run, one per line. A folder name " +
        'excludes everything beneath it ("Templates"); glob patterns work too ' +
        '("Journal/*", "*.excalidraw.md", "**/private/**").',
    )
    .addTextArea((t) =>
      t
        .setValue(settings.excludedFolders.join("\n"))
        .setPlaceholder("Templates\nJournal/*")
        .onChange((v) => {
          settings.excludedFolders = parseExcludeList(v);
          commit();
        }),
    );
  toggle(
    "Off the main thread",
    "Run prediction, autocorrect and model building in a background worker so typing never stutters. Turn off only to debug.",
    "offMainThread",
  );
  row()
    .setName("WASM SIMD acceleration")
    .setDesc(
      "Run the neural model on a fast in-browser SIMD kernel (about 10x quicker than " +
        "plain JavaScript). Recommended on. It is NOT a silent fallback: if your device " +
        "can't run it, the line below says so.",
    )
    .addToggle((t) =>
      t.setValue(settings.wasmSimd).onChange(async (v) => {
        settings.wasmSimd = v;
        await onChange(); // persist + push the setting into the worker FIRST
        await acceleration?.reload(); // then re-init the model on/off the kernel now
        accelState.status = undefined; // the answer just changed, so ask the engine again
        redraw?.(); // re-render so the status line reflects what actually happened
      }),
    );
  // Live status - the whole point of the setting: state what IS happening, never leave
  // a scalar fallback silent. Fetched once and cached (see AccelerationState), because the
  // pane is re-described on every render and re-asking each time would loop.
  if (acceleration) {
    if (accelState.status === undefined && !accelState.pending) {
      accelState.pending = true;
      void Promise.all([acceleration.status(), acceleration.missingAssets()])
        .then(([st, missing]) => {
          accelState.status = !st.lstmLoaded
            ? "The neural model isn't installed (word_lstm.bin missing), so there is nothing to accelerate. Predictions use the word-frequency model only."
            : !settings.wasmSimd
              ? "Turned off. The neural model is running on the slower scalar-JS path by your choice."
              : st.accelerated
                ? "Active. The neural model is running on the WASM-SIMD kernel."
                : "Enabled, but this device has no WASM-SIMD support, so the neural model fell back to the slower scalar-JS path (older mobile webviews; needs iOS 16.4+ on iPhone/iPad).";
          accelState.missing = missing;
        })
        .catch(() => {
          accelState.status = "Could not read acceleration status.";
          accelState.missing = 0;
        })
        .finally(() => {
          accelState.pending = false;
          redraw?.();
        });
    }
    b.note(accelState.status ?? "Checking acceleration…");
  }
  // Offer the one-time model download here too: a user who declined the first-run
  // prompt (or whose download failed) needs a way back that isn't reinstalling.
  if (acceleration && accelState.missing) {
    const n = accelState.missing;
    row()
      .setName("Download language model")
      .setDesc(
        `${n} model file${n === 1 ? " is" : "s are"} missing, so predictions are running ` +
          `on your vault alone. The model is downloaded once from the plugin's GitHub ` +
          `release; nothing is ever uploaded.`,
      )
      .addButton((btn) =>
        btn
          .setButtonText("Download")
          .setCta()
          .onClick(() => {
            void acceleration.installAssets().then((ok) => {
              if (!ok) return;
              accelState.status = undefined; // a model just landed: re-read both answers
              redraw?.();
            });
          }),
      );
  }
  row()
    .setName("Suggestion style")
    .setDesc(
      "How completions appear. A popup list lets you pick from a few options; inline ghost text " +
        "shows just the top one as dimmed text ahead of the cursor. Either way, Tab accepts.",
    )
    .addDropdown((d) =>
      d
        .addOptions({ popup: "Popup list", ghost: "Inline ghost text" })
        .setValue(settings.ghostText ? "ghost" : "popup")
        .onChange((v) => {
          settings.ghostText = v === "ghost";
          commit();
        }),
    );
  row()
    .setName("Keyboard layout")
    .setDesc("Used by the typo model's key-distance prior.")
    .addDropdown((d) =>
      d
        .addOptions({ qwerty: "QWERTY", qwertz: "QWERTZ", azerty: "AZERTY", dvorak: "Dvorak" })
        .setValue(settings.keyboardLayout)
        .onChange((v) => {
          settings.keyboardLayout = v as typeof settings.keyboardLayout;
          commit();
        }),
    );

  // --- personalization management ----------------------------------------
  if (!personalization) return b.groups;
  b.group("Personalization");
  const stats = personalization.getStats();
  // Headline gamification stat: characters saved, streak, and estimated time saved.
  b.custom("Keystrokes saved", (el) => {
    const saved = el.createEl("p", { cls: "setting-item-description" });
    saved.createEl("strong", { text: `⌨️ ${stats.charsSaved.toLocaleString()}` });
    const hrs =
      stats.minutesSaved >= 60
        ? `${(stats.minutesSaved / 60).toFixed(1)} hrs`
        : `${Math.round(stats.minutesSaved)} min`;
    saved.appendText(` keystrokes saved · ≈ ${hrs} of typing`);
    if (stats.streak > 1)
      saved.appendText(` · 🔥 ${stats.streak}-day streak (best ${stats.bestStreak})`);
  });

  row()
    .setName("Writing stats")
    .setDesc("Your streak, time saved, milestones, and what the plugin has learned. Stored with your vault, so the numbers are the same on every device.")
    .addButton((b) => b.setButtonText("See your stats").setCta().onClick(() => personalization.onOpenStats()))
    .addButton((b) => b.setButtonText("Reset statistics").setWarning().onClick(() => personalization.onResetStats()));

  // Support / Buy me a coffee - placed next to the feel-good stat.
  b.custom("Support, buy me a coffee", (el) => {
    const support = el.createDiv({ cls: "smart-autocorrect-support" });
    const supportText = support.createEl("p", { cls: "setting-item-description" });
    supportText.appendText("Enjoying the plugin? You can ");
    const link = supportText.createEl("a", { text: "buy me a coffee ☕", href: BMC_URL });
    link.setAttribute("target", "_blank");
    link.setAttribute("rel", "noopener");
    supportText.appendText(", or scan the code.");
    const qr = support.createEl("img", { cls: "smart-autocorrect-qr" });
    qr.src = BMC_QR_DATA_URI;
    qr.alt = "Buy Me a Coffee QR code";
    qr.width = 130;
    qr.height = 130;
  });
  b.note(
    settings.personalizationEnabled
      ? `So far: ${stats.accepts} suggestions accepted, ${stats.corrections} typos fixed, ${stats.reverts} undone, and ${stats.learnListSize} of your words on the don't-touch list. All of this lives in personalization.json in the plugin folder, travels with your vault, and stays out of your notes.`
      : `Personalisation is off, so nothing new is being learned or used right now. Your ${stats.accepts} past accepts, ${stats.corrections} fixes and ${stats.reverts} undos are still saved and will kick back in the moment you turn it on.`,
  );

  row()
    .setName("Personalise to me")
    .setDesc(
      "Learn from your corrections and accepts, and use what was learned. Turn off to " +
        "keep suggestions identical for everyone: nothing new is recorded and existing " +
        "learning is ignored (but kept, so you can turn this back on).",
    )
    .addToggle((t) =>
      t.setValue(settings.personalizationEnabled).onChange((v) => {
        settings.personalizationEnabled = v;
        commit();
        // The blurb above and the controls below both depend on this, so redraw.
        redraw?.();
      }),
    );

  row()
    .setName("Share file (vault path)")
    .setDesc(
      "Only used by the Export/Import buttons below. This is not where personalization " +
        "lives; the live data is always personalization.json in the plugin folder.",
    )
    .addText((t) =>
      t
        .setValue(settings.personalizationSharePath)
        .onChange((v) => {
          settings.personalizationSharePath = v.trim() || "predictive-personalization.json";
          commit();
        }),
    );

  row()
    .setName("Export / import personalization")
    .addButton((b) =>
      b.setButtonText("Export").onClick(() => void personalization.onExport(settings.personalizationSharePath)),
    )
    .addButton((b) =>
      b.setButtonText("Import (replace)").onClick(() => void personalization.onImport(settings.personalizationSharePath, false)),
    )
    .addButton((b) =>
      b.setButtonText("Import (merge)").onClick(() => void personalization.onImport(settings.personalizationSharePath, true)),
    );

  row()
    .setName("Reset personalization")
    .setDesc("Clear all learned adaptation (keyboard model, ranking, protected words).")
    .addButton((b) =>
      b
        .setButtonText("Reset")
        .setWarning()
        .onClick(() => void personalization.onReset()),
    );
  return b.groups;
}

/**
 * Pre-1.13 entry point, kept so the pane still renders on the Obsidian versions this plugin
 * supports (manifest minAppVersion is below 1.13). Same groups, drawn imperatively.
 */
export function renderPredictiveSettings(
  containerEl: HTMLElement,
  settings: PredictiveSettings,
  onChange: () => void | Promise<void>,
  personalization?: PersonalizationHandlers,
  redraw?: () => void,
  acceleration?: AccelerationHandlers,
  accelState: AccelerationState = {},
): void {
  renderPaneGroups(
    containerEl,
    buildPredictiveSettingGroups(settings, onChange, personalization, redraw, acceleration, accelState),
  );
}
