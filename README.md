# Smart Autocorrect

On-device predictive text and autocorrect for [Obsidian](https://obsidian.md), powered by a small **local neural language model**. It suggests your next word, fixes typos as you type, completes whole phrases, and gets the capitalisation right, all **fully offline**. Nothing you write ever leaves your vault: no servers, no accounts, no telemetry.

---

## What it does

- **Next-word prediction.** A popup offers the words you are most likely to type next, read from the surrounding sentence (the neural model reads the whole note up to your cursor, not a fixed window; a word-frequency model backs it up). Accept with **Tab**.
- **Autocorrect.** Misspelt words are fixed on space or punctuation, phone-keyboard style (`teh` to `the`, `definately` to `definitely`, `fone` to `phone`). It weighs how the letters sit on your keyboard *and* how sound-alike the word is, against what the sentence expects.
- **Multi-word phrases.** When a whole phrase is likely, it is offered as one completion and ranked by how many keystrokes it actually saves.
- **Smart capitalisation.** Real sentence starts and proper nouns are capitalised (`london` to `London`), genuine lowercase stays lowercase (`I'll polish it`), and abbreviations like `e.g.`, `w.r.t.`, `U.S.` are handled correctly.
- **Link and tag suggestions.** Text matching an existing note title or alias is subtly underlined; hover to preview the note, click to turn it into a `[[link]]`. Or run the "Suggest links in this note" and "Suggest tags for this note" commands to pick from a list. Retrieval-based, so it only ever proposes notes that already exist and never inserts anything on its own.
- **Learns your vault.** Suggestions bias toward the words and phrasing you actually use in your own notes. This learning is stored locally and can be turned off.
- **Profanity and NSFW filter** (on by default). Swear words, slurs, and explicit terms are never *suggested* or used as an autocorrection. It only filters what the plugin offers; anything **you** type is left exactly as written.
- **Writing stats.** A status-bar tally shows how many keystrokes you have saved; click it (or run "Show writing stats") for a dashboard with your streak, time saved, and progress to the next milestone.

### Undo is a first-class control

Made an unwanted correction? Press **Ctrl/Cmd-Z** (or Backspace right after). That restores what you typed *and* teaches the plugin to leave that word alone in future. Undo is how you train it, not just how you fix a mistake.

---

## Getting started

1. Install and enable the plugin.
2. Accept the one-time model download (86 MB) when prompted. Obsidian's installer can't carry files that large, so the model comes from this repo's release.
3. Start typing in any note. As you type a word, a popup shows completions; press **Tab** to accept the highlighted one, or keep typing to ignore it.
4. After a space or punctuation, an obvious typo is corrected automatically. If you did not want that, **Ctrl/Cmd-Z** undoes it and remembers your choice.

There is nothing to configure to get started. Everything below is optional tuning.

---

## Languages

The neural model is trained on **English**, so prediction, phrase completion, and capitalisation are English-language features. The typo model supports several **physical keyboard layouts** for its key-distance model (**QWERTY, QWERTZ, AZERTY, and Dvorak**), so keyboard-based corrections match the keys you actually press. Your **personal dictionary** and vault-learning work in any language you write in.

---

## Privacy

100% local. The model runs inside Obsidian on your device (accelerated with WebAssembly SIMD where available). There is no telemetry and no cloud. The only network request is the one-time model download when you first enable the plugin, which you can decline. What you learn stays in `personalization.json` inside the plugin folder, which travels with your vault (for example via Obsidian Sync) and is hidden from your notes.

---

## Settings

**Predictions and autocorrect**
- **Predictive text.** Master switch for the next-word popup.
- **Autocorrect typos when you press space.** The phone-style fix-on-space behaviour.
- **Autocorrect strength.** How surprising a typed word must be, versus the best alternative, before it is replaced. Lower corrects eagerly; higher only fixes very unlikely words.
- **Auto-capitalise sentences and proper nouns.** Sentence starts, `THe` to `The`, learned proper-noun casing.
- **Learn my writing style from this vault** / **Vault influence.** Bias toward your own notes, and how strongly.
- **Neural vs. word-frequency blend.** How much the neural model drives suggestions vs the frequency model.
- **Trust typing vs. context.** For an ambiguous typo, trust the exact letters vs what the sentence expects.
- **Personal dictionary.** Words that are always correct as written, never autocorrected or re-cased (case-sensitive, for example `NixOS`, `kubeCTL`). Adding a word here also un-blocks it from the profanity filter.
- **Filter profanity and NSFW words.** Never suggest or autocorrect to offensive words (on by default). Only affects what the plugin *offers*, never your own typing.

**Accuracy boosters.** Keyboard-typo strength, sound-alike strength, real-word correction (`form` to `from`), missing-space splits (`alot` to `a lot`), context ranking, and adaptive learning from your keyboard slips and choices.

**Links and tags**
- **Suggest wiki-links as you write.** Ambient underline of text that matches an existing note; hover to preview, click to link. Or use the "Suggest links in this note" and "Suggest tags for this note" commands for a list.

**Triggers and display**
- **Suggestions shown.** How many completions appear at once.
- **Start suggesting after.** How many letters of a word before completions appear (raise it to cut noise on short prefixes).
- **Accept suggestion with.** Tab (default), Enter, Right arrow, or End. Only the chosen key accepts; the others keep their normal behaviour.
- **Inline ghost text.** An alternative to the popup: show the top prediction as dimmed text ahead of the cursor, Tab to accept.
- **Tab indents bullets only from the start.** Word-style, so a missed Tab-accept can't shove a bullet right.

**Where it works**
- **Don't touch code, math, links and tags.** Never run inside code blocks, LaTeX, `[[wikilinks]]`, URLs, `#tags`, or frontmatter.
- **Excluded folders and files.** Folders or glob patterns where the plugin never runs (`Templates`, `Journal/*`, `*.excalidraw.md`).
- **Off the main thread** / **WASM SIMD acceleration.** Run the engine in a background worker on a fast kernel so typing never stutters.

**Personalization.** Shows a running keystrokes-saved count and streak, and lets you export, import, or reset everything the plugin has learned.

---

## Acknowledgements

This plugin began as a fork of [Various Complements](https://github.com/tadashi-aikawa/obsidian-various-complements-plugin) by Tadashi Aikawa. Its engine has since been rewritten from scratch around a local neural language model; the build setup is still derived from it. MIT licensed.

