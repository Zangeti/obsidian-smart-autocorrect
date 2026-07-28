# Smart Autocorrect

[![release](https://img.shields.io/github/v/release/Zangeti/obsidian-smart-autocorrect)](https://github.com/Zangeti/obsidian-smart-autocorrect/releases/latest)
![downloads](https://img.shields.io/github/downloads/Zangeti/obsidian-smart-autocorrect/total)

**Phone-style autocorrect and next-word prediction**, directly in your vault. It runs completely offline, so your notes stay private and never leave your device. 🔒

![Smart Autocorrect predicting the next word as you type](docs/demo.gif)

## Core features

| Feature | What it does |
| --- | --- |
| **Real-time autocorrect** | Resolves typos automatically as you write. |
| **Phrase completion** | Predicts your next word, and sometimes whole phrases. Press **Tab** to accept. |
| **Contextual suggestions** | Aware of your notes and the sentence you're typing. |
| **100% local** | Runs entirely on-device for absolute data privacy. |

## Also included

| Feature | What it does |
| --- | --- |
| **Suggest alternatives** ✨ | Right-click any word for more eloquent wording (good to substantial, help to facilitate, big to immense). On-device too. |
| **Smart capitalisation** | Sentence starts, names and places, and abbreviations like `e.g.` and `U.S.` |
| **Learns your writing** | Words from your own notes come up first. Wrong correction? Just undo (**Ctrl/Cmd-Z**): it restores your text, adds the word to your dictionary, and learns from it. |
| **Note links** (experimental) | Text that matches one of your notes is underlined. Hover to preview, click to link. |
| **Writing stats** | Keystrokes saved, typing time saved, and your daily streak. |

<img src="docs/stats.png" alt="The writing stats dashboard, showing keystrokes saved, typing time saved and current streak" width="440">

## Getting started

1. Install and enable the plugin.
2. Accept the one-time model download (86 MB) when prompted.
3. Start typing. There's nothing to configure; everything in settings is optional.

## Privacy

100% local. There is no telemetry and no cloud. The only network request is the one-time model download, which you can decline. What the plugin learns stays in `personalization.json` inside the plugin folder, so it travels with your vault and stays out of your notes.

English only for prediction and capitalisation. The personal dictionary works in any language.

## Support

If you find this useful, you can [buy me a coffee](https://buymeacoffee.com/zangeti).

<a href="https://buymeacoffee.com/zangeti"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy me a coffee" height="50"></a>

## Credits

Started as a fork of [Various Complements](https://github.com/tadashi-aikawa/obsidian-various-complements-plugin) by Tadashi Aikawa. The engine has since been rewritten around a local neural language model. MIT licensed.
