/**
 * Mobile-style autocorrect + auto-capitalisation, driven by keystrokes.
 *
 * On a word boundary (space/enter/punctuation) we look at the token just
 * completed, ask the engine whether to correct it, and optionally capitalise a
 * new sentence's first word. A one-shot revert (Ctrl/Cmd-Z immediately after a
 * correction) restores the original - exactly like a phone keyboard.
 *
 * Obsidian-side glue; decision logic lives in ./engine (unit-tested).
 */
import { MarkdownView, Plugin } from "obsidian";
import type { Editor, EditorPosition } from "obsidian";
import { isolateHistory, undo } from "@codemirror/commands";
import { EditorView } from "@codemirror/view";
import type { Extension, Transaction } from "@codemirror/state";
import {
  RevertBuffer,
  applyAutoCapitalization,
  capitalizeFirst,
  shouldCapitalizeNext,
  defaultSentenceCaseConfig,
  matchCase,
  classifyMarkdownContext,
  pathExcluded,
  isDoubledWord,
  type SentenceCaseConfig,
} from "./engine/index";
import { contextWords } from "./context";
import type { PredictiveEngineController } from "./PredictiveEngineController";
import type { PredictiveSettings } from "./PredictiveSettings";

const BOUNDARY_KEYS = new Set([" ", "Enter", ".", ",", ";", ":", "!", "?"]);

/** Keys that are modifiers only. Pressing Ctrl-Z fires TWO keydowns - "Control"
 *  then "z" - so these must NOT count as "the user typed something else" or the
 *  pending-correction flag is cleared before the "z" ever arrives. */
const MODIFIER_KEYS = new Set(["Control", "Meta", "Shift", "Alt", "AltGraph", "CapsLock"]);

export class AutocorrectController {
  private plugin: Plugin;
  private engine: PredictiveEngineController;
  private settings: PredictiveSettings;
  private revert = new RevertBuffer();
  private caseCfg: SentenceCaseConfig;
  private justCorrected = false;
  /** Personal dictionary, EXACT surface forms. Rebuilt on settings change. */
  private dictionary = new Set<string>();

  // --- deferred, revert-driven dictionary/abbreviation learning -----------------------------
  // A bounded history of recent corrections so a revert is recognised even when other edits
  // happened in between (#12). A revert does NOT add immediately; it arms a pending decision
  // that is resolved on the NEXT action - only if the reverted word is still there (#9/#14),
  // which is what stops a second Ctrl-Z (undo the typing too) from wrongly learning the word.
  private recent: { from: string; to: string; abbrev?: string }[] = [];
  private pending: { word: string; pos: number; abbrev?: string; createdAt: number } | null = null;
  private updateSeq = 0;
  private pendingTimer: number | null = null;

  /** Called with the original word when a correction is undone AND the word is kept, so the
   *  plugin can add it to the personal dictionary (gated on the setting and on the word not
   *  already being known). */
  onRevert?: (word: string) => void;

  /** Called when a sentence-initial capitalisation is reverted, with the abbreviation before
   *  the full stop that caused it (e.g. reverting "etc. Then"→"then" learns "etc"), so the
   *  plugin can stop capitalising after it (#10). */
  onLearnAbbreviation?: (abbrev: string) => void;

  constructor(
    plugin: Plugin,
    engine: PredictiveEngineController,
    settings: PredictiveSettings,
  ) {
    this.plugin = plugin;
    this.engine = engine;
    this.settings = settings;
    this.caseCfg = defaultSentenceCaseConfig(settings.extraAbbreviations);
    this.dictionary = new Set(settings.userDictionary);
  }

  updateSettings(settings: PredictiveSettings): void {
    this.settings = settings;
    this.caseCfg = defaultSentenceCaseConfig(settings.extraAbbreviations);
    this.dictionary = new Set(settings.userDictionary);
  }

  register(): void {
    // CAPTURE phase, and it matters: CodeMirror's own undo keymap is bound on the
    // editor's contentDOM, which is a DESCENDANT of document. A bubble-phase
    // listener here would run AFTER CodeMirror had already undone something, and
    // preventDefault() at that point is far too late to stop it.
    this.plugin.registerDomEvent(
      document,
      "keydown",
      (evt: KeyboardEvent) => {
        this.onKeyDown(evt);
      },
      { capture: true },
    );
    // Watch document changes so a reverted correction is learned robustly (see `recent`/
    // `pending`). This fires for BOTH our own Ctrl-Z interception and Obsidian's native undo,
    // so a revert after intervening edits (#12) is still caught.
    this.plugin.registerEditorExtension(this.correctionExtension());
  }

  private correctionExtension(): Extension {
    return EditorView.updateListener.of((u) => {
      if (!u.docChanged) return;
      this.updateSeq++;
      // Keep the pending word's position current, then resolve it once a LATER action lands.
      if (this.pending) {
        this.pending.pos = u.changes.mapPos(this.pending.pos, 1);
        if (this.pending.createdAt < this.updateSeq) this.resolvePending(u.view);
      }
      for (const tr of u.transactions) {
        if (tr.docChanged && tr.isUserEvent("undo")) this.detectRevert(tr);
      }
    });
  }

  /** Record a correction so its later revert can be recognised and (deferred) learned. */
  private trackCorrection(from: string, to: string, abbrev?: string): void {
    this.revert.record(from, to); // legacy-editor immediate-revert path
    this.recent.push({ from, to, abbrev });
    if (this.recent.length > 24) this.recent.shift();
  }

  /** An undo transaction restored a tracked correction (its target reverted to the original):
   *  arm a pending learn, to be confirmed on the next action. */
  private detectRevert(tr: Transaction): void {
    tr.changes.iterChanges((fromA, toA, _fromB, toB, inserted) => {
      const ins = inserted.toString();
      const old = tr.startState.doc.sliceString(fromA, toA);
      const idx = this.recent.findIndex((c) => c.from === ins && c.to === old);
      if (idx < 0) return;
      const corr = this.recent.splice(idx, 1)[0];
      this.engine.recordRevert(corr.from); // the original was actually intended
      this.pending = { word: corr.from, pos: toB - ins.length, abbrev: corr.abbrev, createdAt: this.updateSeq };
      if (this.pendingTimer) window.clearTimeout(this.pendingTimer);
      // Fallback for "revert then stop typing": confirm after a short idle too.
      this.pendingTimer = window.setTimeout(() => {
        const ed = this.activeEditor();
        const view = ed ? this.viewOf(ed) : null;
        if (view) this.resolvePending(view);
      }, 1500);
    });
  }

  /** Confirm or drop a pending learn: only learn if the reverted word is still present (the
   *  user kept it) - if a second Ctrl-Z removed it, drop silently. */
  private resolvePending(view: EditorView): void {
    const p = this.pending;
    if (!p) return;
    this.pending = null;
    if (this.pendingTimer) {
      window.clearTimeout(this.pendingTimer);
      this.pendingTimer = null;
    }
    const doc = view.state.doc;
    if (p.pos < 0 || p.pos + p.word.length > doc.length) return;
    if (doc.sliceString(p.pos, p.pos + p.word.length) !== p.word) return; // word gone → don't learn
    if (p.abbrev) this.onLearnAbbreviation?.(p.abbrev);
    else this.onRevert?.(p.word);
  }

  private activeEditor(): Editor | null {
    const view = this.plugin.app.workspace.getActiveViewOfType(MarkdownView);
    return view?.editor ?? null;
  }

  /** The CM6 view behind an Obsidian Editor. Internal API, so it is duck-typed
   *  and every caller tolerates null. */
  private viewOf(editor: Editor): EditorView | null {
    const cm = (editor as unknown as { cm?: EditorView }).cm;
    return cm && typeof cm.dispatch === "function" && cm.state ? cm : null;
  }

  private onKeyDown(evt: KeyboardEvent): void {
    // IME safety: never act during a composition (CJK, accents, etc.).
    if (evt.isComposing || evt.key === "Process") {
      this.justCorrected = false;
      return;
    }
    // A bare modifier keydown is not "the user moved on" - Ctrl-Z arrives as
    // "Control" THEN "z", so clearing the flag here would disarm the revert
    // before the "z" ever landed. This is what made Ctrl-Z fall through to
    // CodeMirror, which undid the correction and the typing as one group and so
    // appeared to just delete the word.
    if (MODIFIER_KEYS.has(evt.key)) return;

    // One-shot revert of the last autocorrect.
    if (this.justCorrected && (evt.ctrlKey || evt.metaKey) && evt.key.toLowerCase() === "z") {
      const editor = this.activeEditor();
      if (editor && this.revertLast(editor)) {
        this.justCorrected = false;
        evt.preventDefault();
        evt.stopPropagation();
        return;
      }
    }
    this.justCorrected = false;

    if (!BOUNDARY_KEYS.has(evt.key)) return;
    if (!this.settings.autocorrectOnSpace && !this.settings.autoCapitalize) return;

    const editor = this.activeEditor();
    if (!editor) return;
    // Enter finishes the current word just as space does, so it must autocorrect it too.
    // But Enter also inserts a newline that moves the caret to a fresh (empty) line before
    // the deferred handler runs - so we snapshot the caret NOW (still at the end of the word,
    // this is capture phase, before the editor processes the key) and correct the word on the
    // line we are leaving. The split keeps everything before the caret on the old line, so the
    // captured position still points at the word.
    const at = evt.key === "Enter" ? editor.getCursor() : null;
    // Defer to after the boundary character is inserted by the editor.
    window.setTimeout(() => void this.handleBoundary(editor, at), 0);
  }

  /**
   * The personal-dictionary entry `token` is an instance of, or null.
   *
   * The dictionary is case-SENSITIVE by design - it exists to pin spellings the model cannot
   * represent ("NixOS", "kubeCTL"), and a case-insensitive match would defeat that. But the
   * casing WE would legitimately apply must not break the match, or a pinned word stops being
   * recognised as soon as it starts a sentence. So an entry also matches through exactly the
   * two transformations this controller itself performs: a leading capital, and all caps.
   */
  private pinnedEntry(token: string): string | null {
    if (this.dictionary.has(token)) return token;
    for (const entry of this.dictionary) {
      if (capitalizeFirst(entry) === token || entry.toUpperCase() === token) return entry;
    }
    return null;
  }

  /**
   * Write a pinned word with the casing its POSITION calls for: caps if the user typed it in
   * caps, a leading capital at a sentence start, otherwise exactly as pinned. Nothing here
   * touches spelling - that is what being in the dictionary buys.
   */
  private applyPinnedCasing(
    editor: Editor,
    cursor: EditorPosition,
    tokenStartCh: number,
    token: string,
    pinned: string,
    precedingText: string,
  ): void {
    const typedUpper =
      token.length > 1 && token === token.toUpperCase() && token !== token.toLowerCase();
    let out = pinned;
    if (typedUpper) out = out.toUpperCase();
    else if (this.settings.autoCapitalize && shouldCapitalizeNext(precedingText, this.caseCfg))
      out = capitalizeFirst(out);
    if (out === token) return;
    if (editor.getLine(cursor.line).slice(tokenStartCh, tokenStartCh + token.length) !== token)
      return;
    this.applyCorrection(
      editor,
      { line: cursor.line, ch: tokenStartCh },
      { line: cursor.line, ch: tokenStartCh + token.length },
      out,
    );
    // Tracked like any other correction, so Ctrl-Z reverts it and the revert is understood
    // (a pinned word re-cased at a sentence start is still a change the user may reject).
    this.trackCorrection(token, out);
    this.justCorrected = true;
  }

  private async handleBoundary(editor: Editor, at: EditorPosition | null = null): Promise<void> {
    if (!this.settings.pluginEnabled) return; // master switch off: no autocorrect / capitalisation
    const cursor = at ?? editor.getCursor();
    // Never autocorrect in an excluded folder/file.
    const file = this.plugin.app.workspace.getActiveFile();
    if (file && pathExcluded(file.path, this.settings.excludedFolders)) return;
    // Markdown/LaTeX awareness: never correct inside code, math, links, etc.
    if (this.settings.markdownAware) {
      const textBefore = editor.getRange({ line: 0, ch: 0 }, cursor);
      if (classifyMarkdownContext(textBefore).suppressAutocorrect) return;
    }
    // Text of the current line up to (but not including) the char just typed.
    const line = editor.getLine(cursor.line);
    const before = line.slice(0, cursor.ch);
    // The boundary character is the last char; the token is what precedes it.
    const uptoToken = before.replace(/[\s.,;:!?]+$/, "");
    const tokenMatch = uptoToken.match(/([A-Za-z][A-Za-z'-]*)$/);
    if (!tokenMatch) return;

    const token = tokenMatch[1];
    const tokenStartCh = uptoToken.length - token.length;
    const precedingText = uptoToken.slice(0, tokenStartCh);

    // Personal dictionary: the SPELLING is correct as written, so no spelling correction and no
    // proper-noun re-casing. But "pinned spelling" was never meant to mean "exempt from the
    // ordinary rules of written English": a dictionary word still starts a sentence with a
    // capital, and still comes out in caps if you typed it in caps. The old blanket `return`
    // skipped both, so a pinned word at a sentence start silently stayed lowercase.
    const pinned = this.pinnedEntry(token);
    if (pinned !== null) {
      this.applyPinnedCasing(editor, cursor, tokenStartCh, token, pinned, precedingText);
      return;
    }
    // A single letter that is part of a dotted initialism / abbreviation ("e.g.", "w.r.t.",
    // "U.S.", "a.m.") - it is either preceded by a "." (an earlier initial, like the "g" in
    // "e.g" or the "t" in "w.r.t") or immediately followed by the "." just typed. The boundary
    // handler sees each letter as its own token, so capitalising them is what turned "w.r.t."
    // into "W.R.T." and, on the trailing SPACE, "e.g." into "e.G.". Leave such a letter EXACTLY
    // as typed, whatever the boundary character (space or the next dot).
    if (/^[A-Za-z]$/.test(token) && (precedingText.endsWith(".") || before.slice(-1) === ".")) return;

    // Accidental doubled word ("the the" -> "the"): if the word just typed duplicates the
    // one before it and is never validly doubled, delete this copy plus the space before it,
    // then stop (nothing to correct). Undoable with Ctrl/Cmd-Z like any edit.
    if (this.settings.removeDoubledWords) {
      const prevMatch = precedingText.match(/([A-Za-z][A-Za-z'-]*)\s*$/);
      if (prevMatch && isDoubledWord(prevMatch[1], token)) {
        const prevWordEnd = tokenStartCh - (precedingText.length - precedingText.trimEnd().length);
        editor.replaceRange(
          "",
          { line: cursor.line, ch: prevWordEnd },
          { line: cursor.line, ch: uptoToken.length },
        );
        return;
      }
    }

    const context = contextWords(precedingText, this.settings.extraAbbreviations);

    // Spurious-space merge (#15/#18): "th e"→"the", "haven t"→"haven't". Only when this token is
    // separated from the previous one by exactly one space and the engine judges the join a real
    // word / contraction far likelier than the two apart. Checked before single-token correction
    // because a stray space is the more fundamental error.
    if (this.settings.autocorrectOnSpace) {
      const prevMatch = precedingText.match(/([A-Za-z][A-Za-z'-]*) $/);
      if (prevMatch) {
        const prevTok = prevMatch[1];
        const prevStart = tokenStartCh - (prevTok.length + 1);
        const mergeCtx = contextWords(uptoToken.slice(0, prevStart), this.settings.extraAbbreviations);
        let merged: string | null = null;
        try {
          merged = await this.engine.mergeDecision(prevTok, token, mergeCtx);
        } catch {
          merged = null;
        }
        if (
          merged &&
          editor.getLine(cursor.line).slice(prevStart, tokenStartCh + token.length) ===
            `${prevTok} ${token}`
        ) {
          const applied = matchCase(prevTok, merged);
          this.applyCorrection(
            editor,
            { line: cursor.line, ch: prevStart },
            { line: cursor.line, ch: tokenStartCh + token.length },
            applied,
          );
          this.trackCorrection(`${prevTok} ${token}`, applied);
          this.justCorrected = true;
          return;
        }
      }
    }

    let replacement = token;

    if (this.settings.autocorrectOnSpace) {
      try {
        const decision = await this.engine.decide(token, context);
        if (decision.correct) replacement = decision.to;
      } catch {
        return; // engine unavailable: leave what the user typed alone
      }
    }

    // ALL-CAPS the user typed deliberately is never "corrected" back down.
    //
    // Note this is per-word and needs NO run, unlike the suggestion-casing rule: a
    // word you typed in caps is direct evidence about THAT word, so re-casing it is
    // always wrong ("NASA" -> "Nasa" is the obvious failure). Requiring a run would
    // only protect the second word onward and mangle the first one. Suggestions are a
    // prediction about a word you have NOT typed, so there a lone acronym must not
    // flip the whole editor into caps - hence detectUpperMode's minimum run.
    //
    // Spelling correction still applies: matchCase() maps "TEH" -> "THE", so a genuine
    // typo is fixed IN CAPS rather than silently left wrong.
    // The word itself is the evidence: if you typed it in caps, keep it in caps. This
    // covers CAPS LOCK as well as Shift, needs no key listener, and is direct evidence
    // about THIS word - which is what stops "NASA" being "corrected" to "Nasa".
    const typedUpper =
      token.length > 1 && token === token.toUpperCase() && token !== token.toLowerCase();

    if (typedUpper) replacement = replacement.toUpperCase();

    if (this.settings.autoCapitalize && !typedUpper) {
      // Proper-noun casing now comes from the cased LSTM, in context, rather than
      // a static frequency map - so "paris" -> "Paris" while "i will polish it"
      // is left alone. Sentence-initial capitalisation stays punctuation-driven.
      let canonical: string | null = null;
      try {
        canonical = await this.engine.caseFor(replacement, context);
      } catch {
        /* no casing advice: fall through to sentence-case only */
      }
      replacement = applyAutoCapitalization(replacement, precedingText, {
        ...this.caseCfg,
        canonical,
      });
    }

    if (replacement !== token) {
      // decide() is async (the engine lives in a worker), so the buffer may have
      // moved on while we waited. Only rewrite if the exact word we judged is
      // still sitting where we found it - never clobber newer typing.
      if (editor.getLine(cursor.line).slice(tokenStartCh, tokenStartCh + token.length) !== token)
        return;
      const applied = matchCase(token, replacement);
      const from: EditorPosition = { line: cursor.line, ch: tokenStartCh };
      const to: EditorPosition = { line: cursor.line, ch: tokenStartCh + token.length };
      this.applyCorrection(editor, from, to, applied);
      // A pure sentence-initial capitalisation triggered by a preceding "word." is evidence
      // that "word" might be an abbreviation - remember it so reverting the cap can learn it (#10).
      const capOnly = replacement.toLowerCase() === token.toLowerCase();
      // The token before the full stop, INCLUDING dotted initialisms: "z.b." must capture
      // "z.b", not fail for having a single letter before its last dot. Those are exactly the
      // abbreviations that trip the capitaliser, so they are the ones that must be learnable.
      const abbrevMatch = capOnly ? precedingText.match(/([A-Za-z][A-Za-z.]*)\.\s+$/) : null;
      this.trackCorrection(token, applied, abbrevMatch ? abbrevMatch[1].toLowerCase() : undefined);
      this.justCorrected = true;
      // Teach the confusion model only for genuine spelling corrections
      // (not pure capitalisation), so it learns real key slips.
      if (!capOnly) this.engine.recordCorrection(token, replacement);
    }
  }

  /**
   * Write the correction as its OWN undo step.
   *
   * The correction lands milliseconds after the space that triggered it, so
   * CodeMirror's history happily merges the two into one group - and undoing that
   * group removes the correction AND the word the user typed, which reads as
   * "Ctrl-Z ate my word". isolateHistory("before") forces a group boundary, so an
   * undo rewinds exactly the correction and leaves the typing (and the cursor,
   * which CodeMirror maps through the change) untouched.
   */
  private applyCorrection(
    editor: Editor,
    from: EditorPosition,
    to: EditorPosition,
    text: string,
  ): void {
    const view = this.viewOf(editor);
    if (!view) {
      editor.replaceRange(text, from, to); // legacy editor: no history control
      return;
    }
    view.dispatch({
      changes: {
        from: editor.posToOffset(from),
        to: editor.posToOffset(to),
        insert: text,
      },
      annotations: isolateHistory.of("before"),
    });
  }

  /**
   * Undo the last autocorrect, leaving the cursor exactly where it is.
   *
   * This POPS the correction off CodeMirror's history rather than typing the old
   * word back over it. Writing a fresh edit would leave the correction sitting in
   * the undo stack, so a second Ctrl-Z would helpfully re-apply the very thing the
   * user just rejected. Because applyCorrection isolated the correction into its
   * own group, undo() reverts precisely it - the cursor stays after the space, and
   * the next Ctrl-Z goes on to undo the typing as the user expects.
   */
  private revertLast(editor: Editor): boolean {
    const view = this.viewOf(editor);
    const last = this.revert.revert();
    if (!last) return false;

    if (view) {
      // undo() emits an "undo" transaction that the correction-watcher sees, so recordRevert
      // and the DEFERRED dictionary-add are handled there (uniformly with native Ctrl-Z).
      if (!undo(view)) return false;
      return true;
    }

    // Legacy-editor fallback: restore the word in place by hand (no updateListener here, so
    // learn immediately - legacy editors don't get the deferred/undo-driven path).
    const cursor = editor.getCursor();
    const line = editor.getLine(cursor.line);
    const before = line.slice(0, cursor.ch);
    const idx = before.lastIndexOf(last.to);
    if (idx < 0) return false;
    const from: EditorPosition = { line: cursor.line, ch: idx };
    const to: EditorPosition = { line: cursor.line, ch: idx + last.to.length };
    editor.replaceRange(last.from, from, to);
    editor.setCursor({ line: cursor.line, ch: cursor.ch + (last.from.length - last.to.length) });
    this.engine.recordRevert(last.from);
    this.onRevert?.(last.from);
    return true;
  }
}
