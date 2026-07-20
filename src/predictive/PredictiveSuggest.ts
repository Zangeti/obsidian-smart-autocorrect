/**
 * Self-contained EditorSuggest showing context-aware single-word completions,
 * fuzzy corrections, and (optionally) multi-word phrase completions. Phrases are
 * popup-only and accepted with the configured accept key (Tab by default, see
 * bindKeys); they are never autocorrected. Suppressed inside code/math/links/tags/
 * frontmatter.
 */
import { EditorSuggest } from "obsidian";
import type {
  App,
  Editor,
  EditorPosition,
  EditorSuggestContext,
  EditorSuggestTriggerInfo,
  KeymapEventHandler,
  TFile,
} from "obsidian";
import { insertNewlineAndIndent } from "@codemirror/commands";
import type { EditorView } from "@codemirror/view";
import {
  applySuggestionCase,
  classifyMarkdownContext,
  pathExcluded,
  defaultSentenceCaseConfig,
  upperFromText,
  startsWithTightPunct,
  endsWithTightPunct,
  doubleSpaceStart,
  suggestionCase,
  type SuggestionCase,
} from "./engine/index";
import { contextWords } from "./context";
import { indentEditField } from "./tabIndent";
import type { PredictiveEngineController, SuggestItem } from "./PredictiveEngineController";
import type { PredictiveSettings } from "./PredictiveSettings";

export class PredictiveSuggest extends EditorSuggest<SuggestItem> {
  private engine: PredictiveEngineController;
  private settings: PredictiveSettings;
  private onAccept: (insert: string, saved: number) => void;
  /** The case the popup was DISPLAYED in, so the accept can insert exactly that. */
  private shownCase: SuggestionCase = "none";
  /**
   * Cursor position immediately after an accepted suggestion. While the caret sits
   * there, the word is finished - so predict the NEXT word rather than offering more
   * completions of the one just completed.
   */
  private acceptedAt: EditorPosition | null = null;
  /**
   * The multi-word phrase the user is currently typing out, anchored at the position
   * its first word starts. Suggestions are recomputed from scratch on every keystroke,
   * so without this the phrase you are halfway through vanishes the moment you type a
   * space and the model re-ranks - even though you are visibly still writing it.
   */
  private phrase: { text: string; anchor: EditorPosition } | null = null;

  constructor(
    app: App,
    engine: PredictiveEngineController,
    settings: PredictiveSettings,
    onAccept: (insert: string, saved: number) => void,
  ) {
    super(app);
    this.engine = engine;
    this.settings = settings;
    this.onAccept = onAccept;
    this.limit = settings.maxSuggestions; // cap the popup (e.g. top 3)
    this.bindKeys();
  }

  /** Handlers we registered, so a settings change can take them back off. */
  private bound: KeymapEventHandler[] = [];
  /** Our own bare-Enter handler (newline), kept so we can re-register it LAST. */
  private enterHandler: KeymapEventHandler | null = null;
  /**
   * True from the moment the accept key inserts a suggestion until the NEXT suggestion
   * list has been computed. A second accept-key press in that window would re-fire on the
   * stale, still-displayed list and insert the same word twice - so it is swallowed until
   * getSuggestions() clears this. (Rapid Tab-Tab is exactly this race.)
   */
  private acceptBusy = false;

  /**
   * Bind the accept key, and ONLY the accept key.
   *
   * Shift+<key> is registered too, and it is NOT redundant: while typing in CAPS the
   * user is holding Shift, so the accept arrives as Shift+Tab and would otherwise fall
   * through to Obsidian's outdent.
   *
   * Enter is handled separately, in freeEnter() - see there for why removing Obsidian's
   * built-in binding (rather than registering over it) is the only thing that works.
   */
  private bindKeys(): void {
    for (const h of this.bound) this.scope.unregister(h);
    this.bound = [];

    const accept = (evt: KeyboardEvent) => {
      const self = this as unknown as {
        suggestions?: { useSelectedItem: (e: Event) => void };
      };
      if (self.suggestions) {
        // Swallow a repeat press that lands before the list has refreshed - otherwise it
        // re-accepts the same (stale) item and inserts the word twice. Cleared in
        // getSuggestions() once the next list is ready.
        if (this.acceptBusy) return false;
        this.acceptBusy = true;
        self.suggestions.useSelectedItem(evt);
        return false;
      }
      return true;
    };

    const key = this.settings.acceptKey;
    this.bound.push(this.scope.register([], key, accept));
    this.bound.push(this.scope.register(["Shift"], key, accept));
    this.freeEnter();
    this.ensureEnterNewline();
  }

  /**
   * Register OUR OWN bare-Enter handler that dismisses the popup and inserts a newline,
   * so Enter is reliably "next line" when it is not the accept key.
   *
   * This is what makes it deterministic where freeEnter() alone was not: Obsidian
   * registers its Enter->select binding lazily inside open(), which runs AFTER
   * getSuggestions() - so freeEnter()'s removal races it and misses. A Scope tries the
   * MOST RECENTLY registered handler first and stops on the first that returns false, so
   * re-registering ours last (every getSuggestions, i.e. every keystroke the popup is up)
   * guarantees it beats Obsidian's Enter no matter when that was installed. Returning
   * false consumes the key, so the editor gets no native newline - we insert one
   * ourselves (list/quote continuation preserved) and the caret moves to a fresh line,
   * which also makes the autocorrect boundary handler find no word and stay quiet.
   *
   * When Enter IS the accept key we remove our handler and leave Obsidian's select alone.
   */
  private ensureEnterNewline(): void {
    if (this.enterHandler) {
      this.scope.unregister(this.enterHandler);
      this.enterHandler = null;
    }
    if (this.settings.acceptKey === "Enter") return; // Enter accepts: nothing to override
    this.enterHandler = this.scope.register([], "Enter", () => {
      const editor = this.context?.editor;
      this.close();
      if (editor) this.insertNewline(editor);
      return false; // consume: Obsidian's Enter->select must not also fire
    });
  }

  /**
   * Take Obsidian's built-in bare-Enter "select" binding back off the popup's scope, so
   * that when the accept key is NOT Enter, pressing Enter is an ordinary newline -
   * CodeMirror's Enter, with list/quote continuation intact - instead of accepting.
   *
   * It has to REMOVE the base handler, not register over it: in a Scope, any handler
   * that returns false consumes the key and stops the rest, and the base's Enter handler
   * does exactly that. So the only way Enter can reach the editor is for no suggest
   * handler to claim it - our own Enter handler could never win the race.
   *
   * Obsidian hands back no reference to its own binding, so we match on shape. And we
   * call this REPEATEDLY (from getSuggestions, i.e. whenever the popup is about to show)
   * rather than once at construction, because the base can install its Enter handler
   * lazily on first open - a one-shot removal would miss it and Enter would accept
   * forever after. That lazy timing is the bug this replaces. No empty catch: if the
   * scope shape is ever unrecognised we simply remove nothing, and the selectSuggestion
   * guard still stops Enter from accepting - never a silent wrong-key accept.
   */
  private freeEnter(): void {
    if (this.settings.acceptKey === "Enter") return; // Enter IS the accept key: leave it
    const scope = this.scope as unknown as {
      keys?: Array<{ modifiers: string | null; key: string | null }>;
    };
    const keys = scope.keys;
    if (!Array.isArray(keys)) return;
    for (const h of [...keys]) {
      const bareEnter = h.key === "Enter" && !(h.modifiers && h.modifiers.length);
      if (bareEnter && !this.bound.includes(h as unknown as KeymapEventHandler))
        this.scope.unregister(h as unknown as KeymapEventHandler);
    }
  }

  updateSettings(settings: PredictiveSettings): void {
    this.settings = settings;
    this.limit = settings.maxSuggestions;
    this.bindKeys(); // the accept key may have changed
  }

  /** True when the last document edit was an indent/outdent (Tab/Shift-Tab on a bullet), so
   *  we should NOT pop a suggestion: indenting is structural, not typing. */
  private lastEditWasIndent(editor: Editor): boolean {
    const cm = (editor as unknown as { cm?: EditorView }).cm;
    if (!cm || !cm.state) return false;
    return cm.state.field(indentEditField, false) === true;
  }

  onTrigger(
    cursor: EditorPosition,
    editor: Editor,
    file: TFile,
  ): EditorSuggestTriggerInfo | null {
    if (!this.settings.pluginEnabled) return null;
    if (!this.settings.enablePredictions || !this.engine.ready) return null;
    // Inline ghost text is an ALTERNATIVE to the popup: when it's on, suppress the
    // dropdown so the two don't both show for the same word.
    if (this.settings.ghostText) return null;
    // Skip excluded folders/files entirely.
    if (file && pathExcluded(file.path, this.settings.excludedFolders)) return null;
    // Just indented/outdented a bullet: that's structuring a list, not typing a word - don't
    // pop a suggestion off the back of it.
    if (this.lastEditWasIndent(editor)) return null;

    // Markdown/LaTeX awareness: bail inside code, math, links, tags, etc.
    if (this.settings.markdownAware) {
      const textBefore = editor.getRange({ line: 0, ch: 0 }, cursor);
      if (classifyMarkdownContext(textBefore).suppressPrediction) return null;
    }

    const line = editor.getLine(cursor.line);
    const before = line.slice(0, cursor.ch);
    const m = before.match(/([A-Za-z][A-Za-z'-]*)$/);
    const query = m ? m[1] : "";
    const startCh = cursor.ch - query.length;

    // Just accepted a suggestion and the caret has not moved: that word is DONE. Ask
    // for the next word instead of re-completing what was just inserted - otherwise
    // accepting "hello" immediately offers "helloween", and Tab would replace the very
    // word the user just chose. The empty range means we insert AT the caret.
    if (this.acceptedAt?.line === cursor.line && this.acceptedAt.ch === cursor.ch) {
      return { start: cursor, end: cursor, query: "" };
    }
    this.acceptedAt = null; // the caret moved or text changed: back to normal completion

    if (query.length === 0) {
      // Next-word / phrase prediction only right after "word ".
      if (!/\w\s$/.test(before)) return null;
    } else if (query.length < this.settings.minChars) {
      // Word being typed is shorter than the trigger threshold: wait for more letters.
      return null;
    }
    return { start: { line: cursor.line, ch: startCh }, end: cursor, query };
  }

  /** Async: the engine lives in a worker, so nothing here blocks typing.
   *  (Obsidian's EditorSuggest accepts `T[] | Promise<T[]>`.) */
  async getSuggestions(context: EditorSuggestContext): Promise<SuggestItem[]> {
    this.freeEnter(); // the popup is about to show - make sure Enter is still a newline
    this.ensureEnterNewline(); // and re-assert our newline handler as the top Enter binding
    const before = context.editor
      .getLine(context.start.line)
      .slice(0, context.start.ch);
    const ctxWords = contextWords(before, this.settings.extraAbbreviations);
    let items: SuggestItem[] = [];
    try {
      // Ask for MORE than we show: the phrase-follow, dictionary merge and dedup steps
      // below all drop candidates, and without headroom the popup silently shrinks to
      // one or two entries. Obsidian caps the rendered list at `limit` anyway.
      items = await this.engine.getSuggestions(ctxWords, context.query, (this.limit ?? 3) + 3);
    } catch {
      items = []; // a worker hiccup shows no popup rather than throwing at the user
    }
    const merged = this.withPhrase(context, items);
    const out = this.cased(before, context.query, this.withDictionary(context.query, merged));
    this.acceptBusy = false; // a fresh list is ready - the accept key may fire again
    // A candidate that saves no keystrokes (e.g. a word already typed out in full) is
    // dropped upstream by the expected-keystrokes-saved ranking in EngineCore.getSuggestions
    // - its saving is <= 0 - so no display-side exact-match filter is needed here.
    return out;
  }

  /**
   * Keep the phrase being typed out at the top, and let the other slots re-rank freely.
   *
   * The phrase is offered as its REMAINDER measured from the anchor, which makes it
   * work identically whether a word is half-typed or not: "Running fo" replaces "fo"
   * with "for the united states"; "Running " inserts it at the caret. In both cases the
   * text from the anchor to the replace-start is a prefix of the phrase, so the
   * remainder is just the rest of the string - no word-boundary arithmetic, and the
   * spacing is whatever the phrase itself contains.
   */
  private withPhrase(ctx: EditorSuggestContext, items: SuggestItem[]): SuggestItem[] {
    const active = this.activePhrase(ctx);
    if (!active) {
      // Start following the top suggestion if it is a phrase, anchored where the word
      // being typed begins.
      const top = items[0];
      if (top && top.kind === "phrase") this.phrase = { text: top.insert, anchor: ctx.start };
      return items;
    }
    // Drop any fresh item that duplicates the phrase, then let the rest fill the
    // remaining slots - they keep updating even while the top one is being typed out.
    const rest = items.filter((i) => i.insert !== active);
    return [{ insert: active, display: active, kind: "phrase", score: Infinity }, ...rest];
  }

  /** The remainder of the phrase still to be typed, or null if it no longer applies. */
  private activePhrase(ctx: EditorSuggestContext): string | null {
    if (!this.phrase) return null;
    const { editor } = ctx;
    const anchorOff = editor.posToOffset(this.phrase.anchor);
    const startOff = editor.posToOffset(ctx.start);
    // The caret moved back before the anchor, or to another line/note: not ours.
    if (anchorOff > startOff) return (this.phrase = null);
    const typed = editor.getRange(this.phrase.anchor, ctx.start);
    // Case-insensitive: the phrase may be capitalised for its position while the user
    // types it in whatever case they like.
    if (!this.phrase.text.toLowerCase().startsWith(typed.toLowerCase())) return (this.phrase = null);
    const remainder = this.phrase.text.slice(typed.length);
    // Fully typed out: nothing left to offer, so stop following it.
    if (!remainder.trim()) return (this.phrase = null);
    // The word being typed RIGHT NOW must also still be going the phrase's way.
    // Without this, "Running x" would keep offering "for the united states" and
    // silently replace the "x" - following a phrase the user has plainly abandoned.
    if (!remainder.toLowerCase().startsWith(ctx.query.toLowerCase())) return (this.phrase = null);
    return remainder;
  }

  /** The case suggestions must take here, from the text before the word being typed. */
  private caseFor(before: string, query: string): SuggestionCase {
    // Read the WHOLE text up to the cursor, current partial word included: that is
    // what carries caps across a word boundary ("THE QUICK " -> the next word's
    // suggestions are caps immediately) while still dropping them the instant a
    // lowercase character appears in the word being typed.
    return suggestionCase(before, defaultSentenceCaseConfig(this.settings.extraAbbreviations), {
      upper: upperFromText(before + query),
    });
  }

  /**
   * Case every suggestion for its position BEFORE it is displayed.
   *
   * Previously the popup showed whatever case each channel happened to produce and
   * autocorrect fixed it up after the word was committed - so on an empty document you
   * were offered "the", accepted "the", and watched it turn into "The". And because
   * the LSTM, n-gram and fuzzy channels disagree about case, a sentence start could
   * offer "The", "there", "Their" all at once. Case is a property of the POSITION, not
   * of whichever channel won, so it is decided once and applied to the whole list.
   *
   * Dictionary entries are exempt: the user pinned their exact spelling, and
   * "NixOS" -> "Nixos" at a sentence start would destroy the point of them.
   */
  private cased(before: string, query: string, items: SuggestItem[]): SuggestItem[] {
    const mode = this.caseFor(before, query);
    this.shownCase = mode;
    const out =
      mode === "none"
        ? items
        : items.map((i) =>
            i.kind === "dictionary"
              ? i
              : { ...i, insert: applySuggestionCase(i.insert, mode), display: applySuggestionCase(i.display, mode) },
          );
    // The engine now emits both cases of a homograph as separate candidates ("polish"
    // AND "Polish"). A forced position case (a sentence start Titles both) can collapse
    // them to the same surface, so dedupe by insert, keeping the first/best occurrence.
    const seen = new Set<string>();
    return out.filter((i) => (seen.has(i.insert) ? false : (seen.add(i.insert), true)));
  }

  /**
   * Merge personal-dictionary matches in front of the model's suggestions.
   *
   * Done HERE, on the main thread, rather than inside the engine: the engine lives in
   * a Web Worker and would need the dictionary shipped across the RPC boundary and
   * kept in sync on every settings change. The dictionary is small and the popup is
   * capped at a few entries, so a prefix scan costs nothing and needs no plumbing.
   *
   * Lookup is case-INSENSITIVE (typing "nix" should find "NixOS") but the inserted
   * text is the entry VERBATIM - that is the whole reason the feature exists.
   */
  private withDictionary(query: string, items: SuggestItem[]): SuggestItem[] {
    if (!this.settings.suggestUserDictionary || !query) return items;
    const q = query.toLowerCase();
    const hits = this.settings.userDictionary.filter(
      (w) => w.toLowerCase().startsWith(q) && w !== query,
    );
    if (hits.length === 0) return items;
    const seen = new Set(hits);
    const dict: SuggestItem[] = hits.map((w) => ({
      insert: w,
      display: w,
      kind: "dictionary",
      score: Infinity, // the user pinned it; it outranks anything the model guessed
    }));
    // Drop model suggestions that duplicate a dictionary entry, so the verbatim
    // spelling wins instead of appearing twice in two different casings.
    const rest = items.filter((i) => !seen.has(i.insert));
    return [...dict, ...rest];
  }

  renderSuggestion(value: SuggestItem, el: HTMLElement): void {
    el.addClass("predictive-suggestion");
    const left = el.createSpan({ cls: "predictive-left" });
    // A small accent mark on every row so the plugin's completions read as distinct from
    // Obsidian's own plain text suggestions.
    left.createSpan({ cls: "predictive-mark", text: "✦" });
    left.createSpan({ text: value.display });
    if (value.kind === "phrase") {
      el.createSpan({ text: "⏎ phrase", cls: "predictive-kind" });
    } else if (value.kind === "dictionary") {
      el.createSpan({ text: "📖 yours", cls: "predictive-kind" });
    }
  }

  selectSuggestion(value: SuggestItem, evt: MouseEvent | KeyboardEvent): void {
    const ctx = this.context;
    if (!ctx) return;
    // Safety net for freeEnter(): if Obsidian's built-in Enter binding ever survives
    // removal, Enter must STILL never accept when it is not the accept key - we told the
    // user Enter is "next line", so silently completing a word on it is exactly the
    // surprise to avoid. Dismiss the popup and let Enter be Enter.
    if (evt instanceof KeyboardEvent && evt.key === "Enter" && this.settings.acceptKey !== "Enter") {
      // freeEnter() removes Obsidian's built-in Enter binding so Enter falls through to
      // the editor as a newline. If we are HERE, that removal missed and the popup's
      // Enter handler fired instead - which consumes the key, so the editor would get no
      // newline at all (the "Enter does nothing / weird insert" bug: with the caret still
      // sitting after the word, the autocorrect boundary handler then rewrites it in
      // place). Dismiss and insert the newline OURSELVES so Enter is always "next line".
      // Inserting it also moves the caret to a fresh line, so the pending autocorrect
      // boundary sees no token and correctly does nothing.
      this.close();
      this.insertNewline(ctx.editor);
      return;
    }
    // A dictionary entry is inserted EXACTLY as the user stored it. Running it
    // through matchCaseToQuery would rewrite "kubeCTL" to "KubeCTL" the moment the
    // query started with a capital - destroying the one thing the entry is for.
    // Insert EXACTLY what the popup showed. `value.insert` was already cased for this
    // position by cased(), so re-deriving the decision here could disagree with the
    // list the user was looking at - which is the very bug this whole path exists to
    // fix. matchCaseToQuery only applies when no case decision was made ("none"),
    // where it still does useful work ("Par" -> "Paris").
    const insert =
      value.kind === "dictionary" || this.shownCase !== "none"
        ? value.insert
        : this.matchCaseToQuery(ctx.query, value.insert);
    // The model predicts punctuation as ordinary tokens, so a suggestion may open
    // with one. English puts no space before "," or "." - but the popup triggers
    // right after "word ", so replacing from the cursor would give "world ,".
    // Extend the replaced range back over that space and eat it.
    let spaced = this.spaceBefore(ctx.editor, ctx.start, insert);
    let from = this.eatSpaceBefore(ctx.editor, ctx.start, spaced);
    from = this.eatDoubleSpace(ctx.editor, from, spaced);
    // A suggestion ending in punctuation ("polish it.", or a bare ",") should leave
    // the caret ready to type the next word, not hard against the mark - so append a
    // trailing space. Skip it when the text already continues with whitespace, so we
    // never stack two spaces.
    spaced = this.spaceAfter(ctx.editor, ctx.end, spaced);
    ctx.editor.replaceRange(spaced, from, ctx.end);
    const end: EditorPosition = { line: from.line, ch: from.ch + spaced.length };
    ctx.editor.setCursor(end);
    // Remember where we landed so the next popup predicts the FOLLOWING word.
    this.acceptedAt = end;
    // Keystrokes saved = the characters of the accepted surface the user did NOT type
    // (they had typed `ctx.query` already). Feeds the running "keystrokes saved" counter.
    const saved = Math.max(0, value.insert.length - ctx.query.length);
    this.onAccept(value.insert, saved);
  }

  /**
   * Add the separating space when inserting a next-word prediction straight after a
   * completed word.
   *
   * Accepting "hello" leaves the caret hard against the "o" with no space typed yet.
   * The following prediction is inserted AT that caret (an empty range), so without
   * this it would land as "helloworld". Only applies to that empty-range case -
   * ordinary completions replace the typed word and bring their own spacing - and
   * never before punctuation, which is tight against the word by definition.
   */
  private spaceBefore(editor: Editor, at: EditorPosition, insert: string): string {
    if (at.ch === 0 || !insert) return insert;
    if (startsWithTightPunct(insert)) return insert;
    const line = editor.getLine(at.line);
    const prev = line[at.ch - 1];
    return prev && !/\s/.test(prev) ? " " + insert : insert;
  }

  /**
   * Append a trailing space when the accepted text ends in punctuation, so the caret
   * lands ready for the next word rather than tight against the mark.
   *
   * `end` is where the replaced range stops; the char there is whatever already
   * followed the caret. If that is already whitespace (or end of line about to get
   * one), adding another would double it, so bail in that case.
   */
  private spaceAfter(editor: Editor, end: EditorPosition, insert: string): string {
    if (!insert || !endsWithTightPunct(insert)) return insert;
    const next = editor.getLine(end.line)[end.ch];
    return next && /\s/.test(next) ? insert : insert + " ";
  }

  /**
   * Collapse a double space before the word being completed (settings.collapseDoubleSpace).
   *
   * Extends the replaced range back over the extra space rather than issuing a second
   * edit, so the whole completion stays ONE undo step. The rule itself is
   * doubleSpaceStart() in the engine, where it is unit-tested against the edge cases
   * that matter (3+ spaces, tabs, indentation).
   */
  private eatDoubleSpace(editor: Editor, at: EditorPosition, insert: string): EditorPosition {
    if (!this.settings.collapseDoubleSpace) return at;
    // `insert` may itself open with the separator space added by spaceBefore(); then the
    // text before `at` is the word we just completed, not a space run.
    if (insert.startsWith(" ")) return at;
    const ch = doubleSpaceStart(editor.getLine(at.line), at.ch);
    return ch === at.ch ? at : { line: at.line, ch };
  }

  /** If `insert` starts with tight punctuation and `at` sits just after a space,
   *  return the position of that space so it gets replaced away. */
  private eatSpaceBefore(editor: Editor, at: EditorPosition, insert: string): EditorPosition {
    if (!startsWithTightPunct(insert) || at.ch === 0) return at;
    const line = editor.getLine(at.line);
    return line[at.ch - 1] === " " ? { line: at.line, ch: at.ch - 1 } : at;
  }

  /**
   * Insert a newline as if the user had pressed Enter in the editor - used when the popup
   * swallowed the real Enter (see selectSuggestion, ensureEnterNewline).
   *
   * "Identical in all respects" to a normal Enter means Obsidian's OWN handling must run:
   * continuing a bullet/number/quote, and removing the marker on an empty item. Obsidian
   * registers that as a CodeMirror keymap on the editor's contentDOM, so we re-dispatch a
   * synthetic Enter keydown there and let its handler fire exactly as for a real press. If
   * nothing consumes it (no CM Enter binding at all), we fall back to CodeMirror's generic
   * newline, then to a literal "\n" on the legacy editor.
   */
  private insertNewline(editor: Editor): void {
    const cm = (editor as unknown as { cm?: EditorView }).cm;
    if (cm && typeof cm.dispatch === "function") {
      const dom = (cm as unknown as { contentDOM?: HTMLElement }).contentDOM;
      if (dom) {
        const ev = new KeyboardEvent("keydown", {
          key: "Enter",
          code: "Enter",
          keyCode: 13,
          which: 13,
          bubbles: true,
          cancelable: true,
        });
        dom.dispatchEvent(ev);
        if (ev.defaultPrevented) return; // Obsidian's list/quote-aware Enter handled it
      }
      insertNewlineAndIndent(cm); // no handler claimed it: generic indented newline
      return;
    }
    editor.replaceSelection("\n");
  }

  private matchCaseToQuery(query: string, text: string): string {
    if (query && query[0] === query[0].toUpperCase())
      return text[0].toUpperCase() + text.slice(1);
    return text;
  }
}
