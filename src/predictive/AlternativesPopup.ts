/**
 * A small keyboard-navigable popup anchored AT the word, for the "suggest alternatives" action.
 *
 * It deliberately mirrors the word-completion popup rather than the right-click Menu it replaces:
 * it appears next to the word (not under the mouse), is styled like the prediction list, is
 * navigated with the arrow keys, and is accepted with the SAME key as a completion (Tab by
 * default) or Enter. Unlike the completion popup it can show more than three rows - however many
 * good alternatives the model found. Esc, a click elsewhere, or losing the editor dismisses it.
 *
 * It is a standalone floating element (not an Obsidian EditorSuggest) because EditorSuggest only
 * opens off a typing trigger, whereas this is invoked on demand from the context menu. Keyboard
 * handling is a capture-phase document listener so Tab/Enter/arrows are claimed before CodeMirror
 * sees them; the editor never loses focus (rows accept on mousedown with preventDefault).
 */
import type { Editor, EditorPosition } from "obsidian";
import type { EditorView } from "@codemirror/view";

export interface AlternativesTarget {
  editor: Editor;
  /** The range the chosen alternative replaces (the word, or the selection). */
  from: EditorPosition;
  to: EditorPosition;
  /** The alternatives, already cased to match the original word. */
  items: string[];
  /** The user's accept key (Tab by default); Enter always works too. */
  acceptKey: string;
  /** Apply the chosen replacement (also records the stat, moves the caret, etc.). */
  onAccept: (replacement: string) => void;
}

export class AlternativesPopup {
  private el: HTMLElement | null = null;
  private rows: HTMLElement[] = [];
  private items: string[] = [];
  private index = 0;
  private acceptKey = "Tab";
  private onAccept: (r: string) => void = () => {};

  private readonly onKey = (e: KeyboardEvent): void => this.handleKey(e);
  private readonly onPointer = (e: MouseEvent): void => {
    if (this.el && !this.el.contains(e.target as Node)) this.close();
  };

  /** True while a popup is on screen. */
  get isOpen(): boolean {
    return this.el !== null;
  }

  open(target: AlternativesTarget): void {
    this.close();
    if (target.items.length === 0) return;
    this.items = target.items;
    this.onAccept = target.onAccept;
    this.acceptKey = target.acceptKey;
    this.index = 0;

    const el = document.body.createDiv({
      cls: "suggestion-container smart-autocorrect-alt-popup",
    });
    const list = el.createDiv({ cls: "suggestion" });
    this.rows = this.items.map((word, i) => {
      const row = list.createDiv({ cls: "suggestion-item predictive-suggestion" });
      const left = row.createSpan({ cls: "predictive-left" });
      left.createSpan({ cls: "predictive-mark", text: "✦" });
      left.createSpan({ text: word });
      row.createSpan({ cls: "predictive-kind", text: "✎ alt" });
      // mousedown, not click: preventDefault keeps focus in the editor so the replacement lands
      // where the caret is.
      row.addEventListener("mousedown", (ev) => {
        ev.preventDefault();
        this.accept(i);
      });
      row.addEventListener("mouseenter", () => this.highlight(i));
      return row;
    });
    this.el = el;
    this.position(target.editor, target.from);
    this.highlight(0);

    document.addEventListener("keydown", this.onKey, true);
    document.addEventListener("mousedown", this.onPointer, true);
  }

  close(): void {
    if (!this.el) return;
    document.removeEventListener("keydown", this.onKey, true);
    document.removeEventListener("mousedown", this.onPointer, true);
    this.el.remove();
    this.el = null;
    this.rows = [];
    this.items = [];
  }

  /** Place the popup just under the start of the word, using CodeMirror's caret geometry. */
  private position(editor: Editor, at: EditorPosition): void {
    if (!this.el) return;
    const cm = (editor as unknown as { cm?: EditorView }).cm;
    let left = 0;
    let top = 0;
    if (cm && typeof cm.coordsAtPos === "function") {
      const coords = cm.coordsAtPos(editor.posToOffset(at));
      if (coords) {
        left = coords.left;
        top = coords.bottom + 4;
      }
    }
    // Keep it on screen if the word sits near the right/bottom edge.
    const rect = this.el.getBoundingClientRect();
    left = Math.min(left, window.innerWidth - rect.width - 8);
    if (top + rect.height > window.innerHeight) top = Math.max(4, top - rect.height - 20);
    this.el.style.left = `${Math.max(4, left)}px`;
    this.el.style.top = `${top}px`;
  }

  private highlight(i: number): void {
    if (this.rows.length === 0) return;
    this.index = (i + this.rows.length) % this.rows.length;
    this.rows.forEach((r, j) => r.toggleClass("is-selected", j === this.index));
    this.rows[this.index]?.scrollIntoView({ block: "nearest" });
  }

  private accept(i: number): void {
    const word = this.items[i];
    this.close();
    if (word !== undefined) this.onAccept(word);
  }

  private handleKey(e: KeyboardEvent): void {
    if (!this.el) return;
    const isAccept = e.key === this.acceptKey || e.key === "Enter";
    if (e.key === "ArrowDown") {
      this.highlight(this.index + 1);
    } else if (e.key === "ArrowUp") {
      this.highlight(this.index - 1);
    } else if (isAccept) {
      this.accept(this.index);
    } else if (e.key === "Escape") {
      this.close();
    } else {
      return; // let every other key through to the editor
    }
    e.preventDefault();
    e.stopPropagation();
  }
}
