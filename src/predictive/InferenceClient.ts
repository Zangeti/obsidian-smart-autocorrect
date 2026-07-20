/**
 * Main-thread handle on the engine. Sends every call to the inference Web Worker
 * so the LSTM never runs on the UI thread - that synchronous work inside
 * CodeMirror's update cycle was what made typing stutter.
 *
 * If Workers aren't available (or the user turns `offMainThread` off) it falls
 * back to running the SAME EngineCore inline, so behaviour is identical and only
 * the thread differs. Either way the API is async, so callers can't accidentally
 * depend on synchronous results.
 */
import { EngineCore } from "./EngineCore";
import { WORKER_SRC } from "./generated/workerSource";
import type { PredictiveSettings } from "./PredictiveSettings";
import type { PersonalizationState } from "./engine/index";

export class InferenceClient {
  private worker: Worker | null = null;
  private url: string | null = null;
  private inline: EngineCore | null = null;
  private seq = 0;
  private pending = new Map<number, { resolve: (v: never) => void; reject: (e: unknown) => void }>();

  static workersSupported(): boolean {
    return typeof Worker !== "undefined" && typeof Blob !== "undefined" && typeof URL !== "undefined";
  }

  /** True when calls are actually leaving the main thread. */
  get threaded(): boolean {
    return this.worker !== null;
  }

  constructor(settings: PredictiveSettings, personalization: PersonalizationState | null) {
    if (settings.offMainThread && InferenceClient.workersSupported()) {
      try {
        this.url = URL.createObjectURL(new Blob([WORKER_SRC], { type: "text/javascript" }));
        this.worker = new Worker(this.url);
        this.worker.onmessage = (ev: MessageEvent) => {
          const { id, ok, result, error } = ev.data as { id: number; ok: boolean; result?: unknown; error?: string };
          const p = this.pending.get(id);
          if (!p) return;
          this.pending.delete(id);
          if (ok) p.resolve(result as never);
          else p.reject(new Error(error ?? "worker call failed"));
        };
        this.worker.onerror = (ev) => {
          console.warn("[predictive] inference worker error; falling back inline", ev.message);
          this.failInline(settings, personalization);
        };
      } catch (e) {
        console.warn("[predictive] could not start inference worker; running inline", e);
        this.worker = null;
      }
    }
    if (this.worker) void this.call("init", [settings, personalization]);
    else this.inline = new EngineCore(settings, personalization);
  }

  /** A worker that dies mid-session must not take prediction down with it. */
  private failInline(settings: PredictiveSettings, personalization: PersonalizationState | null): void {
    for (const [, p] of this.pending) p.reject(new Error("worker died"));
    this.pending.clear();
    this.worker?.terminate();
    this.worker = null;
    if (!this.inline) this.inline = new EngineCore(settings, personalization);
  }

  private call<T>(op: string, args: unknown[], transfer: Transferable[] = []): Promise<T> {
    const w = this.worker;
    if (!w) {
      // Inline: run it now, but still hand back a promise so callers are uniform.
      try {
        const core = this.inline as unknown as Record<string, (...a: unknown[]) => unknown>;
        return Promise.resolve(core[op].apply(this.inline, args) as T);
      } catch (e) {
        return Promise.reject(e);
      }
    }
    const id = ++this.seq;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as never, reject });
      w.postMessage({ id, op, args }, transfer);
    });
  }

  // --- model loading (ArrayBuffers are TRANSFERRED: no copy of a 57 MB model) --
  loadGlobalPacked(buf: ArrayBuffer): Promise<void> { return this.call("loadGlobalPacked", [buf], [buf]); }
  loadGlobalText(text: string | null): Promise<void> { return this.call("loadGlobalText", [text]); }
  loadLstm(buf: ArrayBuffer): Promise<boolean> { return this.call("loadLstm", [buf], [buf]); }
  loadWordlist(buf: ArrayBuffer): Promise<void> { return this.call("loadWordlist", [buf], [buf]); }
  isKnownWord(word: string): Promise<boolean> { return this.call("isKnownWord", [word]); }
  documentFrequencies(words: string[]): Promise<number[] | null> { return this.call("documentFrequencies", [words]); }
  packGlobal(): Promise<ArrayBuffer | null> { return this.call("packGlobal", []); }

  rebuildPersonal(entries: { path: string; text: string }[] | null): Promise<void> {
    return this.call("rebuildPersonal", [entries]);
  }
  setFile(path: string, text: string): Promise<boolean> { return this.call("setFile", [path, text]); }
  removeFile(path: string): Promise<void> { return this.call("removeFile", [path]); }
  renameFile(a: string, b: string): Promise<void> { return this.call("renameFile", [a, b]); }

  setActiveDocument(text: string): Promise<void> { return this.call("setActiveDocument", [text]); }
  updateSettings(s: PredictiveSettings): Promise<void> { return this.call("updateSettings", [s]); }

  getSuggestions(context: string[], typed: string, k: number, includePhrases = true): Promise<import("./EngineCore").SuggestItem[]> {
    return this.call("getSuggestions", [context, typed, k, includePhrases]);
  }
  decide(typed: string, context: string[]): Promise<import("./engine/index").CorrectionDecision> {
    return this.call("decide", [typed, context]);
  }
  mergeDecision(prev: string, cur: string, context: string[]): Promise<string | null> {
    return this.call("mergeDecision", [prev, cur, context]);
  }
  runEvaluation(text: string): Promise<import("./engine/index").EvalResult | null> {
    return this.call("runEvaluation", [text]);
  }
  /** Contextual proper-noun casing from the cased LSTM (replaced the CaseMap). */
  caseFor(word: string, context: string[]): Promise<string | null> {
    return this.call("caseFor", [word, context]);
  }

  recordAccept(insert: string, saved = 0): Promise<void> { return this.call("recordAccept", [insert, saved]); }
  recordCorrection(from: string, to: string): Promise<void> { return this.call("recordCorrection", [from, to]); }
  recordRevert(original: string): Promise<void> { return this.call("recordRevert", [original]); }
  personalizationState(): Promise<PersonalizationState> { return this.call("personalizationState", []); }
  setPersonalization(s: PersonalizationState | null): Promise<void> { return this.call("setPersonalization", [s]); }
  status(): Promise<{ ready: boolean; accelerated: boolean; lstmLoaded: boolean }> { return this.call("status", []); }
  embed(text: string): Promise<number[] | null> { return this.call("embed", [text]); }
  embedDim(): Promise<number> { return this.call("embedDim", []); }
  rarities(words: string[]): Promise<number[]> { return this.call("rarities", [words]); }

  dispose(): void {
    this.worker?.terminate();
    this.worker = null;
    if (this.url) URL.revokeObjectURL(this.url);
    this.url = null;
    this.pending.clear();
    this.inline = null;
  }
}
