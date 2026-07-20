/**
 * Entry point for the inference Web Worker. Bundled to a standalone IIFE at build
 * time and inlined into main.js as a string (see esbuild.config.mjs +
 * generated/workerSource.ts) - Obsidian's community-plugin installer only fetches
 * main.js/manifest.json/styles.css, so a sidecar worker file would never survive
 * an install.
 *
 * It is a thin RPC shell: every method here just forwards to EngineCore, which is
 * the same class the main thread runs when Workers are unavailable.
 */
import { EngineCore } from "./EngineCore";

type Req = { id: number; op: string; args: unknown[] };

let core: EngineCore | null = null;

function call(op: string, args: unknown[]): unknown {
  if (op === "init") {
    core = new EngineCore(args[0] as never, args[1] as never);
    return true;
  }
  const c = core;
  if (!c) throw new Error("engine not initialised");
  switch (op) {
    case "loadGlobalPacked": return c.loadGlobalPacked(args[0] as ArrayBuffer);
    case "loadGlobalText": return c.loadGlobalText(args[0] as string | null);
    case "loadLstm": return c.loadLstm(args[0] as ArrayBuffer);
    case "loadWordlist": return c.loadWordlist(args[0] as ArrayBuffer);
    case "isKnownWord": return c.isKnownWord(args[0] as string);
    case "documentFrequencies": return c.documentFrequencies(args[0] as string[]);
    case "packGlobal": return c.packGlobal();
    case "rebuildPersonal": return c.rebuildPersonal(args[0] as never);
    case "setFile": return c.setFile(args[0] as string, args[1] as string);
    case "removeFile": return c.removeFile(args[0] as string);
    case "renameFile": return c.renameFile(args[0] as string, args[1] as string);
    case "setActiveDocument": return c.setActiveDocument(args[0] as string);
    case "updateSettings": return c.updateSettings(args[0] as never);
    case "getSuggestions": return c.getSuggestions(args[0] as string[], args[1] as string, args[2] as number, args[3] as boolean | undefined);
    case "getCandidates": return c.getCandidates(args[0] as string[], args[1] as string, args[2] as number);
    case "decide": return c.decide(args[0] as string, args[1] as string[]);
    case "mergeDecision": return c.mergeDecision(args[0] as string, args[1] as string, args[2] as string[]);
    case "runEvaluation": return c.runEvaluation(args[0] as string);
    case "caseFor": return c.caseFor(args[0] as string, args[1] as string[]);
    case "recordAccept": return c.recordAccept(args[0] as string, args[1] as number);
    case "recordCorrection": return c.recordCorrection(args[0] as string, args[1] as string);
    case "recordRevert": return c.recordRevert(args[0] as string);
    case "personalizationState": return c.personalizationState();
    case "setPersonalization": return c.setPersonalization(args[0] as never);
    case "status": return c.status();
    case "embed": return c.embed(args[0] as string);
    case "embedDim": return c.embedDim();
    case "rarities": return c.rarities(args[0] as string[]);
    default: throw new Error(`unknown op ${op}`);
  }
}

self.onmessage = (e: MessageEvent<Req>) => {
  const { id, op, args } = e.data;
  try {
    const result = call(op, args);
    // packGlobal returns a fresh ArrayBuffer - hand ownership back rather than
    // copying it.
    const transfer = result instanceof ArrayBuffer ? [result] : [];
    (self as unknown as Worker).postMessage({ id, ok: true, result }, transfer);
  } catch (err) {
    (self as unknown as Worker).postMessage({ id, ok: false, error: String(err) });
  }
};
