/**
 * Build an in-memory language model from raw corpus text. The real offline
 * pipeline (Python, §8 of PROPOSAL.md) produces a packed binary; this builder
 * is used for the vault-derived personal model and for tests/demos.
 */
import { buildAbbreviationSet } from "../text/abbreviations.ts";
import { splitSentences } from "../text/tokenize.ts";
import {
  InMemoryLanguageModel,
  NgramCounts,
  type BlendConfig,
  DEFAULT_BLEND,
} from "./model.ts";

export interface BuildOptions {
  abbreviations?: Set<string>;
  blend?: BlendConfig;
  /** drop unigrams occurring fewer than this many times (prune). */
  minCount?: number;
}

/** Build raw n-gram counts from corpus text (exposed so it can be packed). */
export function buildCountsFromText(text: string, opts: BuildOptions = {}): NgramCounts {
  const abbreviations = opts.abbreviations ?? buildAbbreviationSet();
  const counts = new NgramCounts();
  for (const sentence of splitSentences(text, { abbreviations })) {
    if (sentence.length) counts.addSentence(sentence);
  }
  return counts;
}

export function buildModelFromText(
  text: string,
  opts: BuildOptions = {},
): InMemoryLanguageModel {
  return new InMemoryLanguageModel(buildCountsFromText(text, opts), opts.blend ?? DEFAULT_BLEND);
}
