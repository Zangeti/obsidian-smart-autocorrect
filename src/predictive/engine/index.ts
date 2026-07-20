/**
 * Public engine API. The Obsidian plugin layer depends only on this barrel, so
 * the engine can be developed and tested independently of Obsidian.
 */
export * from "./text/tokenize.ts";
export * from "./text/abbreviations.ts";
export * from "./text/sentenceCase.ts";
export * from "./text/caseMode.ts";
export * from "./text/markdownContext.ts";
export * from "./text/normalize.ts";
export * from "./text/caseFix.ts";
export * from "./text/doubleSpace.ts";
export * from "./text/profanity.ts";
export * from "./text/exclude.ts";
export * from "./text/doubledWord.ts";
export * from "./text/wordOracle.ts";
export * from "./text/linkMatch.ts";
export * from "./text/vector.ts";
export * from "./text/segment.ts";
export * from "./ngram/model.ts";
export * from "./ngram/build.ts";
export * from "./ngram/serialize.ts";
export * from "./ngram/workerKernel.ts";
export * from "./ngram/incremental.ts";
export * from "./ngram/packed.ts";
export * from "./ngram/prune.ts";
export * from "./lstm/model.ts";
export * from "./lstm/lowercase.ts";
export * from "./channel/keyboard.ts";
export * from "./channel/costModel.ts";
export * from "./channel/confusion.ts";
export * from "./channel/phonetic.ts";
export * from "./channel/editDistance.ts";
export * from "./predict/fuzzyTrie.ts";
export * from "./predict/predictor.ts";
export * from "./predict/reranker.ts";
export * from "./predict/segmentation.ts";
export * from "./predict/sequence.ts";
export * from "./autocorrect/autocorrect.ts";
export * from "./personalization/state.ts";
export * from "./eval/harness.ts";

import { buildModelFromText, type BuildOptions } from "./ngram/build.ts";
import {
  CacheLanguageModel,
  MixtureLanguageModel,
  type LanguageModel,
} from "./ngram/model.ts";
import { FuzzyTrie } from "./predict/fuzzyTrie.ts";
import { RealWordCorrector } from "./predict/segmentation.ts";

/**
 * Convenience façade wiring a global model + optional personal (vault) model +
 * fuzzy trie + real-word corrector together. This is what the plugin uses.
 */
export class Engine {
  readonly model: MixtureLanguageModel;
  readonly cache: CacheLanguageModel;
  readonly index: FuzzyTrie;
  readonly realWord: RealWordCorrector;

  constructor(
    global: LanguageModel,
    personal: LanguageModel | null,
    alpha: number,
    opts: { cacheGamma?: number } = {},
  ) {
    this.model = new MixtureLanguageModel(global, personal, alpha);
    this.cache = new CacheLanguageModel(this.model, opts.cacheGamma ?? 0);
    this.index = new FuzzyTrie(this.model.vocabulary());
    this.realWord = new RealWordCorrector(this.model.vocabulary());
  }

  static fromText(globalText: string, opts?: BuildOptions): Engine {
    return new Engine(buildModelFromText(globalText, opts), null, 0);
  }
}
