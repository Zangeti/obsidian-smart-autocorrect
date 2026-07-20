/**
 * The bridge between the LSTM's token stream and the n-gram/vault side.
 *
 * This file used to hold CasedLstmModel, which marginalised each lowercase
 * candidate over its cased variants ("paris" = paris + Paris). That job is GONE:
 * the LSTM vocab is now lowercase and capitalisation comes out of a separate case
 * head (see model.ts renderCased), so there are no variants to sum over and the
 * LSTM is already a lowercase model. It reads the context directly, lowercasing
 * internally, and keeps the punctuation - the continuous stream it was trained on,
 * spanning sentences, so it can SEE where a sentence ends rather than guess.
 *
 * What still needs adapting is the other direction: the n-gram was built from
 * per-sentence, punctuation-free, lowercase word context, and the LSTM's wider view
 * must not leak into it.
 */
import { SENTENCE_END, normalizeWord } from "../text/tokenize.ts";
import type { Scored, LanguageModel } from "../ngram/model.ts";

/** Cache the derived context by ARRAY IDENTITY: the predictor passes the same
 *  array object for every candidate, so this turns a per-candidate transform into
 *  one transform per prediction. */
class CtxMemo {
  private lastIn: string[] | null = null;
  private lastOut: string[] = [];
  get(context: string[], make: (c: string[]) => string[]): string[] {
    if (context === this.lastIn) return this.lastOut;
    this.lastIn = context;
    this.lastOut = make(context);
    return this.lastOut;
  }
}

/**
 * Presents a lowercase, sentence-local model (the n-gram + vault side) to the
 * engine's cross-sentence context. Truncating at the last SOS marker reproduces
 * exactly the per-sentence context the n-gram counts were built from - the LSTM's
 * wider view must not leak into it.
 */
export class LowercaseModel implements LanguageModel {
  private inner: LanguageModel;
  private memo = new CtxMemo();

  constructor(inner: LanguageModel) {
    this.inner = inner;
  }

  private ctxFor(context: string[]): string[] {
    return this.memo.get(context, (c) => {
      // Back to the last sentence terminator: the n-gram's counts are per-sentence,
      // so the LSTM's wider cross-sentence view must not leak into it.
      let start = 0;
      for (let i = c.length - 1; i >= 0; i--) {
        if (SENTENCE_END.has(c[i])) { start = i + 1; break; }
      }
      const out: string[] = [];
      for (let i = start; i < c.length; i++) {
        const n = normalizeWord(c[i]); // drops punctuation: n-gram is word-only
        if (n) out.push(n);
      }
      return out;
    });
  }

  logProb(word: string, context: string[]): number {
    return this.inner.logProb(normalizeWord(word), this.ctxFor(context));
  }
  predict(context: string[], k: number): Scored[] {
    return this.inner.predict(this.ctxFor(context), k);
  }
  hasWord(word: string): boolean {
    return this.inner.hasWord(normalizeWord(word));
  }
  vocabulary(): IterableIterator<string> {
    return this.inner.vocabulary();
  }
  size() {
    return this.inner.size();
  }
}
