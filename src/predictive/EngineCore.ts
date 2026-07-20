/**
 * The prediction engine, with ZERO Obsidian dependencies - so the exact same code
 * runs either inside the inference Web Worker (the normal case) or inline on the
 * main thread (fallback when Workers are unavailable).
 *
 * Everything that touches the vault or the filesystem stays in
 * PredictiveEngineController; this class only ever sees plain data (text, counts,
 * ArrayBuffers, JSON), which is exactly what can cross a postMessage boundary.
 *
 * It also owns the Personalization state, because inference READS it (reranker
 * orders candidates, confusion model is the channel cost) and the learning hooks
 * MUTATE it. Keeping it here avoids shipping it back and forth per keystroke; the
 * main thread pulls a JSON snapshot only when it's time to persist.
 */
import {
  buildCountsFromText,
  buildSerializedCountsFromDocs,
  IncrementalCorpus,
  InMemoryLanguageModel,
  packCounts,
  PackedLanguageModel,
  MixtureLanguageModel,
  CacheLanguageModel,
  FuzzyTrie,
  RealWordCorrector,
  predict,
  decideCorrection,
  decideRespace,
  tokenizeWords,
  isProfane,
  evaluate,
  LstmLanguageModel,
  LowercaseModel,
  Personalization,
  WordOracle,
  DEFAULT_CHANNEL,
  type Candidate,
  type CorrectionDecision,
  type LanguageModel,
  type ChannelConfig,
  type NgramCounts,
  type EvalResult,
  type SerializedCounts,
  type PersonalizationState,
} from "./engine/index";
// Type-only: erased at build time, so the worker bundle never pulls in the
// settings TAB (which imports obsidian).
import type { PredictiveSettings } from "./PredictiveSettings";

export interface SuggestItem {
  /** text inserted on accept. */
  insert: string;
  /** text shown in the popup. */
  display: string;
  /** "dictionary" = a user's personal-dictionary word: inserted VERBATIM, never re-cased. */
  kind: "word" | "correction" | "phrase" | "dictionary";
  score: number;
}

/** A vault file's text, as handed over from the main thread. */
export interface DocEntry {
  path: string;
  text: string;
}

/** Hard cap on phrase length. The expected-keystrokes-saved ranking and the sentence
 *  boundary in phraseCandidates decide the real length; this is just a compute budget, which
 *  is why it is a constant rather than a user setting. 4 words is plenty in practice (the
 *  utility argmax rarely favours a longer phrase) and each extra word is another whole beam of
 *  the expensive f32 recurrent steps. */
const MAX_PHRASE_WORDS = 4;

/**
 * Choose which candidates to show and in what order so the WHOLE menu maximises expected
 * keystrokes saved, charging each slot its Down-arrow reach cost: the item in slot `j` needs
 * `j` extra key presses to accept, so its realised saving is `saved - j` and the menu's
 * expected return is Σ_i P_i·(saved_i - pos_i). For any fixed displayed set the optimal order
 * is by P descending (exchange argument: swapping an adjacent higher-P item earlier always
 * helps, independent of the savings), so we sort by P descending and run a DP over the number
 * of items placed so far (= the next slot index) that decides include/skip per candidate.
 * O(N·K), exact. Items whose saving cannot cover their slot cost are dropped, which also caps
 * the menu length on its own.
 */
function orderMenu<T extends { p: number; saved: number }>(cands: T[], k: number): T[] {
  const items = cands.slice().sort((a, b) => b.p - a.p);
  const n = items.length;
  const K = Math.min(k, n);
  if (K === 0) return [];
  const NEG = -Infinity;
  const gain = (i: number, slot: number) => items[i].p * (items[i].saved - slot);
  const hist: number[][] = []; // hist[i] = dp state BEFORE processing item i
  let dp = new Array<number>(K + 1).fill(NEG);
  dp[0] = 0;
  for (let i = 0; i < n; i++) {
    hist.push(dp.slice());
    const ndp = dp.slice(); // skip item i: state carries over unchanged
    for (let m = 0; m < K; m++) {
      if (dp[m] === NEG) continue;
      const val = dp[m] + gain(i, m); // include item i at slot m
      if (val > ndp[m + 1]) ndp[m + 1] = val;
    }
    dp = ndp;
  }
  let bestM = 0;
  for (let m = 1; m <= K; m++) if (dp[m] > dp[bestM]) bestM = m;
  // Backtrack against the saved history: item i was included iff the after-state at count m
  // equals the before-state at count m-1 plus its slot gain.
  const chosen: T[] = [];
  let m = bestM;
  let after = dp;
  const eq = (a: number, b: number) => Math.abs(a - b) <= 1e-9 * (1 + Math.abs(a));
  for (let i = n - 1; i >= 0; i--) {
    const before = hist[i];
    if (m > 0 && before[m - 1] !== NEG && eq(after[m], before[m - 1] + gain(i, m - 1))) {
      chosen.push(items[i]);
      m--;
    }
    after = before;
  }
  return chosen.reverse();
}

export class EngineCore {
  private settings: PredictiveSettings;
  private personalization: Personalization;
  private globalModel: LanguageModel | null = null;
  private globalCounts: NgramCounts | null = null;
  private personal: LanguageModel | null = null;
  private corpus: IncrementalCorpus | null = null;
  private base: MixtureLanguageModel | null = null;
  private cache: CacheLanguageModel | null = null;
  /** mid-word scoring model: n-gram mixed with the LSTM context prior. */
  private midModel: LanguageModel | null = null;
  private index: FuzzyTrie | null = null;
  private realWord: RealWordCorrector | null = null;
  private channel: ChannelConfig = DEFAULT_CHANNEL;
  private lastShown: Candidate[] = [];
  /** The LSTM. Its vocab is LOWERCASE, so it plugs straight into the lowercase
   *  pipeline with no adapter - capitalisation is a separate head (renderCased),
   *  not a property of the token. */
  private lstm: LstmLanguageModel | null = null;
  /** Membership oracle for real-but-rare words outside the 120k LM vocab (protect-only;
   *  never feeds prediction/suggestion/candidate generation). Empty until wordlist.bin loads. */
  private wordOracle: WordOracle = WordOracle.empty();

  /** Lower-cased personal-dictionary words, which the profanity filter must never block -
   *  adding a word to your personal dictionary is how you opt it back in. Rebuilt whenever
   *  settings change (cheap; a handful of words). */
  private profAllow: ReadonlySet<string> = new Set();

  /** Lower-cased personal-dictionary words, as LEXICON: words the user has declared real. The
   *  same list as profAllow, used for a different question - "is this a word?" rather than
   *  "may we show it?" - so the two are kept separate even though they are rebuilt together. */
  private lexicon: ReadonlySet<string> = new Set();

  constructor(settings: PredictiveSettings, personalization: PersonalizationState | null) {
    this.settings = settings;
    this.personalization = personalization
      ? new Personalization(personalization, settings.beta)
      : Personalization.empty(settings.beta);
    this.applyChannel();
    this.rebuildProfanityAllow();
  }

  private rebuildProfanityAllow(): void {
    const words = new Set((this.settings.userDictionary ?? []).map((w) => w.toLowerCase()));
    this.profAllow = words;
    this.lexicon = words;
  }

  /** True if the profanity filter is on and any word of `surface` (a single word OR a
   *  multi-word phrase) is blocklisted and not in the personal dictionary. Used to keep
   *  blocked words out of OUR suggestions and out of autocorrect targets - never to touch
   *  what the user typed. */
  private blockedSurface(surface: string): boolean {
    if (!this.settings.filterProfanity) return false;
    for (const w of tokenizeWords(surface)) if (isProfane(w, this.profAllow)) return true;
    return false;
  }

  get ready(): boolean {
    return this.cache !== null && this.index !== null;
  }

  /** True when the LSTM is running on the WASM SIMD kernel. */
  get accelerated(): boolean {
    return this.lstm?.accelerated ?? false;
  }

  /** Whether a neural model is loaded at all. Lets the settings UI tell "no model
   *  installed" apart from "model loaded but on the slow scalar path" - otherwise
   *  both look like accelerated === false and the status line would lie. */
  get lstmLoaded(): boolean {
    return this.lstm !== null;
  }

  /** Single source for the status RPC - called by the worker entry AND, when there is
   *  no worker, inline via InferenceClient.call. */
  status(): { ready: boolean; accelerated: boolean; lstmLoaded: boolean } {
    return { ready: this.ready, accelerated: this.accelerated, lstmLoaded: this.lstmLoaded };
  }

  /** Semantic topic fingerprint of a passage (see LstmLanguageModel.embed). Null when
   *  the neural model isn't loaded - the related-link index then falls back to keywords
   *  only. Returned as a plain array so it survives the worker boundary cheaply. */
  embed(text: string): number[] | null {
    if (!this.lstm) return null;
    return Array.from(this.lstm.embed(text));
  }

  /** Embedding dimensionality, or 0 when no neural model is loaded. */
  embedDim(): number {
    return this.lstm ? this.lstm.embeddingDim : 0;
  }

  /** Per-word descriptiveness scores (see LstmLanguageModel.rarity), for niche tag bias.
   *  Falls back to 0.5 for every word when no neural model is loaded. */
  rarities(words: string[]): number[] {
    if (!this.lstm) return words.map(() => 0.5);
    return words.map((w) => this.lstm!.rarity(w));
  }

  private p() {
    return this.personalization;
  }

  private blendCfg() {
    return { unkFloor: 1e-9, wbGuard: 1e-9, useContinuation: this.settings.useContinuation };
  }

  private applyChannel(): void {
    this.channel = {
      ...DEFAULT_CHANNEL,
      sigma: this.settings.channelSigma,
      layout: this.settings.keyboardLayout,
    };
    this.personalization.confusion.setChannel(this.channel);
  }

  // --- model loading ------------------------------------------------------

  loadGlobalText(corpusText: string | null): void {
    if (corpusText) {
      this.globalCounts = buildCountsFromText(corpusText, { blend: this.blendCfg() });
      this.globalModel = new InMemoryLanguageModel(this.globalCounts, this.blendCfg());
    } else {
      this.globalModel = null;
      this.globalCounts = null;
    }
    this.rebuild();
  }

  loadGlobalPacked(buffer: ArrayBuffer): void {
    this.globalModel = PackedLanguageModel.fromBuffer(buffer);
    this.globalCounts = null; // packed model can't be re-packed
    this.rebuild();
  }

  loadLstm(buffer: ArrayBuffer): boolean {
    try {
      this.lstm = LstmLanguageModel.fromBuffer(buffer, this.settings.wasmSimd ?? true);
    } catch (e) {
      console.warn("[predictive] failed to load LSTM model", e);
      this.lstm = null;
    }
    this.rebuild();
    return this.lstm !== null;
  }

  /** Load the real-word oracle (wordlist.bin). Absent asset = empty oracle (no protection). */
  loadWordlist(buffer: ArrayBuffer): void {
    this.wordOracle = WordOracle.fromBuffer(buffer);
  }

  /**
   * Is `word` already "known" to the predictor - i.e. in the 120k LM vocab OR the user's
   * personal additions? This is the scope for "don't re-add a word that already exists":
   * such a word is already featured (predictable / suggestable), so adding it is redundant.
   * NOTE: deliberately does NOT consult the big oracle - an oracle-only word (e.g. "debitor")
   * is a real word we won't CORRECT, but it is not yet featured, so the user is still allowed
   * to add it to their personal dictionary to make it suggestable.
   */
  isKnownWord(word: string): boolean {
    return this.midModel?.hasWord(word.toLowerCase()) ?? false;
  }

  /**
   * Is `word` a real word of the LANGUAGE? The lexicality test behind the autocorrect regime,
   * answered ONLY from curated sources: the bundled model vocabulary, the bundled word oracle
   * (∪ its morphological stems), and the user's own dictionary.
   *
   * Deliberately NOT `midModel.hasWord`. That mixes in the personal n-gram built from the
   * vault, where any misspelling occurring twice becomes "known" - which silently promoted the
   * user's own typos to real words and made them uncorrectable forever after ("pruchrase",
   * "morehesitant", "thje"). Vault frequency is evidence about which words are LIKELY here; it
   * is not evidence that a string is a word, so it informs the prior and nothing else.
   *
   * The vault does get ONE vote, but on the right evidence: a word used in several DIFFERENT
   * notes (document frequency, not raw count) is part of the user's working vocabulary - their
   * project names, jargon and proper nouns - and must keep its protection. A string repeated
   * inside a single note is still just one mistake, which is precisely the case that broke.
   */
  private isRealWord(word: string): boolean {
    const w = word.toLowerCase();
    if (this.globalModel?.hasWord(w)) return true;
    if (this.lexicon.has(w)) return true;
    if (this.wordOracle.has(w)) return true;
    if (this.vaultLexical(w)) return true;
    return this.wordOracle.stemKnown(w, (x) =>
      (this.globalModel?.hasWord(x) ?? false) || this.lexicon.has(x) || this.wordOracle.has(x),
    );
  }

  /**
   * Minimum number of DISTINCT vault notes a word must appear in before the vault itself is
   * taken as evidence that it is a real word. Two is the smallest number that can distinguish
   * "the user writes this" from "the user mistyped this (possibly more than once) in one note",
   * which is the whole point of using document frequency instead of a count.
   */
  private static readonly VAULT_LEXICAL_MIN_DOCS = 2;

  /** Is `word` established across the user's own notes? (See VAULT_LEXICAL_MIN_DOCS.) */
  private vaultLexical(word: string): boolean {
    if (!this.corpus) return false;
    return this.corpus.documentFrequency(word) >= EngineCore.VAULT_LEXICAL_MIN_DOCS;
  }

  /**
   * How many distinct vault notes contain each of `words` - the signal behind "this word just
   * left the vault entirely", so the personal dictionary can be pruned the instant it happens
   * instead of after a periodic re-read of every note.
   *
   * Returns null when there is no corpus to ask (personal learning off, or not built yet).
   * That is deliberately distinct from an array of zeros: a caller that prunes on this must
   * never mistake "we don't know" for "it's gone".
   */
  documentFrequencies(words: string[]): number[] | null {
    if (!this.corpus) return null;
    return words.map((w) => this.corpus!.documentFrequency(w));
  }

  packGlobal(): ArrayBuffer | null {
    if (!this.globalModel || !this.globalCounts) return null;
    return packCounts(this.globalModel, this.globalCounts, { topK: 24 });
  }

  // --- personal corpus (texts arrive from the main thread) -----------------

  /**
   * Full (re)build from every vault document. Counting + case-mapping the whole
   * vault happens HERE, which is why this call is worth its postMessage: it used
   * to need a second, separate build worker.
   */
  rebuildPersonal(entries: DocEntry[] | null): void {
    if (!this.settings.personalBias || !entries) {
      this.personal = null;
      this.corpus = null;
      this.rebuild();
      return;
    }
    // minCount 2: a word must appear at least twice in the vault to be suggested,
    // so one-off typos never leak into recommendations.
    this.corpus = new IncrementalCorpus(this.blendCfg(), 2);
    for (const e of entries) this.corpus.set(e.path, buildSerializedCountsFromDocs([e.text]));
    this.personal = this.corpus.build();
    this.rebuild();
  }

  /** A single file changed: re-count just that file and patch the corpus. */
  setFile(path: string, text: string): boolean {
    if (!this.settings.personalBias) return true;
    if (!this.corpus) return false; // caller must do a full rebuild
    const counts: SerializedCounts = buildSerializedCountsFromDocs([text]);
    this.corpus.set(path, counts);
    this.personal = this.corpus.build();
    this.rebuild();
    return true;
  }

  removeFile(path: string): void {
    if (!this.corpus) return;
    this.corpus.remove(path);
    this.personal = this.corpus.build();
    this.rebuild();
  }

  renameFile(oldPath: string, newPath: string): void {
    this.corpus?.rename(oldPath, newPath);
  }

  /**
   * The capitalisation this context calls for, or null when we have no opinion.
   * Contextual, straight from the LSTM's case head - this is what replaced the
   * static CaseMap: "paris" -> "Paris", but "i will polish it" stays lowercase
   * because the model has actually seen both usages, and "nasa" -> "NASA" because
   * the head carries a per-word case bias.
   */
  caseFor(word: string, context: string[]): string | null {
    // Autocorrect REWRITES the user's typed word, so gate on confidence: an ambiguous
    // homograph typed lowercase ("to polish") is left alone, only a confident proper
    // noun ("paris" -> "Paris") is re-cased. (Display casing uses renderCased directly.)
    return this.lstm ? this.lstm.casedConfident(word, context) : null;
  }

  setActiveDocument(text: string): void {
    this.cache?.setDocument(tokenizeWords(text));
  }

  updateSettings(settings: PredictiveSettings): void {
    this.settings = settings;
    this.applyChannel();
    this.rebuildProfanityAllow();
    if (this.base) this.base.setAlpha(settings.personalBias ? settings.alpha : 0);
    if (this.cache) this.cache.setGamma(settings.cacheGamma);
    // Live-adjust the LSTM<->n-gram blend without a full rebuild.
    if (this.lstm && this.midModel instanceof MixtureLanguageModel)
      this.midModel.setAlpha(settings.lstmWeight);
  }

  private rebuild(): void {
    if (!this.globalModel && !this.personal) {
      this.base = null;
      this.cache = null;
      this.index = null;
      this.realWord = null;
      this.midModel = null;
      return;
    }
    const baseModel = this.globalModel ?? this.personal!;
    const personal = this.globalModel ? this.personal : null;
    this.base = new MixtureLanguageModel(
      baseModel,
      personal,
      this.settings.personalBias ? this.settings.alpha : 0,
    );
    this.cache = new CacheLanguageModel(this.base, this.settings.cacheGamma);
    // The pipeline's context is CASED with sentence markers. The n-gram side gets
    // it lowercased + truncated to the current sentence; the LSTM gets the
    // continuous cased stream and marginalises over case variants, so the blend
    // compares like with like.
    const ngram = new LowercaseModel(this.cache);
    this.midModel = this.lstm
      ? new MixtureLanguageModel(ngram, this.lstm, this.settings.lstmWeight)
      : ngram;
    this.index = new FuzzyTrie(this.base.vocabulary());
    this.realWord = new RealWordCorrector(this.base.vocabulary());
  }

  // --- inference ----------------------------------------------------------

  runEvaluation(text: string): EvalResult | null {
    if (!this.cache || !this.index) return null;
    return evaluate(this.cache, this.index, text, {
      beta: this.settings.beta,
      costModel: this.settings.adaptiveKeyboard ? this.p().confusion : undefined,
      layout: this.settings.keyboardLayout,
    });
  }

  getCandidates(context: string[], typed: string, k: number): Candidate[] {
    if (!this.midModel || !this.index) return [];
    const cands = predict(this.midModel, this.index, {
      context,
      typed,
      k,
      beta: this.settings.beta,
      channel: this.channel,
      mode: "prefix",
      maxCost: this.settings.maxEditCost,
      costModel: this.settings.adaptiveKeyboard ? this.p().confusion : undefined,
      reranker: this.settings.learnedRanking ? this.p().reranker : undefined,
      fuzzyStrength: this.settings.fuzzyStrength,
      phoneticStrength: this.settings.phoneticStrength,
    });
    this.lastShown = cands;
    return cands;
  }

  getSuggestions(context: string[], typed: string, k: number, includePhrases = true): SuggestItem[] {
    if (!this.cache) return [];
    let seeds: { word: string; score: number }[];
    if (typed) {
      seeds = this.getCandidates(context, typed, k).map((c) => ({ word: c.word, score: c.score }));
    } else {
      const last = context[context.length - 1];
      seeds = this.getCandidates(context, "", k + 1)
        .filter((c) => c.word !== last)
        .slice(0, k)
        .map((c) => ({ word: c.word, score: c.score }));
    }
    if (seeds.length === 0) return [];

    // Rank every candidate - a single word OR a multi-word phrase - by EXPECTED KEYSTROKES
    // SAVED, then order the whole menu to maximise its expected return (below). P(the user
    // wants a candidate) is a softmax over the reranker/blend scores (times the phrase's real
    // joint continuation probability); the saving is a keystroke count. Nothing here is a
    // tuned threshold.
    const maxScore = Math.max(...seeds.map((s) => s.score));
    const weights = seeds.map((s) => Math.exp(s.score - maxScore));
    const wSum = weights.reduce((a, b) => a + b, 0) || 1;

    const BEAM = 3; // beam width / #seeds to phrase-extend: a compute budget, not a threshold
    const maxWords = MAX_PHRASE_WORDS;
    // Keystrokes saved by accepting `insert` when `typed` is already in the box, at menu
    // position 0. To finish by hand the user types the remaining characters; accepting costs
    // one key. Two adjustments the user asked for:
    //   +1  a small bias so a one-character completion ("productivit"->"productivity") still
    //       shows - accepting is at worst as fast as typing the last letter, and surfacing it
    //       is what people expect. (Nets out to: saved = remaining chars.)
    //   +space  when accepting also auto-inserts the following separator (a suggestion that
    //       ends in punctuation - see PredictiveSuggest.spaceAfter), that is one more key the
    //       user does not press. Word completions do NOT auto-insert a trailing space (the
    //       user types it either way), so they get no such bonus.
    // A candidate already typed out in full saves nothing (remaining <= 0) and is dropped -
    // which is exactly what makes a fully-typed word disappear from the popup.
    const savingOf = (insert: string): number => {
      const remaining = insert.length - typed.length;
      if (remaining <= 0) return 0;
      return remaining + (/[.,!?;:]$/.test(insert) ? 1 : 0);
    };
    // A recommendation that is only punctuation ("...", ",") is never worth a slot.
    const hasAlnum = (s: string): boolean => /[A-Za-z0-9]/.test(s);

    const best = new Map<string, { insert: string; kind: SuggestItem["kind"]; p: number; saved: number }>();
    const consider = (insert: string, kind: SuggestItem["kind"], pWant: number): void => {
      if (!hasAlnum(insert)) return;
      // Never proactively surface a profane/NSFW word (a whole phrase is dropped if any
      // word in it is blocked). This filters OUR suggestions only.
      if (this.blockedSurface(insert)) return;
      const saved = savingOf(insert);
      if (saved <= 0) return;
      const prev = best.get(insert);
      // Keep the higher-probability route to the same surface.
      if (!prev || prev.p < pWant) best.set(insert, { insert, kind, p: pWant, saved });
    };

    seeds.forEach((s, i) => {
      const pSeed = weights[i] / wSum; // P(this first word), among the shown candidates
      if (!this.lstm) {
        consider(s.word, "word", pSeed);
        return;
      }
      // Bare word: each plausible CASE is its own candidate (homograph "polish"/"Polish").
      for (const v of this.lstm.caseVariants(s.word, context)) consider(v, "word", pSeed);
      // Multi-word continuations, each weighted by its real joint extension probability. Only
      // the top few seeds are extended (a compute budget). A phrase is offered ONLY while its
      // first word is still being completed: once the user has typed the whole first word
      // (typed covers the seed) the phrase has served its lookahead purpose and would just
      // re-type what is already there, so it disappears - the continuation belongs to the
      // next-word menu after the space.
      if (includePhrases && maxWords >= 2 && i < BEAM && typed.length < s.word.length) {
        for (const p of this.lstm.phraseCandidates(context, s.word, maxWords, BEAM))
          consider(p.words.join(" "), "phrase", pSeed * Math.exp(p.extLogProb));
      }
    });

    // --- order the whole menu, not each slot in isolation ---------------------------------
    // Accepting the item in slot j costs j extra Down-arrow presses, so its realised saving is
    // (saved_j - j). The menu's expected return is Σ_i P_i·(saved_i - pos_i); we choose the
    // displayed subset AND their order to maximise it. An exchange argument shows that for ANY
    // fixed subset the optimal order is by P descending (pairing the most-likely-wanted item
    // with the cheapest-to-reach slot), independent of the savings. So we sort by P descending
    // and run a small DP over "how many items placed so far" (= the next slot index) that
    // decides include/skip for each - O(N·k), exact. An item whose saving cannot cover its
    // slot cost is simply left out, which also caps the menu length naturally.
    return orderMenu([...best.values()], k).map(({ insert, kind, p, saved }, pos) => ({
      insert,
      display: insert,
      kind,
      score: p * Math.max(0, saved - pos),
    }));
  }

  decide(typed: string, context: string[]): CorrectionDecision {
    if (!this.midModel || !this.index)
      return { correct: false, from: typed, to: typed, reason: "not-ready" };
    // The user typed a profane/NSFW word deliberately - never "correct" it away, even if
    // the model would otherwise replace it (a blocked word is rare, so the model may treat
    // it as a typo). This is separate from filtering: it protects the user's OWN input.
    if (this.settings.filterProfanity && isProfane(typed, this.profAllow))
      return { correct: false, from: typed, to: typed, reason: "profanity-kept" };
    const decision = decideCorrection(this.midModel, this.index, typed, context, {
      infoGainThreshold: this.settings.infoGainThreshold,
      beta: this.settings.beta,
      channel: this.channel,
      costModel: this.settings.adaptiveKeyboard ? this.p().confusion : undefined,
      reranker: this.settings.learnedRanking ? this.p().reranker : undefined,
      realWord: this.settings.realWordCorrection ? this.realWord ?? undefined : undefined,
      enableSplit: this.settings.splitCorrection,
      learnList: this.p().learnList,
      fuzzyStrength: this.settings.fuzzyStrength,
      phoneticStrength: this.settings.phoneticStrength,
      isRealWord: (w) => this.isRealWord(w),
    });
    // Never autocorrect a benign typo INTO a profane/NSFW word (the target can be a
    // split like "a lot", so check the whole surface).
    if (decision.correct && this.blockedSurface(decision.to))
      return { correct: false, from: typed, to: typed, reason: "profanity-target" };
    return decision;
  }

  /** Should an accidental space be removed, merging `prev` with `cur`? (#15/#18) Returns the
   *  joined surface, or null. Gated by the same split-correction setting. */
  mergeDecision(prev: string, cur: string, context: string[]): string | null {
    if (!this.midModel || !this.settings.splitCorrection) return null;
    if (this.settings.filterProfanity && isProfane(prev + cur, this.profAllow)) return null;
    const m = decideRespace(this.midModel, prev, cur, context);
    if (!m) return null;
    if (this.blockedSurface(m.to)) return null;
    return m.to;
  }

  // --- learning hooks -----------------------------------------------------

  recordAccept(insert: string, saved = 0): void {
    const tokens = tokenizeWords(insert);
    const shown = this.lastShown;
    const idx = shown.findIndex((c) => c.word === tokens[0]);
    if (tokens.length === 1 && idx >= 0 && this.settings.learnedRanking) {
      const feats = shown.map((c) => c.features!).filter(Boolean);
      if (feats.length === shown.length) this.p().reranker.learn(feats, idx);
    }
    this.cache?.observe(tokens); // phrases seed the whole span
    this.p().stats.accepts++;
    if (saved > 0) this.p().stats.charsSaved += saved;
  }

  recordCorrection(from: string, to: string): void {
    if (this.settings.adaptiveKeyboard) this.p().confusion.learn(to, from);
    this.p().stats.corrections++;
  }

  recordRevert(original: string): void {
    this.p().learnList.add(original.toLowerCase());
    this.p().stats.reverts++;
  }

  /** Snapshot for persistence (the main thread owns the file). */
  personalizationState(): PersonalizationState {
    return this.personalization.toState();
  }

  /** Replace the learned state (load / import / reset). */
  setPersonalization(state: PersonalizationState | null): void {
    this.personalization = state
      ? new Personalization(state, this.settings.beta)
      : Personalization.empty(this.settings.beta);
    this.applyChannel();
  }
}
