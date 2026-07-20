/**
 * Mobile-style autocorrect decision (pure). On a word boundary we ask whether
 * the just-typed token should be replaced. Sources of a better answer, in order
 * of strength:
 *   1. split/join fix        ("alot" -> "a lot", "thebank" -> "the bank")
 *   2. real-word alternative  ("form" -> "from" when context demands it)
 *   3. fuzzy typo correction  ("markt" -> "market")
 *
 * A confidence margin protects legitimate words. Ctrl-Z revert uses RevertBuffer.
 */
import { GeometryCostModel, type CostModel } from "../channel/costModel.ts";
import { type ChannelConfig, DEFAULT_CHANNEL } from "../channel/keyboard.ts";
import { phoneticCost } from "../channel/phonetic.ts";
import { structuralEdit } from "../channel/editDistance.ts";
import type { LanguageModel } from "../ngram/model.ts";
import { type Candidate, type CandidateIndex, predict } from "../predict/predictor.ts";
import type { Reranker } from "../predict/reranker.ts";
import { RealWordCorrector, sequenceLogProb, suggestSplit } from "../predict/segmentation.ts";
import { fixContraction } from "../text/caseFix.ts";

export interface AutocorrectConfig {
  /**
   * Information-gain gate (nats): the SOLE strength control. We replace the typed
   * word only if its Shannon surprisal exceeds the best candidate's by at least
   * this much (see {@link informationGain}). Small → correct aggressively; large
   * → only when the typed word is very unlikely vs. the chosen word.
   */
  infoGainThreshold?: number;
  learnList?: Set<string>;
  beta?: number;
  channel?: ChannelConfig;
  costModel?: CostModel;
  reranker?: Reranker;
  realWord?: RealWordCorrector;
  enableSplit?: boolean;
  /** keyboard-geometry channel weight (higher = keyboard typos cheaper). */
  fuzzyStrength?: number;
  /** phonetic channel weight (higher = sound-alike typos cheaper). */
  phoneticStrength?: number;
  /**
   * Use the two-regime decision: a non-word typo is priced by the LM's genuine
   * unknown-word floor and judged only against structurally-plausible corrections, so
   * obvious typos correct reliably (default true). Set false for the legacy single-regime
   * behaviour (typed prior = the mixture's own probability, which inflates a typo via the
   * neural <unk> mass). Exposed mainly for A/B evaluation.
   */
  nonWordRegime?: boolean;
  /**
   * "Is this a real, but rare, English word that the language model's 120k vocab does
   * not contain?" (the bundled word-oracle ∪ morphological stem). Such a word is a
   * legitimate token the LM simply cannot score - so we must not treat it as a typo and
   * "correct" it to a near neighbour ("debitor"→"debtor", "reallocates"→"reallocate").
   * When this returns true for the typed token we leave it EXACTLY as written. The oracle
   * never enters candidate generation - it only protects. Words IN the 120k vocab are
   * handled by the normal real-word regime (they keep their true contextual prior), so
   * this predicate is consulted only for out-of-vocab tokens.
   */
  oracleReal?: (word: string) => boolean;
  /**
   * "Is this token a real word of the LANGUAGE?" - the lexicality test that picks the
   * decision regime. It must be answered by a CURATED lexicon (the bundled model vocab ∪
   * word-oracle ∪ the user's own dictionary), never by whatever text happens to be around.
   *
   * This exists because `model.hasWord` cannot answer it. The live model mixes the bundled
   * vocabulary with a personal n-gram built from the user's vault, so any misspelling that
   * appears a couple of times in their own notes becomes "a word the model knows" - which put
   * it in the real-word regime, gave it a genuine contextual prior, and made it permanently
   * uncorrectable. A typo that protects itself the moment you repeat it is exactly backwards,
   * so lexicality is decided here and vault frequency is left to do what it should: inform the
   * PRIOR, not confer wordhood. Defaults to `model.hasWord` when not supplied.
   */
  isRealWord?: (word: string) => boolean;
}

export const DEFAULT_AUTOCORRECT: AutocorrectConfig = { infoGainThreshold: 2.5 };

/**
 * Prior we assign to a NON-WORD you typed, as a candidate "intended word". A random typo
 * is far less likely to be what you meant than any real rare word, so it must NOT inherit
 * the neural model's <unk> mass (which stands for rare real words). log(1e-9) matches the
 * n-gram's own unknown-word floor (DEFAULT_BLEND.unkFloor), i.e. the LM's genuine estimate
 * that a novel token appears here. Using it is what lets an obvious typo clear the gate.
 */
const LOG_OOV_TYPO_PRIOR = Math.log(1e-9);

/**
 * How many edit operations a genuine typo of an `len`-character word may plausibly contain.
 * Short words must be near-exact (an unrelated 2-edit "correction" of a 3-letter non-word is
 * not a typo fix); longer words tolerate proportionally more slips. Keeps the non-word regime
 * from "correcting" intentional gibberish or jargon to a far-off real word.
 */
function editBudget(len: number): number {
  return Math.max(1, Math.round(len / 4));
}

/**
 * Shannon information-gain of correcting `typed` → the most likely intended word.
 *
 * Given the posterior distribution over intended words P(w | typed, context) ∝
 * P(w | context)·P(typed | w) (the noisy channel), the surprisal (self-information)
 * of an outcome is −log P(w). We compare the surprisal of what was typed against
 * the surprisal of the distribution's mode:
 *
 *     ΔI = I(typed) − I(best) = −log P(typed) − (−log P(best)) = log P(best) − log P(typed)
 *
 * The softmax normaliser Z cancels in the difference, so ΔI is just the gap in
 * log-posterior scores. ΔI ≈ 0 means the typed word is about as likely as the
 * best option (leave it alone); a large ΔI means the typed word is very unlikely
 * compared to the chosen word (correct it). A flat/high-entropy distribution
 * (many plausible options) shrinks P(best), which naturally shrinks ΔI, so we
 * stay cautious exactly when the correction is ambiguous.
 */
export interface InfoGain {
  deltaI: number; // surprisal(typed) − surprisal(best), in nats
  best: string;
  pBest: number;
  pTyped: number;
  entropy: number; // entropy of the posterior, in nats
}

export function informationGain(
  cands: { word: string; logPost: number }[],
  typed: string,
  typedLogPost: number,
): InfoGain | null {
  if (cands.length === 0) return null;
  let best = cands[0];
  for (const c of cands) if (c.logPost > best.logPost) best = c;

  // Full posterior (candidates ∪ typed) so the surprisals are proper log-probs.
  const logPosts = new Map<string, number>();
  for (const c of cands) {
    const prev = logPosts.get(c.word);
    if (prev === undefined || c.logPost > prev) logPosts.set(c.word, c.logPost);
  }
  const prevTyped = logPosts.get(typed);
  if (prevTyped === undefined || typedLogPost > prevTyped) logPosts.set(typed, typedLogPost);

  let max = -Infinity;
  for (const v of logPosts.values()) if (v > max) max = v;
  let z = 0;
  for (const v of logPosts.values()) z += Math.exp(v - max);
  const logZ = max + Math.log(z);
  let entropy = 0;
  for (const v of logPosts.values()) {
    const p = Math.exp(v - logZ);
    if (p > 0) entropy -= p * Math.log(p);
  }
  const pBest = Math.exp(best.logPost - logZ);
  const pTyped = Math.exp(typedLogPost - logZ);
  // ΔI = log P(best) − log P(typed); the logZ cancels.
  return { deltaI: best.logPost - typedLogPost, best: best.word, pBest, pTyped, entropy };
}

/**
 * How much THIS POSITION - as opposed to the words' general frequency - favours `candidate`
 * over `typed`, in nats:
 *
 *     [log P(cand | context) − log P(typed | context)] − [log P(cand) − log P(typed)]
 *
 * the difference of the two words' pointwise mutual information with the context. Positive
 * means the context genuinely points at the candidate; zero means the preference is pure
 * frequency (and with an empty context it is exactly zero, by construction). Used to hold
 * real-word substitutions to contextual evidence rather than to a popularity contest.
 */
export function contextualEvidence(
  model: LanguageModel,
  typed: string,
  candidate: string,
  context: string[],
): number {
  const inContext = model.logProb(candidate, context) - model.logProb(typed, context);
  const contextFree = model.logProb(candidate, []) - model.logProb(typed, []);
  return inContext - contextFree;
}

export interface CorrectionDecision {
  correct: boolean;
  from: string;
  to: string;
  reason: string;
}

export function decideCorrection(
  model: LanguageModel,
  index: CandidateIndex,
  typedToken: string,
  context: string[],
  cfg: AutocorrectConfig = DEFAULT_AUTOCORRECT,
): CorrectionDecision {
  const typed = typedToken.toLowerCase();
  const channel = cfg.channel ?? DEFAULT_CHANNEL;
  const cm = cfg.costModel ?? new GeometryCostModel(channel);
  const beta = cfg.beta ?? 1.0;
  const no = (reason: string): CorrectionDecision => ({
    correct: false,
    from: typedToken,
    to: typedToken,
    reason,
  });
  const yes = (to: string, reason: string): CorrectionDecision => ({
    correct: true,
    from: typedToken,
    to: matchCase(typedToken, to),
    reason,
  });

  if (!typed || typed.length < 2) return no("too-short");
  if (cfg.learnList?.has(typed)) return no("learn-listed");

  // Lexicality: decided by a curated lexicon, NOT by what the surrounding vault contains.
  // Everything below branches on this rather than on model.hasWord (see AutocorrectConfig).
  const isReal = cfg.isRealWord
    ? cfg.isRealWord(typed)
    : model.hasWord(typed) || (cfg.oracleReal?.(typed) ?? false);

  // An all-caps token is an acronym/initialism (SEC, NASA, USA, HTTP) or shouting,
  // not a misspelling of a lowercase word - never "correct" it (SEC → SEE).
  if (typedToken.length >= 2 && typedToken === typedToken.toUpperCase() && /[A-Z]/.test(typedToken))
    return no("acronym");

  // 0. contraction failsafe (deterministic): "dont" -> "don't".
  const contraction = fixContraction(typed);
  if (contraction) return yes(contraction, "contraction");

  // 0b. Real-but-rare word protection. If the typed token is a legitimate English word the
  // LM's 120k vocab simply doesn't contain (bundled word-oracle ∪ morphological stem), it is
  // NOT a typo: leave it exactly as written and skip candidate generation entirely. Words IN
  // the 120k vocab keep their true prior and fall through to the real-word regime below, so
  // this only fires for out-of-vocab tokens (checked first because it's the cheap common miss).
  if (isReal && !model.hasWord(typed)) return no("real-word-rare");

  // 1. split/join fix.
  if (cfg.enableSplit) {
    const split = suggestSplit(typed, model, context);
    if (split) return yes(split.join(" "), "split");
  }

  // 2+3. Unified correction: keyboard/phonetic typo candidates AND real-word
  // homophone confusions ("form"→"from") compete in ONE information-gain
  // decision, so the globally best explanation wins rather than whichever path
  // fires first. (Previously a rare in-vocab word like "worf" was grabbed by the
  // real-word path as "wharf" before the keyboard fix "word" was ever considered.)
  // Pull a WIDE pool: with weak context the right word can rank below near-misses.
  const cands = predict(model, index, {
    context,
    typed,
    k: 32,
    mode: "full",
    beta,
    channel,
    costModel: cm,
    reranker: cfg.reranker,
    fuzzyStrength: cfg.fuzzyStrength,
    phoneticStrength: cfg.phoneticStrength,
  });

  // Add the best homophone peer (their/there, form/from, to/too - pairs the fuzzy
  // generator might not reach) as another candidate, priced by its phonetic
  // confusion cost, then let the same posterior rank it against the rest.
  if (cfg.realWord && isReal) {
    const peer = cfg.realWord.bestAlternative(typed, context, model, -Infinity);
    if (peer && peer !== typed && !cands.some((c) => c.word === peer)) {
      cands.push({
        word: peer,
        score: 0,
        logProbLM: model.logProb(peer, context),
        channelCost: phoneticCost(typed, peer),
      });
    }
  }
  if (cands.length === 0) return no("no-candidate");

  // Posterior log-score of an intended word: log P(w|context) + log P(typed|w).
  // The channel cost is the −log P(typed|w) term (keyboard geometry OR phonetic
  // distortion, whichever explains the typo better); typed→itself has cost 0.
  // A far-fetched correction pays a large channel cost, so it loses the posterior
  // on its own - no separate plausibility cutoff needed.
  const logPost = (c: Candidate) => c.logProbLM - beta * c.channelCost;

  // Two regimes, split on whether what you typed is a REAL word. This is the fix for
  // "obvious typos sometimes don't correct": the neural model maps every out-of-vocab
  // string to a single <unk> token, so the mixture gave a *typo* the probability mass of
  // "some rare real word" - inflating its own prior and pushing clear fixes under the
  // information-gain gate. A non-word has no business as an intended word, so we price it
  // by the language model's genuine unknown-word floor and judge it only against
  // STRUCTURALLY PLAUSIBLE corrections (a believable slip, not a stretch to an unrelated
  // word). A real word keeps its true contextual prior, so overriding it still needs the
  // strong contextual evidence the gate demands (their→there, form→from).
  let pool = cands;
  let typedLogPost: number;
  if (isReal || cfg.nonWordRegime === false) {
    typedLogPost = model.logProb(typed, context);
  } else {
    const budget = editBudget(typed.length);
    pool = cands.filter((c) => c.word === typed || structuralEdit(typed, c.word, budget) <= budget);
    typedLogPost = LOG_OOV_TYPO_PRIOR;
  }

  const info = informationGain(
    pool.map((c) => ({ word: c.word, logPost: logPost(c) })),
    typed,
    typedLogPost,
  );
  if (!info || info.best === typed) return no("already-best");

  // Information-gain decision (kept as the single strength control): substitute iff the
  // typed word is enough more surprising than the best candidate. For a non-word the floor
  // above makes a plausible typo clear the gate reliably; for a real word its true prior
  // keeps the bar high - the threshold still tunes eagerness in both regimes.
  const igThreshold = cfg.infoGainThreshold ?? DEFAULT_AUTOCORRECT.infoGainThreshold!;
  if (info.deltaI < igThreshold) return no("insufficient-info-gain");

  // Replacing a REAL word additionally requires that CONTEXT is what favours the alternative.
  // The information gain alone cannot tell the two sources of evidence apart: at a sentence
  // start there is no context at all, so the comparison collapses to raw frequency and the
  // commoner word wins on nothing but being commoner - which is how a correctly-typed "Thin"
  // became "Then". Subtracting the context-free log-ratio leaves exactly the part of the
  // evidence that this position contributes, and we require the correction to be favoured by
  // it. A typo needs no such test: there the question is which word was meant, not whether a
  // word was wrong.
  if (isReal && contextualEvidence(model, typed, info.best, context) <= 0)
    return no("no-contextual-evidence");

  return yes(info.best, "corrected");
}

/**
 * How much more probable (nats) a re-spaced reading must be than the arrangement as typed
 * before we act on it. Measured on the real model, the "advantage"
 * logP(ab) − [logP(a)+logP(b|a)] cleanly separates genuine fragment-splits ("th e"→"the",
 * "wor d"→"word": advantage ≥ ~11) from deliberately-spaced word pairs ("to day", "in to",
 * "so on": advantage ≤ ~3). 8.0 sits in that gap with margin on both sides, biased toward NOT
 * merging (a wrongly-removed space is more disruptive than a missed one). Contractions are
 * handled separately (they need no margin - a curated join like "haven t"→"haven't" is safe).
 */
const MERGE_MARGIN = 8.0;

/** Longest pair of adjacent tokens we will try to re-space. Beyond this the split search is
 *  not worth its language-model calls, and a genuine slip is never this long. */
const MAX_RESPACE_LENGTH = 24;

/**
 * A single-letter piece is only ever a word in English as "a" or "i". Any other lone letter is
 * a fragment, so an arrangement containing one is not a legitimate reading ("you d", "haven t")
 * even though the model happily assigns those letters probabilities as tokens.
 */
function isFragment(piece: string): boolean {
  return piece.length === 1 && piece !== "a" && piece !== "i";
}

/**
 * Fix a MISPLACED space between the previous token and the one just completed.
 *
 * The space bar can go wrong three ways, and they are the same error: the boundary landed in
 * the wrong place. Joining ("th e"→"the", "haven t"→"haven't") is only the special case where
 * the right number of boundaries is zero; "int he"→"in the" needs the boundary MOVED, which no
 * amount of joining or splitting alone can express. So we treat the two tokens as one character
 * run and ask the language model which arrangement of it - whole, or split at any point - reads
 * best, judging every option on equal terms.
 *
 * The as-typed arrangement is the incumbent and only loses by MERGE_MARGIN, so deliberately
 * spaced pairs ("to day", "a lot") stay exactly as written. Arrangements containing a
 * non-word or a stray single letter are not readings at all and never compete.
 */
export function decideRespace(
  model: LanguageModel,
  prev: string,
  cur: string,
  context: string[],
): { to: string } | null {
  const a = prev.toLowerCase();
  const b = cur.toLowerCase();
  if (a.length < 1 || b.length < 1) return null;
  const joined = a + b;
  if (joined.length > MAX_RESPACE_LENGTH) return null;
  // Contraction failsafe: "havent"→"haven't", "youd"→"you'd". A curated join, so no margin needed.
  const contr = fixContraction(joined);
  if (contr) return { to: contr };

  // Every arrangement is scored the same way - jointly, in context - so readings with different
  // piece counts stay comparable. Shared with suggestSplit for exactly that reason.
  const score = (pieces: string[]): number => sequenceLogProb(model, pieces, context);

  const typedScore = score([a, b]);
  let best: string[] | null = null;
  let bestScore = -Infinity;
  const consider = (pieces: string[]): void => {
    // Cheap structural rejects first, so the model is only asked about real readings.
    if (pieces.some((p) => isFragment(p) || !model.hasWord(p))) return;
    const s = score(pieces);
    if (s > bestScore) {
      bestScore = s;
      best = pieces;
    }
  };

  consider([joined]);
  for (let i = 1; i < joined.length; i++) {
    if (i === a.length) continue; // the as-typed arrangement: it is the incumbent, not a rival
    consider([joined.slice(0, i), joined.slice(i)]);
  }

  if (!best || bestScore <= typedScore + MERGE_MARGIN) return null;
  return { to: (best as string[]).join(" ") };
}

export function matchCase(original: string, replacement: string): string {
  if (original.length === 0) return replacement;
  if (original === original.toUpperCase() && original.length > 1)
    return replacement.toUpperCase();
  if (original[0] === original[0].toUpperCase())
    return replacement[0].toUpperCase() + replacement.slice(1);
  return replacement;
}

export class RevertBuffer {
  private last: { from: string; to: string } | null = null;
  record(from: string, to: string): void {
    this.last = { from, to };
  }
  revert(): { from: string; to: string } | null {
    const l = this.last;
    this.last = null;
    return l;
  }
  clear(): void {
    this.last = null;
  }
}
