/**
 * WordOracle – a pure "is this a real English word?" membership test, backed by a
 * comprehensive (~274k) sorted wordlist blob (see build_model/build_wordlist.mjs).
 *
 * ROLE (deliberately narrow): the oracle NEVER contributes to prediction, to
 * autocomplete suggestions, or to correction-candidate generation – those all come
 * from the language model's 120k vocab, because a word the LM can't score is useless
 * as a suggestion or a correction target. The oracle answers ONE question, used only
 * to PROTECT a real word from being "corrected" and to keep the personal-dictionary
 * flow from offering a word that is already legitimate.
 *
 * It is consulted only as a fallback, AFTER the O(1) `model.hasWord` check misses, so
 * it costs nothing on the common path. Lookup is an O(log n) binary search directly
 * over the sorted blob (no multi-MB JS Set), so memory ≈ file size (~2.8 MB) – which
 * matters on mobile.
 *
 * Binary format (little-endian):
 *   u32 MAGIC ('WDL1' = 0x57444c31) | u32 count | u32 blobBytes | blob
 * where `blob` is the lowercase words joined by '\n', pre-sorted lexicographically.
 */

const MAGIC = 0x57444c31;

/** Common English inflectional suffixes, longest first so "-ies"/"-ing" beat "-s". */
const SUFFIXES = ["ically", "iness", "ously", "ment", "ness", "ing", "ies", "ied", "ier", "iest", "ly", "es", "ed", "er", "est", "s"];

export class WordOracle {
  private blob: Uint8Array | null;
  private dec = new TextDecoder();
  readonly count: number;

  private constructor(blob: Uint8Array | null, count: number) {
    this.blob = blob;
    this.count = count;
  }

  /** Empty oracle (`has` always false) – used when the wordlist asset is absent. */
  static empty(): WordOracle {
    return new WordOracle(null, 0);
  }

  static fromBuffer(buf: ArrayBuffer): WordOracle {
    if (buf.byteLength < 12) return WordOracle.empty();
    const dv = new DataView(buf);
    if (dv.getUint32(0, true) !== MAGIC) return WordOracle.empty();
    const count = dv.getUint32(4, true);
    const blobLen = dv.getUint32(8, true);
    if (12 + blobLen > buf.byteLength) return WordOracle.empty();
    return new WordOracle(new Uint8Array(buf, 12, blobLen), count);
  }

  /** Exact membership: is `word` (any case) a real English word in the list? */
  has(word: string): boolean {
    const blob = this.blob;
    if (!blob || !word) return false;
    const target = word.toLowerCase();
    let lo = 0;
    let hi = blob.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      // Snap to the start of the line containing `mid`.
      let s = mid;
      while (s > 0 && blob[s - 1] !== 10) s--;
      let e = s;
      while (e < blob.length && blob[e] !== 10) e++;
      const w = this.dec.decode(blob.subarray(s, e));
      if (w === target) return true;
      if (w < target) lo = e + 1;
      else hi = s;
    }
    return false;
  }

  /**
   * Recognise a real word by stripping a single common inflectional suffix and
   * checking whether the stem is a known word (per the supplied `known` predicate,
   * normally `model.hasWord` OR `this.has`). Catches forms the wordlist might miss,
   * e.g. "reallocates" → "reallocate", "carrying" → "carry". Restores the dropped
   * -e ("making" → "make") and undoes y→ie ("carries" → "carry") and consonant
   * doubling ("running" → "run") where the naive stem fails.
   */
  stemKnown(word: string, known: (w: string) => boolean): boolean {
    const w = word.toLowerCase();
    if (w.length < 4) return false;
    for (const suf of SUFFIXES) {
      if (!w.endsWith(suf) || w.length - suf.length < 3) continue;
      const base = w.slice(0, w.length - suf.length);
      if (known(base)) return true;
      // -ing/-ed dropped a silent 'e': "making"→"make", "used"→"use".
      if ((suf === "ing" || suf === "ed") && known(base + "e")) return true;
      // y→ie/i before -es/-ed/-er/-est: "carries"→"carry", "happier"→"happy".
      if ((suf === "ies" || suf === "ied" || suf === "ier" || suf === "iest") && known(base + "y")) return true;
      // Consonant doubling: "running"→"run", "stopped"→"stop".
      if ((suf === "ing" || suf === "ed") && base.length >= 3 && base[base.length - 1] === base[base.length - 2] && known(base.slice(0, -1))) return true;
    }
    return false;
  }
}
