/**
 * Lightweight bag-of-words vectors for topical similarity between chunks of note text.
 * This is the "keyword" half of related-link suggestion (the "semantic" half comes from
 * averaging the LSTM's word embeddings, in lstm/model.ts). Both are deterministic and
 * fully offline.
 *
 * A term vector is a sparse Map<term, weight>. Weights are TF-IDF: term frequency in the
 * chunk times inverse document frequency across the vault, so a rare, distinctive word
 * ("photosynthesis") counts for far more than a common one ("about"). Cosine similarity
 * of two such vectors is the classic measure of "these two passages are about the same
 * thing".
 */

/** Function words and filler that carry no topic signal, so they never enter a vector. */
export const STOP_WORDS = new Set([
  "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for", "is", "it",
  "as", "by", "be", "we", "i", "you", "he", "she", "they", "this", "that", "with", "from",
  "was", "are", "not", "have", "has", "had", "will", "would", "can", "could", "should",
  "all", "one", "so", "if", "up", "out", "no", "do", "does", "did", "my", "me", "us",
  "your", "our", "their", "his", "her", "its", "them", "then", "than", "there", "here",
  "what", "when", "where", "which", "who", "how", "why", "into", "over", "such", "some",
  "more", "most", "other", "any", "each", "also", "just", "only", "very", "much", "many",
  "been", "being", "were", "am", "these", "those", "about", "after", "before", "because",
  "while", "between", "through", "during", "under", "again", "further", "once", "both",
  "get", "got", "make", "made", "like", "well", "even", "back", "still", "way", "may",
]);

/** Split text into lower-cased content terms (letters, length ≥ 3, no stop words). */
export function terms(text: string): string[] {
  const out: string[] = [];
  const re = /[a-zA-Z][a-zA-Z'-]{2,}/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    const w = m[0].toLowerCase().replace(/^['-]+|['-]+$/g, "");
    if (w.length >= 3 && !STOP_WORDS.has(w)) out.push(w);
  }
  return out;
}

/** Raw term-frequency counts for a chunk. */
export function termFreq(text: string): Map<string, number> {
  const tf = new Map<string, number>();
  for (const t of terms(text)) tf.set(t, (tf.get(t) ?? 0) + 1);
  return tf;
}

/**
 * Document frequencies over a set of chunks: how many chunks each term appears in.
 * Feed into `idf` to weight rare terms up.
 */
export function documentFrequencies(docs: Iterable<Map<string, number>>): Map<string, number> {
  const df = new Map<string, number>();
  for (const doc of docs) for (const t of doc.keys()) df.set(t, (df.get(t) ?? 0) + 1);
  return df;
}

/** Smoothed inverse document frequency for `term` given `docCount` total documents. */
export function idf(term: string, df: Map<string, number>, docCount: number): number {
  return Math.log((docCount + 1) / ((df.get(term) ?? 0) + 1)) + 1;
}

/** Build a normalised (unit-length) TF-IDF vector from term frequencies. */
export function tfidf(tf: Map<string, number>, df: Map<string, number>, docCount: number): Map<string, number> {
  const v = new Map<string, number>();
  let norm = 0;
  for (const [t, c] of tf) {
    const w = c * idf(t, df, docCount);
    v.set(t, w);
    norm += w * w;
  }
  if (norm > 0) {
    const inv = 1 / Math.sqrt(norm);
    for (const [t, w] of v) v.set(t, w * inv);
  }
  return v;
}

/** Cosine similarity of two sparse unit vectors (iterates the smaller one). */
export function cosineSparse(a: Map<string, number>, b: Map<string, number>): number {
  const [small, big] = a.size <= b.size ? [a, b] : [b, a];
  let dot = 0;
  for (const [t, w] of small) {
    const o = big.get(t);
    if (o !== undefined) dot += w * o;
  }
  return dot;
}

/** Cosine similarity of two dense vectors (unit-normalised or not). */
export function cosineDense(a: Float32Array | number[], b: Float32Array | number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / Math.sqrt(na * nb);
}
