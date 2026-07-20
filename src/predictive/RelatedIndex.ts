/**
 * Vault-wide index for RELATED-note suggestion, at SEGMENT granularity so a suggestion
 * can point at the specific heading/passage it matches (`[[Note#Heading]]`), not just the
 * whole note.
 *
 * Relevance blends two local, deterministic signals:
 *   - semantic: cosine of the LSTM's averaged word-embedding fingerprint, MEAN-CENTRED.
 *     Raw averaged embeddings share a large common component that makes everything look
 *     ~0.8 similar (measured on-vault: different-note pairs sit at p50≈0.78, AUC≈0.59).
 *     Subtracting the global mean embedding removes that component and makes the signal
 *     discriminative (different-note p50≈0.0, AUC≈0.72). Thresholds below are percentiles
 *     of the *measured* centred background, not invented constants.
 *   - keyword: TF-IDF cosine over shared distinctive words.
 *
 * Per-segment vectors are cached by note mtime; a rebuild only re-reads and re-embeds
 * changed notes, then re-centres everything (cheap in-memory).
 */
import type { App } from "obsidian";
import { segmentText, termFreq, tfidf, cosineSparse, documentFrequencies } from "./engine/index";
import type { PredictiveEngineController } from "./PredictiveEngineController";

const MAX_CHARS = 8000;

/**
 * Confidence bar by sensitivity, expressed as a PERCENTILE of the vault's OWN background
 * similarity distribution (the cosine between unrelated sections). A candidate must be more
 * similar than this fraction of random section pairs to be shown. So the actual cosine
 * threshold is read off the measured data, per vault, not hard-coded: sensitivity picks how
 * far into the tail we insist on (1 = only the top ~1% of the background, 5 = top ~10%).
 */
const BG_PERCENTILE: Record<number, number> = { 1: 0.99, 2: 0.98, 3: 0.96, 4: 0.93, 5: 0.9 };
/** Vectors sampled to estimate the top principal components (bounds build cost on big vaults). */
const PC_SAMPLE = 3000;
/** How many shared directions to strip (SIF). Top few, not one, sharpens topical similarity. */
const PC_COMPONENTS = 2;
/** Random section pairs sampled to estimate the background similarity distribution. */
const BG_SAMPLE = 4000;

export interface RelatedCandidate {
  target: string; // note basename (link target)
  display: string; // title to show
  path: string;
  heading?: string; // matched section within the note, for [[Note#Heading]]
  score: number; // blended 0..1
  semantic: number; // centred cosine
  keyword: number;
  snippet: string; // one-line preview of the matched section
  sectionText: string; // the matched section's text, for the side preview panel
}

interface SegVec {
  raw: Float32Array; // uncentred embedding (kept so the mean can be recomputed on rebuild)
  cen: Float32Array; // centred + normalised (what queries compare against)
  tf: Map<string, number>;
  tfidf: Map<string, number>;
  heading?: string;
  snippet: string;
  text: string; // the section's own text (capped), for the preview panel
}

interface NoteEntry {
  title: string;
  mtime: number;
  tf: Map<string, number>; // note-level, for df + shortlisting
  segs: SegVec[];
  noteCen: Float32Array; // note-level centred vector, for fast semantic shortlisting
}

const dot = (a: Float32Array, b: Float32Array): number => {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += a[i] * b[i];
  return d;
};
const normalise = (v: Float32Array): Float32Array => {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  if (n > 0) {
    const inv = 1 / Math.sqrt(n);
    for (let i = 0; i < v.length; i++) v[i] *= inv;
  }
  return v;
};

export class RelatedIndex {
  private app: App;
  private engine: PredictiveEngineController;
  private notes = new Map<string, NoteEntry>();
  private df = new Map<string, number>();
  private docCount = 0;
  private inverted = new Map<string, Set<string>>();
  private mean: Float32Array | null = null;
  /** Top principal components of the centred vectors (SIF common-component removal). Removing
   *  the top FEW shared directions - not just one - sharpens topical discrimination further. */
  private pcs: Float32Array[] = [];
  /** Sorted sample of background section-pair similarities, for the percentile threshold. */
  private bgSorted: number[] = [];
  private dim = 0;
  private building: Promise<void> | null = null;
  private built = false;
  /** Cache of query-text -> raw embedding, so re-running suggestions over a note only
   *  re-embeds the block you actually changed (the rest are cache hits). Bounded. */
  private queryCache = new Map<string, number[] | null>();

  private wSem = 0.7;
  private wKw = 0.3;

  constructor(app: App, engine: PredictiveEngineController) {
    this.app = app;
    this.engine = engine;
  }

  ensureBuilt(): Promise<void> {
    if (this.building) return this.building;
    if (this.built) return Promise.resolve();
    return this.rebuild();
  }

  rebuild(): Promise<void> {
    if (this.building) return this.building;
    this.building = this.doBuild().finally(() => {
      this.building = null;
    });
    return this.building;
  }

  private async doBuild(): Promise<void> {
    const files = this.app.vault.getMarkdownFiles();
    const live = new Set(files.map((f) => f.path));
    for (const p of [...this.notes.keys()]) if (!live.has(p)) this.notes.delete(p);

    for (const f of files) {
      const cached = this.notes.get(f.path);
      if (cached && cached.mtime === f.stat.mtime) continue;
      try {
        const raw = stripFrontmatter(await this.app.vault.cachedRead(f)).slice(0, MAX_CHARS);
        const noteTf = termFreq(raw);
        if (noteTf.size === 0) {
          this.notes.delete(f.path);
          continue;
        }
        const segs: SegVec[] = [];
        for (const seg of segmentText(raw)) {
          // Fold the owning heading into what we match on, so a section is found by its
          // heading's words too (e.g. a bullet list under "Photosynthesis").
          const matchText = seg.heading ? `${seg.heading}. ${seg.text}` : seg.text;
          const arr = await this.engine.embed(matchText);
          if (!arr || arr.length === 0) continue;
          const raw32 = Float32Array.from(arr);
          segs.push({
            raw: raw32,
            cen: raw32, // placeholder; centred after the global mean is known
            tf: termFreq(matchText),
            tfidf: new Map(),
            heading: seg.heading,
            snippet: snippetOf(seg.text, seg.heading),
            text: seg.text.slice(0, 600),
          });
          if (!this.dim) this.dim = raw32.length;
        }
        this.notes.set(f.path, { title: f.basename, mtime: f.stat.mtime, tf: noteTf, segs, noteCen: new Float32Array(this.dim) });
      } catch {
        /* skip unreadable */
      }
    }

    // Global mean over every segment embedding, then centre + normalise each.
    if (this.dim > 0) {
      const mean = new Float32Array(this.dim);
      let count = 0;
      for (const n of this.notes.values())
        for (const s of n.segs) {
          for (let i = 0; i < this.dim; i++) mean[i] += s.raw[i];
          count++;
        }
      if (count > 0) for (let i = 0; i < this.dim; i++) mean[i] /= count;
      this.mean = mean;
      // First pass: mean-centre + normalise every section vector.
      const all: Float32Array[] = [];
      for (const n of this.notes.values())
        for (const s of n.segs) {
          const c = new Float32Array(this.dim);
          for (let i = 0; i < this.dim; i++) c[i] = s.raw[i] - mean[i];
          normalise(c);
          s.cen = c;
          all.push(c);
        }
      // SIF common-component removal: after centring, the few most-shared directions still
      // dominate and blur topics together. Estimate the top principal components (power
      // iteration + deflation on a sample) and project them out of every vector. Removing the
      // top TWO rather than one strips more of the generic "this is English prose" signal, so
      // the remaining similarity is more specifically about content.
      this.pcs = topPrincipalComponents(all, this.dim, PC_COMPONENTS);
      for (const c of all) removeComponents(c, this.pcs);
      // Note-level vector = normalised mean of its (now sharpened) section vectors.
      for (const n of this.notes.values()) {
        const acc = new Float32Array(this.dim);
        for (const s of n.segs) for (let i = 0; i < this.dim; i++) acc[i] += s.cen[i];
        n.noteCen = normalise(acc);
      }
      // Background: similarity between random, mostly-unrelated section pairs. The
      // suggestion threshold is a percentile of THIS, so it adapts to the vault.
      this.bgSorted = sampleBackground(all, BG_SAMPLE);
    }

    // Keyword stats: df, per-segment TF-IDF, inverted index over note-level terms.
    this.docCount = this.notes.size;
    this.df = documentFrequencies([...this.notes.values()].map((n) => n.tf));
    this.inverted.clear();
    for (const [path, n] of this.notes) {
      for (const s of n.segs) s.tfidf = tfidf(s.tf, this.df, this.docCount);
      for (const t of n.tf.keys()) {
        let set = this.inverted.get(t);
        if (!set) this.inverted.set(t, (set = new Set()));
        set.add(path);
      }
    }
    this.built = true;
  }

  /**
   * Best related sections for `text`, most relevant first. Points at the matched section
   * (`heading`) within each note. `excludePath` and any note titles in `exclude` are
   * dropped, and only candidates whose centred similarity clears the sensitivity floor
   * (or have a strong keyword overlap) are returned - so a chunk with nothing close yields
   * an empty list.
   */
  async candidatesFor(
    text: string,
    excludePath: string | undefined,
    exclude: Set<string>,
    limit: number,
    sensitivity: number,
    /** For the AMBIENT link icons: only surface a candidate when TWO independent signals
     *  agree - a strong topical (semantic) match AND genuine lexical overlap - so an icon
     *  means a relatively strong connection, not a semantic coincidence. A match that is
     *  exceptionally strong semantically (top ~1% of background) still qualifies on its own.
     *  The explicit `[[` picker leaves this off (there the user has already asked to link). */
    requireCorroboration = false,
  ): Promise<RelatedCandidate[]> {
    await this.ensureBuilt();
    const tf = termFreq(text);
    if (tf.size === 0) return [];
    const qTfidf = tfidf(tf, this.df, this.docCount);
    let arr: number[] | null = null;
    if (this.mean) {
      if (this.queryCache.has(text)) arr = this.queryCache.get(text)!;
      else {
        arr = await this.engine.embed(text);
        if (this.queryCache.size > 400) this.queryCache.clear();
        this.queryCache.set(text, arr);
      }
    }
    let qC: Float32Array | null = null;
    if (arr && arr.length && this.mean) {
      qC = new Float32Array(this.dim);
      for (let i = 0; i < this.dim; i++) qC[i] = arr[i] - this.mean[i];
      normalise(qC);
      removeComponents(qC, this.pcs); // same common-component removal as the index
    }
    // Confidence bar read off the vault's OWN background distribution (see BG_PERCENTILE):
    // a section only shows as a candidate if it out-matches this fraction of random section pairs.
    const floor = this.backgroundFloor(sensitivity);
    // The strictest background percentile - what a single "very strong on its own" match looks
    // like. Used only as the reference scale for the aggregate icon gate below (data-derived, not
    // a hand-picked cosine).
    const strongFloor = this.backgroundFloor(1);

    // Shortlist: notes sharing a term (keyword) ∪ the top notes by note-level semantic
    // similarity (so a match with no shared words is still found).
    const shortlist = new Set<string>();
    for (const t of tf.keys()) {
      const set = this.inverted.get(t);
      if (set) for (const p of set) shortlist.add(p);
      if (shortlist.size > 600) break;
    }
    if (qC) {
      const ranked: Array<[string, number]> = [];
      for (const [path, n] of this.notes) ranked.push([path, dot(qC, n.noteCen)]);
      ranked.sort((a, b) => b[1] - a[1]);
      for (let i = 0; i < Math.min(30, ranked.length); i++) shortlist.add(ranked[i][0]);
    }

    const out: RelatedCandidate[] = [];
    for (const path of shortlist) {
      if (path === excludePath) continue;
      const n = this.notes.get(path);
      if (!n) continue;
      if (exclude.has(n.title.toLowerCase())) continue;
      // Per-SECTION candidates: the same note can appear more than once when different
      // sections match (they link to different `[[Note#Heading]]`s). Dedupe by heading and
      // keep the best per heading; a note contributes at most two sections.
      const byHeading = new Map<string, { seg: SegVec; sem: number; kw: number; score: number }>();
      for (const s of n.segs) {
        const sem = qC ? dot(qC, s.cen) : 0;
        // Gate on the semantic signal against the data-derived confidence bar. Keyword
        // overlap only sharpens ranking; it is not a back door for weak, generic matches.
        if (!qC || sem < floor) continue;
        const kw = cosineSparse(qTfidf, s.tfidf);
        const score = this.wSem * sem + this.wKw * kw;
        const key = (s.heading ?? "").toLowerCase();
        const prev = byHeading.get(key);
        if (!prev || score > prev.score) byHeading.set(key, { seg: s, sem, kw, score });
      }
      const perNote = [...byHeading.values()].sort((a, b) => b.score - a.score).slice(0, 2);
      for (const c of perNote)
        out.push({
          target: n.title,
          display: n.title,
          path,
          heading: c.seg.heading,
          score: c.score,
          semantic: c.sem < 0 ? 0 : c.sem,
          keyword: c.kw,
          snippet: c.seg.snippet,
          sectionText: c.seg.text,
        });
    }
    out.sort((a, b) => b.score - a.score);
    if (!requireCorroboration) return out.slice(0, limit);
    // Aggregate gate for the AMBIENT icon - no hard-coded counts. Each candidate above the floor
    // contributes "excess" evidence = how far its similarity sits above the display bar. We show
    // the icon when the TOTAL excess reaches what a single very-strong match would contribute
    // (strongFloor − floor). So one very-strong match qualifies on its own, OR several medium ones
    // whose evidence adds up do - the number needed emerges from their strengths, not a magic
    // constant. A lone weak match contributes almost nothing, so it never shows an icon. Both bars
    // are the vault's own data-derived background percentiles.
    const reference = Math.max(1e-6, strongFloor - floor);
    let evidence = 0;
    for (const c of out) evidence += Math.max(0, c.semantic - floor);
    if (evidence < reference) return [];
    return out.slice(0, limit);
  }

  /** The cosine bar for `sensitivity`, read as a percentile of the vault's measured
   *  background similarity. Returns 1 (nothing qualifies) until the background is built. */
  private backgroundFloor(sensitivity: number): number {
    if (this.bgSorted.length === 0) return 1;
    const pct = BG_PERCENTILE[sensitivity] ?? BG_PERCENTILE[3];
    const idx = Math.min(this.bgSorted.length - 1, Math.max(0, Math.floor(pct * this.bgSorted.length)));
    return this.bgSorted[idx];
  }
}

/** Top `k` principal components of `vectors` via power iteration + deflation (deterministic
 *  seed, sampled for cost). Fewer components when there aren't enough vectors to justify them. */
function topPrincipalComponents(vectors: Float32Array[], dim: number, k: number): Float32Array[] {
  if (vectors.length < 8) return [];
  const step = vectors.length > PC_SAMPLE ? Math.ceil(vectors.length / PC_SAMPLE) : 1;
  const sample: Float32Array[] = [];
  for (let i = 0; i < vectors.length; i += step) sample.push(vectors[i].slice()); // copies: we deflate them
  // Only ask for as many components as the sample can meaningfully support.
  const want = Math.max(1, Math.min(k, Math.floor(sample.length / 8)));
  const comps: Float32Array[] = [];
  const v = new Float32Array(dim);
  for (let c = 0; c < want; c++) {
    let u = new Float32Array(dim);
    for (let i = 0; i < dim; i++) u[i] = Math.sin(i * 0.1 + 1 + c * 7.13);
    normalise(u);
    for (let it = 0; it < 12; it++) {
      v.fill(0);
      for (const x of sample) {
        let d = 0;
        for (let i = 0; i < dim; i++) d += x[i] * u[i];
        for (let i = 0; i < dim; i++) v[i] += d * x[i];
      }
      normalise(v);
      u.set(v);
    }
    comps.push(u.slice());
    // Deflate: remove this component from the sample so the next iteration finds the NEXT one.
    for (const x of sample) removeComponent(x, u);
  }
  return comps;
}

/** Project the `u` direction out of `x` and renormalise (in place). */
function removeComponent(x: Float32Array, u: Float32Array): void {
  let d = 0;
  for (let i = 0; i < x.length; i++) d += x[i] * u[i];
  for (let i = 0; i < x.length; i++) x[i] -= d * u[i];
  normalise(x);
}

/** Project every direction in `us` out of `x` (in place). */
function removeComponents(x: Float32Array, us: Float32Array[]): void {
  for (const u of us) removeComponent(x, u);
}

/** Sorted sample of cosines between random vector pairs - the background distribution. */
function sampleBackground(vectors: Float32Array[], n: number): number[] {
  const L = vectors.length;
  if (L < 4) return [];
  let seed = 123457;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const pairs = Math.min(n, (L * (L - 1)) / 2);
  const out: number[] = [];
  for (let k = 0; k < pairs; k++) {
    const i = Math.floor(rnd() * L);
    let j = Math.floor(rnd() * L);
    if (j === i) j = (j + 1) % L;
    out.push(dot(vectors[i], vectors[j]));
  }
  out.sort((a, b) => a - b);
  return out;
}

/** Drop a leading YAML frontmatter block so its fields (tags, date, aliases) are never
 *  embedded, matched, or linked as if they were prose. */
function stripFrontmatter(text: string): string {
  return text.replace(/^\uFEFF?---\r?\n[\s\S]*?\r?\n---[ \t]*\r?\n?/, "");
}

/**
 * A one-line preview: the section's FIRST meaningful line (its title/lead-in), with an
 * ellipsis if there's more below it - rather than cramming as many lines as fit up to a
 * character cap. Adapts to however the note was segmented (a "Title / - bullets" block
 * previews as "Title …").
 */
function snippetOf(text: string, heading?: string): string {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.replace(/^\s*(?:[-*+]|\d+[.)])\s+/, "").replace(/[#>*_`]/g, "").trim())
    .filter((l) => l.length > 0);
  const first = lines[0] ?? heading ?? "";
  const hasMore = lines.length > 1 || first.length > 100;
  const head = first.length > 100 ? first.slice(0, 100).trimEnd() : first;
  return hasMore ? `${head} …` : head;
}
