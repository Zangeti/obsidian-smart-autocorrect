/**
 * Self-contained n-gram counting kernel. Deliberately has NO imports and no
 * external references, so its `.toString()` is runnable code that can be
 * embedded verbatim into a Blob Web Worker (see the plugin's buildWorker.ts).
 * The same function is used on the main-thread fallback path, so both paths run
 * identical logic.
 *
 * Returns the SerializedCounts shape from ./serialize.ts (kept structurally in
 * sync; a round-trip test guards it).
 */
export function buildSerializedCountsFromDocs(docs: string[]): {
  v: number;
  vocab: string[];
  uni: number[];
  totalUni: number;
  bi: [number, [number, number][]][];
  tri: [number, [number, [number, number][]][]][];
} {
  const SOS = "<s>";
  const ABBR = new Set([
    "e.g", "i.e", "etc", "vs", "mr", "mrs", "ms", "dr", "prof", "st",
    "u.s", "u.k", "u.n", "a.m", "p.m", "jan", "feb", "mar", "apr", "jun",
    "jul", "aug", "sep", "sept", "oct", "nov", "dec", "inc", "ltd", "co",
    "no", "fig", "vol", "cf", "al",
  ]);

  const vocab: string[] = [];
  const wordId = new Map<string, number>();
  const uni: number[] = [];
  let totalUni = 0;
  const bi = new Map<number, Map<number, number>>();
  const tri = new Map<number, Map<number, Map<number, number>>>();

  const intern = (w: string): number => {
    let id = wordId.get(w);
    if (id === undefined) {
      id = vocab.length;
      vocab.push(w);
      wordId.set(w, id);
      uni.push(0);
    }
    return id;
  };
  const bump2 = (m: Map<number, Map<number, number>>, a: number, b: number) => {
    let inner = m.get(a);
    if (!inner) {
      inner = new Map();
      m.set(a, inner);
    }
    inner.set(b, (inner.get(b) ?? 0) + 1);
  };

  const norm = (w: string): string =>
    w.toLowerCase().replace(/^[^a-z0-9]+/i, "").replace(/[^a-z0-9]+$/i, "");

  for (const text of docs) {
    // sentence split (abbreviation-aware, simplified)
    const raw = text.split(/\s+/).filter((t) => t.length > 0);
    let sentence: string[] = [];
    const flush = () => {
      if (!sentence.length) return;
      const ids = [intern(SOS), intern(SOS), ...sentence.map(intern)];
      for (let i = 0; i < ids.length; i++) {
        const w = ids[i];
        if (vocab[w] !== SOS) {
          uni[w]++;
          totalUni++;
        }
        if (i >= 1) bump2(bi, ids[i - 1], w);
        if (i >= 2) {
          let mid = tri.get(ids[i - 2]);
          if (!mid) {
            mid = new Map();
            tri.set(ids[i - 2], mid);
          }
          bump2(mid, ids[i - 1], w);
        }
      }
      sentence = [];
    };
    for (const tok of raw) {
      const n = norm(tok);
      if (n) sentence.push(n);
      if (/[.!?…]["')\]]?$/.test(tok)) {
        const stem = tok.replace(/["')\]]+$/, "").replace(/\.$/, "").toLowerCase();
        if (/[!?…]/.test(tok) || (!ABBR.has(stem) && !/\d$/.test(stem) && !/^([a-z]\.)+[a-z]?$/i.test(tok.replace(/["')\]]+$/, "")))) {
          flush();
        }
      }
    }
    flush();
  }

  const biArr: [number, [number, number][]][] = [];
  for (const [w1, inner] of bi) biArr.push([w1, [...inner.entries()]]);
  const triArr: [number, [number, [number, number][]][]][] = [];
  for (const [w2, mid] of tri) {
    const midArr: [number, [number, number][]][] = [];
    for (const [w1, inner] of mid) midArr.push([w1, [...inner.entries()]]);
    triArr.push([w2, midArr]);
  }
  return { v: 1, vocab, uni, totalUni, bi: biArr, tri: triArr };
}
