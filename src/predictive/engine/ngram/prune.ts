/**
 * Prune an NgramCounts to words seen at least `minUni` times AND n-gram entries
 * seen at least `minNgram` times, remapping to a compact id space. Dropping rare
 * words removes the "words that don't exist" problem; dropping rare n-grams
 * removes unreliable one-off contexts, which is what makes a large-corpus model
 * both accurate and small/fast to pack.
 */
import { NgramCounts } from "./model.ts";
import { SOS } from "../text/tokenize.ts";

export function pruneCounts(
  c: NgramCounts,
  minUni: number,
  minNgram = 1,
  keepWord?: (w: string) => boolean,
): NgramCounts {
  const keep = new Set<number>();
  for (let id = 0; id < c.vocab.length; id++) {
    const w = c.vocab[id];
    if (w === SOS || ((c.uni[id] ?? 0) >= minUni && (!keepWord || keepWord(w)))) keep.add(id);
  }

  const nc = new NgramCounts();
  const remap = new Map<number, number>();
  for (const oldId of keep) remap.set(oldId, nc.intern(c.vocab[oldId]));

  let total = 0;
  for (const oldId of keep) {
    const nid = remap.get(oldId)!;
    const cnt = c.uni[oldId] ?? 0;
    nc.uni[nid] = cnt;
    if (c.vocab[oldId] !== SOS) total += cnt;
  }
  nc.totalUni = total;

  for (const [w1, inner] of c.bi) {
    if (!keep.has(w1)) continue;
    const nw1 = remap.get(w1)!;
    let m: Map<number, number> | undefined;
    for (const [next, cnt] of inner) {
      if (!keep.has(next) || cnt < minNgram) continue;
      if (!m) {
        m = new Map();
        nc.bi.set(nw1, m);
      }
      m.set(remap.get(next)!, cnt);
    }
  }

  for (const [w2, mid] of c.tri) {
    if (!keep.has(w2)) continue;
    const nw2 = remap.get(w2)!;
    let dm: Map<number, Map<number, number>> | undefined;
    for (const [w1, inner] of mid) {
      if (!keep.has(w1)) continue;
      const nw1 = remap.get(w1)!;
      let m: Map<number, number> | undefined;
      for (const [next, cnt] of inner) {
        if (!keep.has(next) || cnt < minNgram) continue;
        if (!dm) {
          dm = new Map();
          nc.tri.set(nw2, dm);
        }
        if (!m) {
          m = new Map();
          dm.set(nw1, m);
        }
        m.set(remap.get(next)!, cnt);
      }
    }
  }
  return nc;
}
