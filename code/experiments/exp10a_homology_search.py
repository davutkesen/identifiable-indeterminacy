"""Sequence identity within and across the predefined splits (needs MMseqs2).

For each of the three benchmarks (seqdata.py):

  test_to_train  every hit of a test protein against the training proteins
                 with identity >= 0.25, identity being identical aligned
                 residues over the length of the shorter sequence; the
                 stratified evaluation (exp10b) reads the largest of them;
  pooled_edges   every pair of proteins of the pooled training and test sets
                 with identity >= 0.30, from an all-against-all search; the
                 homology-aware cross-validation (exp12) builds its folds
                 from the connected components of this graph.

Searches use MMseqs2 easy-search at sensitivity 7.5 (the most sensitive
preset), keep alignments with E-value <= 1e-3 (the MMseqs2 default, which
keeps chance alignments of very short peptides out), allow up to 1,000
targets per query, and take identities from full alignments
(--alignment-mode 3).  The results are written to homology.json.
"""
import json
import time

import seqdata as sd

LOW_TEST, LOW_EDGE = 0.25, 0.30


def main():
    sd.check_manifest()
    out = {"mmseqs_version": sd.mmseqs_version(),
           "parameters": {"sensitivity": 7.5, "evalue": 1e-3, "max_seqs": 1000,
                          "alignment_mode": 3,
                          "identity": "identical aligned residues / length of the shorter sequence",
                          "test_to_train_min": LOW_TEST, "pooled_edge_min": LOW_EDGE}}
    for bench in sd.BENCHMARKS:
        t0 = time.time()
        tr, te = sd.records(bench, "train"), sd.records(bench, "test")
        hits = sd.mmseqs_search(te, tr)
        tt = {}
        for q, t, ident in hits:
            if ident >= LOW_TEST:
                tt.setdefault(q, {})
                tt[q][t] = max(tt[q].get(t, 0.0), round(ident, 4))
        pooled = tr + te
        edges = {}
        for q, t, ident in sd.mmseqs_search(pooled, pooled):
            if ident >= LOW_EDGE:
                key = (q, t) if q < t else (t, q)
                edges[key] = max(edges.get(key, 0.0), round(ident, 4))
        out[bench] = {
            "n_train": len(tr), "n_test": len(te),
            "test_to_train": {q: sorted(v.items(), key=lambda kv: -kv[1])
                              for q, v in tt.items()},
            "pooled_edges": [[a, b, v] for (a, b), v in sorted(edges.items())],
        }
        n30 = sum(1 for v in out[bench]["test_to_train"].values() if v[0][1] >= 0.30)
        print(f"{bench:5s} test {len(te):5d}: {n30} with a training protein at >= 30% "
              f"identity; pooled graph {len(edges)} edges  [{time.time() - t0:.0f}s]",
              flush=True)
    with open("homology.json", "w") as f:
        json.dump(out, f)
    print("written homology.json")


if __name__ == "__main__":
    main()
