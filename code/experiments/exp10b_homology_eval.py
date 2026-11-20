"""Test proteins with and without a training homologue (transport benchmark).

The compilers report a 30% identity filter, but the sequences they
distribute show test proteins with close training homologues
(exp10a_homology_search.py).  A nearest-neighbour rule can exploit such
pairs, so every rule of the paper is re-scored separately on the test
proteins whose largest identity to any training protein of the same
feature file is below a threshold ("remote") and on the rest
("homologous").  Nothing is re-tuned: each rule keeps the configuration it
was selected with (rules.configs_from_stored), and its predictions on the
whole test split are first checked against the stored scores.

For the comparisons of the proposed construction with the plain rule,
N-KNN and the direct construction, the paired bootstrap (10,000 resamples)
is run within each stratum, and the difference between the two strata in
that gain is given a bootstrap interval from independent resamples of the
two strata.  The primary threshold is 30% identity; 40% and 50% are
reported as sensitivity analyses.
"""
import json

import numpy as np

import rules as R
import seqdata as sd

REPS = ["AAC", "DPC", "PSSM"]
THRESHOLDS = [0.30, 0.40, 0.50]
COMPARE = [("proposed", "knn"), ("proposed", "nknn"), ("proposed", "direct")]
BOOT_REPS, BOOT_SEED = 10000, 17
BINS = [0.0, 0.30, 0.50, 0.90, 1.0001]


def stored_f1(rule, rep, wk, ab, fz, cb, bl):
    return {"knn": lambda: wk[rep]["knn"]["test"]["F1"],
            "direct": lambda: wk[rep]["v1"]["test"]["F1"],
            "proposed": lambda: wk[rep]["v2"]["test"]["F1"],
            "nknn": lambda: ab[rep]["nknn"]["test"]["F1"],
            "nsknn": lambda: fz[rep]["NS-kNN"]["F1"],
            "centroid": lambda: cb[rep]["nearest centroid"]["F1"]}.get(
        rule, lambda: bl[rep][rule]["F1"])()


def split_stats(H):
    """Counts of homologous pairs across and within the predefined splits of
    every benchmark, over all its sequences."""
    out = {}
    for bench in sd.BENCHMARKS:
        tr, te = sd.records(bench, "train"), sd.records(bench, "test")
        lab_te = {a: lab for a, lab, _ in te}
        best = {a: (v[0][1] if v else 0.0)
                for a, v in ((a, H[bench]["test_to_train"].get(a, [])) for a in lab_te)}
        train_ids = {a for a, _, _ in tr}
        within = set()
        for a, b, v in H[bench]["pooled_edges"]:
            if a in train_ids and b in train_ids:
                within.update((a, b))
        o = {"n_train": len(tr), "n_test": len(te),
             "train_positives": sum(lab for _, lab, _ in tr),
             "test_positives": sum(lab_te.values()),
             "train_with_train_homologue_30": len(within)}
        lab_tr = {a: lab for a, lab, _ in tr}
        same = {a: max((v for t, v in H[bench]["test_to_train"].get(a, [])
                        if lab_tr.get(t) == lab_te[a]), default=0.0) for a in lab_te}
        for thr in (0.30, 0.50, 0.70, 0.90):
            hit = [a for a, v in best.items() if v >= thr]
            o[f"test_with_train_homologue_{int(thr * 100)}"] = len(hit)
            o[f"positives_with_train_homologue_{int(thr * 100)}"] = sum(lab_te[a] for a in hit)
            # the same, counting only training proteins of the test protein's class
            o[f"test_with_same_class_train_homologue_{int(thr * 100)}"] = sum(
                1 for v in same.values() if v >= thr)
        out[bench] = o
    return out


def mapping_stats():
    """How the rows of the feature files were traced to their sequences."""
    from common import load_rep
    out = {}
    for split in ("train", "test"):
        Xa, _ = load_rep("AAC", split)
        recs = sd.records("ET", split)
        A = np.array([sd.aac(s) for _, _, s in recs])
        D = sd.sqdist_rows(Xa, A)
        out[f"ET_composition_{split}"] = {"rows": int(len(Xa)), "sequences": len(recs),
                                          "exact": int((D.min(1) < 1e-4).sum())}
        for bench in ("ET", "SNARE"):
            X, y, accs = sd.features(bench, "PSSM", split)
            rmap = {a: (a, lab, s) for a, lab, s in sd.records(bench, split)}
            rows = [rmap[a] for a in accs]
            informative = sum(1 for xr, (_, _, s) in zip(X, rows)
                              if any(c not in s for c in sd.AA))
            out[f"{bench}_profile_{split}"] = {
                "rows": int(len(X)), "rows_with_absent_residue": informative,
                "mismatches": sd.absent_residue_mismatches(X, rows)}
    return out


def main():
    H = json.load(open("homology.json"))
    hom = H["ET"]["test_to_train"]
    wk, ab, fz, cb, bl = [json.load(open(f)) for f in (
        "weights_k.json", "ablation.json", "fuzzy_baselines.json",
        "centroid_baselines.json", "baselines.json")]
    out = {"thresholds": THRESHOLDS, "primary": 0.30, "boot_reps": BOOT_REPS,
           "mmseqs_version": H["mmseqs_version"], "split_stats": split_stats(H),
           "mapping": mapping_stats()}
    print(out["mapping"], flush=True)
    for b, v in out["split_stats"].items():
        print(b, v, flush=True)
    seed = BOOT_SEED
    for rep in REPS:
        Xtr, ytr, atr = sd.features("ET", rep, "train")
        Xte, yte, ate = sd.features("ET", rep, "test")
        in_train = set(atr)
        maxid = np.array([max([v for t, v in hom.get(a, []) if t in in_train], default=0.0)
                          for a in ate])
        cfg = R.configs_from_stored(rep, wk, ab, fz, bl)
        preds = {}
        for rule, c in cfg.items():
            yp, s = R.predict(rule, c, Xtr, ytr, Xte)
            f1 = R.summarise(yte, yp, s)["F1"]
            ref = stored_f1(rule, rep, wk, ab, fz, cb, bl)
            assert abs(f1 - ref) < 1e-9, f"{rep} {rule}: {f1} != stored {ref}"
            preds[rule] = (yp, s)
        hist = np.histogram(maxid, bins=BINS)[0]
        o = {"n_test": int(len(yte)),
             "identity_bins": {f"[{a:.2f},{min(b, 1.0):.2f})": int(h)
                               for a, b, h in zip(BINS, BINS[1:], hist)},
             "max_identity": maxid.round(4).tolist(), "accessions": ate,
             "strata": {}}
        for thr in THRESHOLDS:
            remote = maxid < thr
            st = {}
            boots = {}
            for name, mask in (("remote", remote), ("homologous", ~remote)):
                y = yte[mask]
                st[name] = {"n": int(mask.sum()), "positives": int(y.sum()),
                            "rules": {r: R.summarise(y, p[mask], None if s is None else s[mask])
                                      for r, (p, s) in preds.items()},
                            "compare": {}}
                for a, b in COMPARE:
                    res = R.boot_diff(y, preds[a][0][mask], preds[b][0][mask],
                                      reps=BOOT_REPS, seed=seed)
                    seed += 1
                    boots[(name, a, b)] = res.pop("boot")
                    st[name]["compare"][f"{a}-{b}"] = res
            st["interaction"] = {}
            for a, b in COMPARE:
                d = boots[("remote", a, b)] - boots[("homologous", a, b)]
                obs = (st["remote"]["compare"][f"{a}-{b}"]["point"]
                       - st["homologous"]["compare"][f"{a}-{b}"]["point"])
                st["interaction"][f"{a}-{b}"] = {
                    "point": float(obs),
                    "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
            o["strata"][f"{thr:.2f}"] = st
            r, h = st["remote"], st["homologous"]
            print(f"{rep:4s} thr={thr:.2f}  remote n={r['n']:4d} (pos {r['positives']:3d})  "
                  f"homologous n={h['n']:4d} (pos {h['positives']:3d})", flush=True)
            for rule in ("knn", "direct", "proposed", "nknn", "Gradient boosting"):
                print(f"      {rule:18s} F1 remote {r['rules'][rule]['F1']:.4f}   "
                      f"homologous {h['rules'][rule]['F1']:.4f}", flush=True)
            for a, b in COMPARE:
                cr, ch = r["compare"][f"{a}-{b}"], h["compare"][f"{a}-{b}"]
                it = st["interaction"][f"{a}-{b}"]
                print(f"      {a}-{b:8s} remote {cr['point']:+.4f} [{cr['ci95'][0]:+.4f},{cr['ci95'][1]:+.4f}]"
                      f"   homologous {ch['point']:+.4f} [{ch['ci95'][0]:+.4f},{ch['ci95'][1]:+.4f}]"
                      f"   difference {it['point']:+.4f} [{it['ci95'][0]:+.4f},{it['ci95'][1]:+.4f}]",
                      flush=True)
        out[rep] = o
    json.dump(out, open("homology_eval.json", "w"), indent=1)
    print("written homology_eval.json")


if __name__ == "__main__":
    main()
