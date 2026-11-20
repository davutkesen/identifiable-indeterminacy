"""Paired comparisons over thirty resamplings of the pooled data (Table S4).

The smallest p value a two-sided Wilcoxon signed-rank test can return on n
paired differences is 2/2^n: about 0.002 at n = 10, which says no more than
"all ten splits agreed in sign", and 1.86e-9 at n = 30.  Here we

  * draw thirty stratified 80/20 resamplings of the pooled data;
  * apply Holm correction within each family of three comparisons;
  * report the mean paired difference with a bootstrap 95% confidence
    interval and the paired standardised effect size;
  * state the floor explicitly so a reader can see the limit of the test.

The thirty test sets overlap, so the p values summarise agreement in sign
rather than give exact tail probabilities; the inference of the paper is the
fixed-split bootstrap of exp6_symmetry_boot.py.

Two families are tested, and every configuration in them is the one
selected by five-fold CV on the predefined training split (weights_k.json),
held fixed across the thirty resamplings; nothing is re-tuned inside a
resampling.  Each resampling is a stratified 80/20 split of the pooled data.
Family 1: the proposed transformation against the plain rule, the latter
with k and class weighting tuned under the same protocol.  Family 2: the
proposed transformation against the direct transformation, both tuned.
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from common import (REPS, blocks_v1, blocks_v2, block_sqdists, class_weights,
                    load_rep, vote_predict)

N_SPLITS = 30
SEED = 42
BOOT = 10000
TRANSFORMS = {"v1": blocks_v1, "v2": blocks_v2}

def nknn_f1(Xtr, ytr, Xte, yte, tf, w, k, cwon):
    Btr, Bte = TRANSFORMS[tf](Xtr, ytr, [Xtr, Xte])
    D = np.zeros_like(block_sqdists(Bte[:1], Btr[:1])[0])
    for Di, wi in zip(block_sqdists(Bte, Btr), w):
        if wi:
            D += wi * Di
    np.sqrt(np.maximum(D, 0, out=D), out=D)
    yp, _ = vote_predict(D, ytr, k, class_weights(ytr, cwon))
    return f1_score(yte, yp, average="macro")


def knn_f1(Xtr, ytr, Xte, yte, k, cwon=True):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    D = (B * B).sum(1)[:, None] + (A * A).sum(1)[None] - 2 * (B @ A.T)
    np.sqrt(np.maximum(D, 0, out=D), out=D)
    yp, _ = vote_predict(D, ytr, k, class_weights(ytr, cwon))
    return f1_score(yte, yp, average="macro")


def boot_ci(d, reps=BOOT, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    means = d[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvals, names):
    order = np.argsort(pvals)
    m, adj, running = len(pvals), {}, 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[names[i]] = min(1.0, running)
    return adj


def floor_p(n):
    """Smallest attainable two-sided p of the signed-rank test at n pairs."""
    return float(wilcoxon(np.arange(1, n + 1), np.zeros(n))[1])


def paired_test(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    stat, p = wilcoxon(a, b)
    lo, hi = boot_ci(d)
    sd = d.std(ddof=1)
    return {"mean_a": float(a.mean()), "sd_a": float(a.std(ddof=1)),
            "mean_b": float(b.mean()), "sd_b": float(b.std(ddof=1)),
            "mean_diff": float(d.mean()), "ci95": [lo, hi],
            "dz": float(d.mean() / sd) if sd > 0 else float("inf"),
            "wins": int((d > 0).sum()), "n": len(d),
            "W": float(stat), "p_raw": float(p),
            "scores_a": [round(float(v), 4) for v in a],
            "scores_b": [round(float(v), 4) for v in b]}


def main():
    tuned = json.load(open("weights_k.json"))
    print("using the tuned configurations of weights_k.json")
    out = {"n_splits": N_SPLITS, "wilcoxon_floor_p": floor_p(N_SPLITS),
           "family1": {}, "family2": {}}
    print(f"two-sided Wilcoxon floor at n={N_SPLITS}: "
          f"p = {out['wilcoxon_floor_p']:.2e}  (at n=10 it is {floor_p(10):.4f})\n")

    for rep in REPS:
        Xtr, ytr = load_rep(rep, "train")
        Xte, yte = load_rep(rep, "test")
        X = np.vstack([Xtr, Xte])
        y = np.concatenate([ytr, yte])
        sss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.2,
                                     random_state=SEED)
        t = tuned[rep]
        a1, b1, a2, b2 = [], [], [], []
        for i, (tr, te) in enumerate(sss.split(X, y)):
            Xa, Xb, ya, yb = X[tr], X[te], y[tr], y[te]
            f2 = nknn_f1(Xa, ya, Xb, yb, "v2", t["v2"]["best_w"],
                         t["v2"]["best_k"], t["v2"]["best_cw"])
            a1.append(f2)
            b1.append(knn_f1(Xa, ya, Xb, yb, t["knn"]["best_k"],
                             t["knn"]["best_cw"]))
            a2.append(f2)
            b2.append(nknn_f1(Xa, ya, Xb, yb, "v1", t["v1"]["best_w"],
                              t["v1"]["best_k"], t["v1"]["best_cw"]))
            if (i + 1) % 10 == 0:
                print(f"  {rep} {i+1}/{N_SPLITS}", flush=True)
        out["family1"][rep] = paired_test(a1, b1)
        out["family2"][rep] = paired_test(a2, b2)
        r = out["family1"][rep]
        print(f"== {rep} family1  v2-knn={r['mean_diff']:+.4f} "
              f"CI[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] dz={r['dz']:+.2f} "
              f"wins={r['wins']}/{r['n']} p={r['p_raw']:.2e}", flush=True)
        if True:
            r = out["family2"][rep]
            print(f"== {rep} family2  v2-v1={r['mean_diff']:+.4f} "
                  f"CI[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] dz={r['dz']:+.2f} "
                  f"wins={r['wins']}/{r['n']} p={r['p_raw']:.2e}", flush=True)

    for fam in ("family1", "family2"):
        if not out[fam]:
            continue
        names = list(out[fam])
        adj = holm([out[fam][n]["p_raw"] for n in names], names)
        for n in names:
            out[fam][n]["p_holm"] = adj[n]
    json.dump(out, open("stats30.json", "w"), indent=1)
    print("\nwritten stats30.json")


if __name__ == "__main__":
    main()
