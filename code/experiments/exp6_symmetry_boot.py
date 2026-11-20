"""Three analyses at the selected configurations.

  A  label-swap symmetry (Table S1).  The selected weights load only one class block,
     (w_I, w_{M_0}, w_{M_1}) = (0.9, 0.0, 0.1).  Re-running the whole
     cross-validated search with the class labels exchanged tests whether
     the selection follows the labelling (the weight should move to the
     other block) and whether the predictive performance is unchanged.

  B  guard-constant sweep at the SELECTED configuration (Table S17), so that
     its reference column is a number reported elsewhere in the paper.

  C  paired bootstrap on the fixed predefined test split (Table 4,
     last column, and Table S2).  The resampling
     analysis pools and re-splits the data, so its thirty differences share
     test instances.  Resampling the test instances of the predefined split
     with replacement, and recomputing the paired difference in macro F1 on
     each resample, respects the split and makes the unit of analysis
     explicit: the test instance.
"""
import json

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import (REPS, blocks_v1, blocks_v2, block_sqdists, class_weights,
                    load_rep, simplex_grid, vote_predict)
from exp2_weights_k import GRID, KS, KMAX, neighbours, vote_from_neighbours

SEED = 0
TRANSFORMS = {"v1": blocks_v1, "v2": blocks_v2}


def cv_best(X, y, fn, folds=5):
    """Return the CV-selected (w, k, cw) and its CV score."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    acc = np.zeros((len(GRID), len(KS), 2))
    for tr, va in skf.split(X, y):
        Btr, Bva = fn(X[tr], y[tr], [X[tr], X[va]])
        Db = block_sqdists(Bva, Btr)
        cws = [class_weights(y[tr], False), class_weights(y[tr], True)]
        for gi, w in enumerate(GRID):
            idx, dk = neighbours(Db, w, KMAX)
            for ki, k in enumerate(KS):
                for ci, cw in enumerate(cws):
                    yp, _ = vote_from_neighbours(idx, dk, y[tr], k, cw)
                    acc[gi, ki, ci] += f1_score(y[va], yp, average="macro")
    acc /= folds
    gi, ki, ci = np.unravel_index(acc.argmax(), acc.shape)
    return list(GRID[gi]), KS[ki], bool(ci), float(acc[gi, ki, ci])


def predict(Xtr, ytr, Xte, fn, w, k, cwon, eps_s=1e-8):
    if fn is blocks_v2:
        Btr, Bte = fn(Xtr, ytr, [Xtr, Xte], eps_s=eps_s)
    else:
        Btr, Bte = fn(Xtr, ytr, [Xtr, Xte])
    Db = block_sqdists(Bte, Btr)
    idx, dk = neighbours(Db, w, k)
    yp, p1 = vote_from_neighbours(idx, dk, ytr, k, class_weights(ytr, cwon))
    return yp, p1


def knn_predict(Xtr, ytr, Xte, k, cwon):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    D = (B * B).sum(1)[:, None] + (A * A).sum(1)[None] - 2 * (B @ A.T)
    np.sqrt(np.maximum(D, 0, out=D), out=D)
    return vote_predict(D, ytr, k, class_weights(ytr, cwon))


# ---------------------------------------------------------------- A
def label_swap():
    out = {}
    for rep in REPS:
        Xtr, ytr = load_rep(rep, "train")
        Xte, yte = load_rep(rep, "test")
        out[rep] = {}
        for lab, (ya, yb) in [("original", (ytr, yte)),
                              ("swapped", (1 - ytr, 1 - yte))]:
            w, k, cw, cvf = cv_best(Xtr, ya, blocks_v2)
            yp, _ = predict(Xtr, ya, Xte, blocks_v2, w, k, cw)
            out[rep][lab] = {"w": w, "k": k, "cw": cw, "cv_f1": cvf,
                             "test_f1": float(f1_score(yb, yp, average="macro")),
                             "test_mcc": float(matthews_corrcoef(yb, yp))}
            print(f"  {rep} {lab:8s} w={w} k={k} cw={cw} cv={cvf:.4f} "
                  f"test F1={out[rep][lab]['test_f1']:.4f}", flush=True)
    return out


# ---------------------------------------------------------------- B
def eps_at_selected(tuned):
    out = {}
    for rep in REPS:
        Xtr, ytr = load_rep(rep, "train")
        Xte, yte = load_rep(rep, "test")
        t = tuned[rep]["v2"]
        out[rep] = {}
        for e in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
            yp, _ = predict(Xtr, ytr, Xte, blocks_v2, t["best_w"], t["best_k"],
                            t["best_cw"], eps_s=e)
            out[rep][f"{e:.0e}"] = float(f1_score(yte, yp, average="macro"))
        print(f"  eps at selected config, {rep}: "
              + ", ".join(f"{k}:{v:.4f}" for k, v in out[rep].items()), flush=True)
    return out


# ---------------------------------------------------------------- C
def paired_bootstrap(tuned, reps=10000, seed=11):
    rng = np.random.default_rng(seed)
    out = {}
    for rep in REPS:
        Xtr, ytr = load_rep(rep, "train")
        Xte, yte = load_rep(rep, "test")
        t1, t2 = tuned[rep]["v1"], tuned[rep]["v2"]
        y1, _ = predict(Xtr, ytr, Xte, blocks_v1, t1["best_w"], t1["best_k"], t1["best_cw"])
        y2, _ = predict(Xtr, ytr, Xte, blocks_v2, t2["best_w"], t2["best_k"], t2["best_cw"])
        # the plain rule with k and class weighting tuned under the same protocol
        tk = tuned[rep]["knn"]
        yk, _ = knn_predict(Xtr, ytr, Xte, tk["best_k"], tk["best_cw"])
        n = len(yte)
        pos = np.where(yte == 1)[0]
        neg = np.where(yte == 0)[0]
        observed = {"v2_minus_v1": f1_score(yte, y2, average="macro") - f1_score(yte, y1, average="macro"),
                    "v2_minus_knn": f1_score(yte, y2, average="macro") - f1_score(yte, yk, average="macro")}
        res = {"plain": {"v2_minus_v1": [], "v2_minus_knn": []},
               "stratified": {"v2_minus_v1": [], "v2_minus_knn": []}}
        for _ in range(reps):
            for scheme in ("plain", "stratified"):
                if scheme == "plain":
                    idx = rng.integers(0, n, n)
                else:   # resample within each class, so the class ratio is preserved
                    idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
                f2 = f1_score(yte[idx], y2[idx], average="macro")
                res[scheme]["v2_minus_v1"].append(f2 - f1_score(yte[idx], y1[idx], average="macro"))
                res[scheme]["v2_minus_knn"].append(f2 - f1_score(yte[idx], yk[idx], average="macro"))
        out[rep] = {}
        for key in observed:
            d = np.array(res["plain"][key])
            ds = np.array(res["stratified"][key])
            out[rep][key] = {"point": float(observed[key]),        # the observed difference
                             "boot_mean": float(np.mean(d)),
                             "ci95": [float(np.percentile(d, 2.5)),
                                      float(np.percentile(d, 97.5))],
                             "ci95_stratified": [float(np.percentile(ds, 2.5)),
                                                 float(np.percentile(ds, 97.5))],
                             "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
                             "p_two_sided_stratified": float(2 * min((ds <= 0).mean(), (ds >= 0).mean()))}
            r = out[rep][key]
            print(f"  bootstrap {rep} {key:16s} observed {r['point']:+.4f} "
                  f"CI[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] p={r['p_two_sided']:.4f}  "
                  f"stratified CI[{r['ci95_stratified'][0]:+.4f},{r['ci95_stratified'][1]:+.4f}] "
                  f"p={r['p_two_sided_stratified']:.4f}", flush=True)
    return out


def main():
    tuned = json.load(open("weights_k.json"))
    out = {}
    print("A. label-swap symmetry", flush=True)
    out["label_swap"] = label_swap()
    print("B. guard constant at the selected configuration", flush=True)
    out["eps_at_selected"] = eps_at_selected(tuned)
    print("C. paired bootstrap on the predefined test split", flush=True)
    out["paired_bootstrap"] = paired_bootstrap(tuned)
    json.dump(out, open("symmetry_boot.json", "w"), indent=1)
    print("written symmetry_boot.json")


if __name__ == "__main__":
    main()
