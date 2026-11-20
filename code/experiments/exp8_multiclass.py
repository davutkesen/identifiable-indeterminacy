"""A multiclass experiment: does the construction behave the same way when
the class blocks are not redundant?

With two classes the partition of unity makes one class block a function of
the other two components, and every binary number in the paper is, formally,
about an intuitionistic fuzzy triple.  With |C| > 2 the construction has
|C| + 1 genuinely independent blocks.  Two real multiclass datasets shipped
with scikit-learn are used:

  digits   10 classes, 64 features; subsampled to an imbalanced class
           distribution (class c keeps a decreasing share of its instances),
           so that the smallest class is about a tenth of the largest.
  wine     3 classes, 13 features, mildly imbalanced as distributed.

Protocol: stratified 75/25 split, stratified five-fold CV on the training
part for selection, one evaluation on the held-out part.  The weight vector
is parametrised as (w_I, (1-w_I)/|C|, ..., (1-w_I)/|C|): one weight on the
indeterminacy block and the remainder shared equally among the class
blocks, so that w_I is the quantity whose selected value answers the
question.  k and class weighting are searched over the usual grid.  Both
constructions and the plain rule, tuned over the same grid, are reported,
with macro F1 and the multiclass Matthews coefficient.
"""
import json

import numpy as np
from sklearn.datasets import load_digits, load_wine
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from common import blocks_v1, blocks_v2, block_sqdists

SEED = 0
KS = [1, 3, 5, 7, 9, 11, 15, 21]
WI = [round(0.1 * i, 1) for i in range(11)]


def dmat(Db, w):
    D = np.zeros_like(Db[0])
    for Di, wi in zip(Db, w):
        if wi:
            D += wi * Di
    return np.sqrt(np.maximum(D, 0, out=D), out=D)


def vote_multi(D, ytr, k, classes, lam, eps=1e-8):
    kk = min(k, D.shape[1])
    idx = np.argpartition(D, kk - 1, axis=1)[:, :kk]
    inv = 1.0 / (np.take_along_axis(D, idx, 1) + eps)
    lab = ytr[idx]
    score = np.stack([(inv * (lab == c)).sum(1) * lam[c] for c in classes], 1)
    return classes[score.argmax(1)]


def weights(tag, wi, nb):
    if tag == "v1":
        return [1 - wi, wi, 0.0] if False else None   # not used; v1 searched on (wT, wI, wF)
    return [wi] + [(1 - wi) / (nb - 1)] * (nb - 1)


def run(X, y, name):
    classes = np.unique(y)
    C = len(classes)
    lam_on = {c: len(y) / (C * (y == c).sum()) for c in classes}
    lam_off = {c: 1.0 for c in classes}
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr, te = next(sss.split(X, y))
    Xa, ya, Xb, yb = X[tr], y[tr], X[te], y[te]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    out = {"n": int(len(y)), "d": int(X.shape[1]), "classes": int(C),
           "class_sizes": [int((y == c).sum()) for c in classes]}

    # the plain rule on standardised inputs, k and class weighting tuned
    best = (-1, None, None)
    for k in KS:
        for cwon, lam in ((False, lam_off), (True, lam_on)):
            vals = []
            for a, b in skf.split(Xa, ya):
                sc = StandardScaler().fit(Xa[a])
                A, B = sc.transform(Xa[a]), sc.transform(Xa[b])
                D = dmat([((B * B).sum(1)[:, None] + (A * A).sum(1)[None] - 2 * B @ A.T)], [1.0])
                vals.append(f1_score(ya[b], vote_multi(D, ya[a], k, classes, lam), average="macro"))
            if np.mean(vals) > best[0]:
                best = (float(np.mean(vals)), k, cwon)
    sc = StandardScaler().fit(Xa)
    A, B = sc.transform(Xa), sc.transform(Xb)
    D = dmat([((B * B).sum(1)[:, None] + (A * A).sum(1)[None] - 2 * B @ A.T)], [1.0])
    yp = vote_multi(D, ya, best[1], classes, lam_on if best[2] else lam_off)
    out["knn"] = {"k": best[1], "cw": best[2], "cv_f1": best[0],
                  "test_f1": float(f1_score(yb, yp, average="macro")),
                  "test_mcc": float(matthews_corrcoef(yb, yp)),
                  "test_errors": int((yp != yb).sum())}
    out["n_test"] = int(len(yb))

    # the two constructions; v1 over the (wT, wI, wF) simplex with step 0.1
    from common import simplex_grid
    for tag, fn in [("v1", blocks_v1), ("v2", blocks_v2)]:
        grid = simplex_grid(0.1) if tag == "v1" else [weights("v2", wi, C + 1) for wi in WI]
        acc = np.zeros((len(grid), len(KS), 2))
        for a, b in skf.split(Xa, ya):
            Btr, Bva = fn(Xa[a], ya[a], [Xa[a], Xa[b]])
            Db = block_sqdists(Bva, Btr)
            for gi, w in enumerate(grid):
                D = dmat(Db, w)
                for ki, k in enumerate(KS):
                    for ci, lam in enumerate((lam_off, lam_on)):
                        acc[gi, ki, ci] += f1_score(ya[b], vote_multi(D, ya[a], k, classes, lam),
                                                    average="macro")
        acc /= 5
        gi, ki, ci = np.unravel_index(acc.argmax(), acc.shape)
        w, k, cwon = grid[gi], KS[ki], bool(ci)
        Btr, Bte = fn(Xa, ya, [Xa, Xb])
        D = dmat(block_sqdists(Bte, Btr), w)
        yp = vote_multi(D, ya, k, classes, lam_on if cwon else lam_off)
        out[tag] = {"w": [round(float(x), 3) for x in w], "k": k, "cw": cwon,
                    "cv_f1": float(acc[gi, ki, ci]),
                    "test_f1": float(f1_score(yb, yp, average="macro")),
                    "test_mcc": float(matthews_corrcoef(yb, yp)),
                    "test_errors": int((yp != yb).sum())}
        if tag == "v2":
            # the whole w_I profile at the selected k and class weighting
            out["v2_wI_profile"] = {str(wi): float(acc[i, ki, ci]) for i, wi in enumerate(WI)}
    print(f"  {name}: C={C} sizes={out['class_sizes']}", flush=True)
    for key in ("knn", "v1", "v2"):
        r = out[key]
        print(f"     {key:7s} " + (f"w={r['w']} " if 'w' in r else "") +
              f"k={r['k']} cv={r['cv_f1']:.4f} test F1={r['test_f1']:.4f} "
              f"MCC={r['test_mcc']:.4f}", flush=True)
    print("     v2 w_I profile:", {k: round(v, 3) for k, v in out["v2_wI_profile"].items()}, flush=True)
    return out


def binary_tied():
    """The binary benchmark under the multiclass parametrisation: w_I on the
    indeterminacy block and (1 - w_I)/2 on each class block, w_I in steps of
    0.1, with k and class weighting searched as everywhere else.  This
    separates the effect of having more classes from the effect of tying the
    class-block weights."""
    from common import REPS, class_weights, load_rep
    from exp2_weights_k import neighbours, vote_from_neighbours
    out = {}
    for rep in REPS:
        Xtr, ytr = load_rep(rep, "train")
        Xte, yte = load_rep(rep, "test")
        grid = [weights("v2", wi, 3) for wi in WI]
        acc = np.zeros((len(grid), len(KS), 2))
        for a, b in StratifiedKFold(n_splits=5, shuffle=True,
                                    random_state=SEED).split(Xtr, ytr):
            Btr, Bva = blocks_v2(Xtr[a], ytr[a], [Xtr[a], Xtr[b]])
            Db = block_sqdists(Bva, Btr)
            cws = [class_weights(ytr[a], False), class_weights(ytr[a], True)]
            for gi, w in enumerate(grid):
                idx, dk = neighbours(Db, w, max(KS))
                for ki, k in enumerate(KS):
                    for ci, cw in enumerate(cws):
                        yp, _ = vote_from_neighbours(idx, dk, ytr[a], k, cw)
                        acc[gi, ki, ci] += f1_score(ytr[b], yp, average="macro")
        acc /= 5
        gi, ki, ci = np.unravel_index(acc.argmax(), acc.shape)
        Btr, Bte = blocks_v2(Xtr, ytr, [Xtr, Xte])
        idx, dk = neighbours(block_sqdists(Bte, Btr), grid[gi], KS[ki])
        yp, _ = vote_from_neighbours(idx, dk, ytr, KS[ki], class_weights(ytr, bool(ci)))
        out[rep] = {"w_I": WI[gi], "k": KS[ki], "cw": bool(ci),
                    "cv_f1": float(acc[gi, ki, ci]),
                    "test_f1": float(f1_score(yte, yp, average="macro")),
                    "test_mcc": float(matthews_corrcoef(yte, yp)),
                    "profile": {str(wi): float(acc[i, ki, ci]) for i, wi in enumerate(WI)}}
        print(f"  binary, tied class weights, {rep}: w_I={WI[gi]} k={KS[ki]} "
              f"cw={bool(ci)} cv={acc[gi, ki, ci]:.4f} test F1={out[rep]['test_f1']:.4f}",
              flush=True)
    return out


def main():
    rng = np.random.default_rng(SEED)
    out = {}
    d = load_digits()
    keep = []
    for c in range(10):
        idx = np.where(d.target == c)[0]
        frac = 1.0 - 0.09 * c            # 100% of class 0 down to 19% of class 9
        keep.append(rng.choice(idx, int(round(frac * len(idx))), replace=False))
    keep = np.concatenate(keep)
    out["digits-imbalanced"] = run(d.data[keep], d.target[keep], "digits-imbalanced")
    w = load_wine()
    out["wine"] = run(w.data, w.target, "wine")
    out["binary_tied"] = binary_tied()
    json.dump(out, open("multiclass.json", "w"), indent=1)
    print("written multiclass.json")


if __name__ == "__main__":
    main()
