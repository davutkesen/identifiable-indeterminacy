"""Cross-validated search over the weight simplex and over k (Table 4,
Figure 3).

  (1) The component weights are selected, not fixed: the point of the
      proposed transformation is that its indeterminacy weight is
      identifiable.  The full simplex w_1 + w_2 + w_3 = 1 is searched in
      steps of 0.1 (66 points).

  (2) k is searched over {1,3,5,7,9,11,15,21}, together with class weighting
      on or off; the plain rule is tuned over the same grid.

Selection is by stratified 5-fold CV on the TRAINING split only; the selected
configuration is then evaluated once on the predefined test split.

A by-product: for the direct construction the indeterminacy block is constant
across instances, so its contribution to every pairwise distance is exactly
zero and the CV surface must be flat along the w_I axis.  We record the block
distance maxima so that this can be stated as a measured fact rather than an
algebraic claim.
"""
import json
import time

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import (REPS, blocks_v1, blocks_v2, block_sqdists, class_weights,
                    load_rep, scores, simplex_grid, vote_predict)

SEED = 0
KS = [1, 3, 5, 7, 9, 11, 15, 21]
KMAX = max(KS)
GRID = simplex_grid(0.1)
TRANSFORMS = {"v1": blocks_v1, "v2": blocks_v2}


def neighbours(Db, w, kmax):
    """Indices and distances of the kmax nearest training points."""
    D = np.zeros_like(Db[0])
    for Di, wi in zip(Db, w):
        if wi:
            D += wi * Di
    np.sqrt(np.maximum(D, 0, out=D), out=D)
    kk = min(kmax, D.shape[1])
    idx = np.argpartition(D, kk - 1, axis=1)[:, :kk]
    dk = np.take_along_axis(D, idx, 1)
    order = np.argsort(dk, axis=1)
    return np.take_along_axis(idx, order, 1), np.take_along_axis(dk, order, 1)


def vote_from_neighbours(idx, dk, y_tr, k, cw, eps=1e-8):
    inv = 1.0 / (dk[:, :k] + eps)
    lab = y_tr[idx[:, :k]]
    s1 = (inv * (lab == 1)).sum(1) * cw[1]
    s0 = (inv * (lab == 0)).sum(1) * cw[0]
    return (s1 > s0).astype(int), s1 / (s0 + s1 + eps)


def cv_surface(X, y, fn, folds=5):
    """Mean CV macro F1 for every (weight, k, class-weighting) combination."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    acc = np.zeros((len(GRID), len(KS), 2))
    blockmax = np.zeros(3)
    for tr, va in skf.split(X, y):
        Btr, Bva = fn(X[tr], y[tr], [X[tr], X[va]])
        Db = block_sqdists(Bva, Btr)
        blockmax = np.maximum(blockmax, [float(np.sqrt(D.max())) for D in Db])
        cws = [class_weights(y[tr], False), class_weights(y[tr], True)]
        for gi, w in enumerate(GRID):
            idx, dk = neighbours(Db, w, KMAX)
            for ki, k in enumerate(KS):
                for ci, cw in enumerate(cws):
                    yp, _ = vote_from_neighbours(idx, dk, y[tr], k, cw)
                    acc[gi, ki, ci] += f1_score(y[va], yp, average="macro")
    return acc / folds, blockmax


def plain_neighbours(Xtr, Xq, kmax):
    """Neighbours of Xq in Xtr in the standardised input space, the space the
    plain rule works in throughout the paper (scaler fitted on Xtr)."""
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xq)
    D = (B * B).sum(1)[:, None] + (A * A).sum(1)[None] - 2.0 * (B @ A.T)
    return neighbours([np.maximum(D, 0)], [1.0], kmax)


def knn_cv_surface(X, y, folds=5):
    """The plain rule tuned over the same grid of k and class weighting as
    the two constructions, on the same five folds."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    acc = np.zeros((len(KS), 2))
    for tr, va in skf.split(X, y):
        idx, dk = plain_neighbours(X[tr], X[va], KMAX)
        cws = [class_weights(y[tr], False), class_weights(y[tr], True)]
        for ki, k in enumerate(KS):
            for ci, cw in enumerate(cws):
                yp, _ = vote_from_neighbours(idx, dk, y[tr], k, cw)
                acc[ki, ci] += f1_score(y[va], yp, average="macro")
    return acc / folds


def test_eval(Xtr, ytr, Xte, yte, fn, w, k, cwon):
    Btr, Bte = fn(Xtr, ytr, [Xtr, Xte])
    Db = block_sqdists(Bte, Btr)
    idx, dk = neighbours(Db, w, k)
    yp, p1 = vote_from_neighbours(idx, dk, ytr, k, class_weights(ytr, cwon))
    return scores(yte, yp, p1)


def main():
    out = {}
    for rep in REPS:
        Xtr, ytr = load_rep(rep, "train")
        Xte, yte = load_rep(rep, "test")
        out[rep] = {}
        print(f"\n################ {rep}  train={Xtr.shape} ################",
              flush=True)
        for tag, fn in TRANSFORMS.items():
            t0 = time.time()
            surf, blockmax = cv_surface(Xtr, ytr, fn)
            gi, ki, ci = np.unravel_index(surf.argmax(), surf.shape)
            w, k, cwon = GRID[gi], KS[ki], bool(ci)
            te = test_eval(Xtr, ytr, Xte, yte, fn, w, k, cwon)
            out[rep][tag] = {
                "best_w": list(w), "best_k": k, "best_cw": cwon,
                "cv_f1": float(surf[gi, ki, ci]),
                "test": {m: float(v) for m, v in te.items()},
                "block_max_dist": blockmax.tolist(),
                "surface": surf.tolist(),
                "grid": [list(g) for g in GRID], "ks": KS,
                "seconds": round(time.time() - t0, 1),
            }
            print(f"  {tag}: best w={w} k={k} CW={cwon}  CV F1={surf[gi,ki,ci]:.4f}"
                  f"  ->  test F1={te['F1']:.4f} MCC={te['MCC']:.4f}"
                  f"  [{round(time.time()-t0,1)}s]", flush=True)
            print(f"      max block distance (b1,b2,b3) = "
                  f"{blockmax[0]:.3e} {blockmax[1]:.3e} {blockmax[2]:.3e}",
                  flush=True)
        # the plain rule, tuned under the same protocol
        t0 = time.time()
        surf = knn_cv_surface(Xtr, ytr)
        ki, ci = np.unravel_index(surf.argmax(), surf.shape)
        k, cwon = KS[ki], bool(ci)
        idx, dk = plain_neighbours(Xtr, Xte, k)
        yp, p1 = vote_from_neighbours(idx, dk, ytr, k, class_weights(ytr, cwon))
        te = scores(yte, yp, p1)
        out[rep]["knn"] = {"best_k": k, "best_cw": cwon,
                           "cv_f1": float(surf[ki, ci]),
                           "test": {m: float(v) for m, v in te.items()},
                           "surface": surf.tolist(), "ks": KS,
                           "seconds": round(time.time() - t0, 1)}
        print(f"  knn: best k={k} CW={cwon}  CV F1={surf[ki,ci]:.4f}  ->  "
              f"test F1={te['F1']:.4f} MCC={te['MCC']:.4f}", flush=True)
    json.dump(out, open("weights_k.json", "w"), indent=1)
    print("\nwritten weights_k.json")


if __name__ == "__main__":
    main()
