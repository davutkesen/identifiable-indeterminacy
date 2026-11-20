"""Does the selected weight vector depend on the 0.1 grid?

The weight search of the paper (exp2_weights_k.py) covers the simplex in
steps of 0.1.  Here, on the transport-protein benchmark and with the same
five folds, k grid and class-weighting options, the proposed construction
is searched again

  (a) on the simplex in steps of 0.05 (231 points), which contains the 66
      points of the main grid, and
  (b) on the tied family (w_I, (1-w_I)/2, (1-w_I)/2) with w_I in steps of
      0.01, the near-continuous version of the tied check of the paper.

Each selection is evaluated once on the predefined test split.  The cross-
validated scores at the 66 shared points are checked against the stored
surface of exp2 (same folds, so they must agree to rounding).  Results are
written to finegrid.json.
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import time

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

from common import REPS, block_sqdists, blocks_v2, class_weights, load_rep, simplex_grid
from exp2_weights_k import KMAX, KS, SEED, neighbours, vote_from_neighbours

FINE = simplex_grid(0.05)
TIED = [(round(a / 100, 2), round((1 - a / 100) / 2, 3), round((1 - a / 100) / 2, 3))
        for a in range(101)]


def surface(X, y, grid):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    acc = np.zeros((len(grid), len(KS), 2))
    for tr, va in skf.split(X, y):
        Btr, Bva = blocks_v2(X[tr], y[tr], [X[tr], X[va]])
        Db = block_sqdists(Bva, Btr)
        cws = [class_weights(y[tr], False), class_weights(y[tr], True)]
        for gi, w in enumerate(grid):
            idx, dk = neighbours(Db, w, KMAX)
            for ki, k in enumerate(KS):
                for ci, cw in enumerate(cws):
                    yp, _ = vote_from_neighbours(idx, dk, y[tr], k, cw)
                    acc[gi, ki, ci] += f1_score(y[va], yp, average="macro")
    return acc / 5


def select_and_test(grid, acc, Xtr, ytr, Xte, yte):
    gi, ki, ci = np.unravel_index(acc.argmax(), acc.shape)
    w, k, cw = grid[gi], KS[ki], bool(ci)
    Btr, Bte = blocks_v2(Xtr, ytr, [Xtr, Xte])
    idx, dk = neighbours(block_sqdists(Bte, Btr), w, k)
    yp, _ = vote_from_neighbours(idx, dk, ytr, k, class_weights(ytr, cw))
    return {"w": [float(v) for v in w], "k": k, "cw": cw, "cv_f1": float(acc[gi, ki, ci]),
            "test_f1": float(f1_score(yte, yp, average="macro")),
            "test_mcc": float(matthews_corrcoef(yte, yp))}


def main():
    wk = json.load(open("weights_k.json"))
    out = {"fine_step": 0.05, "tied_step": 0.01, "results": {}}
    for rep in REPS:
        t0 = time.time()
        Xtr, ytr = load_rep(rep, "train")
        Xte, yte = load_rep(rep, "test")
        acc = surface(Xtr, ytr, FINE)
        # the 66 points of the main grid, against the stored surface of exp2
        main_grid = [tuple(g) for g in wk[rep]["v2"]["grid"]]
        stored = np.array(wk[rep]["v2"]["surface"])
        pos = {tuple(round(v, 10) for v in g): i for i, g in enumerate(FINE)}
        shared = np.array([acc[pos[tuple(round(v, 10) for v in g)]] for g in main_grid])
        dev = float(np.abs(shared - stored).max())
        assert dev < 1e-12, f"{rep}: fine-grid surface differs from exp2 by {dev}"
        fine = select_and_test(FINE, acc, Xtr, ytr, Xte, yte)
        acc_t = surface(Xtr, ytr, TIED)
        tied = select_and_test(TIED, acc_t, Xtr, ytr, Xte, yte)
        best_ki_ci = np.unravel_index(acc_t.argmax(), acc_t.shape)[1:]
        tied["profile"] = {f"{TIED[i][0]:.2f}": float(acc_t[(i,) + tuple(best_ki_ci)])
                           for i in range(len(TIED))}
        out["results"][rep] = {"main": {"w": wk[rep]["v2"]["best_w"], "k": wk[rep]["v2"]["best_k"],
                                        "cw": wk[rep]["v2"]["best_cw"],
                                        "cv_f1": wk[rep]["v2"]["cv_f1"],
                                        "test_f1": wk[rep]["v2"]["test"]["F1"]},
                               "fine": fine, "tied_fine": tied,
                               "shared_point_deviation": dev,
                               "seconds": round(time.time() - t0, 1)}
        print(f"{rep:4s} main w={wk[rep]['v2']['best_w']} test={wk[rep]['v2']['test']['F1']:.4f} | "
              f"0.05: w={fine['w']} k={fine['k']} cw={fine['cw']} cv={fine['cv_f1']:.4f} "
              f"test={fine['test_f1']:.4f} | tied 0.01: w_I={tied['w'][0]:.2f} k={tied['k']} "
              f"cw={tied['cw']} cv={tied['cv_f1']:.4f} test={tied['test_f1']:.4f}  "
              f"[{time.time() - t0:.0f}s]", flush=True)
        json.dump(out, open("finegrid.json", "w"), indent=1)
    print("written finegrid.json")


if __name__ == "__main__":
    main()
