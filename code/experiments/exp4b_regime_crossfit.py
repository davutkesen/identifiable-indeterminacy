"""Regime relation without mathematical coupling, and with tuning on both sides.

In

    x = F1(k-NN) - F1(majority),      y = F1(proposed) - F1(k-NN)

the same estimate of F1(k-NN) enters x with a positive sign and y with a
negative one, so sampling noise in that one number pushes x up and y down
together and manufactures a negative association on its own.  The remedy
is cross-fitting: the held-out part of every split is divided into two
disjoint stratified halves, x is computed on the first half only and y on
the second half only.  The two estimates of F1(k-NN) then come from
different instances and their noise is independent.

The relation of exp4_regime.py is measured with the transformation at
uniform weights and k = 5, whereas the headline results
use tuned configurations.  Here both arms are also tuned on the training
part of each split (k for the plain rule; the weight simplex and k for the
proposed construction), and the cross-fitted coordinates are recomputed.

Both the coupled and the cross-fitted coordinates are stored so that the
size of the coupling artefact can be reported.
"""
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from common import blocks_v2, block_sqdists, class_weights, vote_predict
from exp2_weights_k import GRID, KS, KMAX, neighbours, vote_from_neighbours
from exp4_regime import SEED, W_V2, source_a, source_b, source_c, spearman_boot

K_FIXED = 5


def plain_dist(A, B):
    D = (B * B).sum(1)[:, None] + (A * A).sum(1)[None] - 2 * (B @ A.T)
    return np.sqrt(np.maximum(D, 0, out=D), out=D)


def f1(y, yp):
    return f1_score(y, yp, average="macro")


def maj_pred(ytr, n):
    return np.full(n, np.bincount(ytr.astype(int)).argmax())


def tune_knn(Xtr, ytr, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    best = (-1, KS[0])
    for k in KS:
        vals = []
        for tr, va in skf.split(Xtr, ytr):
            sc = StandardScaler().fit(Xtr[tr])
            D = plain_dist(sc.transform(Xtr[tr]), sc.transform(Xtr[va]))
            yp, _ = vote_predict(D, ytr[tr], k, class_weights(ytr[tr], False))
            vals.append(f1(ytr[va], yp))
        m = float(np.mean(vals))
        if m > best[0]:
            best = (m, k)
    return best[1]


def tune_v2(Xtr, ytr, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    acc = np.zeros((len(GRID), len(KS)))
    cw = {0: 1.0, 1: 1.0}
    for tr, va in skf.split(Xtr, ytr):
        Btr, Bva = blocks_v2(Xtr[tr], ytr[tr], [Xtr[tr], Xtr[va]])
        Db = block_sqdists(Bva, Btr)
        for gi, w in enumerate(GRID):
            idx, dk = neighbours(Db, w, KMAX)
            for ki, k in enumerate(KS):
                yp, _ = vote_from_neighbours(idx, dk, ytr[tr], k, cw)
                acc[gi, ki] += f1(ytr[va], yp)
    gi, ki = np.unravel_index(acc.argmax(), acc.shape)
    return list(GRID[gi]), KS[ki]


def evaluate(Xtr, ytr, Xte, yte, k_knn, w, k_v2):
    """Predictions of the three rules on one held-out set."""
    sc = StandardScaler().fit(Xtr)
    D = plain_dist(sc.transform(Xtr), sc.transform(Xte))
    p_knn, _ = vote_predict(D, ytr, k_knn, class_weights(ytr, False))
    Btr, Bte = blocks_v2(Xtr, ytr, [Xtr, Xte])
    idx, dk = neighbours(block_sqdists(Bte, Btr), w, k_v2)
    p_v2, _ = vote_from_neighbours(idx, dk, ytr, k_v2, {0: 1.0, 1: 1.0})
    return maj_pred(ytr, len(yte)), p_knn, p_v2


def point(X, y, label, n_rep=3, seed=SEED):
    if len(np.unique(y)) < 2 or min(np.bincount(y.astype(int))) < 16:
        return None
    sss = StratifiedShuffleSplit(n_splits=n_rep, test_size=0.25, random_state=seed)
    rec = {"coupled": [], "crossfit": [], "coupled_tuned": [], "crossfit_tuned": []}
    for s, (tr, te) in enumerate(sss.split(X, y)):
        Xa, ya, Xb, yb = X[tr], y[tr], X[te], y[te]
        # two disjoint stratified halves of the held-out part
        hs = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + s)
        h1, h2 = next(hs.split(Xb, yb))
        try:
            k_t = tune_knn(Xa, ya)
            w_t, k2_t = tune_v2(Xa, ya)
        except Exception:
            return None
        for tag, (kk, ww, k2) in [("", (K_FIXED, W_V2, K_FIXED)),
                                  ("_tuned", (k_t, w_t, k2_t))]:
            pm, pk, pv = evaluate(Xa, ya, Xb, yb, kk, ww, k2)
            # coupled: everything on the whole held-out part
            xc = f1(yb, pk) - f1(yb, pm)
            yc = f1(yb, pv) - f1(yb, pk)
            # cross-fitted: x on half 1, y on half 2
            xx = f1(yb[h1], pk[h1]) - f1(yb[h1], pm[h1])
            yy = f1(yb[h2], pv[h2]) - f1(yb[h2], pk[h2])
            rec["coupled" + tag].append((xc, yc))
            rec["crossfit" + tag].append((xx, yy))
    out = {"task": label, "n": int(len(y)), "d": int(X.shape[1]),
           "pos_rate": float(y.mean())}
    for key, vals in rec.items():
        v = np.array(vals)
        out[key] = {"x": float(v[:, 0].mean()), "y": float(v[:, 1].mean())}
    return out


def collect():
    """The same tasks as exp4_regime, with the cross-fitting point() function."""
    import exp4_regime as base
    base.point = point                       # monkey-patch the evaluator
    pts = []
    print("source A", flush=True); pts += base.source_a()
    print("source B", flush=True); pts += base.source_b()
    print("source C", flush=True); pts += base.source_c()
    return pts


def summarise(pts, key):
    x = np.array([p[key]["x"] for p in pts])
    y = np.array([p[key]["y"] for p in pts])
    rho, p, lo, hi = spearman_boot(x, y)
    bands = {}
    for lo_, hi_, lab in [(0, 0.05, "x<0.05"), (0, 0.10, "x<0.10"),
                          (0.30, 9, "x>=0.30")]:
        m = (x >= lo_) & (x < hi_)
        bands[lab] = {"n": int(m.sum()),
                      "help_pct": float(100 * (y[m] > 0).mean()) if m.sum() else None,
                      "mean_gain": float(y[m].mean()) if m.sum() else None}
    return {"n": len(x), "rho": rho, "p": p, "ci95": [lo, hi], "bands": bands}


def main():
    pts = collect()
    out = {"points": pts, "summary": {}}
    for key in ("coupled", "crossfit", "coupled_tuned", "crossfit_tuned"):
        for src in ["ALL", "A transport-protein feature subsets", "B external real", "C synthetic"]:
            sub = pts if src == "ALL" else [p for p in pts if p["source"] == src]
            if len(sub) < 5:
                continue
            s = summarise(sub, key)
            out["summary"][f"{key}|{src}"] = s
            print(f"{key:15s} {src:28s} n={s['n']:3d} rho={s['rho']:+.3f} "
                  f"CI[{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}]  "
                  + "  ".join(f"{b}:{v['help_pct']:.0f}%/{v['mean_gain']:+.3f}"
                              for b, v in s["bands"].items() if v["n"]),
                  flush=True)
    json.dump(out, open("regime_crossfit.json", "w"), indent=1)
    print("written regime_crossfit.json")


if __name__ == "__main__":
    main()
