"""The regime relation over 133 tasks at a fixed configuration (Table S10,
Figure 4a).

The claim under test is that the transformation helps when the untransformed
metric is close to degenerate and hurts when the untransformed metric already
works.  Three feature representations give only three points, so the
relation is measured over a larger collection of tasks.

For every task we compute two coordinates

    x = F1_macro(plain k-NN) - F1_macro(majority)      how well the raw metric works
    y = F1_macro(neutrosophic k-NN) - F1_macro(plain k-NN)   what the transform adds

and test for a negative association.  Points come from three sources:

  A  the real transport-protein data, with random feature subsets of varying
     size.  This
     holds the domain fixed and varies only the quality of the raw metric,
     which is exactly the axis the claim is about.
  B  real external binary tasks built one-vs-rest from datasets shipped inside
     scikit-learn, each at its natural positive rate and, where that rate
     exceeds 0.20, once more with the positives subsampled to a rate of 0.20.
  C  a controlled synthetic sweep over class separation and dimension.

The configuration is held FIXED across tasks (k = 5, no class weighting on
either side) so that y isolates the transformation.  Tuning per task would
make the result a statement about tuning, not about the transformation;
exp4b_regime_crossfit.py re-measures the relation with both arms tuned.
"""
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import (load_breast_cancer, load_digits, load_iris,
                              load_wine, make_classification)
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from common import (REPS, blocks_v2, block_sqdists, class_weights, load_rep,
                    vote_predict)

K = 5
SEED = 7
W_V2 = [1 / 3, 1 / 3, 1 / 3]      # symmetric; no per-task tuning


def _knn_f1(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    D = (B * B).sum(1)[:, None] + (A * A).sum(1)[None] - 2 * (B @ A.T)
    np.sqrt(np.maximum(D, 0, out=D), out=D)
    yp, _ = vote_predict(D, ytr, K, class_weights(ytr, False))
    return f1_score(yte, yp, average="macro")


def _nknn_f1(Xtr, ytr, Xte, yte):
    Btr, Bte = blocks_v2(Xtr, ytr, [Xtr, Xte])
    Db = block_sqdists(Bte, Btr)
    D = np.zeros_like(Db[0])
    for Di, wi in zip(Db, W_V2):
        D += wi * Di
    np.sqrt(np.maximum(D, 0, out=D), out=D)
    yp, _ = vote_predict(D, ytr, K, class_weights(ytr, False))
    return f1_score(yte, yp, average="macro")


def _majority_f1(ytr, yte):
    maj = np.bincount(ytr.astype(int)).argmax()
    return f1_score(yte, np.full_like(yte, maj), average="macro")


def point(X, y, label, n_rep=3, seed=SEED):
    """Average the two coordinates over a few stratified 75/25 splits."""
    if len(np.unique(y)) < 2 or min(np.bincount(y.astype(int))) < 8:
        return None
    sss = StratifiedShuffleSplit(n_splits=n_rep, test_size=0.25,
                                 random_state=seed)
    xs, ys = [], []
    for tr, te in sss.split(X, y):
        Xa, Xb, ya, yb = X[tr], X[te], y[tr], y[te]
        try:
            mj, kn, nk = (_majority_f1(ya, yb), _knn_f1(Xa, ya, Xb, yb),
                          _nknn_f1(Xa, ya, Xb, yb))
        except Exception:
            return None
        xs.append(kn - mj)
        ys.append(nk - kn)
    return {"task": label, "n": int(len(y)), "d": int(X.shape[1]),
            "pos_rate": float(y.mean()), "x": float(np.mean(xs)),
            "y": float(np.mean(ys))}


# ------------------------------------------------------------------ source A
def source_a():
    rng = np.random.default_rng(SEED)
    pts = []
    for rep in REPS:
        Xtr, ytr = load_rep(rep, "train")
        Xte, yte = load_rep(rep, "test")
        X = np.vstack([Xtr, Xte])
        y = np.concatenate([ytr, yte])
        if len(y) > 2500:                       # keep the sweep affordable
            sel = rng.choice(len(y), 2500, replace=False)
            X, y = X[sel], y[sel]
        d = X.shape[1]
        sizes = [s for s in (5, 10, 20, 50, 100, 200, 400) if s <= d]
        for s in sizes:
            draws = 1 if s == d else 5
            for r in range(draws):
                cols = (np.arange(d) if s == d
                        else rng.choice(d, s, replace=False))
                p = point(X[:, cols], y, f"{rep}/d={s}/r{r}")
                if p:
                    p["source"] = "A transport-protein feature subsets"
                    pts.append(p)
        print(f"  source A: {rep} done ({len(pts)} points so far)", flush=True)
    return pts


# ------------------------------------------------------------------ source B
def source_b():
    rng = np.random.default_rng(SEED)
    pts = []
    sets = [("digits", load_digits()), ("wine", load_wine()),
            ("iris", load_iris())]
    tasks = []
    for name, ds in sets:
        for c in np.unique(ds.target):
            tasks.append((f"{name}-vs-rest-{c}", ds.data,
                          (ds.target == c).astype(int)))
    bc = load_breast_cancer()
    tasks.append(("breast-cancer", bc.data, bc.target))
    for label, X, y in tasks:
        for rate in (None, 0.20):
            Xs, ys = X, y
            if rate is not None:
                pos = np.where(y == 1)[0]
                neg = np.where(y == 0)[0]
                keep = max(8, int(round(rate * len(neg) / (1 - rate))))
                if keep >= len(pos):
                    # the natural positive rate is already at or below the
                    # target (the digits tasks): a second version would be an
                    # exact copy of the first, so none is made
                    continue
                pos = rng.choice(pos, keep, replace=False)
                sel = np.concatenate([pos, neg])
                Xs, ys = X[sel], y[sel]
            p = point(Xs, ys, f"{label}{'' if rate is None else '/imb'}")
            if p:
                p["source"] = "B external real"
                pts.append(p)
    print(f"  source B: {len(pts)} points", flush=True)
    return pts


# ------------------------------------------------------------------ source C
def source_c():
    pts = []
    i = 0
    for d in (10, 50, 200):
        for sep in (0.2, 0.4, 0.7, 1.0, 1.5, 2.5):
            for inf in (0.25, 0.75):
                i += 1
                X, y = make_classification(
                    n_samples=1200, n_features=d,
                    n_informative=max(2, int(d * inf)), n_redundant=0,
                    n_clusters_per_class=2, class_sep=sep, flip_y=0.02,
                    weights=[0.8, 0.2], random_state=SEED + i)
                p = point(X, y, f"synth/d={d}/sep={sep}/inf={inf}")
                if p:
                    p["source"] = "C synthetic"
                    pts.append(p)
    print(f"  source C: {len(pts)} points", flush=True)
    return pts


def spearman_boot(x, y, reps=10000, seed=3):
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    rs = np.array([spearmanr(x[i], y[i])[0] for i in idx])
    rs = rs[~np.isnan(rs)]
    return (float(rho), float(p),
            float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5)))


def mean_boot(y, reps=10000, seed=5):
    """Percentile bootstrap 95% interval of a mean, with the task as the unit."""
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    m = y[rng.integers(0, len(y), size=(reps, len(y)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    pts = []
    print("source A (real transport-protein data, feature subsets)", flush=True)
    pts += source_a()
    print("source B (external real tasks)", flush=True)
    pts += source_b()
    print("source C (controlled synthetic)", flush=True)
    pts += source_c()

    out = {"config": {"k": K, "weights": W_V2, "class_weighting": False},
           "points": pts, "spearman": {}}
    for name in ["A transport-protein feature subsets", "B external real",
                 "C synthetic", "ALL"]:
        sub = pts if name == "ALL" else [p for p in pts if p["source"] == name]
        if len(sub) < 5:
            continue
        x = np.array([p["x"] for p in sub])
        y = np.array([p["y"] for p in sub])
        rho, p, lo, hi = spearman_boot(x, y)
        out["spearman"][name] = {"n": len(sub), "rho": rho, "p": p,
                                 "ci95": [lo, hi],
                                 "frac_help": float((y > 0).mean())}
        print(f"{name:32s} n={len(sub):4d}  rho={rho:+.3f} "
              f"CI[{lo:+.3f},{hi:+.3f}] p={p:.2e}  helps on "
              f"{100*(y>0).mean():.0f}%", flush=True)

    json.dump(out, open("regime.json", "w"), indent=1)
    print("\nwritten regime.json")


if __name__ == "__main__":
    main()
