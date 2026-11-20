"""Diagnostics.

  confusion   confusion matrices, which explain the accuracy collapse of the
              proposed transformation combined with class weighting on DPC
              (Table S3).
  redundancy  a redundancy audit.  The ARFF files carry feature vectors only,
              with no peptide sequences or identifiers, so CD-HIT clustering
              cannot be run here.  What can be measured is near-duplication in
              feature space, and whether the predefined split behaves
              differently from random splits of the pooled data -- which
              matters because random resampling destroys whatever structure
              the predefined split encodes (Section 5).
  eps_s       sensitivity to the guard constant in the class-share denominator.
  scaling     whether standardising the inputs changes any prediction of the
              two constructions (Proposition 2).
  runtime     wall-clock scaling with sample size and dimension (Table S16).
"""
import json
import platform
import time

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from common import (REPS, blocks_v1, blocks_v2, block_sqdists, class_weights,
                    load_rep, vote_predict)

W_PAPER = [1 / 3, 1 / 3, 1 / 3]   # uniform, untuned; stated as such in the paper
K = 5


def dmat(Db, w):
    D = np.zeros_like(Db[0])
    for Di, wi in zip(Db, w):
        if wi:
            D += wi * Di
    return np.sqrt(np.maximum(D, 0, out=D), out=D)


# ---------------------------------------------------- confusion matrices
def confusions():
    out = {}
    for rep in REPS:
        Xtr, ytr = load_rep(rep, "train")
        Xte, yte = load_rep(rep, "test")
        out[rep] = {}
        sc = StandardScaler().fit(Xtr)
        A, B = sc.transform(Xtr), sc.transform(Xte)
        D = (B * B).sum(1)[:, None] + (A * A).sum(1)[None] - 2 * (B @ A.T)
        np.sqrt(np.maximum(D, 0, out=D), out=D)
        for tag, cwon in [("KNN", False), ("KNN+CW", True)]:
            yp, _ = vote_predict(D, ytr, K, class_weights(ytr, cwon))
            out[rep][tag] = confusion_matrix(yte, yp).tolist()
        for name, fn in [("v1", blocks_v1), ("v2", blocks_v2)]:
            Btr, Bte = fn(Xtr, ytr, [Xtr, Xte])
            Dw = dmat(block_sqdists(Bte, Btr), W_PAPER)
            for suffix, cwon in [("", False), ("+CW", True)]:
                yp, _ = vote_predict(Dw, ytr, K, class_weights(ytr, cwon))
                cm = confusion_matrix(yte, yp)
                out[rep][name + suffix] = cm.tolist()
        print(f"  confusion matrices: {rep} done", flush=True)
    return out


# ------------------------------------------------------ redundancy audit
def redundancy():
    out = {}
    for rep in REPS:
        Xtr, ytr = load_rep(rep, "train")
        Xte, yte = load_rep(rep, "test")
        sc = StandardScaler().fit(Xtr)
        A, B = sc.transform(Xtr), sc.transform(Xte)
        Dte = (B * B).sum(1)[:, None] + (A * A).sum(1)[None] - 2 * (B @ A.T)
        np.sqrt(np.maximum(Dte, 0, out=Dte), out=Dte)
        nn_te = Dte.min(1)
        Dtr = (A * A).sum(1)[:, None] + (A * A).sum(1)[None] - 2 * (A @ A.T)
        np.sqrt(np.maximum(Dtr, 0, out=Dtr), out=Dtr)
        np.fill_diagonal(Dtr, np.inf)
        nn_tr = Dtr.min(1)
        scale = float(np.median(nn_tr))
        out[rep] = {
            "test_to_train_nn": {
                "min": float(nn_te.min()), "median": float(np.median(nn_te)),
                "frac_below_1pct_of_median_train_nn":
                    float((nn_te < 0.01 * scale).mean()),
                "frac_below_10pct": float((nn_te < 0.10 * scale).mean()),
                "exact_duplicates": int((nn_te == 0).sum())},
            "train_internal_nn": {
                "median": scale,
                "exact_duplicate_rows":
                    int(len(A) - len(np.unique(A.round(10), axis=0)))},
        }
        print(f"  redundancy: {rep} done", flush=True)
    return out


def split_difficulty():
    """Is the predefined split harder than a random split of the pooled data?

    The resampling analysis pools train and test and re-splits at random.  If
    random splits score systematically higher, that analysis is measuring an
    easier problem than the one the predefined split poses.
    """
    out = {}
    for rep in REPS:
        Xtr, ytr = load_rep(rep, "train")
        Xte, yte = load_rep(rep, "test")

        def run(Xa, ya, Xb, yb):
            sc = StandardScaler().fit(Xa)
            A, B = sc.transform(Xa), sc.transform(Xb)
            D = (B * B).sum(1)[:, None] + (A * A).sum(1)[None] - 2 * (B @ A.T)
            np.sqrt(np.maximum(D, 0, out=D), out=D)
            yp, _ = vote_predict(D, ya, K, class_weights(ya, True))
            return f1_score(yb, yp, average="macro")

        pre = run(Xtr, ytr, Xte, yte)
        X = np.vstack([Xtr, Xte])
        y = np.concatenate([ytr, yte])
        sss = StratifiedShuffleSplit(n_splits=10, test_size=len(yte) / len(y),
                                     random_state=42)
        rnd = [run(X[a], y[a], X[b], y[b]) for a, b in sss.split(X, y)]
        out[rep] = {"predefined_split_f1": float(pre),
                    "random_split_f1_mean": float(np.mean(rnd)),
                    "random_split_f1_sd": float(np.std(rnd, ddof=1)),
                    "inflation": float(np.mean(rnd) - pre)}
        print(f"  split difficulty: {rep} predefined={pre:.4f} "
              f"random={np.mean(rnd):.4f} (+{np.mean(rnd)-pre:.4f})", flush=True)
    return out


# ------------------------------------------------ guard-constant sweep
def eps_sensitivity():
    out = {}
    for rep in REPS:
        Xtr, ytr = load_rep(rep, "train")
        Xte, yte = load_rep(rep, "test")
        out[rep] = {}
        for e in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
            Btr, Bte = blocks_v2(Xtr, ytr, [Xtr, Xte], eps_s=e)
            Dw = dmat(block_sqdists(Bte, Btr), W_PAPER)
            yp, _ = vote_predict(Dw, ytr, K, class_weights(ytr, False))
            out[rep][f"{e:.0e}"] = float(f1_score(yte, yp, average="macro"))
        print(f"  eps sweep: {rep} {out[rep]}", flush=True)
    return out


def scaling_sensitivity():
    """Does standardising the inputs before the transformation matter?"""
    out = {}
    for rep in REPS:
        Xtr, ytr = load_rep(rep, "train")
        Xte, yte = load_rep(rep, "test")
        out[rep] = {}
        for name, fn in [("v1", blocks_v1), ("v2", blocks_v2)]:
            sc = StandardScaler().fit(Xtr)
            preds = {}
            for tag, A, B in [("raw", Xtr, Xte),
                              ("standardised", sc.transform(Xtr),
                               sc.transform(Xte))]:
                Btr, Bte = fn(A, ytr, [A, B])
                Dw = dmat(block_sqdists(Bte, Btr), W_PAPER)
                yp, _ = vote_predict(Dw, ytr, K, class_weights(ytr, False))
                preds[tag] = yp
                out[rep][f"{name}/{tag}"] = float(
                    f1_score(yte, yp, average="macro"))
            out[rep][f"{name}/identical_predictions"] = bool(
                np.array_equal(preds["raw"], preds["standardised"]))
        print(f"  scaling: {rep} {out[rep]}", flush=True)
    return out


# --------------------------------------------------------- runtime scaling
REPEATS = 7


def _timed(f):
    """Median wall-clock time of REPEATS calls of f()."""
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def runtime():
    rng = np.random.default_rng(0)
    rows = []
    Xtr_f, ytr_f = load_rep("DPC", "train")

    def parts(X, y, q):
        holder = {}

        def tf():
            holder["B"] = blocks_v2(X, y, [X, q])

        def sr():
            Btr, Bq = holder["B"]
            Dw = dmat(block_sqdists(Bq, Btr), [1 / 3, 1 / 3, 1 / 3])
            vote_predict(Dw, y, K, class_weights(y, True))
        sc = StandardScaler().fit(X)
        A, B = sc.transform(X), sc.transform(q)

        def pl():
            D = (B * B).sum(1)[:, None] + (A * A).sum(1)[None] - 2 * (B @ A.T)
            np.sqrt(np.maximum(D, 0, out=D), out=D)
            vote_predict(D, y, K, class_weights(y, False))
        return tf, sr, pl

    # warm-up: the first call of each code path pays one-off library and
    # cache initialisation, which would otherwise be recorded in the first
    # row of the table
    w_idx = np.random.default_rng(1).choice(len(ytr_f), 500, replace=False)
    X0, y0 = Xtr_f[w_idx, :50], ytr_f[w_idx]
    for f in parts(X0, y0, X0[:100]):
        f()
    for n in (500, 1000, 2000, 4000):
        for d in (50, 100, 200, 400):
            idx = rng.choice(len(ytr_f), n, replace=False)
            cols = rng.choice(Xtr_f.shape[1], d, replace=False)
            X, y = Xtr_f[np.ix_(idx, cols)], ytr_f[idx]
            q = X[: max(1, n // 5)]
            tf, sr, pl = parts(X, y, q)
            t_tf = _timed(tf)
            t_sr = _timed(sr)
            t_pl = _timed(pl)
            rows.append({"n": n, "d": d, "n_query": int(len(q)),
                         "transform_s": round(t_tf, 4),
                         "search_s": round(t_sr, 4), "plain_knn_s": round(t_pl, 4),
                         "ratio": round((t_tf + t_sr) / max(t_pl, 1e-9), 1)})
    print("  runtime grid done", flush=True)
    return {"repeats": REPEATS,
            "host": {"platform": platform.platform(),
                     "processor": platform.processor() or "x86_64",
                     "python": platform.python_version(),
                     "numpy": np.__version__},
            "grid": rows}


def main():
    out = {}
    print("confusion matrices", flush=True)
    out["confusion"] = confusions()
    print("redundancy audit (feature space)", flush=True)
    out["redundancy"] = redundancy()
    print("predefined vs random split", flush=True)
    out["split_difficulty"] = split_difficulty()
    print("guard-constant sweep", flush=True)
    out["eps_s"] = eps_sensitivity()
    print("input scaling sensitivity", flush=True)
    out["scaling"] = scaling_sensitivity()
    print("runtime scaling", flush=True)
    out["runtime"] = runtime()
    json.dump(out, open("diagnostics.json", "w"), indent=1)
    print("\nwritten diagnostics.json")


if __name__ == "__main__":
    main()
