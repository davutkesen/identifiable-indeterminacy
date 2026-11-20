"""Verification of the two constructions as implemented.

  1. The ablation at k = 5 and the uniform weight vector (1/3, 1/3, 1/3):
     plain k-NN with and without the class-weighted vote, and the direct
     (Definition 3, "v1") and proposed (Definition 4, "v2") transformations
     with and without it.  These are the untuned configurations of
     Table S3.
  2. The partition of unity of the proposed construction, max |I + sum_c M_c
     - 1| on the three test sets (Section 4).
  3. Table 1: the largest pairwise distance that each block contributes over
     the five training folds of the weight search, every block difference
     formed by direct subtraction, and the share of falsity entries that the
     clipping in eq. (6) sets to zero.

Output: verify.json.
"""
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import (REPS, blocks_v1, blocks_v2, block_maxdist_direct,
                    block_sqdists, class_weights, load_rep, majority_scores,
                    partition_residual, scores, vote_predict)

W_UNIFORM = [1 / 3, 1 / 3, 1 / 3]
K = 5


def weighted_dist(Db, w):
    D = np.zeros_like(Db[0])
    for Di, wi in zip(Db, w):
        if wi:
            D += wi * Di
    np.sqrt(D, out=D)
    return D


def run(rep):
    Xtr, ytr = load_rep(rep, "train")
    Xte, yte = load_rep(rep, "test")
    uni = {"majority": majority_scores(ytr, yte)}

    # --- plain KNN in the standardised input space --------------------------
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr).astype(np.float64), sc.transform(Xte).astype(np.float64)
    D = ((B * B).sum(1)[:, None] + (A * A).sum(1)[None] - 2 * (B @ A.T))
    np.sqrt(np.maximum(D, 0, out=D), out=D)
    for tag, cwon in [("KNN", False), ("KNN+CW", True)]:
        yp, p1 = vote_predict(D, ytr, K, class_weights(ytr, cwon))
        uni[tag] = scores(yte, yp, p1)

    # --- the two transformations, on the raw features ----------------------
    for tag, fn in [("v1", blocks_v1), ("v2", blocks_v2)]:
        Btr, Bte = fn(Xtr, ytr, [Xtr, Xte])
        Dw = weighted_dist(block_sqdists(Bte, Btr), W_UNIFORM)
        for suffix, cwon in [("", False), ("+CW", True)]:
            yp, p1 = vote_predict(Dw, ytr, K, class_weights(ytr, cwon))
            uni[tag + suffix] = scores(yte, yp, p1)

    resid = partition_residual(Xtr, ytr, Xte)
    # direct indeterminacy block, compared by direct subtraction: exactly 0
    Btr, Bte = blocks_v1(Xtr, ytr, [Xtr, Xte])
    iblock_direct = float(np.abs(Bte[1][:, None, :] - Btr[1][None, :64, :]).max())
    return uni, resid, iblock_direct


def defects(rep, folds=5, seed=0):
    """Table 1: the largest pairwise distance each block contributes, over the
    same five training folds the weight search uses, computed by direct
    subtraction for every block; and the share of falsity entries that the
    clipping in eq. (6) sets to zero on the training split."""
    Xtr, ytr = load_rep(rep, "train")
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    mx = {"v1": np.zeros(3), "v2": np.zeros(3)}
    for tr, va in skf.split(Xtr, ytr):
        for tag, fn in [("v1", blocks_v1), ("v2", blocks_v2)]:
            Btr, Bva = fn(Xtr[tr], ytr[tr], [Xtr[tr], Xtr[va]])
            mx[tag] = np.maximum(mx[tag], block_maxdist_direct(Bva, Btr))
    T, I, F = blocks_v1(Xtr, ytr, [Xtr])[0]
    clip = float(100.0 * (F == 0).mean())
    return {"v1": mx["v1"].tolist(), "v2": mx["v2"].tolist(),
            "clip_share_pct": clip}


def main():
    uniform, resids, iblock = {}, {}, {}
    for rep in REPS:
        uni, resid, ibd = run(rep)
        resids[rep] = resid
        iblock[rep] = ibd
        # stored at full precision; the tables round once, when printed
        uniform[rep] = {k: {m: float(v) for m, v in sc.items()}
                        for k, sc in uni.items()}
    print(f"Ablation at k = {K} and the uniform weight vector "
          f"{[round(w, 4) for w in W_UNIFORM]}:")
    for rep in REPS:
        for tag in ["KNN", "KNN+CW", "v1", "v1+CW", "v2", "v2+CW"]:
            print(f"   {rep:5s} {tag:7s} F1={uniform[rep][tag]['F1']:.4f} "
                  f"acc={uniform[rep][tag]['ACC']:.4f}")
    print("\npartition of unity, max |I + sum_c M_c - 1| on the test sets:")
    for rep, r in resids.items():
        print(f"   {rep:5s} {r:.3e}")
    print("\ndirect indeterminacy block, max |I_j(x)-I_j(z)| by direct subtraction:")
    for rep, v in iblock.items():
        print(f"   {rep:5s} {v:.1e}")
    print("\nlargest block distance over the training folds, direct subtraction:")
    dfx = {}
    for rep in REPS:
        dfx[rep] = defects(rep)
        r = dfx[rep]
        print(f"   {rep:5s} direct (T,I,F) = {r['v1'][0]:.4f} {r['v1'][1]:.1e} "
              f"{r['v1'][2]:.4f}   proposed (I,M0,M1) = {r['v2'][0]:.4f} "
              f"{r['v2'][1]:.4f} {r['v2'][2]:.4f}   F clipped {r['clip_share_pct']:.2f}%",
              flush=True)
    json.dump({"uniform": uniform, "iblock_direct": iblock, "defects": dfx,
               "weights_uniform": W_UNIFORM, "partition_residual": resids},
              open("verify.json", "w"), indent=1)
    print("written verify.json")


if __name__ == "__main__":
    main()
