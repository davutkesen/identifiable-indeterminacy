"""Shared machinery for the experiments.

The key trick used throughout: the neutrosophic distance is

    d(x,z)^2 = sum_b w_b * || Z_b(x) - Z_b(z) ||^2

so the per-block squared distance matrices D_b can be computed ONCE per fold
and then recombined for any weight vector at negligible cost.  That is what
makes a 66-point simplex search over 8 values of k affordable.
"""

import os
import sys
import time

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EPS_R = 0.0     # no range guard: exact normalisation by the training range
EPS_D = 1e-8    # distance/vote guard,  eq. (3)
EPS_S = 1e-8    # inverse-distance guard in s_{c,j}, eq. (10)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "raw")
CACHE = os.path.join(ROOT, "data", "cache")
REPS = ["AAC", "DPC", "PSSM"]


# --------------------------------------------------------------------- data
def load_arff(path):
    import pandas as pd
    from scipy.io import arff
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    col = [c for c in df.columns if "class" in c.lower()][0]
    X = df.drop(columns=[col]).values.astype(np.float64)
    raw = (df[col].astype(str).str.lower().str.strip()
           .str.replace("b'", "", regex=False).str.replace("'", "", regex=False)
           .str.replace("\r", "", regex=False))
    return X, (raw == "yes").astype(int).values


def load_rep(rep, split):
    """Load one representation/split, caching the parsed arrays as .npz."""
    os.makedirs(CACHE, exist_ok=True)
    cf = os.path.join(CACHE, f"{rep}_{split}.npz")
    if os.path.exists(cf):
        z = np.load(cf)
        return z["X"], z["y"]
    X, y = load_arff(os.path.join(DATA, f"et_tp_{rep}_{split}.arff"))
    np.savez_compressed(cf, X=X, y=y)
    return X, y


# ----------------------------------------------------------- transformations
def feature_range(Xtr):
    """Exact training range of every feature, and a mask of the non-constant
    ones.  Normalising by the exact range (no additive guard) makes every
    component exactly invariant under positive per-feature affine maps; a
    constant feature, for which the range is zero, is mapped to delta = 0 for
    every instance, which is the only choice that is also invariant."""
    rng = Xtr.max(0) - Xtr.min(0)
    nz = rng > 0
    return np.where(nz, rng, 1.0), nz


def blocks_v1(Xtr, ytr, Xq, eps_r=EPS_R):
    """Direct construction (Definition 3): T = 1 - min_c |x-mu_c|/r, I = sigma/r,
    F = max(0, 1-T-I).

    Returns a list of (n_q, d) blocks in the order [T, I, F].
    """
    classes = np.unique(ytr)
    mu = np.array([Xtr[ytr == c].mean(0) for c in classes])
    sigma = Xtr.std(0)
    rng, nz = feature_range(Xtr)
    out = []
    for X in Xq:
        md = np.min(np.abs(X[None] - mu[:, None]), axis=0)
        T = np.clip(1 - np.where(nz, md / rng, 0.0), 0, 1)
        I = np.clip(np.broadcast_to(np.where(nz, sigma / rng, 0.0), X.shape), 0, 1)
        F = np.clip(1 - T - I, 0, 1)
        out.append([T, I, F])
    return out


def blocks_v2(Xtr, ytr, Xq, eps_r=EPS_R, eps_s=EPS_S):
    """Proposed construction (Definition 4): I = min_c delta_c, M_c = (1 - I) * s_c.

    Returns a list of (n_q, d) blocks in the order [I, M_0, M_1, ...].
    Satisfies  I_j + sum_c M_{c,j} = 1  exactly for every instance.
    """
    classes = np.unique(ytr)
    mu = np.array([Xtr[ytr == c].mean(0) for c in classes])
    rng, nz = feature_range(Xtr)
    out = []
    for X in Xq:
        delta = np.where(nz, np.abs(X[None] - mu[:, None]) / rng, 0.0)   # (c, n, d)
        I = np.clip(delta.min(axis=0), 0, 1)
        inv = 1.0 / (delta + eps_s)
        s = inv / inv.sum(axis=0, keepdims=True)
        M = (1.0 - I)[None] * s
        out.append([I] + [M[c] for c in range(len(classes))])
    return out


def partition_residual(Xtr, ytr, Xq, eps_r=EPS_R, eps_s=EPS_S):
    """max | I_j + sum_c M_{c,j} - 1 | over the query set.  Should be ~1e-16."""
    bl = blocks_v2(Xtr, ytr, [Xq], eps_r, eps_s)[0]
    tot = bl[0] + sum(bl[1:])
    return float(np.max(np.abs(tot - 1.0)))


# ------------------------------------------------------------ fast distances
def block_sqdists(B_te, B_tr, dtype=np.float64):
    """Squared Euclidean distance matrix per block: list of (n_te, n_tr)."""
    out = []
    for Zte, Ztr in zip(B_te, B_tr):
        a = Zte.astype(dtype)
        b = Ztr.astype(dtype)
        D = (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2.0 * (a @ b.T)
        np.maximum(D, 0, out=D)
        out.append(D)
    return out


def block_maxdist_direct(B_q, B_tr, chunk=8):
    """Largest pairwise Euclidean distance per block, by direct subtraction.

    The expansion |a|^2 + |b|^2 - 2 a.b used for the search is accurate for
    every distance that matters to a neighbour ranking, but its cancellation
    error makes an exactly-zero block look like ~1e-8.  This routine forms
    a - b explicitly, so a block that is constant across instances comes out
    as exactly 0.0.  Returns one float per block.
    """
    out = []
    for A, B in zip(B_q, B_tr):
        A = np.ascontiguousarray(A, dtype=np.float64)
        B = np.ascontiguousarray(B, dtype=np.float64)
        m = 0.0
        for i in range(0, A.shape[0], chunk):
            diff = A[i:i + chunk, None, :] - B[None, :, :]
            m = max(m, float(np.einsum("ijk,ijk->ij", diff, diff).max()))
        out.append(float(np.sqrt(m)))
    return out


def vote_predict(D, y_tr, k, cw, eps_d=EPS_D):
    """Distance-weighted, optionally class-weighted vote over the k nearest.

    D is a (n_te, n_tr) distance matrix (not squared).  cw maps class -> factor.
    Returns (pred, proba_of_class_1).
    """
    k = min(k, D.shape[1])
    idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    dk = np.take_along_axis(D, idx, 1).astype(np.float64) + eps_d
    inv = 1.0 / dk
    lab = y_tr[idx]
    s1 = (inv * (lab == 1)).sum(1) * cw[1]
    s0 = (inv * (lab == 0)).sum(1) * cw[0]
    tot = s0 + s1 + eps_d
    return (s1 > s0).astype(int), s1 / tot


def class_weights(y, enabled):
    if not enabled:
        return {0: 1.0, 1: 1.0}
    cl, cnt = np.unique(y, return_counts=True)
    return {int(c): len(y) / (len(cl) * n) for c, n in zip(cl, cnt)}


# ---------------------------------------------------------------- evaluation
def scores(y_true, y_pred, p1=None):
    out = {"ACC": accuracy_score(y_true, y_pred),
           "F1": f1_score(y_true, y_pred, average="macro"),
           "MCC": matthews_corrcoef(y_true, y_pred)}
    try:
        out["AUC"] = roc_auc_score(y_true, p1) if p1 is not None else float("nan")
    except Exception:
        out["AUC"] = float("nan")
    return out


def majority_scores(y_tr, y_te):
    maj = np.bincount(y_tr.astype(int)).argmax()
    yp = np.full_like(y_te, maj)
    return scores(y_te, yp, None)


# ------------------------------------------------------------------- simplex
def simplex_grid(step=0.1):
    """All (w1,w2,w3) with wi in multiples of `step` summing to 1.  66 points."""
    n = int(round(1.0 / step))
    pts = []
    for a in range(n + 1):
        for b in range(n - a + 1):
            c = n - a - b
            pts.append((a / n, b / n, c / n))
    return pts


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.dt = time.time() - self.t
