"""Fuzzy and intuitionistic-fuzzy nearest-neighbour baselines.

What the construction adds over the three-component nearest-neighbour
classifiers it resembles (Table 5).  Two baselines are run under the
same protocol as everything else (k and class weighting selected by
stratified five-fold CV on the training split; one evaluation on the
predefined test split; inputs standardised, as for the plain rule).

  FKNN     the fuzzy k-nearest-neighbour rule of Keller, Gray and Givens
           (1985).  Each training point x_i receives class memberships from
           its K nearest training neighbours,
               u_c(x_i) = 0.51 + 0.49 n_c/K   if x_i belongs to class c,
               u_c(x_i) = 0.49 n_c/K          otherwise,
           where n_c counts neighbours of class c.  A query is assigned the
           class maximising sum_j u_c(x_j) d_j^{-2} over its k neighbours.

  IF-KNN   an intuitionistic-fuzzy variant defined here, not attributed to
           any published formulation.  A training point whose neighbourhood
           is evenly mixed is given hesitation
               pi_i = 1 - |n_1 - n_0| / K,
           and intuitionistic membership and non-membership
               mu_{c,i} = (1 - pi_i) u_c(x_i),   nu_{c,i} = (1 - pi_i)(1 - u_c(x_i)),
           so that mu + nu + pi = 1 for every point.  A query is assigned the
           class maximising sum_j (mu_{c,j} - nu_{c,j}) d_j^{-2}: neighbours
           that sit in mixed regions count for less.  This is the natural way
           to put an Atanassov triple into a nearest-neighbour vote, and it is
           the comparison the construction of the paper should be held to.

Class weighting, where selected, multiplies the contribution of class-c
neighbours by lambda_c as in the paper.  In both rules the training-phase
neighbourhood K is tied to the voting neighbourhood, K = k, and the
neighbour weights are inverse squared distances (fuzzifier m = 2).  The
ranking score used for the AUC is (S_1 - S_0) / (|S_0| + |S_1|), with S_c the
class score defined above.

  NS-kNN   the neutrosophic k-nearest-neighbour classifier of Akbulut,
           Sengur, Guo and Smarandache (Symmetry 9:179, 2017), implemented
           from the equations of that paper.  Inputs are scaled to [-1, 1]
           by the training minimum and maximum, as there.  Cluster centres
           are the class means of the training data (the supervised use of
           neutrosophic c-means described there), and every training point
           receives the memberships of its eqs. (3)-(5),
               T_ij = |x_i - c_j|^e / D_i,   I_i = |x_i - cbar_i|^e / D_i,
               F_i  = delta^e / D_i,          e = -2/(m-1),
               D_i  = sum_j |x_i - c_j|^e + |x_i - cbar_i|^e + delta^e,
           with cbar_i the mean of the two centres of largest truth.  A query
           u is assigned the class maximising its eq. (6),
               mu_j(u) = sum_i d_i (T_ij + I_i - F_i) / sum_i d_i,
               d_i = |u - x_i|^{-2/(q-1)},
           over its k nearest training points.  The paper fixes the fuzzifier
           of fuzzy k-NN at m = 2 and leaves q as "an integer"; we take
           m = q = 2.  delta is searched over {2^-10, 2^-8, ..., 2^10}, the
           grid of the paper, and k over the grid used for every rule here.
           Class weighting, where selected, multiplies the truth term of
           class j by lambda_j; because I_i and F_i carry no class index,
           the term sum_i d_i (I_i - F_i) is common to every class and drops
           out of the argmax with or without it.  The AUC score is
           S_1 / (S_0 + S_1) with S_j = lambda_j sum_i d_i T_ij.
"""
import json

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import REPS, class_weights, load_rep, scores

SEED = 0
KS = [1, 3, 5, 7, 9, 11, 15, 21]
EPS = 1e-8


def sqdist(A, B):
    D = (A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * (A @ B.T)
    return np.maximum(D, 0, out=D)


def train_memberships(Xtr, ytr, K):
    """Keller memberships u[:, c] and the balance hesitation pi for each
    training point, from its K nearest OTHER training points."""
    D = sqdist(Xtr, Xtr)
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, K - 1, axis=1)[:, :K]
    n1 = (ytr[idx] == 1).sum(1)
    n0 = K - n1
    frac = np.stack([n0, n1], 1) / K                      # (n, 2)
    u = 0.49 * frac
    u[np.arange(len(ytr)), ytr] += 0.51
    pi = 1.0 - np.abs(n1 - n0) / K
    return u, pi


def predict(Xtr, ytr, Xte, k, cw, kind):
    u, pi = train_memberships(Xtr, ytr, k)
    D = sqdist(Xte, Xtr)
    idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    w = 1.0 / (np.take_along_axis(D, idx, 1) + EPS)      # d^-2 weights
    lab = ytr[idx]
    lam = np.where(lab == 1, cw[1], cw[0])
    if kind == "FKNN":
        s = u[idx]                                        # (nte, k, 2)
        score = np.stack([(w * lam * s[:, :, 0]).sum(1),
                          (w * lam * s[:, :, 1]).sum(1)], 1)
    else:
        mu = (1 - pi[idx])[:, :, None] * u[idx]
        nu = (1 - pi[idx])[:, :, None] * (1 - u[idx])
        s = mu - nu
        score = np.stack([(w * lam * s[:, :, 0]).sum(1),
                          (w * lam * s[:, :, 1]).sum(1)], 1)
    pred = (score[:, 1] > score[:, 0]).astype(int)
    tot = np.abs(score).sum(1) + EPS
    p1 = 0.5 + 0.5 * (score[:, 1] - score[:, 0]) / tot
    return pred, p1


def cv_select(Xtr, ytr, kind, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    best = (-1, None, None)
    for k in KS:
        for cwon in (False, True):
            vals = []
            for tr, va in skf.split(Xtr, ytr):
                sc = StandardScaler().fit(Xtr[tr])
                A, B = sc.transform(Xtr[tr]), sc.transform(Xtr[va])
                yp, _ = predict(A, ytr[tr], B, k, class_weights(ytr[tr], cwon), kind)
                vals.append(f1_score(ytr[va], yp, average="macro"))
            m = float(np.mean(vals))
            if m > best[0]:
                best = (m, k, cwon)
    return best


# ------------------------------------------------------------------ NS-kNN
DELTAS = [2.0 ** e for e in range(-10, 11, 2)]
M_NCM = 2.0
Q_VOTE = 2.0
KMAX = max(KS)


def scale_pm1(Xtr, Xq):
    """[-1, 1] by the training minimum and maximum; a constant feature -> 0."""
    lo, hi = Xtr.min(0), Xtr.max(0)
    rng = hi - lo
    nz = rng > 0
    return [np.where(nz, 2.0 * (X - lo) / np.where(nz, rng, 1.0) - 1.0, 0.0)
            for X in Xq]


def ncm_parts(Xtr, ytr):
    """The delta-independent parts of eqs. (3)-(5): distances to the class
    centres and to the mean of the two centres of largest truth."""
    classes = np.unique(ytr)
    C = np.array([Xtr[ytr == c].mean(0) for c in classes])
    dc = np.sqrt(np.maximum(sqdist(Xtr, C), 0)) + 1e-12            # (n, c)
    two = np.argsort(dc, axis=1)[:, :2]       # largest truth = nearest centres
    cbar = 0.5 * (C[two[:, 0]] + C[two[:, 1]])
    dbar = np.sqrt(((Xtr - cbar) ** 2).sum(1)) + 1e-12
    return dc, dbar


def ncm_memberships(dc, dbar, delta):
    e = -2.0 / (M_NCM - 1.0)
    a, b, f = dc ** e, dbar ** e, delta ** e
    D = a.sum(1) + b + f
    return a / D[:, None], b / D, f / D            # T (n, c), I (n,), F (n,)


def ncm_iterated(Xtr, ytr, delta, tol=1e-6, max_iter=200):
    """Second reading of [27]: the class means only initialise the centres,
    and neutrosophic c-means (Guo and Sengur, 2015) is then iterated, with
    the centre update c_j = sum_i T_ij^m x_i / sum_i T_ij^m (all three NCM
    weights equal to one, as in eqs. (3)-(5) of [27]), until the centres
    move by less than tol.  Returns the memberships at convergence."""
    classes = np.unique(ytr)
    C = np.array([Xtr[ytr == c].mean(0) for c in classes])
    for _ in range(max_iter):
        dc = np.sqrt(np.maximum(sqdist(Xtr, C), 0)) + 1e-12
        two = np.argsort(dc, axis=1)[:, :2]
        cbar = 0.5 * (C[two[:, 0]] + C[two[:, 1]])
        dbar = np.sqrt(((Xtr - cbar) ** 2).sum(1)) + 1e-12
        T, I, F = ncm_memberships(dc, dbar, delta)
        Wm = T ** M_NCM
        Cn = (Wm.T @ Xtr) / Wm.sum(0)[:, None]
        if np.abs(Cn - C).max() < tol:
            C = Cn
            break
        C = Cn
    dc = np.sqrt(np.maximum(sqdist(Xtr, C), 0)) + 1e-12
    two = np.argsort(dc, axis=1)[:, :2]
    cbar = 0.5 * (C[two[:, 0]] + C[two[:, 1]])
    dbar = np.sqrt(((Xtr - cbar) ** 2).sum(1)) + 1e-12
    return ncm_memberships(dc, dbar, delta)


def ns_neighbours(Xtr, Xq, kmax):
    D = np.sqrt(sqdist(Xq, Xtr))
    kk = min(kmax, D.shape[1])
    idx = np.argpartition(D, kk - 1, axis=1)[:, :kk]
    dk = np.take_along_axis(D, idx, 1)
    o = np.argsort(dk, axis=1)
    return np.take_along_axis(idx, o, 1), np.take_along_axis(dk, o, 1)


def ns_predict(T, I, F, idx, dk, k, lam):
    w = 1.0 / (dk[:, :k] ** (2.0 / (Q_VOTE - 1.0)) + EPS)
    nb = idx[:, :k]
    common = (w * (I[nb] - F[nb])).sum(1)                 # same for every class
    S = np.stack([lam[j] * (w * T[nb, j]).sum(1) for j in range(T.shape[1])], 1)
    mu = (S + common[:, None]) / w.sum(1)[:, None]        # eq. (6) when lam = 1
    return mu.argmax(1), S[:, 1] / (S.sum(1) + EPS)


def nsknn_select(Xtr, ytr, folds=5, iterate=False):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    acc = np.zeros((len(DELTAS), len(KS), 2))
    for tr, va in skf.split(Xtr, ytr):
        A, B = scale_pm1(Xtr[tr], [Xtr[tr], Xtr[va]])
        dc, dbar = ncm_parts(A, ytr[tr])
        idx, dk = ns_neighbours(A, B, KMAX)
        lams = [class_weights(ytr[tr], False), class_weights(ytr[tr], True)]
        for di, delta in enumerate(DELTAS):
            T, I, F = (ncm_iterated(A, ytr[tr], delta) if iterate
                       else ncm_memberships(dc, dbar, delta))
            for ki, k in enumerate(KS):
                for ci, lam in enumerate(lams):
                    yp, _ = ns_predict(T, I, F, idx, dk, k, lam)
                    acc[di, ki, ci] += f1_score(ytr[va], yp, average="macro")
    acc /= folds
    di, ki, ci = np.unravel_index(acc.argmax(), acc.shape)
    return float(acc[di, ki, ci]), DELTAS[di], KS[ki], bool(ci)


def nsknn_fit_predict(Xtr, ytr, Xte, delta, k, cwon, iterate=False):
    A, B = scale_pm1(Xtr, [Xtr, Xte])
    if iterate:
        T, I, F = ncm_iterated(A, ytr, delta)
    else:
        dc, dbar = ncm_parts(A, ytr)
        T, I, F = ncm_memberships(dc, dbar, delta)
    idx, dk = ns_neighbours(A, B, k)
    return ns_predict(T, I, F, idx, dk, k, class_weights(ytr, cwon))


def main():
    out = {}
    for rep in REPS:
        Xtr, ytr = load_rep(rep, "train")
        Xte, yte = load_rep(rep, "test")
        out[rep] = {}
        for tag, it in (("NS-kNN", False), ("NS-kNN-iterated", True)):
            cvf, delta, k, cwon = nsknn_select(Xtr, ytr, iterate=it)
            yp, p1 = nsknn_fit_predict(Xtr, ytr, Xte, delta, k, cwon, iterate=it)
            s = scores(yte, yp, p1)
            s.update({"cv_f1": cvf, "k": k, "cw": cwon, "delta": delta})
            out[rep][tag] = {m: (float(v) if not isinstance(v, bool) else v)
                             for m, v in s.items()}
            print(f"  {rep} {tag:16s} k={k:2d} delta=2^{int(np.log2(delta)):+d} "
                  f"cw={cwon!s:5s} cv={cvf:.4f} test F1={s['F1']:.4f} "
                  f"MCC={s['MCC']:.4f} AUC={s['AUC']:.4f}", flush=True)
        for kind in ("FKNN", "IF-KNN"):
            cvf, k, cwon = cv_select(Xtr, ytr, kind)
            sc = StandardScaler().fit(Xtr)
            yp, p1 = predict(sc.transform(Xtr), ytr, sc.transform(Xte), k,
                             class_weights(ytr, cwon), kind)
            s = scores(yte, yp, p1)
            s.update({"cv_f1": cvf, "k": k, "cw": cwon})
            out[rep][kind] = {m: (float(v) if not isinstance(v, bool) else v)
                              for m, v in s.items()}
            print(f"  {rep} {kind:7s} k={k:2d} cw={cwon!s:5s} cv={cvf:.4f} "
                  f"test F1={s['F1']:.4f} MCC={s['MCC']:.4f} AUC={s['AUC']:.4f}",
                  flush=True)
    json.dump(out, open("fuzzy_baselines.json", "w"), indent=1)
    print("written fuzzy_baselines.json")


if __name__ == "__main__":
    main()
