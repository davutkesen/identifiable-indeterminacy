"""The two design rules of the proposed construction, ablated, and the
published neutrosophic feature construction N-KNN (Table 4, Table 5 and
Table S2 of the paper).

Rule 1  indeterminacy depends on the instance: I_j = min_c delta_c,j(x)
        instead of the constant sigma_j / r_j.
Rule 2  the remaining mass is resolved by class: M_c,j = (1 - I_j) s_c,j
        instead of the class-blind truth T_j = 1 - min_c delta_c,j with the
        falsity as residual.

  neither rule  the direct construction (Definition 3); weights_k.json "v1"
  rule 1 only   the direct construction with I_j = min_c delta_c,j.  By the
                identity T_j = 1 - I_j (eq. 8) its falsity is then zero and
                the construction collapses to the single block I, which is
                the corner (1, 0, 0) of the proposed construction; only k
                and class weighting are searched
  rule 2 only   the constant I_j = sigma_j / r_j with the class shares
                M_c,j = (1 - I_j) s_c,j; the full weight simplex is searched
  both rules    the proposed construction (Definition 4); weights_k.json "v2"

N-KNN  the neutrosophic feature construction of Ashika et al. (2024), in the
       form of Khan et al. (2025).  The features are min-max scaled with
       training statistics; with m+, m- and m the positive-class,
       negative-class and overall training means,
           T = 1 - ||x - m+|| / max_train ||x' - m+||
           I = 1 - ||x - m||  / max_train ||x' - m||
           F = 1 - ||x - m-|| / max_train ||x' - m-||;
       the three values are appended to the d features, all d + 3 columns
       are min-max scaled with training statistics, and the distance-weighted
       vote is applied in that space.  The published descriptions leave the
       order of scaling, the sign of F and the classifier settings partly
       open; the choices above follow Khan et al. (2025), and the
       neighbourhood size and class weighting are selected by the same
       five-fold cross-validation as for every other nearest-neighbour rule.

The plain rule on min-max scaled features, without the three appended
values, is recorded as well ("minmax_knn"), so that the effect of the
scaling can be told apart from that of T, I and F.

Every statistic is estimated on the training part (of the fold, during
selection).  The paired bootstrap on the predefined test split repeats the
procedure of exp6_symmetry_boot.paired_bootstrap (10,000 resamples, plain
and stratified) for the proposed construction against rule 1 only, rule 2
only and N-KNN, with the same generator seed and the same order of draws, so
that every interval in Table 4 is computed on one set of resamples per
representation.  The run proves this: the direct construction and the plain
rule are evaluated on the same resamples, and their intervals must equal
those stored in symmetry_boot.json to the last digit.

Output: ablation.json.
"""
import json
import time

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from common import (EPS_S, REPS, blocks_v1, blocks_v2, block_sqdists,
                    class_weights, feature_range, load_rep, scores)
from exp2_weights_k import (GRID, KMAX, KS, SEED, cv_surface, neighbours,
                            vote_from_neighbours)
from exp6_symmetry_boot import knn_predict

BOOT_REPS, BOOT_SEED = 10000, 11     # the resamples of exp6_symmetry_boot


# ------------------------------------------------------------ constructions
def blocks_r1(Xtr, ytr, Xq):
    """Rule 1 only: the single, class-blind block I_j = min_c delta_c,j."""
    return [[b[0]] for b in blocks_v2(Xtr, ytr, Xq)]


def blocks_r2(Xtr, ytr, Xq, eps_s=EPS_S):
    """Rule 2 only: constant I_j = sigma_j / r_j and M_c,j = (1 - I_j) s_c,j.
    The components still sum to one for every instance and feature."""
    classes = np.unique(ytr)
    mu = np.array([Xtr[ytr == c].mean(0) for c in classes])
    rng, nz = feature_range(Xtr)
    i_const = np.clip(np.where(nz, Xtr.std(0) / rng, 0.0), 0, 1)
    out = []
    for X in Xq:
        delta = np.where(nz, np.abs(X[None] - mu[:, None]) / rng, 0.0)
        inv = 1.0 / (delta + eps_s)
        s = inv / inv.sum(axis=0, keepdims=True)
        I = np.broadcast_to(i_const, X.shape).copy()
        M = (1.0 - I)[None] * s
        out.append([I] + [M[c] for c in range(len(classes))])
    return out


def _minmax(A):
    lo, hi = A.min(0), A.max(0)
    return lo, np.where(hi > lo, hi - lo, 1.0)


def nknn_space(Xtr, ytr, Xq):
    """N-KNN: min-max scaled features with T, I, F appended, all min-max
    scaled with training statistics.  Returns one array per query set."""
    lo, span = _minmax(Xtr)
    S = (Xtr - lo) / span
    means = (S[ytr == 1].mean(0), S.mean(0), S[ytr == 0].mean(0))   # m+, m, m-

    def dist(A, m):
        return np.sqrt(((A - m) ** 2).sum(1))
    dmax = [dist(S, m).max() for m in means]

    def augment(A):
        return np.column_stack([A] + [1.0 - dist(A, m) / dm for m, dm in zip(means, dmax)])
    lo2, span2 = _minmax(augment(S))
    return [(augment((X - lo) / span) - lo2) / span2 for X in Xq]


def minmax_blocks(Xtr, ytr, Xq):
    """The plain rule on min-max scaled features (training statistics): the
    space N-KNN starts from, without the three appended values."""
    lo, span = _minmax(Xtr)
    return [[(X - lo) / span] for X in Xq]


def nknn_blocks(Xtr, ytr, Xq):
    """N-KNN in the block interface: a single block, the augmented space."""
    return [[Z] for Z in nknn_space(Xtr, ytr, Xq)]


# ------------------------------------------------------------ selection
def cv_k_cw(X, y, fn, folds=5):
    """Mean CV macro F1 over k and class weighting for a single-block metric,
    on the same five folds as every other rule."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    acc = np.zeros((len(KS), 2))
    for tr, va in skf.split(X, y):
        Btr, Bva = fn(X[tr], y[tr], [X[tr], X[va]])
        idx, dk = neighbours(block_sqdists(Bva, Btr), [1.0], KMAX)
        cws = [class_weights(y[tr], False), class_weights(y[tr], True)]
        for ki, k in enumerate(KS):
            for ci, cw in enumerate(cws):
                yp, _ = vote_from_neighbours(idx, dk, y[tr], k, cw)
                acc[ki, ci] += f1_score(y[va], yp, average="macro")
    return acc / folds


def flat_along_wI(surf, ki, ci):
    """Largest variation of the CV surface along w_I, at the selected k and
    class weighting, with the ratio between the two class blocks fixed."""
    curves = {}
    for g, (wi, w0, w1) in enumerate(GRID):
        if w0 + w1 > 0:
            curves.setdefault(round(w1 / (w0 + w1), 6), []).append(surf[g, ki, ci])
    return float(max(np.ptp(v) for v in curves.values() if len(v) >= 2))


def fit_predict(Xtr, ytr, Xte, fn, w, k, cwon):
    Btr, Bte = fn(Xtr, ytr, [Xtr, Xte])
    idx, dk = neighbours(block_sqdists(Bte, Btr), w, k)
    return vote_from_neighbours(idx, dk, ytr, k, class_weights(ytr, cwon))


def record(w, k, cwon, cv, yte, yp, p1, extra=None):
    te = scores(yte, yp, p1)
    out = {"best_w": None if w is None else list(w), "best_k": int(k),
           "best_cw": bool(cwon), "cv_f1": float(cv),
           "test": {m: float(v) for m, v in te.items()},
           "pred_pos_rate": float(np.mean(yp))}
    out.update(extra or {})
    return out


# ------------------------------------------------------------ bootstrap
def paired_bootstrap(yte, y_ref, others, rng):
    """Proposed minus each rule in `others`, on the same resamples."""
    n = len(yte)
    pos, neg = np.where(yte == 1)[0], np.where(yte == 0)[0]
    observed = {key: f1_score(yte, y_ref, average="macro") - f1_score(yte, yo, average="macro")
                for key, yo in others.items()}
    res = {s: {key: [] for key in others} for s in ("plain", "stratified")}
    for _ in range(BOOT_REPS):
        for scheme in ("plain", "stratified"):
            if scheme == "plain":
                idx = rng.integers(0, n, n)
            else:
                idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            f_ref = f1_score(yte[idx], y_ref[idx], average="macro")
            for key, yo in others.items():
                res[scheme][key].append(f_ref - f1_score(yte[idx], yo[idx], average="macro"))
    out = {}
    for key in others:
        d, ds = np.array(res["plain"][key]), np.array(res["stratified"][key])
        out[key] = {"point": float(observed[key]), "boot_mean": float(d.mean()),
                    "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                    "ci95_stratified": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
                    "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
                    "p_two_sided_stratified": float(2 * min((ds <= 0).mean(), (ds >= 0).mean()))}
    return out


def main():
    tuned = json.load(open("weights_k.json"))
    stored_boot = json.load(open("symmetry_boot.json"))["paired_bootstrap"]
    rng = np.random.default_rng(BOOT_SEED)
    out = {}
    corner = [list(g) for g in GRID].index([1.0, 0.0, 0.0])
    for rep in REPS:
        t0 = time.time()
        Xtr, ytr = load_rep(rep, "train")
        Xte, yte = load_rep(rep, "test")
        out[rep] = {}
        # ---- rule 1 only: the single block I, i.e. the corner (1,0,0)
        acc = cv_k_cw(Xtr, ytr, blocks_r1)
        stored = np.array(tuned[rep]["v2"]["surface"])[corner]
        assert np.allclose(acc, stored, atol=1e-12, rtol=0), "rule-1 surface must equal the (1,0,0) corner"
        ki, ci = np.unravel_index(acc.argmax(), acc.shape)
        y1, p1 = fit_predict(Xtr, ytr, Xte, blocks_r1, [1.0], KS[ki], bool(ci))
        # the direct construction at its stored selection, for comparison:
        # where it puts no weight on the falsity block it is the same metric
        td = tuned[rep]["v1"]
        yd, _ = fit_predict(Xtr, ytr, Xte, blocks_v1, td["best_w"], td["best_k"], td["best_cw"])
        assert abs(f1_score(yte, yd, average="macro") - td["test"]["F1"]) < 1e-12
        out[rep]["rule1"] = record(None, KS[ki], ci, acc[ki, ci], yte, y1, p1,
                                   {"surface": acc.tolist(),
                                    "same_predictions_as_direct": bool(np.array_equal(y1, yd))})
        # ---- rule 2 only: constant I with class shares, full simplex
        surf, blockmax = cv_surface(Xtr, ytr, blocks_r2)
        gi, ki, ci = np.unravel_index(surf.argmax(), surf.shape)
        y2, p2 = fit_predict(Xtr, ytr, Xte, blocks_r2, GRID[gi], KS[ki], bool(ci))
        out[rep]["rule2"] = record(GRID[gi], KS[ki], ci, surf[gi, ki, ci], yte, y2, p2,
                                   {"block_max_dist": blockmax.tolist(),
                                    "largest_variation_along_w_I": flat_along_wI(surf, ki, ci),
                                    "surface": surf.tolist()})
        # ---- N-KNN (Ashika et al. 2024; Khan et al. 2025)
        acc = cv_k_cw(Xtr, ytr, nknn_blocks)
        ki, ci = np.unravel_index(acc.argmax(), acc.shape)
        yn, pn = fit_predict(Xtr, ytr, Xte, nknn_blocks, [1.0], KS[ki], bool(ci))
        out[rep]["nknn"] = record(None, KS[ki], ci, acc[ki, ci], yte, yn, pn,
                                  {"surface": acc.tolist()})
        # ---- the same rule without the appended values, to separate the
        #      effect of min-max scaling from that of T, I and F
        acc = cv_k_cw(Xtr, ytr, minmax_blocks)
        ki, ci = np.unravel_index(acc.argmax(), acc.shape)
        ym, pm = fit_predict(Xtr, ytr, Xte, minmax_blocks, [1.0], KS[ki], bool(ci))
        out[rep]["minmax_knn"] = record(None, KS[ki], ci, acc[ki, ci], yte, ym, pm,
                                        {"surface": acc.tolist()})
        # ---- the proposed construction, refitted at its stored selection
        tv = tuned[rep]["v2"]
        yv, _ = fit_predict(Xtr, ytr, Xte, blocks_v2, tv["best_w"], tv["best_k"], tv["best_cw"])
        assert abs(f1_score(yte, yv, average="macro") - tv["test"]["F1"]) < 1e-12
        tk = tuned[rep]["knn"]
        yk, _ = knn_predict(Xtr, ytr, Xte, tk["best_k"], tk["best_cw"])
        boot = paired_bootstrap(
            yte, yv, {"v2_minus_rule1": y1, "v2_minus_rule2": y2, "v2_minus_nknn": yn,
                      "v2_minus_v1": yd, "v2_minus_knn": yk}, rng)
        # the same resamples as the stored comparison with the direct
        # construction and the plain rule, or the intervals are not comparable
        for key in ("v2_minus_v1", "v2_minus_knn"):
            assert boot.pop(key) == stored_boot[rep][key], f"{rep} {key}: resamples differ"
        out[rep]["bootstrap"] = boot
        for key in ("rule1", "rule2", "nknn", "minmax_knn"):
            r = out[rep][key]
            print(f"  {rep:5s} {key:6s} w={r['best_w']} k={r['best_k']:2d} CW={r['best_cw']!s:5s} "
                  f"CV F1={r['cv_f1']:.4f}  test F1={r['test']['F1']:.4f} "
                  f"MCC={r['test']['MCC']:.4f} AUC={r['test']['AUC']:.4f}", flush=True)
        for key, r in out[rep]["bootstrap"].items():
            print(f"  {rep:5s} {key:15s} {r['point']:+.4f} CI[{r['ci95'][0]:+.4f},"
                  f"{r['ci95'][1]:+.4f}] strat CI[{r['ci95_stratified'][0]:+.4f},"
                  f"{r['ci95_stratified'][1]:+.4f}] p={r['p_two_sided']:.4f}", flush=True)
        print(f"  [{time.time() - t0:.0f}s]", flush=True)
    json.dump(out, open("ablation.json", "w"), indent=1)
    print("written ablation.json")


if __name__ == "__main__":
    main()
