"""Every classification rule of the paper behind one interface, for running
the protocol on further benchmarks and on subsets of a test set.

A configuration is what the protocol selects on the training split (by the
same stratified five-fold cross-validation, SEED 0, as everywhere else):

  knn        the plain rule on standardised inputs: k, class weighting
  direct     Definition 3: weight vector on the simplex, k, class weighting
  proposed   Definition 4: the same
  nknn       N-KNN (Ashika et al., 2024; Khan et al., 2025): k, class weighting
  nsknn      NS-k-NN (Akbulut et al., 2017): delta, k, class weighting
  centroid   nearest centroid on standardised inputs (nothing to select)
  trained classifiers of exp1_baselines.grids(): one hyper-parameter each

tune() runs the selection; configs_from_stored() reads the configurations
already selected for the transport-protein benchmark from the stored result
files; predict() refits a configuration on a training set and predicts.
"""
import numpy as np
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

from common import blocks_v1, blocks_v2, class_weights, scores
from exp1_baselines import cv_macro_f1, dec_fun, grids
from exp2_weights_k import GRID, KS, cv_surface, knn_cv_surface
from exp6_symmetry_boot import knn_predict, predict as block_predict
from exp7_fuzzy_baselines import nsknn_fit_predict, nsknn_select
from exp9_ablation import cv_k_cw, fit_predict, nknn_blocks

NN_RULES = ["knn", "direct", "proposed", "nknn", "nsknn", "centroid"]
TRAINED = [name for name, _, _ in grids()]


def _param(s):
    if s in (None, "None"):
        return None
    return float(s) if "." in str(s) else int(s)


def tune(Xtr, ytr, rules=None, log=print):
    """Select every rule's configuration on the training split."""
    rules = rules or NN_RULES + TRAINED
    cfg = {}
    for r in rules:
        if r == "knn":
            surf = knn_cv_surface(Xtr, ytr)
            ki, ci = np.unravel_index(surf.argmax(), surf.shape)
            cfg[r] = {"k": KS[ki], "cw": bool(ci), "cv_f1": float(surf[ki, ci])}
        elif r in ("direct", "proposed"):
            fn = blocks_v1 if r == "direct" else blocks_v2
            surf, _ = cv_surface(Xtr, ytr, fn)
            gi, ki, ci = np.unravel_index(surf.argmax(), surf.shape)
            cfg[r] = {"w": list(GRID[gi]), "k": KS[ki], "cw": bool(ci),
                      "cv_f1": float(surf[gi, ki, ci])}
        elif r == "nknn":
            surf = cv_k_cw(Xtr, ytr, nknn_blocks)
            ki, ci = np.unravel_index(surf.argmax(), surf.shape)
            cfg[r] = {"k": KS[ki], "cw": bool(ci), "cv_f1": float(surf[ki, ci])}
        elif r == "nsknn":
            cvf, delta, k, cw = nsknn_select(Xtr, ytr)
            cfg[r] = {"delta": float(delta), "k": int(k), "cw": bool(cw), "cv_f1": cvf}
        elif r == "centroid":
            cfg[r] = {}
        else:
            factory, grid = {n: (f, g) for n, f, g in grids()}[r]
            cvs = [(cv_macro_f1(factory, p, Xtr, ytr), i) for i, p in enumerate(grid)]
            best_cv, bi = max(cvs)
            cfg[r] = {"param": grid[bi], "cv_f1": float(best_cv)}
        log(f"      tuned {r:26s} {cfg[r]}")
    return cfg


def configs_from_stored(rep, wk, ab, fz, bl):
    """The configurations selected for the transport-protein benchmark, read
    from weights_k.json, ablation.json, fuzzy_baselines.json, baselines.json."""
    cfg = {"knn": {"k": wk[rep]["knn"]["best_k"], "cw": wk[rep]["knn"]["best_cw"]},
           "direct": {"w": wk[rep]["v1"]["best_w"], "k": wk[rep]["v1"]["best_k"],
                      "cw": wk[rep]["v1"]["best_cw"]},
           "proposed": {"w": wk[rep]["v2"]["best_w"], "k": wk[rep]["v2"]["best_k"],
                        "cw": wk[rep]["v2"]["best_cw"]},
           "nknn": {"k": ab[rep]["nknn"]["best_k"], "cw": ab[rep]["nknn"]["best_cw"]},
           "nsknn": {"delta": fz[rep]["NS-kNN"]["delta"], "k": int(fz[rep]["NS-kNN"]["k"]),
                     "cw": bool(fz[rep]["NS-kNN"]["cw"])},
           "centroid": {}}
    for name in TRAINED:
        cfg[name] = {"param": _param(bl[rep][name]["param"])}
    return cfg


def predict(rule, c, Xtr, ytr, Xte):
    """(predicted labels, ranking score) of a configured rule."""
    if rule == "knn":
        return knn_predict(Xtr, ytr, Xte, c["k"], c["cw"])
    if rule in ("direct", "proposed"):
        fn = blocks_v1 if rule == "direct" else blocks_v2
        return block_predict(Xtr, ytr, Xte, fn, c["w"], c["k"], c["cw"])
    if rule == "nknn":
        return fit_predict(Xtr, ytr, Xte, nknn_blocks, [1.0], c["k"], c["cw"])
    if rule == "nsknn":
        return nsknn_fit_predict(Xtr, ytr, Xte, c["delta"], c["k"], c["cw"])
    if rule == "centroid":
        sc = StandardScaler().fit(Xtr)
        A, B = sc.transform(Xtr), sc.transform(Xte)
        nc = NearestCentroid().fit(A, ytr)
        d0 = np.sqrt(((B - nc.centroids_[0]) ** 2).sum(1))
        d1 = np.sqrt(((B - nc.centroids_[1]) ** 2).sum(1))
        return nc.predict(B), d0 - d1
    factory = {n: f for n, f, _ in grids()}[rule]
    est = factory(c["param"]).fit(Xtr, ytr)
    return est.predict(Xte), dec_fun(est, Xte)


# ------------------------------------------------------------- fast scores
def macro_f1_counts(y, p):
    """Macro F1 of binary labels, rows of y and p being separate samples."""
    y = np.atleast_2d(y)
    p = np.atleast_2d(p)
    tp = ((y == 1) & (p == 1)).sum(-1)
    fp = ((y == 0) & (p == 1)).sum(-1)
    fn = ((y == 1) & (p == 0)).sum(-1)
    tn = ((y == 0) & (p == 0)).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f0 = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    return 0.5 * (f1 + f0)


def boot_diff(y, pa, pb, reps=10000, seed=0, chunk=1000):
    """Paired bootstrap of macro F1(pa) - macro F1(pb) over the instances.
    Returns the observed difference, a percentile 95% interval and the
    two-sided bootstrap p value."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for start in range(0, reps, chunk):
        idx = rng.integers(0, n, (min(chunk, reps - start), n))
        d.append(macro_f1_counts(y[idx], pa[idx]) - macro_f1_counts(y[idx], pb[idx]))
    d = np.concatenate(d)
    obs = float(macro_f1_counts(y, pa)[0] - macro_f1_counts(y, pb)[0])
    assert abs(obs - (f1_score(y, pa, average="macro") - f1_score(y, pb, average="macro"))) < 1e-12
    return {"point": obs, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
            "boot": d}


def summarise(y, p, s=None):
    out = scores(y, p, s)
    out = {m: float(v) for m, v in out.items()}
    out.update({"n": int(len(y)), "positives": int(y.sum()),
                "pred_pos_rate": float(np.mean(p))})
    return out
