"""Learned metrics, and the construction inside trained models.

Two questions a nearest-neighbour metric has to answer once trained
classifiers are known to be ahead of every nearest-neighbour rule:

(1) How does a training-free metric compare with a metric learned for the
    same rule?  Neighbourhood components analysis (NCA; Goldberger, Roweis,
    Hinton and Salakhutdinov, 2005), the learned linear metric of
    scikit-learn, is fitted on standardised inputs (PCA initialisation, at
    most 50 components, 100 iterations), and the distance-weighted vote of
    the paper is run in the learned space; k and class weighting are
    selected by the same five-fold cross-validation, with NCA refitted on
    the training part of every fold.  NCA is fitted on a stratified
    subsample of at most 3,000 training proteins, since its memory and time
    grow with the square of the sample; the vote uses all of them.

(2) Does the construction help a trained model?  The proposed image
    [I, M_0, M_1] of every training protein is computed with cross-fitting
    (centroids and ranges from the other four of five folds), and of every
    test protein from the whole training split.  Two models are compared
    with themselves on the raw features, under the grids of exp1_baselines:
      RBF SVM on the image weighted by the selected nearest-neighbour
      weights, so that its Euclidean distance is the neutrosophic metric
      and the kernel is exp(-gamma d^2);
      gradient boosting on the raw features with the image appended, the
      same augmentation N-KNN performs with its three values, at the
      learning rate selected for gradient boosting on the raw features (so
      that the image is the only difference between the two fits).
    The RBF SVM is selected over C as in exp1_baselines on either input.
    The paired bootstrap (10,000 resamples) gives the test difference.
    Results are written after every benchmark and representation, and a
    rerun skips those already in learned.json.
"""

import os

# one BLAS thread: the folds are then reproducible to the last digit, and two
# stages can share a two-core machine without oversubscribing it
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

import rules as R
import seqdata as sd
from common import blocks_v2, class_weights
from exp2_weights_k import KMAX, KS, SEED, neighbours, vote_from_neighbours

NCA_COMPONENTS, NCA_ITER, NCA_MAX_N = 50, 100, 3000
BOOT_REPS, BOOT_SEED = 10000, 29
SVM_C = [0.1, 1.0, 10.0]


# ------------------------------------------------------------------ NCA
def nca_fit(X, y):
    n_comp = min(NCA_COMPONENTS, X.shape[1])
    idx = np.arange(len(y))
    if len(y) > NCA_MAX_N:
        rng = np.random.default_rng(SEED)
        keep = []
        for c in (0, 1):
            ic = idx[y == c]
            keep.append(rng.choice(ic, int(round(NCA_MAX_N * len(ic) / len(y))), replace=False))
        idx = np.sort(np.concatenate(keep))
    model = make_pipeline(StandardScaler(),
                          NeighborhoodComponentsAnalysis(n_components=n_comp, init="pca",
                                                         max_iter=NCA_ITER, random_state=SEED))
    return model.fit(X[idx], y[idx])


def nca_neighbours(model, Xtr, Xq, kmax):
    A, B = model.transform(Xtr), model.transform(Xq)
    D = (B * B).sum(1)[:, None] + (A * A).sum(1)[None] - 2.0 * (B @ A.T)
    return neighbours([np.maximum(D, 0)], [1.0], kmax)


def nca_select_predict(Xtr, ytr, Xte):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    acc = np.zeros((len(KS), 2))
    for tr, va in skf.split(Xtr, ytr):
        m = nca_fit(Xtr[tr], ytr[tr])
        idx, dk = nca_neighbours(m, Xtr[tr], Xtr[va], KMAX)
        cws = [class_weights(ytr[tr], False), class_weights(ytr[tr], True)]
        for ki, k in enumerate(KS):
            for ci, cw in enumerate(cws):
                yp, _ = vote_from_neighbours(idx, dk, ytr[tr], k, cw)
                acc[ki, ci] += f1_score(ytr[va], yp, average="macro")
    acc /= 5
    ki, ci = np.unravel_index(acc.argmax(), acc.shape)
    m = nca_fit(Xtr, ytr)
    idx, dk = nca_neighbours(m, Xtr, Xte, KS[ki])
    yp, p1 = vote_from_neighbours(idx, dk, ytr, KS[ki], class_weights(ytr, bool(ci)))
    return yp, p1, {"k": KS[ki], "cw": bool(ci), "cv_f1": float(acc[ki, ci]),
                    "components": min(NCA_COMPONENTS, Xtr.shape[1])}


# ------------------------------------------------------ image for models
def crossfit_image(Xtr, ytr, Xte):
    """[I, M_0, M_1] of training proteins (cross-fitted) and test proteins."""
    Ztr = np.zeros((len(ytr), 3 * Xtr.shape[1]))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for a, b in skf.split(Xtr, ytr):
        Ztr[b] = np.hstack(blocks_v2(Xtr[a], ytr[a], [Xtr[b]])[0])
    Zte = np.hstack(blocks_v2(Xtr, ytr, [Xte])[0])
    return Ztr, Zte


def weighted(Z, w, d):
    return np.hstack([np.sqrt(wb) * Z[:, i * d:(i + 1) * d] for i, wb in enumerate(w)])


def svm(C):
    return SVC(C=C, gamma="scale", class_weight="balanced", random_state=SEED)


def gb(lr):
    return HistGradientBoostingClassifier(learning_rate=lr, max_iter=300,
                                          class_weight="balanced", random_state=SEED)


def select_fit(factory, grid, Xtr, ytr, Xte):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best = None
    for p in grid:
        v = np.mean([f1_score(ytr[va], factory(p).fit(Xtr[tr], ytr[tr]).predict(Xtr[va]),
                              average="macro") for tr, va in skf.split(Xtr, ytr)])
        if best is None or v > best[0]:
            best = (v, p)
    est = factory(best[1]).fit(Xtr, ytr)
    return est.predict(Xte), {"param": best[1], "cv_f1": float(best[0])}


# ----------------------------------------------------------------- main
def configs(bench, rep):
    if bench == "ET":
        wk = json.load(open("weights_k.json"))[rep]
        gb_lr = R._param(json.load(open("baselines.json"))[rep]["Gradient boosting"]["param"])
        return {"knn": (wk["knn"]["best_k"], wk["knn"]["best_cw"]),
                "proposed": (wk["v2"]["best_w"], wk["v2"]["best_k"], wk["v2"]["best_cw"]),
                "gb_lr": gb_lr}
    pre = json.load(open("prereg.json"))["benchmarks"][bench][rep]["configs"]
    return {"knn": (pre["knn"]["k"], pre["knn"]["cw"]),
            "proposed": (pre["proposed"]["w"], pre["proposed"]["k"], pre["proposed"]["cw"]),
            "gb_lr": pre["Gradient boosting"]["param"]}


def main():
    import os
    out = json.load(open("learned.json")) if os.path.exists("learned.json") else {}
    for i, (bench, rep) in enumerate((b, r) for b in sd.BENCHMARKS
                                     for r in sd.REPRESENTATIONS[b]):
        seed = BOOT_SEED + 10 * i
        if f"{bench}|{rep}" in out:
            continue
        t0 = time.time()
        Xtr, ytr, _ = sd.features(bench, rep, "train")
        Xte, yte, _ = sd.features(bench, rep, "test")
        c = configs(bench, rep)
        yk, _ = R.predict("knn", {"k": c["knn"][0], "cw": c["knn"][1]}, Xtr, ytr, Xte)
        w, k, cw = c["proposed"]
        yv, _ = R.predict("proposed", {"w": w, "k": k, "cw": cw}, Xtr, ytr, Xte)
        yn, pn, ncfg = nca_select_predict(Xtr, ytr, Xte)
        d = Xtr.shape[1]
        Ztr, Zte = crossfit_image(Xtr, ytr, Xte)
        Wtr, Wte = weighted(Ztr, w, d), weighted(Zte, w, d)
        sc = StandardScaler().fit(Xtr)
        ys_raw, cs_raw = select_fit(svm, SVM_C, sc.transform(Xtr), ytr, sc.transform(Xte))
        ys_img, cs_img = select_fit(svm, SVM_C, Wtr, ytr, Wte)
        yg_raw = gb(c["gb_lr"]).fit(Xtr, ytr).predict(Xte)
        yg_aug = gb(c["gb_lr"]).fit(np.hstack([Xtr, Ztr]), ytr).predict(np.hstack([Xte, Zte]))
        cg_raw = cg_aug = {"param": c["gb_lr"]}
        preds = {"knn": yk, "proposed": yv, "nca": yn, "svm_raw": ys_raw, "svm_image": ys_img,
                 "gb_raw": yg_raw, "gb_augmented": yg_aug}
        o = {"rules": {r: R.summarise(yte, p) for r, p in preds.items()},
             "configs": {"nca": ncfg, "svm_raw": cs_raw, "svm_image": cs_img,
                         "gb_raw": cg_raw, "gb_augmented": cg_aug},
             "compare": {}}
        for a, b in (("proposed", "nca"), ("nca", "knn"), ("svm_image", "svm_raw"),
                     ("gb_augmented", "gb_raw")):
            r = R.boot_diff(yte, preds[a], preds[b], reps=BOOT_REPS, seed=seed)
            seed += 1
            r.pop("boot")
            o["compare"][f"{a}-{b}"] = r
        out[f"{bench}|{rep}"] = o
        json.dump(out, open("learned.json", "w"), indent=1, default=float)
        print(f"{bench:5s} {rep:4s} " + "  ".join(f"{r}={v['F1']:.3f}" for r, v in o["rules"].items())
              + f"  [{time.time() - t0:.0f}s]", flush=True)
        for key, v in o["compare"].items():
            print(f"      {key:24s} {v['point']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]",
                  flush=True)
    print("written learned.json")


if __name__ == "__main__":
    main()
