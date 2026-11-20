"""Trained classifiers outside the nearest-neighbour family (Table 5, rows
marked with a dagger).

A comparison confined to nearest-neighbour variants cannot show where that
family stands.  This script runs five standard discriminative learners under
the protocol used for every model in the paper: model selection by
stratified 5-fold CV on the TRAINING split only, then a single evaluation on
the predefined test split.
"""
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from common import REPS, load_rep, majority_scores, scores

SEED = 0


def grids():
    """(name, estimator factory, hyper-parameter grid) under a common protocol."""
    return [
        ("Logistic regression (L2)",
         lambda C: make_pipeline(StandardScaler(),
                                 LogisticRegression(C=C, max_iter=5000,
                                                    class_weight="balanced",
                                                    random_state=SEED)),
         [0.01, 0.1, 1.0, 10.0]),
        ("Linear SVM",
         lambda C: make_pipeline(StandardScaler(),
                                 LinearSVC(C=C, class_weight="balanced",
                                           max_iter=20000, random_state=SEED)),
         [0.001, 0.01, 0.1, 1.0]),
        ("RBF SVM",
         lambda C: make_pipeline(StandardScaler(),
                                 SVC(C=C, gamma="scale", class_weight="balanced",
                                     random_state=SEED)),
         [0.1, 1.0, 10.0]),
        ("Random forest",
         lambda md: RandomForestClassifier(n_estimators=500, max_depth=md,
                                           class_weight="balanced_subsample",
                                           n_jobs=-1, random_state=SEED),
         [None, 10, 20]),
        ("Gradient boosting",
         lambda lr: HistGradientBoostingClassifier(learning_rate=lr,
                                                   max_iter=300,
                                                   class_weight="balanced",
                                                   random_state=SEED),
         [0.03, 0.1, 0.3]),
    ]


def dec_fun(est, X):
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return est.decision_function(X)


def cv_macro_f1(factory, par, X, y, folds=5):
    from sklearn.metrics import f1_score
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    vals = []
    for tr, te in skf.split(X, y):
        est = factory(par).fit(X[tr], y[tr])
        vals.append(f1_score(y[te], est.predict(X[te]), average="macro"))
    return float(np.mean(vals))


def main():
    out = {}
    for rep in REPS:
        Xtr, ytr = load_rep(rep, "train")
        Xte, yte = load_rep(rep, "test")
        out[rep] = {"majority": majority_scores(ytr, yte)}
        print(f"\n=== {rep}  train={Xtr.shape}  test={Xte.shape} ===", flush=True)
        for name, factory, grid in grids():
            t0 = time.time()
            cvs = [(cv_macro_f1(factory, p, Xtr, ytr), p) for p in grid]
            best_cv, best_p = max(cvs)
            est = factory(best_p).fit(Xtr, ytr)
            s = scores(yte, est.predict(Xte), dec_fun(est, Xte))
            s["cv_f1"] = best_cv
            s["param"] = str(best_p)
            s["fit_predict_s"] = round(time.time() - t0, 1)
            out[rep][name] = s
            print(f"  {name:26s} cv={best_cv:.4f} param={best_p!s:6s} "
                  f"test F1={s['F1']:.4f} MCC={s['MCC']:.4f} AUC={s['AUC']:.4f} "
                  f"({s['fit_predict_s']}s)", flush=True)
    json.dump(out, open("baselines.json", "w"), indent=1, default=float)
    print("\nwritten baselines.json")


if __name__ == "__main__":
    main()
