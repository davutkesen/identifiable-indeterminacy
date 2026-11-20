"""Homology-aware repeated cross-validation on the pooled data of each
benchmark.

The predefined splits are single splits, and a random re-split of the
pooled data places homologous proteins on both sides.  Here the training
and test proteins of each benchmark are pooled, and the connected
components of the graph that joins every pair at >= 30% identity
(homology.json, exp10a) are kept whole: stratified group five-fold
cross-validation (sklearn StratifiedGroupKFold) assigns each component to
one fold, so no test protein has a homologue at that level among the
training proteins of its fold.

Inside every outer fold, each nearest-neighbour rule is selected on the
outer training part exactly as in the main protocol (rules.tune: stratified
five-fold cross-validation, SEED 0) and scored once on the outer test fold.
Gradient boosting is run at its most common setting (learning rate 0.1,
otherwise as in exp1_baselines) as the reference for the trained
classifiers.  The same is done once with ordinary stratified folds, which
ignore homology, for the transport-protein and SNARE benchmarks, to measure
how much of the random-split advantage is homology.

Differences between rules are summarised over the J folds of all repeats
by their mean and the corrected resampled t interval of Nadeau and Bengio
(2003), whose variance factor 1/J + n_test/n_train accounts for the overlap
of the training sets.  Fold results are saved as they come, one file per
benchmark and representation under groupcv_folds/, and a rerun resumes from
them; "python exp12_groupcv.py VES:DPC" runs one representation, so that two
can share a two-core machine.  groupcv.json is always written from every
fold file present.
"""

import os

# one BLAS thread: the folds are then reproducible to the last digit, and two
# stages can share a two-core machine without oversubscribing it
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import time

import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import rules as R
import seqdata as sd

REPEATS = {"ET": 2, "SNARE": 2, "VES": 1}
RANDOM_CONTROL = ("ET", "SNARE")
NN = ["knn", "direct", "proposed", "nknn"]
GB = ("Gradient boosting", 0.1)
COMPARE = [("proposed", "knn"), ("proposed", "nknn"), ("proposed", "direct"),
           ("Gradient boosting", "proposed")]
FOLD_DIR = "groupcv_folds"


def components(accs, edges):
    """Connected components of the identity graph, as one integer per accession."""
    parent = {a: a for a in accs}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b, _ in edges:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    roots = {}
    return np.array([roots.setdefault(find(a), len(roots)) for a in accs])


def corrected_interval(d, n_te, n_tr):
    """Nadeau-Bengio corrected resampled t interval for the mean of d."""
    d = np.asarray(d, float)
    J = len(d)
    var = (1.0 / J + n_te / n_tr) * d.var(ddof=1)
    half = stats.t.ppf(0.975, J - 1) * np.sqrt(var)
    t = d.mean() / np.sqrt(var) if var > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), J - 1)
    return [float(d.mean() - half), float(d.mean() + half)], float(p)


def run_fold(X, y, tr, te):
    cfg = R.tune(X[tr], y[tr], rules=NN, log=lambda s: None)
    cfg[GB[0]] = {"param": GB[1]}
    res = {}
    for rule, c in cfg.items():
        yp, s = R.predict(rule, c, X[tr], y[tr], X[te])
        res[rule] = {"F1": float(R.macro_f1_counts(y[te], yp)[0]),
                     "MCC": float(R.summarise(y[te], yp)["MCC"]),
                     "config": {k: v for k, v in c.items() if k != "cv_f1"}}
    return res


def summarise(folds):
    out = {}
    n_te = np.mean([f["n_test"] for f in folds])
    n_tr = np.mean([f["n_train"] for f in folds])
    for rule in NN + [GB[0]]:
        v = [f["rules"][rule]["F1"] for f in folds]
        out[rule] = {"mean_F1": float(np.mean(v)), "sd_F1": float(np.std(v, ddof=1))}
    for a, b in COMPARE:
        d = [f["rules"][a]["F1"] - f["rules"][b]["F1"] for f in folds]
        ci, p = corrected_interval(d, n_te, n_tr)
        out[f"{a}-{b}"] = {"mean": float(np.mean(d)), "ci95_corrected": ci, "p": p,
                           "wins": int(sum(x > 0 for x in d)), "folds": len(d)}
    return out


def fold_file(bench, rep):
    return os.path.join(FOLD_DIR, f"{bench}_{rep}.json")


def load_folds():
    done = {}
    for bench in sd.BENCHMARKS:
        for rep in sd.REPRESENTATIONS[bench]:
            f = fold_file(bench, rep)
            if os.path.exists(f):
                done.update(json.load(open(f)))
    return done


def main():
    import sys
    os.makedirs(FOLD_DIR, exist_ok=True)
    hom = json.load(open("homology.json"))
    only = {tuple(a.split(":")) for a in sys.argv[1:]}
    for bench in sd.BENCHMARKS:
        for rep in sd.REPRESENTATIONS[bench]:
            if only and (bench, rep) not in only:
                continue
            ff = fold_file(bench, rep)
            done = json.load(open(ff)) if os.path.exists(ff) else {}
            Xa, ya, aa = sd.features(bench, rep, "train")
            Xb, yb, ab = sd.features(bench, rep, "test")
            X, y, accs = np.vstack([Xa, Xb]), np.concatenate([ya, yb]), aa + ab
            groups = components(accs, hom[bench]["pooled_edges"])
            sizes = np.bincount(groups)
            schemes = [("group", r) for r in range(REPEATS[bench])]
            if bench in RANDOM_CONTROL:
                schemes.append(("random", 0))
            for scheme, r in schemes:
                if scheme == "group":
                    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r)
                    splits = list(cv.split(X, y, groups))
                else:
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
                    splits = list(cv.split(X, y))
                for f, (tr, te) in enumerate(splits):
                    key = f"{bench}|{rep}|{scheme}|{r}|{f}"
                    if key in done:
                        continue
                    t0 = time.time()
                    if scheme == "group":
                        assert not set(groups[tr]) & set(groups[te])
                    res = run_fold(X, y, tr, te)
                    done[key] = {"n_train": int(len(tr)), "n_test": int(len(te)),
                                 "test_positives": int(y[te].sum()), "rules": res}
                    json.dump(done, open(ff, "w"))
                    print(f"{key:24s} n_te={len(te):5d}  " + "  ".join(
                        f"{k[:8]}={v['F1']:.3f}" for k, v in res.items()) +
                        f"  [{time.time() - t0:.0f}s]", flush=True)
            print(f"{bench} {rep}: {len(sizes)} components, largest {sizes.max()} of {len(y)} proteins",
                  flush=True)
    done = load_folds()
    out = {"repeats": REPEATS, "folds": 5, "identity_threshold": 0.30, "results": {}}
    for bench in sd.BENCHMARKS:
        for rep in sd.REPRESENTATIONS[bench]:
            Xa, ya, aa = sd.features(bench, rep, "train")
            Xb, yb, ab = sd.features(bench, rep, "test")
            accs = aa + ab
            groups = components(accs, hom[bench]["pooled_edges"])
            o = {"n": len(accs), "components": int(groups.max() + 1),
                 "largest_component": int(np.bincount(groups).max())}
            for scheme in ("group", "random"):
                folds = [v for k, v in done.items()
                         if k.startswith(f"{bench}|{rep}|{scheme}|")]
                if folds:
                    o[scheme] = summarise(folds)
            out["results"][f"{bench}|{rep}"] = o
    json.dump(out, open("groupcv.json", "w"), indent=1)
    print("written groupcv.json")


if __name__ == "__main__":
    main()
