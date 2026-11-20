"""Is neighbourhood components analysis handicapped by its budget?

exp13_learned.py fits NCA with at most 50 components on a stratified
subsample of at most 3,000 training proteins.  Here the same protocol
(standardised inputs, PCA initialisation, 100 iterations, the vote of the
paper over all training proteins in the learned space, k and class weighting
selected by five-fold cross-validation with NCA refitted in every fold) is
run at a larger budget: at most 200 components and at most 5,000 training
proteins, which is every training protein of the transport-protein and SNARE
benchmarks.  The stored budget is refitted as well and must reproduce
learned.json.  The paired bootstrap (10,000 resamples) compares the two
budgets and the proposed construction with the larger one.

Where the larger budget changes nothing (the SNARE amino-acid composition:
20 features, 2,878 training proteins) the stored result is copied.
Results are written to nca_budget.json after every representation and a
rerun skips those present.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import time

import rules as R
import seqdata as sd
import exp13_learned as E

SMALL = (50, 3000)          # components, training proteins: the budget of exp13
LARGE = (200, 5000)
BOOT_REPS, BOOT_SEED = 10000, 61


def nca(budget, Xtr, ytr, Xte):
    E.NCA_COMPONENTS, E.NCA_MAX_N = budget
    try:
        return E.nca_select_predict(Xtr, ytr, Xte)
    finally:
        E.NCA_COMPONENTS, E.NCA_MAX_N = SMALL


def main():
    stored = json.load(open("learned.json"))
    out = json.load(open("nca_budget.json")) if os.path.exists("nca_budget.json") else \
        {"small": SMALL, "large": LARGE, "boot_reps": BOOT_REPS, "results": {}}
    for i, (bench, rep) in enumerate((b, r) for b in sd.BENCHMARKS
                                     for r in sd.REPRESENTATIONS[b]):
        key = f"{bench}|{rep}"
        if key in out["results"]:
            continue
        t0 = time.time()
        Xtr, ytr, _ = sd.features(bench, rep, "train")
        Xte, yte, _ = sd.features(bench, rep, "test")
        w, k, cw = E.configs(bench, rep)["proposed"]
        yv, _ = R.predict("proposed", {"w": w, "k": k, "cw": cw}, Xtr, ytr, Xte)
        ys, _, cs = nca(SMALL, Xtr, ytr, Xte)
        fs = R.summarise(yte, ys)
        assert abs(fs["F1"] - stored[key]["rules"]["nca"]["F1"]) < 1e-9, \
            f"{key}: NCA at the stored budget gives {fs['F1']}, stored {stored[key]['rules']['nca']['F1']}"
        same = Xtr.shape[1] <= SMALL[0] and len(ytr) <= SMALL[1]
        if same:
            yl, cl = ys, dict(cs)
        else:
            yl, _, cl = nca(LARGE, Xtr, ytr, Xte)
        o = {"n_train": int(len(ytr)), "d": int(Xtr.shape[1]), "identical_budgets": bool(same),
             "small": dict(fs, **cs), "large": dict(R.summarise(yte, yl), **cl),
             "fitted_on": {"small": int(min(len(ytr), SMALL[1])), "large": int(min(len(ytr), LARGE[1]))},
             "proposed": R.summarise(yte, yv), "compare": {}}
        seed = BOOT_SEED + 10 * i
        for name, (a, b) in (("large-small", (yl, ys)), ("proposed-large", (yv, yl)),
                             ("proposed-small", (yv, ys))):
            r = R.boot_diff(yte, a, b, reps=BOOT_REPS, seed=seed)
            seed += 1
            r.pop("boot")
            o["compare"][name] = r
        o["seconds"] = round(time.time() - t0, 1)
        out["results"][key] = o
        json.dump(out, open("nca_budget.json", "w"), indent=1, default=float)
        c = o["compare"]
        print(f"{bench:5s} {rep:4s} NCA small {o['small']['F1']:.4f}  large {o['large']['F1']:.4f} "
              f"({c['large-small']['point']:+.4f} [{c['large-small']['ci95'][0]:+.4f},"
              f"{c['large-small']['ci95'][1]:+.4f}])  proposed {o['proposed']['F1']:.4f}  "
              f"proposed-large {c['proposed-large']['point']:+.4f} "
              f"[{c['proposed-large']['ci95'][0]:+.4f},{c['proposed-large']['ci95'][1]:+.4f}]  "
              f"[{time.time() - t0:.0f}s]", flush=True)
    print("written nca_budget.json")


if __name__ == "__main__":
    main()
