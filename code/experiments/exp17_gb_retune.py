"""Gradient boosting on the features with the image appended, with its
learning rate re-selected for the augmented input.

exp13_learned.py appends the cross-fitted image [I, M_0, M_1] to the raw
features and runs gradient boosting at the learning rate selected for the
raw features, so that the image is the only difference between the two fits.
A model tuned for d columns need not be tuned for 4d, so here the learning
rate is selected again, over the grid of exp1_baselines, by the same
stratified five-fold cross-validation.  The image is cross-fitted inside
every fold (centroids and ranges of a held-out part never come from that
part, nor from the validation fold), and the selected model is fitted on the
whole training split and scored once on the test split.  The paired
bootstrap (10,000 resamples) compares it with gradient boosting on the raw
features at its own selected learning rate.

Each benchmark and representation is written to gb_retune_parts/ as soon as
it is finished, and a rerun skips those present; arguments such as
"VES:DPC" restrict a run, so that two can share a two-core machine.
gb_retune.json is assembled from every part present.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import rules as R
import seqdata as sd
from exp13_learned import configs, crossfit_image, gb
from exp2_weights_k import SEED

GRID = [0.03, 0.1, 0.3]              # the grid of exp1_baselines for gradient boosting
BOOT_REPS, BOOT_SEED = 10000, 53
PARTS = "gb_retune_parts"


def part(bench, rep):
    return os.path.join(PARTS, f"{bench}_{rep}.json")


def main():
    os.makedirs(PARTS, exist_ok=True)
    stored = json.load(open("learned.json"))
    only = {tuple(a.split(":")) for a in sys.argv[1:]}
    order = [(b, r) for b in sd.BENCHMARKS for r in sd.REPRESENTATIONS[b]]
    for i, (bench, rep) in enumerate(order):
        if (only and (bench, rep) not in only) or os.path.exists(part(bench, rep)):
            continue
        t0 = time.time()
        Xtr, ytr, _ = sd.features(bench, rep, "train")
        Xte, yte, _ = sd.features(bench, rep, "test")
        lr_raw = configs(bench, rep)["gb_lr"]
        y_raw = gb(lr_raw).fit(Xtr, ytr).predict(Xte)
        f_raw = float(f1_score(yte, y_raw, average="macro"))
        assert abs(f_raw - stored[f"{bench}|{rep}"]["rules"]["gb_raw"]["F1"]) < 1e-12
        # learning rate for the augmented input, image cross-fitted inside each fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv = np.zeros(len(GRID))
        for tr, va in skf.split(Xtr, ytr):
            Ztr, Zva = crossfit_image(Xtr[tr], ytr[tr], Xtr[va])
            A, B = np.hstack([Xtr[tr], Ztr]), np.hstack([Xtr[va], Zva])
            for g, lr in enumerate(GRID):
                cv[g] += f1_score(ytr[va], gb(lr).fit(A, ytr[tr]).predict(B), average="macro")
        cv /= 5
        lr_aug = GRID[int(cv.argmax())]
        Ztr, Zte = crossfit_image(Xtr, ytr, Xte)
        y_aug = gb(lr_aug).fit(np.hstack([Xtr, Ztr]), ytr).predict(np.hstack([Xte, Zte]))
        f_aug = float(f1_score(yte, y_aug, average="macro"))
        if lr_aug == lr_raw:        # then it is the fit of exp13, to the last digit
            assert abs(f_aug - stored[f"{bench}|{rep}"]["rules"]["gb_augmented"]["F1"]) < 1e-12
        r = R.boot_diff(yte, y_aug, y_raw, reps=BOOT_REPS, seed=BOOT_SEED + i)
        r.pop("boot")
        o = {"lr_raw": lr_raw, "lr_augmented": lr_aug,
             "cv_f1_augmented": {str(lr): float(v) for lr, v in zip(GRID, cv)},
             "gb_raw": R.summarise(yte, y_raw), "gb_augmented_retuned": R.summarise(yte, y_aug),
             "gb_augmented_at_raw_lr_F1": stored[f"{bench}|{rep}"]["rules"]["gb_augmented"]["F1"],
             "retuned-raw": r, "seconds": round(time.time() - t0, 1)}
        json.dump(o, open(part(bench, rep), "w"), indent=1)
        print(f"{bench:5s} {rep:4s} lr raw {lr_raw} -> augmented {lr_aug}  raw {f_raw:.4f}  "
              f"augmented {f_aug:.4f} (at raw lr {o['gb_augmented_at_raw_lr_F1']:.4f})  "
              f"difference {r['point']:+.4f} [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]  "
              f"[{time.time() - t0:.0f}s]", flush=True)
    out = {"grid": GRID, "boot_reps": BOOT_REPS, "results": {}}
    for bench, rep in order:
        if os.path.exists(part(bench, rep)):
            out["results"][f"{bench}|{rep}"] = json.load(open(part(bench, rep)))
    json.dump(out, open("gb_retune.json", "w"), indent=1)
    print(f"written gb_retune.json with {len(out['results'])} of {len(order)} representations")


if __name__ == "__main__":
    main()
