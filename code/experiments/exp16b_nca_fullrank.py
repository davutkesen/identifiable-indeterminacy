"""Neighbourhood components analysis at full rank on every training protein.

exp16_nca_budget.py raises the budget of NCA to 200 components and 5,000
training proteins.  Where the input has more than 200 dimensions and the
training split has at most 5,000 proteins (the dipeptide and the profile
representation of the transport-protein and SNARE benchmarks) NCA is fitted
here with as many components as the input has dimensions, on every training
protein, under the same protocol: standardised inputs, PCA initialisation,
100 iterations, the vote of the paper over all training proteins in the
learned space, k and class weighting selected by five-fold cross-validation
with NCA refitted in every fold.  On the amino-acid composition of those two
benchmarks the larger budget of exp16 is already full rank on every training
protein, and the vesicular benchmark, with 9,787 training proteins, stays at
the 5,000 of exp16.

The first budget is refitted as well and must reproduce learned.json.  The
paired bootstrap (10,000 resamples) compares the full-rank fit with the first
budget and the proposed construction with the full-rank fit.

Each representation is written to nca_fullrank_parts/ as soon as it is
finished and a rerun skips those present; arguments such as "ET:PSSM"
restrict a run, so that two can share a two-core machine.  nca_fullrank.json
is assembled from every part present.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time

import rules as R
import seqdata as sd
import exp13_learned as E
from exp16_nca_budget import LARGE, SMALL, nca

FULL = (10 ** 6, LARGE[1])   # components: as many as the input has dimensions
BOOT_REPS, BOOT_SEED = 10000, 67
PARTS = "nca_fullrank_parts"


def part(bench, rep):
    return os.path.join(PARTS, f"{bench}_{rep}.json")


def main():
    os.makedirs(PARTS, exist_ok=True)
    stored = json.load(open("learned.json"))
    only = {tuple(a.split(":")) for a in sys.argv[1:]}
    order = [(b, r) for b in sd.BENCHMARKS for r in sd.REPRESENTATIONS[b]]
    eligible = []
    for i, (bench, rep) in enumerate(order):
        Xtr, ytr, _ = sd.features(bench, rep, "train")
        if not (Xtr.shape[1] > LARGE[0] and len(ytr) <= LARGE[1]):
            continue
        eligible.append((bench, rep))
        if (only and (bench, rep) not in only) or os.path.exists(part(bench, rep)):
            continue
        t0 = time.time()
        Xte, yte, _ = sd.features(bench, rep, "test")
        key = f"{bench}|{rep}"
        w, k, cw = E.configs(bench, rep)["proposed"]
        yv, _ = R.predict("proposed", {"w": w, "k": k, "cw": cw}, Xtr, ytr, Xte)
        ys, _, cs = nca(SMALL, Xtr, ytr, Xte)
        fs = R.summarise(yte, ys)
        assert abs(fs["F1"] - stored[key]["rules"]["nca"]["F1"]) < 1e-9, \
            f"{key}: NCA at the stored budget gives {fs['F1']}, stored {stored[key]['rules']['nca']['F1']}"
        yf, _, cf = nca(FULL, Xtr, ytr, Xte)
        assert cf["components"] == Xtr.shape[1]
        o = {"n_train": int(len(ytr)), "d": int(Xtr.shape[1]),
             "small": dict(fs, **cs), "full": dict(R.summarise(yte, yf), **cf),
             "proposed": R.summarise(yte, yv), "compare": {}}
        seed = BOOT_SEED + 10 * i
        for name, (a, b) in (("full-small", (yf, ys)), ("proposed-full", (yv, yf))):
            r = R.boot_diff(yte, a, b, reps=BOOT_REPS, seed=seed)
            seed += 1
            r.pop("boot")
            o["compare"][name] = r
        o["seconds"] = round(time.time() - t0, 1)
        json.dump(o, open(part(bench, rep), "w"), indent=1, default=float)
        c = o["compare"]
        print(f"{bench:5s} {rep:4s} NCA first budget {o['small']['F1']:.4f}  full rank {o['full']['F1']:.4f} "
              f"({c['full-small']['point']:+.4f} [{c['full-small']['ci95'][0]:+.4f},"
              f"{c['full-small']['ci95'][1]:+.4f}])  cv {o['full']['cv_f1']:.4f}  "
              f"proposed-full {c['proposed-full']['point']:+.4f} "
              f"[{c['proposed-full']['ci95'][0]:+.4f},{c['proposed-full']['ci95'][1]:+.4f}]  "
              f"[{time.time() - t0:.0f}s]", flush=True)
    out = {"small": SMALL, "boot_reps": BOOT_REPS, "eligible": [f"{b}|{r}" for b, r in eligible],
           "results": {}}
    for bench, rep in eligible:
        if os.path.exists(part(bench, rep)):
            out["results"][f"{bench}|{rep}"] = json.load(open(part(bench, rep)))
    json.dump(out, open("nca_fullrank.json", "w"), indent=1)
    print(f"written nca_fullrank.json with {len(out['results'])} of {len(eligible)} representations")


if __name__ == "__main__":
    main()
