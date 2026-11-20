"""Two checks on the parity of the comparison with the plain rule.

(1) Scaling of the plain rule.  The plain nearest-neighbour rule of the
    paper works on standardised inputs, whereas the constructions divide by
    the training range.  Here the plain rule is also tuned (k and class
    weighting, same five folds) on min-max scaled inputs, and a third
    variant lets cross-validation choose between the two scalings.  All
    three are evaluated on the predefined test split of every benchmark and
    representation, with the paired bootstrap (10,000 resamples) of the
    proposed construction against each.  For the two further benchmarks
    this lies outside the plan of exp11a: the configurations of the plan
    are untouched and the new rows are labelled as additions.

(2) Class shares alone.  The "rule 2 only" cell of the ablation multiplies
    the shares s_c,j by the constant 1 - sigma_j / r_j, which weights the
    features.  Here the shares are used without any factor, I_j = 0 and
    M_c,j = s_c,j (transport-protein benchmark).  With two classes
    s_0,j = 1 - s_1,j, so every weight vector gives the same metric up to
    scale and the single block s_1 is searched over k and class weighting.

Output: parity.json.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import time

import numpy as np

import rules as R
import seqdata as sd
from common import EPS_S, feature_range
from exp2_weights_k import KS
from exp9_ablation import cv_k_cw, fit_predict, minmax_blocks

BOOT_REPS, BOOT_SEED = 10000, 41


def blocks_shares(Xtr, ytr, Xq, eps_s=EPS_S):
    """Class shares alone: the single block s_1,j (I_j = 0, M_c,j = s_c,j)."""
    classes = np.unique(ytr)
    mu = np.array([Xtr[ytr == c].mean(0) for c in classes])
    rng, nz = feature_range(Xtr)
    out = []
    for X in Xq:
        delta = np.where(nz, np.abs(X[None] - mu[:, None]) / rng, 0.0)
        inv = 1.0 / (delta + eps_s)
        s = inv / inv.sum(axis=0, keepdims=True)
        out.append([s[-1]])
    return out


def stored_configs(bench, rep):
    """(plain config with CV score, proposed config) as selected in the paper."""
    if bench == "ET":
        wk = json.load(open("weights_k.json"))[rep]
        return ({"k": wk["knn"]["best_k"], "cw": wk["knn"]["best_cw"], "cv_f1": wk["knn"]["cv_f1"]},
                {"w": wk["v2"]["best_w"], "k": wk["v2"]["best_k"], "cw": wk["v2"]["best_cw"]})
    c = json.load(open("prereg.json"))["benchmarks"][bench][rep]["configs"]
    return (dict(c["knn"]), {k: c["proposed"][k] for k in ("w", "k", "cw")})


def single_block(Xtr, ytr, Xte, fn):
    acc = cv_k_cw(Xtr, ytr, fn)
    ki, ci = np.unravel_index(acc.argmax(), acc.shape)
    yp, s = fit_predict(Xtr, ytr, Xte, fn, [1.0], KS[ki], bool(ci))
    return yp, s, {"k": KS[ki], "cw": bool(ci), "cv_f1": float(acc[ki, ci])}


def main():
    ab = json.load(open("ablation.json"))
    out = {"boot_reps": BOOT_REPS, "scaling": {}, "shares_alone": {}}
    seed = BOOT_SEED
    for bench in sd.BENCHMARKS:
        for rep in sd.REPRESENTATIONS[bench]:
            t0 = time.time()
            Xtr, ytr, _ = sd.features(bench, rep, "train")
            Xte, yte, _ = sd.features(bench, rep, "test")
            ck, cp = stored_configs(bench, rep)
            yz, sz = R.predict("knn", ck, Xtr, ytr, Xte)
            yv, sv = R.predict("proposed", cp, Xtr, ytr, Xte)
            ym, sm, cm = single_block(Xtr, ytr, Xte, minmax_blocks)
            if bench == "ET":      # the min-max rule of exp9, to the last digit
                ref = ab[rep]["minmax_knn"]
                assert (cm["k"], cm["cw"]) == (ref["best_k"], ref["best_cw"])
                assert abs(R.summarise(yte, ym, sm)["F1"] - ref["test"]["F1"]) < 1e-12
            chosen = "minmax" if cm["cv_f1"] > ck["cv_f1"] else "zscore"
            ys, ss = (ym, sm) if chosen == "minmax" else (yz, sz)
            o = {"zscore": dict(R.summarise(yte, yz, sz), k=ck["k"], cw=ck["cw"], cv_f1=ck["cv_f1"]),
                 "minmax": dict(R.summarise(yte, ym, sm), **cm),
                 "selected": dict(R.summarise(yte, ys, ss), scaling=chosen),
                 "proposed": R.summarise(yte, yv, sv), "compare": {}}
            for key, yo in (("proposed-zscore", yz), ("proposed-minmax", ym),
                            ("proposed-selected", ys)):
                r = R.boot_diff(yte, yv, yo, reps=BOOT_REPS, seed=seed)
                seed += 1
                r.pop("boot")
                o["compare"][key] = r
            out["scaling"][f"{bench}|{rep}"] = o
            print(f"{bench:5s} {rep:4s} z-score {o['zscore']['F1']:.4f} (cv {ck['cv_f1']:.4f})  "
                  f"min-max {o['minmax']['F1']:.4f} (cv {cm['cv_f1']:.4f})  selected: {chosen}  "
                  f"proposed {o['proposed']['F1']:.4f}  proposed-selected "
                  f"{o['compare']['proposed-selected']['point']:+.4f} "
                  f"[{o['compare']['proposed-selected']['ci95'][0]:+.4f},"
                  f"{o['compare']['proposed-selected']['ci95'][1]:+.4f}]  [{time.time() - t0:.0f}s]",
                  flush=True)
            if bench == "ET":
                yh, sh, ch = single_block(Xtr, ytr, Xte, blocks_shares)
                r = R.boot_diff(yte, yv, yh, reps=BOOT_REPS, seed=seed)
                seed += 1
                r.pop("boot")
                out["shares_alone"][rep] = dict(R.summarise(yte, yh, sh), **ch,
                                                proposed_minus=r,
                                                rule2_test_f1=ab[rep]["rule2"]["test"]["F1"])
                print(f"      shares alone: k={ch['k']} cw={ch['cw']} cv {ch['cv_f1']:.4f} "
                      f"test {out['shares_alone'][rep]['F1']:.4f}  (rule 2 alone "
                      f"{ab[rep]['rule2']['test']['F1']:.4f})  proposed-shares {r['point']:+.4f} "
                      f"[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]", flush=True)
            json.dump(out, open("parity.json", "w"), indent=1)
    print("written parity.json")


if __name__ == "__main__":
    main()
