"""Two further benchmarks, step 2 of 2: evaluation on the test splits.

Runs only if prereg.json is unchanged since exp11a_preregister.py wrote it
(its SHA-256 digest must equal the one in prereg.sha256).  Every rule is
refitted on the training split with the configuration recorded there and
evaluated once on the test split.

Reported for each benchmark and representation:
  * every rule's test scores;
  * the paired bootstrap (10,000 resamples of the test proteins) of the
    proposed construction against the plain rule, N-KNN and the direct
    construction;
  * the same comparison with the plain rule on test proteins below and at or
    above 30% identity to every training protein (homology.json);
  * the two frozen predictions, scored: a predicted gain is confirmed when
    the interval of the difference lies above zero and contradicted when it
    lies below; a predicted absence of gain is confirmed when the interval
    lies below zero and contradicted when it lies above; an interval that
    contains zero is inconclusive either way.
"""
import hashlib
import json

import numpy as np

import rules as R
import seqdata as sd

BOOT_REPS, BOOT_SEED = 10000, 23
COMPARE = [("proposed", "knn"), ("proposed", "nknn"), ("proposed", "direct")]
THRESHOLD = 0.30


def verdict(prediction, ci):
    lo, hi = ci
    if prediction == "none":
        return "no prediction"
    if prediction == "gain":
        return "confirmed" if lo > 0 else ("contradicted" if hi < 0 else "inconclusive")
    return "confirmed" if hi < 0 else ("contradicted" if lo > 0 else "inconclusive")


def main():
    text = open("prereg.json").read()
    digest = open("prereg.sha256").read().split()[0]
    if hashlib.sha256(text.encode()).hexdigest() != digest:
        raise SystemExit("prereg.json has changed since it was frozen")
    pre = json.loads(text)
    sd.check_manifest()
    hom = json.load(open("homology.json"))
    out = {"prereg_sha256": digest, "boot_reps": BOOT_REPS, "benchmarks": {}}
    seed = BOOT_SEED
    for bench, reps in pre["benchmarks"].items():
        out["benchmarks"][bench] = {}
        tt = hom[bench]["test_to_train"]
        for rep, p in reps.items():
            Xtr, ytr, atr = sd.features(bench, rep, "train")
            Xte, yte, ate = sd.features(bench, rep, "test")
            in_train = set(atr)
            maxid = np.array([max([v for t, v in tt.get(a, []) if t in in_train], default=0.0)
                              for a in ate])
            preds, res = {}, {}
            for rule, c in p["configs"].items():
                yp, s = R.predict(rule, c, Xtr, ytr, Xte)
                preds[rule] = yp
                res[rule] = R.summarise(yte, yp, s)
                res[rule]["cv_f1"] = c["cv_f1"] if "cv_f1" in c else None
            comp = {}
            for a, b in COMPARE:
                r = R.boot_diff(yte, preds[a], preds[b], reps=BOOT_REPS, seed=seed)
                seed += 1
                r.pop("boot")
                comp[f"{a}-{b}"] = r
            remote = maxid < THRESHOLD
            strata = {}
            for name, mask in (("remote", remote), ("homologous", ~remote)):
                y = yte[mask]
                strata[name] = {"n": int(mask.sum()), "positives": int(y.sum()),
                                "rules": {k: R.summarise(y, v[mask]) for k, v in preds.items()}}
                if 0 < y.sum() < len(y):
                    r = R.boot_diff(y, preds["proposed"][mask], preds["knn"][mask],
                                    reps=BOOT_REPS, seed=seed)
                    seed += 1
                    r.pop("boot")
                    strata[name]["proposed-knn"] = r
            ci = comp["proposed-knn"]["ci95"]
            nn = ["knn", "direct", "proposed", "nknn", "nsknn", "centroid"]
            best_nn = max(nn, key=lambda k: res[k]["F1"])
            best_tr = max(R.TRAINED, key=lambda k: res[k]["F1"])
            out["benchmarks"][bench][rep] = {
                "n_test": int(len(yte)), "test_positives": int(yte.sum()),
                "rules": res, "compare": comp, "strata_30": strata,
                "max_identity": maxid.round(4).tolist(),
                "predictions": {
                    "regime": {"x_cv": p["x_cv"], "predicted": p["regime_prediction"],
                               "verdict": verdict(p["regime_prediction"], ci)},
                    "cv_sign": {"cv_difference": p["cv_difference_proposed_minus_knn"],
                                "predicted": p["cv_sign_prediction"],
                                "verdict": verdict(p["cv_sign_prediction"], ci)}},
                "best_nearest_neighbour": [best_nn, res[best_nn]["F1"]],
                "best_trained": [best_tr, res[best_tr]["F1"]]}
            c = comp["proposed-knn"]
            print(f"{bench:5s} {rep:4s} n_test={len(yte)} (pos {int(yte.sum())})", flush=True)
            for k in nn + R.TRAINED:
                print(f"      {k:26s} F1={res[k]['F1']:.4f} MCC={res[k]['MCC']:.4f}", flush=True)
            for k, v in comp.items():
                print(f"      {k:16s} {v['point']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]", flush=True)
            for name in ("remote", "homologous"):
                s = strata[name]
                if "proposed-knn" in s:
                    v = s["proposed-knn"]
                    print(f"      {name:10s} n={s['n']:4d} pos={s['positives']:3d}  proposed-knn "
                          f"{v['point']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]", flush=True)
            pr = out["benchmarks"][bench][rep]["predictions"]
            print(f"      regime prediction '{pr['regime']['predicted']}' -> {pr['regime']['verdict']};"
                  f"  CV-sign prediction '{pr['cv_sign']['predicted']}' -> {pr['cv_sign']['verdict']}",
                  flush=True)
    json.dump(out, open("benchmarks.json", "w"), indent=1)
    print("written benchmarks.json")


if __name__ == "__main__":
    main()
