"""Two further benchmarks, step 1 of 2: selection on the training splits and
predictions made from the training data alone, fixed in a file whose digest
the test stage checks.  The digest shows that the file has not changed since
it was written, not when it was written.

For the SNARE and vesicular-transport benchmarks (seqdata.py) every rule of
the paper is tuned on the training split by the protocol used throughout
(rules.tune).  This script reads training data only.  It then writes, to
prereg.json, the selected configurations and two predictions for every
representation, made from the training split alone:

  regime    x_cv, the plain rule's cross-validated macro F1 minus that of
            the majority classifier on the same folds, placed in the zones
            of the regime analysis (Section 6.9): x_cv < 0.10 predicts that
            the proposed construction beats the plain rule on the test
            split, x_cv >= 0.30 predicts that it does not, and the zone in
            between makes no prediction;
  cv sign   the sign of the cross-validated difference between the proposed
            construction and the plain rule predicts the sign of the test
            difference.

The analysis plan is written into the same file.  Its SHA-256 digest is
written to prereg.sha256; exp11b_benchmarks.py refuses to run unless the
digest matches, so the configurations and predictions cannot change after
the test splits are opened.
"""
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

import rules as R
import seqdata as sd
from exp2_weights_k import SEED

NEW = {"SNARE": ["AAC", "DPC", "PSSM"], "VES": ["AAC", "DPC"]}
PARTIAL = "prereg_partial.json"      # selections finished so far (training data only)

PLAN = [
    "Primary outcome: for each benchmark and representation, the test macro F1 of the "
    "proposed construction minus that of the tuned plain rule, with a paired-bootstrap "
    "95% interval (10,000 resamples of the test proteins).",
    "Each prediction is scored against the sign of the observed difference; a difference "
    "whose interval contains zero confirms neither a predicted gain nor a predicted loss.",
    "Secondary: the same difference against N-KNN and against the direct construction; "
    "the difference against the plain rule separately on test proteins below and above "
    "30% identity to every training protein; the best trained classifier against the best "
    "nearest-neighbour rule.",
    "Every configuration is the one recorded here; nothing is re-tuned after the test "
    "split is opened, and every outcome is reported whatever its direction.",
]


def majority_cv_f1(y, folds=5):
    from sklearn.metrics import f1_score
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    vals = []
    for tr, va in skf.split(np.zeros(len(y)), y):
        maj = np.bincount(y[tr]).argmax()
        vals.append(f1_score(y[va], np.full(len(va), maj), average="macro"))
    return float(np.mean(vals))


def zone(x):
    return "gain" if x < 0.10 else ("no gain" if x >= 0.30 else "none")


def main():
    if os.path.exists("prereg.json") and "--force" not in sys.argv:
        print("prereg.json exists and is not overwritten (use --force to redo the "
              "selection and freeze a new file)")
        return
    sd.check_manifest()
    out = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "inputs": json.load(open(sd.MANIFEST)), "plan": PLAN, "benchmarks": {}}
    part = json.load(open(PARTIAL)) if os.path.exists(PARTIAL) else {}
    for bench, reps in NEW.items():
        out["benchmarks"][bench] = {}
        for rep in reps:
            if f"{bench}|{rep}" in part:          # selection already done
                out["benchmarks"][bench][rep] = part[f"{bench}|{rep}"]
                continue
            t0 = time.time()
            Xtr, ytr, _ = sd.features(bench, rep, "train")
            print(f"{bench} {rep}: train {Xtr.shape}, positives {int(ytr.sum())}", flush=True)
            cfg = R.tune(Xtr, ytr, log=lambda s: print(s, flush=True))
            x_cv = cfg["knn"]["cv_f1"] - majority_cv_f1(ytr)
            d_cv = cfg["proposed"]["cv_f1"] - cfg["knn"]["cv_f1"]
            out["benchmarks"][bench][rep] = {
                "n_train": int(len(ytr)), "train_positives": int(ytr.sum()),
                "configs": cfg,
                "x_cv": float(x_cv), "regime_prediction": zone(x_cv),
                "cv_difference_proposed_minus_knn": float(d_cv),
                "cv_sign_prediction": "gain" if d_cv > 0 else "no gain",
                "seconds": round(time.time() - t0, 1)}
            part[f"{bench}|{rep}"] = out["benchmarks"][bench][rep]
            json.dump(part, open(PARTIAL, "w"), indent=1, default=float)
            print(f"   x_cv={x_cv:.4f} -> regime prediction '{zone(x_cv)}';  "
                  f"CV proposed-knn={d_cv:+.4f}  [{time.time() - t0:.0f}s]", flush=True)
    text = json.dumps(out, indent=1, default=float)
    open("prereg.json", "w").write(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    open("prereg.sha256", "w").write(digest + "  prereg.json\n")
    print("written prereg.json, SHA-256", digest)


if __name__ == "__main__":
    main()
