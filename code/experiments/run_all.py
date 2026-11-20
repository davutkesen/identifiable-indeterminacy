"""Reproduce every number, table and figure in the paper with one command.

    python run_all.py            # everything
    python run_all.py --quick    # skip the slow stages

Stages, in order:
  0  verify      the untuned configurations at k = 5 and uniform weights,
                 the partition of unity of the proposed construction, and the
                 block distances of Table 1 measured by direct subtraction
  1  baselines   five trained classifiers outside the nearest-neighbour family
  2  weights_k   the weight simplex and the neighbourhood-size grid, and the
                 plain rule tuned over the same k grid and class weighting
  3  stats       thirty paired resamplings, Holm correction, bootstrap CIs
  4  regime      the 133-task regime study
  5  diagnostics confusion matrices, redundancy, guard constant, runtime
  6  regime_cf   the regime relation re-measured with cross-fitting (x and
                 y on disjoint halves) and with both arms tuned (slow)
  7  regime_pmlb the same re-analysis on independent real tasks from PMLB,
                 selected by a fixed rule; downloads the datasets (slow)
  8  symmetry    label-exchange symmetry, guard-constant sweep at the
                 selected configuration, paired bootstrap on the fixed split
  9  fuzzy       fuzzy, intuitionistic-fuzzy and NS-k-NN baselines
 10  centroid    nearest-centroid and per-feature share-sum rules
 11  multiclass  the construction on two multiclass datasets, and the binary
                 benchmark under the multiclass weight parametrisation
 12  ablation    the two design rules crossed, the published N-KNN, and the
                 paired bootstrap of the proposed construction against them
 13  homology_search  sequence identity across the predefined splits and
                 within the pooled data of all three benchmarks (MMseqs2)
 14  homology_eval    every rule on test proteins with and without a
                 training homologue (transport benchmark)
 15  preregister the SNARE and vesicular-transport benchmarks: selection on
                 the training splits and predictions, made from the training
                 data alone, fixed in prereg.json with its SHA-256 digest
 16  benchmarks  their test evaluation, refused if prereg.json has changed
 17  groupcv     homology-aware repeated cross-validation on the pooled data
 18  learned     NCA, and the construction inside an SVM and gradient boosting
 19  finegrid    the weight search repeated on the simplex in steps of 0.05
                 and on the tied family in steps of 0.01
 20  parity      the plain rule on min-max inputs and with its scaling chosen
                 by cross-validation; the class shares alone
 21  nca_budget  NCA with more components and a larger training sample
 22  nca_fullrank  NCA at full rank on every training protein, where the
                 training split has at most 5,000 proteins
 23  gb_retune   gradient boosting on the augmented input with its learning
                 rate re-selected
 24  tables      every table, as a LaTeX fragment in tables/, written from
                 the stored results
 25  figures     every figure, as EPS in figures/ (and as PDF if
                 Ghostscript is installed)

Random seeds are fixed inside each stage.  Expected total runtime on two
x86_64 cores is about sixteen hours, dominated by stages 1, 2, 6, 7, 15, 17,
18 and 21 to 23; stage 13 needs MMseqs2 and the sequence files of data/seq/.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

STAGES = [
    ("verify",      "exp0_verify.py",   HERE, False),
    ("baselines",   "exp1_baselines.py", HERE, True),
    ("weights_k",   "exp2_weights_k.py", HERE, True),
    ("stats",       "exp3_stats.py",    HERE, False),
    ("regime",      "exp4_regime.py",   HERE, False),
    ("diagnostics", "exp5_diag.py",     HERE, False),
    ("regime_cf",   "exp4b_regime_crossfit.py", HERE, True),
    ("regime_pmlb", "exp4c_regime_pmlb.py", HERE, True),
    ("symmetry",    "exp6_symmetry_boot.py", HERE, False),
    ("fuzzy",       "exp7_fuzzy_baselines.py", HERE, False),
    ("centroid",    "exp7b_centroid_baselines.py", HERE, False),
    ("multiclass",  "exp8_multiclass.py", HERE, False),
    ("ablation",    "exp9_ablation.py", HERE, False),
    ("homology_search", "exp10a_homology_search.py", HERE, True),
    ("homology_eval", "exp10b_homology_eval.py", HERE, False),
    ("preregister", "exp11a_preregister.py", HERE, True),
    ("benchmarks",  "exp11b_benchmarks.py", HERE, False),
    ("groupcv",     "exp12_groupcv.py",  HERE, True),
    ("learned",     "exp13_learned.py",  HERE, True),
    ("finegrid",    "exp14_finegrid.py", HERE, False),
    ("parity",      "exp15_parity.py",   HERE, False),
    ("nca_budget",  "exp16_nca_budget.py", HERE, True),
    ("nca_fullrank", "exp16b_nca_fullrank.py", HERE, True),
    ("gb_retune",   "exp17_gb_retune.py", HERE, True),
    ("tables",      "make_tables.py",   HERE, False),
    ("figures",     "make_figures.py",  HERE, False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the slow stages")
    ap.add_argument("--only", default=None,
                    help="comma-separated stage names to run")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None

    t0 = time.time()
    for name, script, cwd, slow in STAGES:
        if only and name not in only:
            continue
        if a.quick and slow:
            print(f"[skip] {name}")
            continue
        print(f"\n=========== {name} ===========", flush=True)
        t = time.time()
        r = subprocess.run([sys.executable, "-u", script], cwd=cwd)
        if r.returncode != 0:
            print(f"stage {name} failed with code {r.returncode}")
            return r.returncode
        print(f"[{name} finished in {time.time()-t:.0f}s]", flush=True)
    print(f"\nall stages complete in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
