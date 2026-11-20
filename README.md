# Identifiable Indeterminacy: A Per-Feature Neutrosophic Metric for Nearest-Neighbour Classification

Hasan Suay Tanın, Davut Kesen and Memet Şahin\
Department of Mathematics, Gaziantep University, 27310 Gaziantep, Türkiye\
Corresponding author: Davut Kesen, davutkesen@gantep.edu.tr

Code and data that reproduce every table and figure of the article, submitted
to *Pattern Analysis and Applications*, and of its Online Resource 1 (the
supplementary methods and tables).

Distances between per-feature neutrosophic triples of truth, indeterminacy and
falsity bring neutrosophic sets into nearest-neighbour classification. In the
map one writes down first, indeterminacy is the feature's dispersion, constant
across instances, so it cancels from every distance and its weight is
unidentifiable. The construction proposed here takes indeterminacy as the
normalised distance to the nearest class centroid and shares the remaining mass
among the classes by inverse distance; its components form an exact partition
of unity, are invariant to per-feature affine rescaling, and make the
indeterminacy weight identifiable.

In the code, `v1` is the direct construction and `v2` the proposed one.

## Reproducing the tables and figures from the stored results

No data and no network are needed, and it takes under a minute:

```
pip install -r code/experiments/requirements.txt
python code/experiments/run_all.py --only tables,figures
```

`tables/` then holds every table as a LaTeX fragment and `figures/` holds
Fig1 to Fig4 as EPS, 337 pt (118.9 mm) wide. The stored results the tables
and figures are built from are the `*.json` files and `groupcv_folds/` in
`code/experiments/`.

`make_figures.py` writes the PostScript of each figure itself, in integer
arithmetic and without a plotting library, so the files come out the same,
byte for byte, on every machine; `figprolog.ps` holds the two embedded fonts
and the drawing procedures, and its head states the licence of the fonts.
`matlab/make_figures.m` is a line-by-line port that writes the same files
from MATLAB (see `matlab/README.txt`).

## Reproducing the whole study

```
python code/experiments/run_all.py
```

runs the 26 stages in order: verify, baselines, weights_k, stats, regime,
diagnostics, regime_cf, regime_pmlb, symmetry, fuzzy, centroid, multiclass,
ablation, homology_search, homology_eval, preregister, benchmarks, groupcv,
learned, finegrid, parity, nca_budget, nca_fullrank, gb_retune, tables,
figures. The head of `run_all.py` says what each stage does. Random seeds are
fixed in every stage. `--quick` skips the slow stages, `--only <stage>` runs
one. On two x86_64 cores the full run takes about sixteen hours, most of it in
`groupcv`. The homology stages need MMseqs2 (version 15-6f452 was used).

The two predictions per representation of the further benchmarks were fixed
before those test splits were touched: `code/experiments/prereg.json` holds
them and `prereg.sha256` its digest, which the `benchmarks` stage checks
before it runs.

## Data

Every data file the experiments read is in `data/`.

`data/seq/` holds the sequences, accession lists and PSSM summaries of the
three protein-function benchmarks (18 files), taken from the compilers'
repositories at the commits recorded in `code/experiments/seq_manifest.json`,
which also holds the SHA-256 digest of every file; `seqdata.check_manifest()`
compares the files with these digests.

| folder | benchmark | source |
|---|---|---|
| `data/seq/et-gru/` | electron transport proteins against other transport proteins | Le, Yapp and Yeh, *BMC Bioinformatics* 20:377, 2019, [github.com/khanhlee/et-gru](https://github.com/khanhlee/et-gru) |
| `data/seq/snare-cnn/` | SNARE proteins against other vesicular transport proteins | Le and Nguyen, *PeerJ Computer Science* 5:e177, 2019, [github.com/khanhlee/snare-cnn](https://github.com/khanhlee/snare-cnn) |
| `data/seq/vesicular-gru/` | vesicular transport proteins against other membrane proteins | Le, Yapp, Nagasundaram, Chua and Yeh, *Computational and Structural Biotechnology Journal* 17:1245–1254, 2019, [github.com/khanhlee/vesicular-gru](https://github.com/khanhlee/vesicular-gru) |

`data/raw/` holds the six ARFF feature files of the electron-transport
benchmark: the amino-acid composition (AAC), dipeptide composition (DPC) and
400-dimensional PSSM summary of its cross-validation (`train`) and
independent (`test`) sets, as used by the compilers of the benchmark. In
`et_tp_DPC_train.arff` every field `0.0` is written `0`, which keeps the file
under the 25 MB that GitHub accepts through the browser; every value, and so
every array the code reads, is unchanged. The amino-acid and dipeptide
compositions of the other two benchmarks are computed from the sequences by
`seqdata.py`, with the definitions that reproduce these files.

`data/pmlb/` holds the 53 datasets of the Penn Machine Learning Benchmarks
on which the regime analysis is evaluated, from
[github.com/EpistasisLab/pmlb](https://github.com/EpistasisLab/pmlb), with
their SHA-256 digests in `code/experiments/pmlb_manifest.json`. They are kept
in the form PMLB distributes them, gzip-compressed TSV, which the code reads
as it is. To apply the selection rule the stage `regime_pmlb` also reads the
PMLB summary table and the datasets that the size rule drops; it downloads
them from PMLB and checks each file against the digest that PMLB records.

The three benchmarks and PMLB are the work of their compilers and keep their
own terms; cite the papers above when you use them.

## Layout

```
code/experiments/   the pipeline (Python 3.13), requirements.txt with the
                    pinned versions, the stored results, figprolog.ps
data/               the input data (see above)
matlab/             the MATLAB port of the figure code
tables/, figures/   written by the last two stages
```
