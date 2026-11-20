"""Sequence-level data for the three benchmarks of Le and co-workers.

    ET     electron transport proteins against other transport proteins
           (Le, Yapp and Yeh, BMC Bioinformatics 20:377, 2019),
           github.com/khanhlee/et-gru
    SNARE  SNARE proteins against other vesicular transport proteins
           (Le and Nguyen, PeerJ Computer Science 5:e177, 2019),
           github.com/khanhlee/snare-cnn
    VES    vesicular transport proteins against other membrane proteins
           (Le, Yapp, Nagasundaram, Chua and Yeh, Computational and
           Structural Biotechnology Journal 17:1245-1254, 2019),
           github.com/khanhlee/vesicular-gru

The files are read from data/seq/<repository>/ (see README.md and
seq_manifest.json, which records the commit and the SHA-256 digest of every
file).  Each benchmark comes with a cross-validation (training) set and an
independent (test) set fixed by its compilers.

Features.  For ET the feature files of the paper are used as they are, and
their rows are mapped to UniProt accessions here:

  * the 400-dimensional PSSM summary has one row per sequence, in the order
    of the compilers' FASTA files (electron transport first); this was
    verified on the 200 PSSM profiles the repository distributes, whose
    summaries reproduce the rows at exactly those positions;
  * the composition files omit some sequences; their rows are in FASTA order
    as well, and each row is matched to the sequence whose amino-acid
    composition it reproduces (the matches are strictly increasing in FASTA
    position, which is checked); the few rows whose composition matches no
    sequence exactly are assigned by position, between their neighbours.

For SNARE and VES, amino-acid composition (count / length) and dipeptide
composition (count / (length - 1)) are computed from the sequences with the
definitions that reproduce the ET files, and for SNARE the compilers'
400-dimensional PSSM summary is used, whose rows follow their accession
lists.

Homology.  mmseqs_identity() runs MMseqs2 (Steinegger and Soeding, 2017) and
returns, for every query sequence, the largest identity to any target
sequence, measured as identical aligned residues over the length of the
shorter of the two sequences.  This normalisation never exceeds the
identity over the aligned columns alone, so a pair it puts above a threshold
is above it under any of the usual definitions.
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np

from common import ROOT, load_rep

SEQ = os.path.join(ROOT, "data", "seq")
AA = "ACDEFGHIKLMNPQRSTVWY"
HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "seq_manifest.json")

BENCHMARKS = ("ET", "SNARE", "VES")
REPRESENTATIONS = {"ET": ["AAC", "DPC", "PSSM"], "SNARE": ["AAC", "DPC", "PSSM"],
                   "VES": ["AAC", "DPC"]}
FILES = {
    "et-gru": ["electron.cv.fasta", "electron.ind.fasta",
               "transport.cv.fasta", "transport.ind.fasta"],
    "snare-cnn": ["snare.fasta", "non_snare.fasta", "snare.cv.list",
                  "snare.ind.list", "non_snare.cv.list", "non_snare.ind.list",
                  "pssm.cv.csv", "pssm.ind.csv"],
    "vesicular-gru": ["vesicular.fasta.format", "membrane_no_vesicular.fasta",
                      "vesicular.cv", "vesicular.ind", "membrane.cv",
                      "membrane.ind"],
}
COMMITS = {"et-gru": "6eb96e323fc37b2fc2e91332966f6217c8a1bd35",
           "snare-cnn": "2e1288cd3b537aa709717225c1bc93c49f5965b1",
           "vesicular-gru": "f5db3aaf03c760fd2b8762a3ed954523ba67bf69"}


# ------------------------------------------------------------------ files
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest():
    man = {"commits": COMMITS, "sha256": {}}
    for repo, names in FILES.items():
        for n in names:
            man["sha256"][f"{repo}/{n}"] = sha256(os.path.join(SEQ, repo, n))
    json.dump(man, open(MANIFEST, "w"), indent=1)
    return man


def check_manifest():
    man = json.load(open(MANIFEST))
    for rel, digest in man["sha256"].items():
        got = sha256(os.path.join(SEQ, rel))
        if got != digest:
            raise RuntimeError(f"{rel}: SHA-256 {got} differs from the manifest")


def parse_fasta(path):
    """Records (accession, sequence).  Robust to a header glued to the end of
    the previous sequence line, which occurs once in the ET files."""
    text = open(path).read()
    out = []
    for chunk in text.split(">")[1:]:
        lines = chunk.strip().splitlines()
        acc = lines[0].split()[0].strip()
        out.append((acc, "".join(l.strip() for l in lines[1:]).upper()))
    return out


def read_list(path):
    return [l.split()[0] for l in open(path) if l.strip()]


# --------------------------------------------------------------- records
def records(bench, split):
    """[(accession, label, sequence)] of a split, positives first, in the
    order of the compilers' files."""
    tag = {"train": "cv", "test": "ind"}[split]
    if bench == "ET":
        d = os.path.join(SEQ, "et-gru")
        pos = parse_fasta(os.path.join(d, f"electron.{tag}.fasta"))
        neg = parse_fasta(os.path.join(d, f"transport.{tag}.fasta"))
        return [(a, 1, s) for a, s in pos] + [(a, 0, s) for a, s in neg]
    if bench == "SNARE":
        d = os.path.join(SEQ, "snare-cnn")
        seq = dict(parse_fasta(os.path.join(d, "snare.fasta")))
        seq.update({a: s for a, s in parse_fasta(os.path.join(d, "non_snare.fasta"))
                    if a not in seq})
        pos = read_list(os.path.join(d, f"snare.{tag}.list"))
        neg = read_list(os.path.join(d, f"non_snare.{tag}.list"))
    elif bench == "VES":
        d = os.path.join(SEQ, "vesicular-gru")
        seq = dict(parse_fasta(os.path.join(d, "vesicular.fasta.format")))
        seq.update({a: s for a, s in
                    parse_fasta(os.path.join(d, "membrane_no_vesicular.fasta"))
                    if a not in seq})
        pos = read_list(os.path.join(d, f"vesicular.{tag}"))
        neg = read_list(os.path.join(d, f"membrane.{tag}"))
    else:
        raise ValueError(bench)
    missing = [a for a in pos + neg if a not in seq]
    if missing:
        raise RuntimeError(f"{bench} {split}: {len(missing)} accessions without a sequence")
    return [(a, 1, seq[a]) for a in pos] + [(a, 0, seq[a]) for a in neg]


# -------------------------------------------------------------- features
def aac(s):
    return np.array([s.count(a) for a in AA], float) / max(len(s), 1)


def dpc(s):
    v = np.zeros(400)
    idx = {a: i for i, a in enumerate(AA)}
    for a, b in zip(s, s[1:]):
        if a in idx and b in idx:
            v[idx[a] * 20 + idx[b]] += 1
    return v / max(len(s) - 1, 1)


def sqdist_rows(A, B):
    """Squared Euclidean distances between the rows of A and of B, without
    forming the (len(A), len(B), d) difference array."""
    D = (A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2.0 * (A @ B.T)
    return np.maximum(D, 0.0)


def et_row_accessions(rep, split):
    """UniProt accession of every row of the ET feature file rep/split."""
    recs = records("ET", split)
    X, y = load_rep(rep, split)
    if rep == "PSSM":
        assert len(recs) == len(X), (rep, split, len(recs), len(X))
        assert all(lab == yy for (_, lab, _), yy in zip(recs, y))
        assert absent_residue_mismatches(X, recs) == 0, "PSSM rows are not in FASTA order"
        return [a for a, _, _ in recs]
    Xa, ya = load_rep("AAC", split)
    A = np.array([aac(s) for _, _, s in recs])
    D = sqdist_rows(Xa, A)
    j = D.argmin(1)
    exact = D[np.arange(len(Xa)), j] < 1e-4
    pos = np.full(len(Xa), -1)
    pos[exact] = j[exact]
    order = pos[exact]
    assert np.all(np.diff(order) > 0), "composition rows are not in FASTA order"
    for r in np.where(~exact)[0]:          # assign by position between neighbours
        lo = max([pos[q] for q in range(r) if pos[q] >= 0], default=-1)
        hi = min([pos[q] for q in range(r + 1, len(pos)) if pos[q] >= 0],
                 default=len(recs))
        cand = [c for c in range(lo + 1, hi)]
        # the free slot whose composition is closest
        best = min(cand, key=lambda c: D[r, c])
        pos[r] = best
    assert np.all(np.diff(pos) > 0)
    assert all(recs[p][1] == yy for p, yy in zip(pos, ya))
    return [recs[p][0] for p in pos]


def _pssm_csv(split):
    tag = {"train": "cv", "test": "ind"}[split]
    rows = [l.strip().split(",") for l in
            open(os.path.join(SEQ, "snare-cnn", f"pssm.{tag}.csv")) if l.strip()]
    X = np.array([[float(v) for v in r[:-1]] for r in rows])
    y = np.array([int(float(r[-1])) for r in rows])
    return X, y


def features(bench, rep, split):
    """(X, y, accessions) for one benchmark, representation and split."""
    if bench == "ET":
        X, y = load_rep(rep, split)
        return X, y, et_row_accessions(rep, split)
    recs = records(bench, split)
    accs = [a for a, _, _ in recs]
    y = np.array([lab for _, lab, _ in recs])
    if rep == "AAC":
        return np.array([aac(s) for _, _, s in recs]), y, accs
    if rep == "DPC":
        return np.array([dpc(s) for _, _, s in recs]), y, accs
    if rep == "PSSM" and bench == "SNARE":
        # the compilers' summary file lists the positives, then the
        # negatives, each sorted by accession; reorder it to the lists
        X, yp = _pssm_csv(split)
        assert len(X) == len(recs) and np.array_equal(np.sort(yp)[::-1], yp)
        npos = int(yp.sum())
        csv_accs = (sorted(a for a, lab, _ in recs if lab == 1) +
                    sorted(a for a, lab, _ in recs if lab == 0))
        assert len(csv_accs) == len(X) and npos == int(y.sum())
        row = {a: i for i, a in enumerate(csv_accs)}
        X = X[[row[a] for a in accs]]
        bad = absent_residue_mismatches(X, recs)
        assert bad == 0, f"PSSM rows do not match their accessions ({bad})"
        return X, y, accs
    raise ValueError((bench, rep))


AA_PSSM = "ARNDCQEGHILKMFPSTWYV"        # the residue order of PSI-BLAST profiles


def absent_residue_mismatches(X, recs):
    """A residue type absent from a sequence leaves its 20 entries of the
    400-dimensional profile summary at zero, and only then.  The number of
    rows whose zero blocks disagree with the residues absent from the
    sequence assigned to them; zero for a correct row order."""
    bad = 0
    for xr, (_, _, s) in zip(X, recs):
        absent_seq = {a for a in AA if a not in s}
        absent_row = {AA_PSSM[i] for i in range(20) if np.all(xr[20 * i:20 * i + 20] == 0)}
        bad += absent_seq != absent_row
    return bad


# -------------------------------------------------------------- homology
def mmseqs_version():
    try:
        return subprocess.run(["mmseqs", "version"], capture_output=True,
                              text=True).stdout.strip()
    except FileNotFoundError:
        return None


def write_fasta(path, recs):
    with open(path, "w") as f:
        for a, _, s in recs:
            f.write(f">{a}\n{s}\n")


def mmseqs_search(query, target, threads=2, sensitivity=7.5, evalue=1e-3,
                  max_seqs=1000):
    """All significant hits (E-value <= evalue) of query against target as
    tuples (q, t, identity), where identity = identical aligned residues /
    length of the shorter sequence.  Self-hits (same accession) are dropped.
    The E-value cut keeps chance alignments of very short peptides, which
    can reach any identity over a handful of residues, out of the result."""
    if shutil.which("mmseqs") is None:
        raise RuntimeError("MMseqs2 is required for this step (apt install mmseqs2)")
    work = tempfile.mkdtemp(prefix="mmseqs_")
    try:
        qf, tf = os.path.join(work, "q.fasta"), os.path.join(work, "t.fasta")
        write_fasta(qf, query)
        write_fasta(tf, target)
        out = os.path.join(work, "hits.m8")
        subprocess.run(["mmseqs", "easy-search", qf, tf, out, os.path.join(work, "tmp"),
                        "-s", str(sensitivity), "-e", str(evalue), "--max-seqs", str(max_seqs),
                        "--alignment-mode", "3", "--threads", str(threads),
                        "--format-output", "query,target,pident,alnlen,qlen,tlen"],
                       check=True, capture_output=True)
        hits = []
        for line in open(out):
            q, t, pid, aln, ql, tl = line.rstrip("\n").split("\t")
            if q == t:
                continue
            ident = float(pid) / 100.0 * int(aln) / min(int(ql), int(tl))
            hits.append((q, t, min(ident, 1.0)))
        return hits
    finally:
        shutil.rmtree(work, ignore_errors=True)


def max_identity(query, target, **kw):
    """{query accession: (largest identity to any target, that target)}."""
    best = {a: (0.0, None) for a, _, _ in query}
    for q, t, ident in mmseqs_search(query, target, **kw):
        if ident > best[q][0]:
            best[q] = (ident, t)
    return best
