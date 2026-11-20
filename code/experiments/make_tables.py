"""Emit every table of the article and of Online Resource 1, as LaTeX
fragments in tables/, directly from the result files.

A number transcribed by hand can drift from the result it reports.
Generating the tables from the stored results removes that whole class of
error: what the paper prints is what the code computed.  "Direct" is the
construction of Definition 3 of the paper (keys "v1" in the result files),
"proposed" the construction of Definition 4 (keys "v2").
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "tables"))
os.makedirs(OUT, exist_ok=True)
REPS = ["AAC", "DPC", "PSSM"]


def load(fn):
    with open(os.path.join(HERE, fn)) as f:
        return json.load(f)


def write(name, body):
    with open(os.path.join(OUT, name + ".tex"), "w") as f:
        f.write(body)
    print("wrote", name + ".tex")


def f4(v):
    """Four decimals; a negative value is set in math mode so that it carries
    a true minus sign rather than a hyphen."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"${v:.4f}$" if v < 0 else f"{v:.4f}"


def sg(v, d=4):
    """Explicitly signed number in math mode: $+0.1059$, $-0.0135$; a value
    that rounds to zero at this precision is printed without a sign."""
    if round(v, d) == 0:
        return f"${0:.{d}f}$"
    return f"${v:+.{d}f}$"


def ci(lo, hi, d=4):
    return f"$[{lo:+.{d}f},\\,{hi:+.{d}f}]$"


BANDS = [(-np.inf, 0.05, "$x<0.05$"), (0.05, 0.10, "$0.05\\le x<0.10$"),
         (0.10, 0.20, "$0.10\\le x<0.20$"), (0.20, 0.30, "$0.20\\le x<0.30$"),
         (0.30, 9.99, "$x\\ge 0.30$")]


# ---------------------------------------- tab_data (main Table 3)
def t_data():
    d = load("datastats.json")
    rows = []
    for rep in REPS:
        for sp in ("train", "test"):
            r = d[f"{rep}_{sp}"]
            rows.append(f"{rep if sp=='train' else ''} & {sp} & {r['d']} & "
                        f"{r['n']} & {r['pos']} & {100*r['rate']:.2f} & "
                        f"{1-r['rate']:.4f}\\\\")
        if rep != REPS[-1]:
            rows.append("\\addlinespace")
    write("tab_data", "\\begin{tabular}{llrrrrr}\n\\toprule\n"
          "representation & split & $d$ & instances & positives & "
          "positive rate (\\%) & majority accuracy\\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# ------------------------------------ tab_pitfalls (main Table 1)
def t_defects():
    d = load("verify.json")["defects"]

    def z(v):
        return "$0$" if v == 0.0 else f"{v:.3f}"
    rows = []
    for rep in REPS:
        r = d[rep]
        rows.append(f"{rep} & {z(r['v1'][0])} & {z(r['v1'][1])} & "
                    f"{z(r['v2'][0])} & {r['clip_share_pct']:.1f}\\\\")
    write("tab_pitfalls", "\\begin{tabular}{lrrrr}\n\\toprule\n"
          "& \\multicolumn{3}{c}{largest pairwise block distance} & \\\\\n"
          "\\cmidrule(lr){2-4}\n"
          "representation & $T$, direct & $I$, direct & $I$, proposed & "
          "$F$ entries clipped (\\%)\\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------- tab_main (main Table 5)
def t_main():
    """Every tuned model on the predefined test split, one row per model and
    one group of columns per representation.  The untuned uniform-weight
    configurations are in tab_confusion (Table S3)."""
    base = load("baselines.json")
    wk = load("weights_k.json")
    ver = load("verify.json")["uniform"]
    ab = load("ablation.json")
    fz = load("fuzzy_baselines.json")
    cb = load("centroid_baselines.json")
    trained = ["Logistic regression (L2)", "Linear SVM", "RBF SVM",
               "Random forest", "Gradient boosting"]
    # (label, kind, score of the model on one representation); kind "nn" is a
    # rule with neighbour search, the family the underline is taken over
    models = [("majority classifier", "ref", lambda r: ver[r]["majority"]),
              ("plain $k$-NN", "nn", lambda r: wk[r]["knn"]["test"]),
              ("plain $k$-NN, min-max", "nn", lambda r: ab[r]["minmax_knn"]["test"]),
              ("N-KNN \\cite{ashika2024,khan2025}", "nn", lambda r: ab[r]["nknn"]["test"]),
              ("direct transform", "nn", lambda r: wk[r]["v1"]["test"]),
              ("proposed transform", "nn", lambda r: wk[r]["v2"]["test"]),
              ("fuzzy $k$-NN \\cite{keller1985}", "nn", lambda r: fz[r]["FKNN"]),
              ("intuitionistic-fuzzy $k$-NN", "nn", lambda r: fz[r]["IF-KNN"]),
              ("NS-$k$-NN \\cite{akbulut2017nsknn}", "nn", lambda r: fz[r]["NS-kNN"]),
              ("nearest centroid$^{\\ddagger}$", "search-free",
               lambda r: cb[r]["nearest centroid"]),
              ("per-feature share sum$^{\\ddagger}$", "search-free",
               lambda r: cb[r]["share sum"])]
    models += [(nm + "$^{\\dagger}$", "trained", (lambda r, nm=nm: base[r][nm]))
               for nm in trained]
    best = {r: max(get(r)["F1"] for _, _, get in models) for r in REPS}
    bestnn = {r: max(get(r)["F1"] for _, kind, get in models if kind == "nn") for r in REPS}
    rows, prev = [], None
    for lab, kind, get in models:
        if prev is not None and kind != prev:
            rows.append("\\addlinespace")
        prev = kind
        cells = []
        for r in REPS:
            s_ = get(r)
            f1 = f4(s_["F1"])
            if abs(s_["F1"] - best[r]) < 1e-9:
                f1 = "\\textbf{" + f1 + "}"
            elif abs(s_["F1"] - bestnn[r]) < 1e-9:
                f1 = "\\underline{" + f1 + "}"
            cells += [f4(s_["ACC"]), f1, f4(s_.get("AUC")), f4(s_["MCC"])]
        rows.append(lab + " & " + " & ".join(cells) + "\\\\")
    head = " & ".join("\\multicolumn{4}{c}{%s}" % r for r in REPS)
    write("tab_main", "\\begin{tabular}{l" + "@{\\hspace{7pt}}rrrr" * 3 + "}\n\\toprule\n"
          "& " + head + "\\\\\n"
          "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\n"
          "model" + " & acc. & F1 & AUC & MCC" * 3 + "\\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def t_weights():
    """Main Table 4: the configurations selected by cross-validation for the
    tuned nearest-neighbour rules built from the same features, including
    the four combinations of the two design rules, with the paired-bootstrap
    difference of the proposed construction to each (plain scheme)."""
    wk = load("weights_k.json")
    tied = load("multiclass.json")["binary_tied"]
    ab = load("ablation.json")
    pb = load("symmetry_boot.json")["paired_bootstrap"]
    yn = lambda b_: "yes" if b_ else "no"

    def wfmt(w):
        return "--" if w is None else "$(" + ",\\,".join(f"{x:.1f}" for x in w) + ")$"

    def diff(r):
        return f"{sg(r['point'], 3)} {ci(*r['ci95'], 3)}"
    rows = []
    for rep, name in zip(REPS, ["amino-acid composition (AAC)", "dipeptide composition (DPC)",
                                "profile representation (PSSM)"]):
        rows.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{name}}}}}\\\\")
        lines = [("plain $k$-NN", None, wk[rep]["knn"]["best_k"], wk[rep]["knn"]["best_cw"],
                  wk[rep]["knn"]["cv_f1"], wk[rep]["knn"]["test"], diff(pb[rep]["v2_minus_knn"])),
                 ("N-KNN", None, ab[rep]["nknn"]["best_k"], ab[rep]["nknn"]["best_cw"],
                  ab[rep]["nknn"]["cv_f1"], ab[rep]["nknn"]["test"],
                  diff(ab[rep]["bootstrap"]["v2_minus_nknn"])),
                 ("direct (neither)", wk[rep]["v1"]["best_w"], wk[rep]["v1"]["best_k"],
                  wk[rep]["v1"]["best_cw"], wk[rep]["v1"]["cv_f1"], wk[rep]["v1"]["test"],
                  diff(pb[rep]["v2_minus_v1"])),
                 ("rule 1 only", None, ab[rep]["rule1"]["best_k"], ab[rep]["rule1"]["best_cw"],
                  ab[rep]["rule1"]["cv_f1"], ab[rep]["rule1"]["test"],
                  diff(ab[rep]["bootstrap"]["v2_minus_rule1"])),
                 ("rule 2 only", ab[rep]["rule2"]["best_w"], ab[rep]["rule2"]["best_k"],
                  ab[rep]["rule2"]["best_cw"], ab[rep]["rule2"]["cv_f1"], ab[rep]["rule2"]["test"],
                  diff(ab[rep]["bootstrap"]["v2_minus_rule2"])),
                 ("proposed (both)", wk[rep]["v2"]["best_w"], wk[rep]["v2"]["best_k"],
                  wk[rep]["v2"]["best_cw"], wk[rep]["v2"]["cv_f1"], wk[rep]["v2"]["test"], "--")]
        for lab, w, k, cw, cv, te, d in lines:
            rows.append(f"\\quad {lab} & {wfmt(w)} & {k} & {yn(cw)} & {cv:.4f} & "
                        f"{te['F1']:.4f} & {f4(te['MCC'])} & {d}\\\\")
        t = tied[rep]
        wm = (1 - t["w_I"]) / 2
        rows.append(f"\\quad proposed, tied$^{{*}}$ & $({t['w_I']:.1f},\\,{wm:.2f},\\,{wm:.2f})$ & "
                    f"{t['k']} & {yn(t['cw'])} & {t['cv_f1']:.4f} & "
                    f"{t['test_f1']:.4f} & {f4(t['test_mcc'])} & --\\\\")
        if rep != REPS[-1]:
            rows.append("\\addlinespace")
    write("tab_weights", "\\begin{tabular}{lccccccc}\n\\toprule\n"
          "& selected $w$ & $k$ & CW & CV F1 & test F1 & test MCC & "
          "proposed $-$ row\\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# ------------------------------------------- tab_stats (Table S4)
def t_stats():
    d = load("stats30.json")
    rows = []
    lab = {"family1": "proposed transformation vs.\\ plain $k$-NN, both tuned",
           "family2": "proposed vs.\\ direct transformation, both tuned"}
    for fam in ("family1", "family2"):
        if not d.get(fam):
            continue
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{lab[fam]}}}}}\\\\")
        for rep in REPS:
            r = d[fam][rep]
            star = "$^{\\ddagger}$" if abs(r["p_raw"] - d["wilcoxon_floor_p"]) \
                < 1e-12 else ""
            rows.append(
                f"\\quad {rep} & {sg(r['mean_diff'])} & "
                f"{ci(r['ci95'][0], r['ci95'][1])} & "
                f"{sg(r['dz'], 2)} & {r['wins']}/{r['n']} & "
                f"{r['p_raw']:.2e}{star} & {r['p_holm']:.2e}\\\\")
        rows.append("\\addlinespace")
    write("tab_stats", "\\begin{tabular}{lrcrrrr}\n\\toprule\n"
          "& mean difference & 95\\% CI & $d_{z}$ & splits won & "
          "$p$ & $p$ (Holm)\\\\\n\\midrule\n"
          + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")


# ------------------------------------------ tab_regime (Table S10)
def t_regime():
    d = load("regime.json")
    P = d["points"]
    x = np.array([p["x"] for p in P])
    y = np.array([p["y"] for p in P])
    rows = []
    for lo, hi, lab in [(-np.inf, 0.05, "$x < 0.05$"), (0.05, 0.10, "$0.05 \\le x < 0.10$"),
                        (0.10, 0.20, "$0.10 \\le x < 0.20$"),
                        (0.20, 0.30, "$0.20 \\le x < 0.30$"),
                        (0.30, 9.99, "$x \\ge 0.30$")]:
        m = (x >= lo) & (x < hi)
        if m.sum() == 0:
            continue
        rows.append(f"{lab} & {int(m.sum())} & {100*(y[m]>0).mean():.0f} & "
                    f"{sg(y[m].mean())} & {sg(np.median(y[m]))}\\\\")
    rows.append("\\midrule")
    for src in ["A transport-protein feature subsets", "B external real", "C synthetic",
                "ALL"]:
        if src not in d["spearman"]:
            continue
        s = d["spearman"][src]
        sub = P if src == "ALL" else [p for p in P if p["source"] == src]
        xs = np.array([p["x"] for p in sub])
        rows.append(
            f"\\emph{{{src}}} & {s['n']} & {100*s['frac_help']:.0f} & "
            f"$\\rho={s['rho']:+.3f}$ & "
            f"$x\\in[{xs.min():.2f},{xs.max():.2f}]$\\\\")
    write("tab_regime", "\\begin{tabular}{lrrrr}\n\\toprule\n"
          "band or source & tasks & helps (\\%) & "
          "mean gain & median gain\\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------- tab_runtime (Table S16)
def t_runtime():
    d = load("diagnostics.json")["runtime"]
    rows = []
    for r in d["grid"]:
        if r["d"] in (50, 400):
            # the ratio is recomputed from the printed four-decimal times so
            # that a reader can reproduce it from the table itself
            ratio = (r["transform_s"] + r["search_s"]) / max(r["plain_knn_s"], 1e-9)
            rows.append(f"{r['n']} & {r['d']} & {r['transform_s']:.4f} & "
                        f"{r['search_s']:.4f} & {r['plain_knn_s']:.4f} & "
                        f"{ratio:.1f}\\\\")
    write("tab_runtime", "\\begin{tabular}{rrrrrr}\n\\toprule\n"
          "$N$ & $d$ & transform (s) & search (s) & plain $k$-NN (s) & "
          "ratio\\\\\n\\midrule\n" + "\n".join(rows)
          + "\n\\bottomrule\n\\end{tabular}\n")


# --------------------------------------- tab_confusion (Table S3)
def t_confusion():
    """The untuned configurations at k = 5 and uniform weights, on all three
    representations: confusion counts and scores (supplementary table)."""
    d = load("diagnostics.json")["confusion"]
    ver = load("verify.json")["uniform"]
    order = [("plain $k$-NN", "KNN"), ("plain $k$-NN + CW", "KNN+CW"),
             ("direct transform", "v1"), ("direct transform + CW", "v1+CW"),
             ("proposed transform", "v2"),
             ("proposed transform + CW", "v2+CW")]
    rows = []
    for rep in REPS:
        pos = np.array(d[rep]["KNN"])[1].sum() / np.array(d[rep]["KNN"]).sum()
        rows.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{rep}}}: share of positives "
                    f"in the test split {pos:.3f}}}\\\\")
        for lab, key in order:
            cm = np.array(d[rep][key])
            tn, fp, fn, tp = cm.ravel()
            tot = cm.sum()
            u = ver[rep][key]
            assert abs(u["ACC"] - (tn + tp) / tot) < 1e-12
            rows.append(f"\\quad {lab} & {tn} & {fp} & {fn} & {tp} & "
                        f"{(fp+tp)/tot:.3f} & {u['ACC']:.4f} & {u['F1']:.4f} & {f4(u['MCC'])}\\\\")
        if rep != REPS[-1]:
            rows.append("\\addlinespace")
    write("tab_confusion", "\\begin{tabular}{lrrrrcccc}\n\\toprule\n"
          "& TN & FP & FN & TP & predicted & accuracy & macro F1 & MCC\\\\\n"
          "& & & & & positive rate & & & \\\\\n\\midrule\n" + "\n".join(rows)
          + "\n\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------- tab_symmetry (Table S1)
def t_symmetry():
    """Supplementary table: the selected weights under exchanged labels
    (symmetry_boot.json) and under finer searches (finegrid.json, exp14)."""
    d = load("symmetry_boot.json")
    fg = load("finegrid.json")["results"]

    def wfmt(x):
        s = f"{x:.3f}".rstrip("0")
        return s + "0" if s.endswith(".") else s
    rows = []
    for rep in REPS:
        entries = [("grid $0.1$", d["label_swap"][rep]["original"]),
                   ("grid $0.1$, labels exchanged", d["label_swap"][rep]["swapped"])]
        for key, lab in (("fine", "grid $0.05$"), ("tied_fine", "tied, step $0.01$")):
            r = fg[rep][key]
            entries.append((lab, {"w": r["w"], "k": r["k"], "cv_f1": r["cv_f1"],
                                  "test_f1": r["test_f1"], "test_mcc": r["test_mcc"]}))
        for i, (lab, r) in enumerate(entries):
            w = ", ".join(wfmt(x) for x in r["w"])
            rows.append(f"{rep if i == 0 else ''} & {lab} & $({w})$ & "
                        f"{r['k']} & {r['cv_f1']:.4f} & {r['test_f1']:.4f} & "
                        f"{r['test_mcc']:.4f}\\\\")
        if rep != REPS[-1]:
            rows.append("\\addlinespace")
    write("tab_symmetry", "\\begin{tabular}{llcrrrr}\n\\toprule\n"
          " & search & $(w_I,w_{M_0},w_{M_1})$ & $k$ & "
          "CV F1 & test F1 & test MCC\\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def t_bootstrap():
    """Supplementary table: the paired bootstrap of the proposed construction
    against every tuned nearest-neighbour rule of Table 4, plain and
    stratified schemes."""
    d = load("symmetry_boot.json")["paired_bootstrap"]
    ab = load("ablation.json")
    blocks = [("v2_minus_v1", "proposed minus direct (neither rule)", lambda rep: d[rep]["v2_minus_v1"]),
              ("v2_minus_rule1", "proposed minus rule 1 only", lambda rep: ab[rep]["bootstrap"]["v2_minus_rule1"]),
              ("v2_minus_rule2", "proposed minus rule 2 only", lambda rep: ab[rep]["bootstrap"]["v2_minus_rule2"]),
              ("v2_minus_knn", "proposed minus plain $k$-NN", lambda rep: d[rep]["v2_minus_knn"]),
              ("v2_minus_nknn", "proposed minus N-KNN", lambda rep: ab[rep]["bootstrap"]["v2_minus_nknn"])]
    rows = []
    for key, lab, get in blocks:
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{lab}}}}}\\\\")
        for rep in REPS:
            r = get(rep)
            # observed difference at full precision; a difference of two rounded
            # entries of tab_main can differ from it in the last digit
            rows.append(f"\\quad {rep} & {sg(r['point'])} & "
                        f"{ci(*r['ci95'])} & {r['p_two_sided']:.3f} & "
                        f"{ci(*r['ci95_stratified'])} & "
                        f"{r['p_two_sided_stratified']:.3f}\\\\")
        rows.append("\\addlinespace")
    write("tab_bootstrap", "\\begin{tabular}{lrcrcr}\n\\toprule\n"
          "& observed difference & \\multicolumn{2}{c}{plain bootstrap} & "
          "\\multicolumn{2}{c}{stratified bootstrap}\\\\\n"
          "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
          "& in macro F1 & 95\\% CI & $p$ & 95\\% CI & $p$\\\\\n\\midrule\n"
          + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")


def t_eps():
    d = load("symmetry_boot.json")["eps_at_selected"]
    keys = list(d["AAC"].keys())
    head = " & ".join("$10^{%d}$" % int(k.split("e")[1]) for k in keys)
    rows = [f"{rep} & " + " & ".join(f"{d[rep][k]:.4f}" for k in keys) + "\\\\"
            for rep in REPS]
    write("tab_eps", "\\begin{tabular}{l" + "r" * len(keys) + "}\n\\toprule\n"
          "$\\varepsilon_s$ & " + head + "\\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def t_multiclass():
    d = load("multiclass.json")
    rows = []
    for name, lab in [("digits-imbalanced", "digits, 10 classes, imbalanced"),
                      ("wine", "wine, 3 classes")]:
        r = d[name]
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{lab}}}: $n={r['n']}$, $d={r['d']}$}}\\\\")
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\quad class sizes "
                    f"{', '.join(str(c) for c in r['class_sizes'])}}}\\\\")
        rows.append(f"\\quad plain $k$-NN & -- & {r['knn']['k']} & "
                    f"{'yes' if r['knn']['cw'] else 'no'} & "
                    f"{r['knn']['test_f1']:.4f} & {r['knn']['test_mcc']:.4f}\\\\")
        v = r["v1"]
        rows.append(f"\\quad direct transform & $w_I={v['w'][1]:.1f}$ & {v['k']} & "
                    f"{'yes' if v['cw'] else 'no'} & {v['test_f1']:.4f} & {v['test_mcc']:.4f}\\\\")
        v = r["v2"]
        rows.append(f"\\quad proposed transform & $w_I={v['w'][0]:.1f}$ & {v['k']} & "
                    f"{'yes' if v['cw'] else 'no'} & {v['test_f1']:.4f} & {v['test_mcc']:.4f}\\\\")
        rows.append("\\addlinespace")
    write("tab_multiclass", "\\begin{tabular}{llcccr}\n\\toprule\n"
          "& selected weight & $k$ & CW & test macro F1 & test MCC\\\\\n\\midrule\n"
          + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
    # the whole w_I profile of the proposed construction, one row per dataset
    keys = list(d["digits-imbalanced"]["v2_wI_profile"].keys())
    prow = []
    for name, lab in [("digits-imbalanced", "digits"), ("wine", "wine")]:
        prof = d[name]["v2_wI_profile"]
        best = max(prof.values())
        cells = [("\\textbf{" + f"{prof[k]:.3f}" + "}") if abs(prof[k] - best) < 1e-9
                 else f"{prof[k]:.3f}" for k in keys]
        prow.append(f"{lab} & " + " & ".join(cells) + "\\\\")
    write("tab_wiprofile", "\\begin{tabular}{l" + "r" * len(keys) + "}\n\\toprule\n"
          "$w_I$ & " + " & ".join(keys) + "\\\\\n\\midrule\n"
          + "\n".join(prow) + "\n\\bottomrule\n\\end{tabular}\n")


COLS4 = [("coupled", "coupled", "untuned"),
         ("crossfit", "cross-fitted", "untuned"),
         ("coupled_tuned", "coupled", "both arms tuned"),
         ("crossfit_tuned", "cross-fitted", "both arms tuned")]


def regime4(pts, name, within=()):
    """Four re-analyses side by side, with the five disjoint bands of
    tab_regime (Table S10), so that the band counts of every column sum to
    the total.  `within` lists (source prefix, label) pairs for which the
    Spearman coefficient within that source is added below the pooled one."""
    from scipy.stats import spearmanr
    from exp4_regime import spearman_boot
    rows = ["tasks & " + " & ".join(str(len(pts)) for _ in COLS4) + "\\\\"]
    rhos, cis = [], []
    for k, _, _ in COLS4:
        x = np.array([p[k]["x"] for p in pts])
        y = np.array([p[k]["y"] for p in pts])
        rho, _, lo, hi = spearman_boot(x, y)
        rhos.append(f"${rho:+.3f}$")
        cis.append(ci(lo, hi, 3))
    rows.append("Spearman $\\rho$ & " + " & ".join(rhos) + "\\\\")
    rows.append("95\\% CI & " + " & ".join(cis) + "\\\\")
    for pre, lab in within:
        q = [p for p in pts if p["source"].startswith(pre)]
        vals = []
        for k, _, _ in COLS4:
            r = spearmanr([p[k]["x"] for p in q], [p[k]["y"] for p in q])[0]
            vals.append(f"${r:+.3f}$")
        rows.append(f"\\quad {lab} & " + " & ".join(vals) + "\\\\")
    from exp4_regime import mean_boot
    mg, mci, hp = [], [], []
    for k, _, _ in COLS4:
        y = np.array([p[k]["y"] for p in pts])
        lo, hi = mean_boot(y)
        mg.append(sg(y.mean(), 3))
        mci.append(ci(lo, hi, 3))
        hp.append(f"{100*(y>0).mean():.0f}")
    rows.append("\\addlinespace")
    rows.append("\\multicolumn{5}{l}{all tasks}\\\\")
    rows.append("\\quad helps (\\%) & " + " & ".join(hp) + "\\\\")
    rows.append("\\quad mean gain & " + " & ".join(mg) + "\\\\")
    rows.append("\\quad 95\\% CI & " + " & ".join(mci) + "\\\\")
    for lo, hi, lab in BANDS:
        cells = {"n": [], "h": [], "g": []}
        for k, _, _ in COLS4:
            x = np.array([p[k]["x"] for p in pts])
            y = np.array([p[k]["y"] for p in pts])
            m = (x >= lo) & (x < hi)
            cells["n"].append(str(int(m.sum())))
            cells["h"].append(f"{100*(y[m]>0).mean():.0f}" if m.sum() else "--")
            cells["g"].append(sg(y[m].mean(), 3) if m.sum() else "--")
        rows.append("\\addlinespace")
        rows.append(f"\\multicolumn{{5}}{{l}}{{band {lab}}}\\\\")
        rows.append("\\quad tasks & " + " & ".join(cells["n"]) + "\\\\")
        rows.append("\\quad helps (\\%) & " + " & ".join(cells["h"]) + "\\\\")
        rows.append("\\quad mean gain & " + " & ".join(cells["g"]) + "\\\\")
    write(name, "\\begin{tabular}{lcccc}\n\\toprule\n"
          "& " + " & ".join(c[1] for c in COLS4) + "\\\\\n"
          "& " + " & ".join(c[2] for c in COLS4) + "\\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def t_regime_main():
    """The main-text regime table: the tasks of sources A-C and the
    independent PMLB tasks, each at k = 5 with coupled coordinates and with
    both rules tuned and the coordinates cross-fitted."""
    from exp4_regime import mean_boot, spearman_boot
    cf = load("regime_crossfit.json")["points"]
    pm = load("regime_pmlb.json")["points"]
    cols = [(cf, "coupled"), (cf, "crossfit_tuned"), (pm, "coupled"), (pm, "crossfit_tuned")]
    XY = [(np.array([p[k]["x"] for p in P]), np.array([p[k]["y"] for p in P])) for P, k in cols]
    rows = ["tasks & " + " & ".join(str(len(x)) for x, _ in XY) + "\\\\"]
    rr, cc = [], []
    for x, y in XY:
        rho, _, lo, hi = spearman_boot(x, y)
        rr.append(f"${rho:+.3f}$")
        cc.append(ci(lo, hi, 3))
    rows.append("Spearman $\\rho$ & " + " & ".join(rr) + "\\\\")
    rows.append("\\quad 95\\% CI & " + " & ".join(cc) + "\\\\")
    for lo, hi, lab in BANDS:
        n_, h_, g_ = [], [], []
        for x, y in XY:
            m = (x >= lo) & (x < hi)
            n_.append(str(int(m.sum())))
            h_.append(f"{100*(y[m]>0).mean():.0f}" if m.sum() else "--")
            g_.append(sg(y[m].mean(), 3) if m.sum() else "--")
        rows.append("\\addlinespace")
        rows.append(f"\\multicolumn{{5}}{{l}}{{band {lab}}}\\\\")
        rows.append("\\quad tasks & " + " & ".join(n_) + "\\\\")
        rows.append("\\quad helps (\\%) & " + " & ".join(h_) + "\\\\")
        rows.append("\\quad mean gain & " + " & ".join(g_) + "\\\\")
    rows.append("\\addlinespace")
    rows.append("\\multicolumn{5}{l}{all tasks}\\\\")
    rows.append("\\quad helps (\\%) & " + " & ".join(
        f"{100*(y>0).mean():.0f}" for _, y in XY) + "\\\\")
    rows.append("\\quad mean gain & " + " & ".join(sg(y.mean(), 3) for _, y in XY) + "\\\\")
    rows.append("\\quad 95\\% CI & " + " & ".join(ci(*mean_boot(y), 3) for _, y in XY) + "\\\\")
    write("tab_regime_main", "\\begin{tabular}{lcccc}\n\\toprule\n"
          f"& \\multicolumn{{2}}{{c}}{{sources A--C, ${len(cf)}$ tasks}} & "
          f"\\multicolumn{{2}}{{c}}{{PMLB, ${len(pm)}$ independent tasks}}\\\\\n"
          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
          "& $k=5$, coupled & tuned, cross-fitted & $k=5$, coupled & tuned, cross-fitted\\\\\n"
          "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def t_crossfit():
    regime4(load("regime_crossfit.json")["points"], "tab_crossfit",
            within=(("A", "source A"), ("B", "source B"),
                    ("C", "source C")))


def t_pmlb():
    d = load("regime_pmlb.json")
    pts = d["points"]
    regime4(pts, "tab_pmlb")
    # the full list, for the supplement
    rows = []
    for p in sorted(pts, key=lambda q: q["crossfit_tuned"]["x"]):
        nm = p["task"].replace("_", "\\_")
        rows.append(f"{nm} & {p['n']} & {p['d']} & {p['pos_rate']:.2f} & "
                    f"{sg(p['crossfit_tuned']['x'], 3)} & "
                    f"{sg(p['crossfit_tuned']['y'], 3)}\\\\")
    write("tab_pmlb_list", "\\begin{tabular}{lrrrrr}\n\\toprule\n"
          "dataset & $n$ & $d$ & positive share & $x$ & $y$\\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# ------------------------------------------- tab_homology (main text)
HOM_RULES = [("plain", "knn"), ("N-KNN", "nknn"), ("proposed", "proposed"),
             ("SVM$^{\\dagger}$", "RBF SVM"), ("GB$^{\\dagger}$", "Gradient boosting")]


def t_homology():
    """Main table: every rule on remote and homologous test proteins (30%)."""
    h = load("homology_eval.json")
    rows = []
    for rep in REPS:
        st = h[rep]["strata"]["0.30"]
        for s_ in ("remote", "homologous"):
            r = st[s_]
            c = r["compare"]["proposed-knn"]
            rows.append(f"{rep if s_ == 'remote' else ''} & {s_} & {r['n']} ({r['positives']}) & "
                        + " & ".join(f"{r['rules'][key]['F1']:.3f}" for _, key in HOM_RULES)
                        + f" & {sg(c['point'], 3)} {{\\scriptsize {ci(*c['ci95'], 3)}}}\\\\")
        if rep != REPS[-1]:
            rows.append("\\addlinespace")
    write("tab_homology", "\\begin{tabular}{llr" + "r" * len(HOM_RULES) + "l}\n\\toprule\n"
          "& & proteins & \\multicolumn{" + str(len(HOM_RULES)) + "}{c}{test macro F1} & "
          "proposed $-$ plain\\\\\n"
          "\\cmidrule(lr){4-" + str(3 + len(HOM_RULES)) + "}\n"
          "& stratum & (positive) & " + " & ".join(lab for lab, _ in HOM_RULES)
          + " & $95\\%$ interval\\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# ------------------------------------- tab_homology_sens (Table S5)
def t_homology_sens():
    """Supplement: the stratified comparison at 30%, 40% and 50% identity."""
    h = load("homology_eval.json")
    rows = []
    for thr in ("0.30", "0.40", "0.50"):
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{threshold {float(thr):.0%}}}}}\\\\".replace("%", "\\%"))
        for rep in REPS:
            st = h[rep]["strata"][thr]
            for s_ in ("remote", "homologous"):
                r = st[s_]
                c = r["compare"]
                rows.append(
                    f"{rep if s_ == 'remote' else ''} & {s_} & {r['n']} ({r['positives']}) & "
                    f"{r['rules']['knn']['F1']:.3f} & {r['rules']['proposed']['F1']:.3f} & "
                    f"{sg(c['proposed-knn']['point'], 3)} {{\\scriptsize {ci(*c['proposed-knn']['ci95'], 3)}}} & "
                    f"{sg(c['proposed-nknn']['point'], 3)} {{\\scriptsize {ci(*c['proposed-nknn']['ci95'], 3)}}}\\\\")
            it = st["interaction"]["proposed-knn"]
            rows.append(f" & difference & & & & {sg(it['point'], 3)} {{\\scriptsize {ci(*it['ci95'], 3)}}} & \\\\")
        if thr != "0.50":
            rows.append("\\addlinespace")
    write("tab_homology_sens", "\\begin{tabular}{llrrrll}\n\\toprule\n"
          "& stratum & $n$ (pos.) & plain & proposed & proposed $-$ plain & proposed $-$ N-KNN\\\\\n"
          "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------- tab_benchmarks (main text)
VERDICT = {"confirmed": "confirmed", "contradicted": "contradicted",
           "inconclusive": "inconclusive", "no prediction": "--"}


def prediction_verdict(prediction, ci):
    """The scoring rule of the plan (exp11b_benchmarks.verdict), restated
    here so that the tables need no sequence data; t_benchmarks checks that
    it reproduces every stored test verdict."""
    lo, hi = ci
    if prediction == "none":
        return "no prediction"
    if prediction == "gain":
        return "confirmed" if lo > 0 else ("contradicted" if hi < 0 else "inconclusive")
    return "confirmed" if hi < 0 else ("contradicted" if lo > 0 else "inconclusive")
TRAINED_NAMES = ["Logistic regression (L2)", "Linear SVM", "RBF SVM", "Random forest",
                 "Gradient boosting"]
SHORT = {"Logistic regression (L2)": "LR", "Linear SVM": "LSVM", "RBF SVM": "SVM",
         "Random forest": "RF", "Gradient boosting": "GB"}
BENCH_NAME = {"ET": "transport", "SNARE": "SNARE", "VES": "vesicular"}


def t_benchmarks():
    """Main table: the two further benchmarks, one column per benchmark and
    representation, every frozen prediction scored."""
    b = load("benchmarks.json")["benchmarks"]
    cols = [(bench, rep) for bench in ("SNARE", "VES") for rep in b[bench]]

    def n(v):
        return f"{v:,}".replace(",", "{,}")

    def row(label, f):
        return label + " & " + " & ".join(f(b[bb][rr]) for bb, rr in cols) + "\\\\"

    def best(r):
        bt = max(TRAINED_NAMES, key=lambda k: r["rules"][k]["F1"])
        return f"{SHORT[bt]} {r['rules'][bt]['F1']:.3f}"
    rows = [row("test proteins", lambda r: f"${n(r['n_test'])}$"),
            row("\\quad positive", lambda r: f"{r['test_positives']}"),
            "\\midrule"]
    pre = load("prereg.json")["benchmarks"]
    for lab, key in (("plain $k$-NN", "knn"), ("proposed", "proposed"), ("N-KNN", "nknn"),
                     ("centroid", "centroid")):
        rows.append(row(lab, lambda r, key=key: f"{r['rules'][key]['F1']:.3f}"))
        if key == "proposed":       # the indeterminacy weight selected on the training split
            rows.append("\\quad selected $w_{I}$ & " + " & ".join(
                f"{pre[bb][rr]['configs']['proposed']['w'][0]:.1f}" for bb, rr in cols) + "\\\\")
    rows.append(row("best trained", best))
    rows.append("\\midrule")
    rows.append(row("proposed $-$ plain", lambda r: sg(r["compare"]["proposed-knn"]["point"], 3)))
    rows.append(row("\\quad $95\\%$ interval", lambda r: "{\\scriptsize " + ci(*r["compare"]["proposed-knn"]["ci95"], 3) + "}"))
    rows.append("\\midrule")
    # the same predictions scored, outside the plan, against the grouped
    # cross-validation (exp12): mean difference and corrected interval
    g = load("groupcv.json")["results"]
    gci = {(bb, rr): g[f"{bb}|{rr}"]["group"]["proposed-knn"]["ci95_corrected"] for bb, rr in cols}
    for bb, rr in cols:
        for kind in ("regime", "cv_sign"):
            pr = b[bb][rr]["predictions"][kind]
            assert prediction_verdict(pr["predicted"], b[bb][rr]["compare"]["proposed-knn"]["ci95"]) \
                == pr["verdict"], "verdict rule differs from exp11b"

    def grouped(kind):
        return lambda r, kind=kind: VERDICT[prediction_verdict(
            r["predictions"][kind]["predicted"], gci[next(c for c in cols if b[c[0]][c[1]] is r)])]
    rows.append(row("$x_{\\text{cv}}$", lambda r: f"{r['predictions']['regime']['x_cv']:.2f}"))
    rows.append(row("regime prediction", lambda r: r["predictions"]["regime"]["predicted"]))
    rows.append(row("\\quad test split", lambda r: VERDICT[r["predictions"]["regime"]["verdict"]]))
    rows.append(row("\\quad grouped CV", grouped("regime")))
    rows.append(row("CV sign prediction", lambda r: r["predictions"]["cv_sign"]["predicted"]))
    rows.append(row("\\quad test split", lambda r: VERDICT[r["predictions"]["cv_sign"]["verdict"]]))
    rows.append(row("\\quad grouped CV", grouped("cv_sign")))
    head = (" & \\multicolumn{3}{c}{SNARE} & \\multicolumn{2}{c}{vesicular transport}\\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\n"
            "& " + " & ".join(rr for _, rr in cols) + "\\\\\n\\midrule\n")
    write("tab_benchmarks", "\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\n" + head
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# ------------------------------------------ tab_groupcv (main text)
def t_groupcv():
    """Main table: homology-aware cross-validation on the pooled data."""
    g = load("groupcv.json")["results"]
    rows = []
    prev = None
    for key, o in g.items():
        bench, rep = key.split("|")
        gr = o["group"]
        d = gr["proposed-knn"]
        if prev is not None and bench != prev:
            rows.append("\\addlinespace")
        rows.append(
            f"{BENCH_NAME[bench] if bench != prev else ''} & {rep} & "
            f"{gr['knn']['mean_F1']:.3f} & {gr['proposed']['mean_F1']:.3f} & "
            f"{gr['nknn']['mean_F1']:.3f} & {gr['Gradient boosting']['mean_F1']:.3f} & "
            f"{sg(d['mean'], 3)} {{\\scriptsize {ci(*d['ci95_corrected'], 3)}}} & "
            f"{d['wins']}/{d['folds']}\\\\")
        prev = bench
    write("tab_groupcv", "\\begin{tabular}{llrrrrlr}\n\\toprule\n"
          "& & \\multicolumn{4}{c}{mean test macro F1} & \\multicolumn{2}{c}{proposed $-$ plain}\\\\\n"
          "\\cmidrule(lr){3-6}\\cmidrule(lr){7-8}\n"
          "& & plain & proposed & N-KNN & GB & mean, corrected $95\\%$ interval & wins\\\\\n"
          "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# ------------------------------------- tab_groupcv_full (supplement)
def t_groupcv_full():
    """Supplement: grouped and random folds, every rule, and the folds' make-up."""
    g = load("groupcv.json")["results"]
    rows = []
    prev = None
    for key, o in g.items():
        bench, rep = key.split("|")
        if prev is not None and bench != prev:
            rows.append("\\addlinespace")
        for scheme in ("group", "random"):
            if scheme not in o:
                continue
            gr = o[scheme]
            lab = (f"{BENCH_NAME[bench]} {rep}" if scheme == "group" else "")
            comp = (f"{o['components']:,} / {o['largest_component']}".replace(",", "{,}")
                    if scheme == "group" else "")
            rows.append(f"{lab} & {comp} & {'grouped' if scheme == 'group' else 'random'} & "
                        f"{gr['proposed-knn']['folds']} & "
                        + " & ".join(f"{gr[k]['mean_F1']:.3f}" for k in
                                     ("knn", "direct", "proposed", "nknn", "Gradient boosting"))
                        + f" & {sg(gr['proposed-nknn']['mean'], 3)} & "
                        f"{sg(gr['Gradient boosting-proposed']['mean'], 3)}\\\\")
        prev = bench
    write("tab_groupcv_full", "\\begin{tabular}{llllrrrrrrr}\n\\toprule\n"
          "& components / & & & \\multicolumn{5}{c}{mean test macro F1} & proposed & GB $-$\\\\\n"
          "\\cmidrule(lr){5-9}\n"
          "& largest & folds & $J$ & plain & direct & proposed & N-KNN & GB & $-$ N-KNN & proposed\\\\\n"
          "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# ------------------------------------------ tab_learned (main text)
def t_learned():
    """Learned metric and trained models with the construction: point
    differences, with their intervals on the line below."""
    L = load("learned.json")
    rows = []
    prev = None
    keys = ("proposed-nca", "nca-knn", "svm_image-svm_raw", "gb_augmented-gb_raw")
    for key, o in L.items():
        bench, rep = key.split("|")
        ru, c = o["rules"], o["compare"]
        if prev is not None and bench != prev:
            rows.append("\\addlinespace")
        rows.append(f"{BENCH_NAME[bench] if bench != prev else ''} & {rep} & {ru['knn']['F1']:.3f} & "
                    f"{ru['nca']['F1']:.3f} & {ru['proposed']['F1']:.3f} & "
                    + " & ".join(sg(c[k]["point"], 3) for k in keys) + "\\\\")
        rows.append("& & & & & " + " & ".join("{\\scriptsize " + ci(*c[k]["ci95"], 3) + "}"
                                               for k in keys) + "\\\\")
        prev = bench
    write("tab_learned", "\\begin{tabular}{llrrrcccc}\n\\toprule\n"
          "& & \\multicolumn{3}{c}{test macro F1} & proposed & NCA & SVM & GB\\\\\n"
          "\\cmidrule(lr){3-5}\n"
          "& & plain & NCA & proposed & $-$ NCA & $-$ plain & image $-$ raw & augm. $-$ raw\\\\\n"
          "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def nca_budgets(key):
    """The budgets NCA was fitted at on one representation, as
    {name: (cross-validated macro F1, test macro F1, proposed minus NCA)},
    and the name of the budget its own cross-validation prefers (the smaller
    budget on a tie).  "first" is the budget of exp13 (at most 50 components
    on at most 3,000 proteins), "larger" that of exp16 (200 and 5,000) and
    "full" the full-rank fit of exp16b on every training protein."""
    L = load("learned.json")[key]
    nb = load("nca_budget.json")["results"][key]
    nf = load("nca_fullrank.json")["results"].get(key)
    out = {"first": (L["configs"]["nca"]["cv_f1"], L["rules"]["nca"]["F1"],
                     L["compare"]["proposed-nca"], None)}
    if not nb["identical_budgets"]:
        out["larger"] = (nb["large"]["cv_f1"], nb["large"]["F1"],
                         nb["compare"]["proposed-large"], nb["compare"]["large-small"])
    if nf is not None:
        out["full"] = (nf["full"]["cv_f1"], nf["full"]["F1"],
                       nf["compare"]["proposed-full"], nf["compare"]["full-small"])
    return out, max(out, key=lambda b: out[b][0])


def t_learned_budget():
    """Supplement, the continuation of the learned-metric table: NCA at
    three budgets, with the one its own cross-validation prefers in bold
    (tab_learned_budget), and gradient boosting on the augmented input with
    its learning rate selected again (tab_learned_gb)."""
    keys = list(load("learned.json"))
    gr = load("gb_retune.json")["results"]

    def small(r):
        return "{\\scriptsize " + ci(*r["ci95"], 3) + "}"
    rows, prev = [], None
    for key in keys:
        bench, rep = key.split("|")
        b, sel = nca_budgets(key)
        if prev is not None and bench != prev:
            rows.append("\\addlinespace")

        def cell(name, j):
            if name not in b:
                return "--"
            v = f"{b[name][j]:.3f}"
            return "\\textbf{" + v + "}" if name == sel else v
        rows.append(f"{BENCH_NAME[bench] if bench != prev else ''} & {rep} & "
                    + " & ".join(cell(n, 0) for n in ("first", "larger", "full")) + " & "
                    + " & ".join(cell(n, 1) for n in ("first", "larger", "full")) + " & "
                    + ("--" if sel == "first" else sg(b[sel][3]["point"], 3)) + " & "
                    + sg(b[sel][2]["point"], 3) + "\\\\")
        rows.append("& & & & & & & & " + ("" if sel == "first" else small(b[sel][3])) + " & "
                    + small(b[sel][2]) + "\\\\")
        prev = bench
    write("tab_learned_budget", "\\begin{tabular}{llrrrrrrcc}\n\\toprule\n"
          "& & \\multicolumn{3}{c}{cross-validated macro F1} & \\multicolumn{3}{c}{test macro F1} & "
          "selected & proposed\\\\\n"
          "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
          "& & first & larger & full rank & first & larger & full rank & $-$ first & $-$ selected\\\\\n"
          "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    rows, prev = [], None
    for key in keys:
        bench, rep = key.split("|")
        g = gr[key]
        if prev is not None and bench != prev:
            rows.append("\\addlinespace")
        rows.append(f"{BENCH_NAME[bench] if bench != prev else ''} & {rep} & "
                    f"{g['lr_raw']:g} & {g['lr_augmented']:g} & {g['gb_raw']['F1']:.3f} & "
                    f"{g['gb_augmented_at_raw_lr_F1']:.3f} & {g['gb_augmented_retuned']['F1']:.3f} & "
                    f"{sg(g['retuned-raw']['point'], 3)} {small(g['retuned-raw'])}\\\\")
        prev = bench
    write("tab_learned_gb", "\\begin{tabular}{llrrrrrc}\n\\toprule\n"
          "& & \\multicolumn{2}{c}{learning rate} & \\multicolumn{3}{c}{test macro F1} & \\\\\n"
          "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\n"
          "& & raw & augm. & raw & augm., rate of raw & augm., own rate & "
          "last $-$ raw\\\\\n"
          "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# ------------------------------- tab_benchmarks_full (supplement)
FULL_RULES = [("plain $k$-NN", "knn"), ("direct", "direct"), ("proposed", "proposed"),
              ("N-KNN", "nknn"), ("NS-$k$-NN", "nsknn"), ("nearest centroid$^{\\ddagger}$", "centroid"),
              ("logistic regression$^{\\dagger}$", "Logistic regression (L2)"),
              ("linear SVM$^{\\dagger}$", "Linear SVM"), ("RBF SVM$^{\\dagger}$", "RBF SVM"),
              ("random forest$^{\\dagger}$", "Random forest"),
              ("gradient boosting$^{\\dagger}$", "Gradient boosting")]


def t_benchmarks_full():
    """Supplement: every rule on the two further benchmarks, F1 / MCC, with
    the cross-validated F1 at selection."""
    b = load("benchmarks.json")["benchmarks"]
    cols = [(bench, rep) for bench in ("SNARE", "VES") for rep in b[bench]]
    rows = []
    for lab, key in FULL_RULES:
        cells = []
        for bb, rr in cols:
            r = b[bb][rr]["rules"][key]
            mcc = f"${r['MCC']:.3f}$" if r["MCC"] < 0 else f"{r['MCC']:.3f}"
            cells.append(f"{r['F1']:.3f} / {mcc}")
        rows.append(f"{lab} & " + " & ".join(cells) + "\\\\")
    # additions outside the plan: the plain rule on min-max inputs (exp15)
    par = load("parity.json")["scaling"]

    def fm(r):
        mcc = f"${r['MCC']:.3f}$" if r["MCC"] < 0 else f"{r['MCC']:.3f}"
        return f"{r['F1']:.3f} / {mcc}"
    rows.insert(1, "plain $k$-NN, min-max$^{\\S}$ & "
                + " & ".join(fm(par[f"{bb}|{rr}"]["minmax"]) for bb, rr in cols) + "\\\\")
    # paired-bootstrap differences in test macro F1, proposed minus ...
    rows.append("\\midrule")
    rows.append(f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\emph{{difference in test macro F1, "
                "proposed minus row, with its $95\\%$ interval}}\\\\")

    def two(label, get):
        rows.append(label + " & " + " & ".join(sg(get(bb, rr)["point"], 3) for bb, rr in cols) + "\\\\")
        rows.append(" & " + " & ".join("{\\scriptsize " + ci(*get(bb, rr)["ci95"], 3) + "}"
                                       for bb, rr in cols) + "\\\\")
    two("N-KNN", lambda bb, rr: b[bb][rr]["compare"]["proposed-nknn"])
    two("direct", lambda bb, rr: b[bb][rr]["compare"]["proposed-direct"])
    two("plain, remote", lambda bb, rr: b[bb][rr]["strata_30"]["remote"]["proposed-knn"])
    two("plain, homologous", lambda bb, rr: b[bb][rr]["strata_30"]["homologous"]["proposed-knn"])
    two("plain, min-max$^{\\S}$", lambda bb, rr: par[f"{bb}|{rr}"]["compare"]["proposed-minmax"])
    two("plain, scaling by CV$^{\\S}$", lambda bb, rr: par[f"{bb}|{rr}"]["compare"]["proposed-selected"])
    head = (" & \\multicolumn{3}{c}{SNARE} & \\multicolumn{2}{c}{vesicular transport}\\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\n"
            "& " + " & ".join(rr for _, rr in cols) + "\\\\\n\\midrule\n")
    write("tab_benchmarks_full", "\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\n" + head
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    t_homology(); t_homology_sens(); t_benchmarks(); t_benchmarks_full()
    t_groupcv(); t_groupcv_full(); t_learned(); t_learned_budget()
    t_data(); t_defects(); t_weights(); t_stats(); t_regime()
    t_runtime(); t_confusion(); t_symmetry(); t_eps(); t_multiclass()
    for fn in (t_bootstrap, t_crossfit, t_pmlb, t_regime_main):
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__} deferred: {e}")
    try:
        t_main()
    except Exception as e:
        print("tab_main deferred (baselines.json incomplete):", e)
