"""The four figures of the article, written as the EPS files Fig1 ... Fig4.

The script writes the PostScript of each figure itself and uses no plotting
library.  figprolog.ps, which is copied into every file, holds the two fonts
and the drawing procedures; the script adds the coordinates.  Every
coordinate is a whole number of thousandths of a point and is computed in
integer arithmetic, so the files come out the same, byte for byte, on every
machine.

Every figure is 337 pt wide, the 119 mm that the journal asks for to the
nearest point, and is meant to be printed at that size: all lettering is
8 pt in a Helvetica-compatible sans serif that is embedded in the file, no
line is thinner than 0.5 pt, and the bounding box is the figure size.
Series are told apart by line style or marker shape as well as by colour.

  1  direct truth and proposed support over a two-feature grid, shared scale
  2  the components of both constructions along one feature axis
  3  cross-validated macro F1 against the indeterminacy weight
  4  the regime relation: 133 tasks of sources A to C and 53 PMLB tasks

"Direct" is the construction of Definition 3 of the article (keys "v1" in the
result files), "proposed" the construction of Definition 4 (keys "v2").
Figures 1 and 2 need no data.  Figures 3 and 4 read the stored results in
this folder; their numbers are rounded to six decimals before they are
drawn.

    python make_figures.py             writes ../../figures/
    python make_figures.py <folder>    writes into <folder>

If Ghostscript is installed, a PDF copy of every figure is written as well.
"""
import json
import math
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = HERE
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "figures"))

MU = (3, 7)        # the two class centroids on every feature, in tenths
K = 1000000        # values are carried as whole numbers of 1 / K


# ------------------------------------------------------------------ numbers
def rdiv(a, b):
    """The quotient a / b of two whole numbers, rounded to the nearest one."""
    if b < 0:
        a, b = -a, -b
    return (2 * a + b) // (2 * b)


def distances(u, g):
    """The point u / g of one feature: its distances d0 and d1 to the two
    class centroids and the smaller of the two, m, in units of 1 / g.  The
    direct truth there is (g - m) / g.  In the proposed construction the
    indeterminacy is m / g, and the supports of the two classes are
    (g - m) / g times d1 / (d0 + d1) and d0 / (d0 + d1)."""
    d0 = abs(u - MU[0] * g // 10)
    d1 = abs(u - MU[1] * g // 10)
    return d0, d1, min(d0, d1)


# --------------------------------------------------------------------- text
def up(s):
    return "[0 8000 0 (%s)]" % s          # upright, 8 pt


def it(s):
    return "[1 8000 0 (%s)]" % s          # italic


def sb(s):
    return "[0 5600 -1300 (%s)]" % s      # upright subscript


def isb(s):
    return "[1 5600 -1300 (%s)]" % s      # italic subscript


def th():
    return "[0 4800 0 ( )]"               # thin space


def txt(*segs):
    return "[" + " ".join(segs) + "]"


# ------------------------------------------------------------ building blocks
def header(name, wd, ht, prolog):
    return ["%!PS-Adobe-3.0 EPSF-3.0",
            "%%%%BoundingBox: 0 0 %d %d" % (wd, ht),
            "%%%%Title: %s" % name,
            "%%Creator: make_figures",
            "%%LanguageLevel: 2",
            "%%DocumentData: Clean7Bit",
            "%%EndComments",
            "%%BeginProlog",
            prolog,
            "%%EndProlog",
            "0.001 0.001 scale"]


def save_eps(L, out, name):
    L = L + ["showpage", "%%EOF"]
    with open(os.path.join(out, name + ".eps"), "wb") as f:
        f.write(("\n".join(L) + "\n").encode("ascii"))
    print("wrote", name + ".eps")


def wrap(v):
    """Whole numbers, twelve to a line."""
    out = []
    for i in range(0, len(v), 12):
        out.append(" ".join("%d" % t for t in v[i:i + 12]))
    return out


def polyline(x0, xs, y0, ys):
    out = ["N %d %d M" % (x0 + xs[0], y0 + ys[0])]
    for i in range(1, len(xs)):
        out.append("%d %d L" % (x0 + xs[i], y0 + ys[i]))
    out.append("S")
    return out


def xaxis(x, y, wd, num, den, labels):
    """Ticks below an axis at the fractions num / den of its width."""
    out = []
    for i in range(len(num)):
        X = x + rdiv(num[i] * wd, den)
        out.append("%d %d %d %d SG" % (X, y, X, y - 3000))
        if len(labels) > 0:
            out.append("%d %d 1 %s T" % (X, y - 12300, txt(up(labels[i]))))
    return out


def yaxis(x, y, ht, num, den, labels):
    """Ticks left of an axis at the fractions num / den of its height."""
    out = []
    for i in range(len(num)):
        Y = y + rdiv(num[i] * ht, den)
        out.append("%d %d %d %d SG" % (x, Y, x - 3000, Y))
        if len(labels) > 0:
            out.append("%d %d 2 %s T" % (x - 6500, Y - 2900, txt(up(labels[i]))))
    return out


def feature_value(kind, u, g):
    """The value of one feature at the point u / g, as the fraction
    A / (g B): the direct truth for kind 1, the support of class 1 in the
    proposed construction for kind 2."""
    d0, d1, m = distances(u, g)
    if kind == 1:
        return g - m, 1
    return (g - m) * d0, d0 + d1


def crossing(kind, i, a, b, c, k, S, first):
    """Where a contour cuts the edge between the nodes i and i + 1 of one
    axis, the other coordinate being a node with the value a / (c b): the
    distance from node i in units of 1 / S, found by bisection.  first tells
    whether the mean of the two values is at least k / 10 at node i."""
    h = S // c
    lo = 0
    hi = h
    while hi - lo > 1:
        mid = (lo + hi) // 2
        au, bu = feature_value(kind, i * h + mid, S)
        if (10 * c * au * b + 10 * S * a * bu - 2 * k * c * S * bu * b >= 0) == first:
            lo = mid
        else:
            hi = mid
    if first:
        return lo
    return hi


def contour_lines(kind, c, k, x0, y0, S):
    """The line on which a panel of Figure 1 takes the value k / 10, as line
    segments in a square of side S with the lower left corner (x0, y0).

    The value at (x1, x2) is the mean of the values of the two features.  It
    is compared with the level at the (c + 1) x (c + 1) nodes of a grid, and
    each cell is cut where the line crosses its edges (marching squares);
    G tells where the value is at least the level."""
    h = S // c
    A = []
    B = []
    for u in range(c + 1):
        a, b = feature_value(kind, u, c)
        A.append(a)
        B.append(b)
    G = []
    for j in range(c + 1):
        G.append([10 * (A[i] * B[j] + A[j] * B[i]) - 2 * k * c * B[i] * B[j] >= 0
                  for i in range(c + 1)])
    out = []
    for j in range(c):
        for i in range(c):
            g00 = G[j][i]
            g10 = G[j][i + 1]
            g01 = G[j + 1][i]
            g11 = G[j + 1][i + 1]
            px = []
            py = []
            if g00 != g10:                               # lower edge
                px.append(x0 + i * h + crossing(kind, i, A[j], B[j], c, k, S, g00))
                py.append(y0 + j * h)
            if g10 != g11:                               # right edge
                px.append(x0 + (i + 1) * h)
                py.append(y0 + j * h + crossing(kind, j, A[i + 1], B[i + 1], c, k, S, g10))
            if g01 != g11:                               # upper edge
                px.append(x0 + i * h + crossing(kind, i, A[j + 1], B[j + 1], c, k, S, g01))
                py.append(y0 + (j + 1) * h)
            if g00 != g01:                               # left edge
                px.append(x0 + i * h)
                py.append(y0 + j * h + crossing(kind, j, A[i], B[i], c, k, S, g00))
            if len(px) == 2:
                pairs = [0, 1]
            elif len(px) == 4:
                # all four edges are cut: the value at the centre of the
                # cell decides which corners belong together
                ctr = 10 * (A[i] * B[i + 1] * B[j] * B[j + 1] + A[i + 1] * B[i] * B[j] * B[j + 1]
                            + A[j] * B[i] * B[i + 1] * B[j + 1] + A[j + 1] * B[i] * B[i + 1] * B[j]) \
                    - 4 * k * c * B[i] * B[i + 1] * B[j] * B[j + 1]
                if g00 == (ctr >= 0):
                    pairs = [0, 1, 2, 3]
                else:
                    pairs = [0, 3, 1, 2]
            else:
                pairs = []
            for t in range(0, len(pairs), 2):
                a = pairs[t]
                b = pairs[t + 1]
                if px[a] != px[b] or py[a] != py[b]:
                    out.append("%d %d %d %d SG" % (px[a], py[a], px[b], py[b]))
    return out


# ------------------------------------------------------------------ figure 1
def fig1(prolog, out):
    L = header("Fig1", 337, 171, prolog)
    S = 120960                       # side of a panel
    X0 = [36000, 169200]             # left edges of the two panels
    Y0 = 29520                       # their lower edge
    n = 500                          # the image of a panel has n x n pixels
    c = 120                          # its contours are traced on c x c cells
    ttl = [txt(up("\\(a\\) direct truth "), it("T")),
           txt(up("\\(b\\) proposed support "), it("M"), sb("1"))]
    lab = ["0", "0.25", "0.5", "0.75", "1"]
    for p in range(2):
        # The value at (x1, x2) is the mean over the two features of a
        # per-feature value: the direct truth in (a), the support of class 1
        # in (b).  The image takes it at the centres of the pixels.
        prof = []
        for i in range(n):
            a, b = feature_value(p + 1, 2 * i + 1, 2 * n)
            prof.append(rdiv(a * K, 2 * n * b))
        L.append("%d %d %d %d %d {" % (X0[p], Y0, S, S, n))
        L += wrap(prof)
        L.append("} %d HM" % K)
        # contours at 0.1, 0.2, ..., 0.9
        L.append("255 255 255 rgb 500 setlinewidth [] 0 setdash 1 setlinecap 1 setlinejoin")
        for k in range(1, 10):
            L += contour_lines(p + 1, c, k, X0[p], Y0, S)
        # the centroids, and in (a) the two mixed points
        a0 = rdiv(MU[0] * S, 10)
        a1 = rdiv(MU[1] * S, 10)
        L.append("1300 setlinewidth 0 setlinecap 0 setlinejoin")
        L.append("%d %d 3000 MOo" % (X0[p] + a0, Y0 + a0))
        L.append("%d %d 3000 MSo" % (X0[p] + a1, Y0 + a1))
        if p == 0:
            L.append("232 38 42 rgb 1600 setlinewidth")
            L.append("%d %d 3500 MX" % (X0[p] + a0, Y0 + a1))
            L.append("%d %d 3500 MX" % (X0[p] + a1, Y0 + a0))
        L.append("0 0 0 rgb 600 setlinewidth")
        L.append("%d %d %d %d RS" % (X0[p], Y0, S, S))
        L += xaxis(X0[p], Y0, S, [0, 1, 2, 3, 4], 4, lab)
        if p == 0:
            L += yaxis(X0[p], Y0, S, [0, 1, 2, 3, 4], 4, lab)
        else:
            L += yaxis(X0[p], Y0, S, [0, 1, 2, 3, 4], 4, [])
        L.append("%d %d 1 %s T" % (X0[p] + S // 2, Y0 + S + 4200, ttl[p]))
        L.append("%d %d 1 %s T" % (X0[p] + S // 2, Y0 - 22000, txt(it("x"), sb("1"))))
    L.append("%d %d 90 1 %s TR" % (8700, Y0 + S // 2, txt(it("x"), sb("2"))))
    # the colour scale
    XB = 296640
    WB = 6120
    lab = ["0", "0.2", "0.4", "0.6", "0.8", "1"]
    L.append("%d %d %d %d CB" % (XB, Y0, WB, S))
    L.append("%d %d %d %d RS" % (XB, Y0, WB, S))
    for i in range(6):
        Y = Y0 + rdiv(i * S, 5)
        L.append("%d %d %d %d SG" % (XB + WB, Y, XB + WB + 3000, Y))
        L.append("%d %d 0 %s T" % (XB + WB + 6500, Y - 2900, txt(up(lab[i]))))
    L.append("%d %d 90 1 %s TR" % (328700, Y0 + S // 2, txt(up("membership value"))))
    save_eps(L, out, "Fig1")


# ------------------------------------------------------------------ figure 2
def fig2(prolog, out):
    L = header("Fig2", 337, 166, prolog)
    PW = 142560                      # width and height of a panel
    PH = 117360
    X0 = [36000, 190080]
    Y0 = 28800
    g = 500                          # the curves are drawn at x = u / g
    # A value v sits at the fraction (v + 0.03) / 1.53 of the height.
    xs = []
    yT = []
    yF = []
    yI = []
    yM0 = []
    yM1 = []
    for u in range(g + 1):
        d0, d1, m = distances(u, g)
        xs.append(rdiv(u * PW, g))
        yT.append(rdiv((100 * (g - m) + 3 * g) * PH, 153 * g))
        yF.append(rdiv((max(0, 100 * m - 22 * g) + 3 * g) * PH, 153 * g))
        yI.append(rdiv((100 * m + 3 * g) * PH, 153 * g))
        yM0.append(rdiv((100 * (g - m) * d1 + 3 * g * (d0 + d1)) * PH, 153 * g * (d0 + d1)))
        yM1.append(rdiv((100 * (g - m) * d0 + 3 * g * (d0 + d1)) * PH, 153 * g * (d0 + d1)))
    yc = rdiv((22 + 3) * PH, 153)            # the constant I = 0.22 of the direct construction
    y1 = rdiv((100 + 3) * PH, 153)           # the sum of the proposed components, 1 everywhere
    sT = "196 78 82 rgb 1500 setlinewidth [] 0 setdash"
    sI = "129 114 179 rgb 1500 setlinewidth [5550 2400] 0 setdash"
    sF = "147 120 96 rgb 1500 setlinewidth [1500 2475] 0 setdash"
    sM0 = "221 132 82 rgb 1500 setlinewidth [] 0 setdash"
    sM1 = "76 114 176 rgb 1500 setlinewidth [9600 2400 1500 2400] 0 setdash"
    s1 = "0 0 0 rgb 900 setlinewidth [900 1485] 0 setdash"
    ttl = [txt(up("\\(a\\) direct")), txt(up("\\(b\\) proposed"))]
    lab = ["0", "0.2", "0.4", "0.6", "0.8", "1"]
    for p in range(2):
        x = X0[p]
        top = Y0 + PH
        L.append("gsave %d %d %d %d rectclip" % (x, Y0, PW, PH))
        L.append("179 179 179 rgb 600 setlinewidth [] 0 setdash 0 setlinecap 1 setlinejoin")
        for t in range(2):
            X = x + rdiv(MU[t] * PW, 10)
            L.append("%d %d %d %d SG" % (X, Y0, X, top))
        if p == 0:
            L.append(sT)
            L += polyline(x, xs, Y0, yT)
            L.append(sI)
            L.append("%d %d %d %d SG" % (x, Y0 + yc, x + PW, Y0 + yc))
            L.append(sF)
            L += polyline(x, xs, Y0, yF)
        else:
            L.append(sI)
            L += polyline(x, xs, Y0, yI)
            L.append(sM0)
            L += polyline(x, xs, Y0, yM0)
            L.append(sM1)
            L += polyline(x, xs, Y0, yM1)
            L.append(s1)
            L.append("%d %d %d %d SG" % (x, Y0 + y1, x + PW, Y0 + y1))
        L.append("grestore")
        # legend: two columns of line samples
        r1 = top - 11100
        r2 = top - 21500
        if p == 0:
            c2 = x + 93280
            L.append(sT)
            L.append("%d %d %d %d SG" % (x + 22680, r1 + 2800, x + 37880, r1 + 2800))
            L.append(sI)
            L.append("%d %d %d %d SG" % (x + 22680, r2 + 2800, x + 37880, r2 + 2800))
            L.append(sF)
            L.append("%d %d %d %d SG" % (c2, r1 + 2800, c2 + 15200, r1 + 2800))
            L.append("0 0 0 rgb")
            L.append("%d %d 0 %s T" % (x + 44280, r1, txt(it("T"))))
            L.append("%d %d 0 %s T" % (x + 44280, r2, txt(it("I"), up(" \\(constant\\)"))))
            L.append("%d %d 0 %s T" % (c2 + 21600, r1, txt(it("F"))))
        else:
            c2 = x + 62280
            L.append(sI)
            L.append("%d %d %d %d SG" % (x + 22680, r1 + 2800, x + 37880, r1 + 2800))
            L.append(sM0)
            L.append("%d %d %d %d SG" % (x + 22680, r2 + 2800, x + 37880, r2 + 2800))
            L.append(sM1)
            L.append("%d %d %d %d SG" % (c2, r1 + 2800, c2 + 15200, r1 + 2800))
            L.append(s1)
            L.append("%d %d %d %d SG" % (c2, r2 + 2800, c2 + 15200, r2 + 2800))
            L.append("0 0 0 rgb")
            L.append("%d %d 0 %s T" % (x + 44280, r1, txt(it("I"))))
            L.append("%d %d 0 %s T" % (x + 44280, r2, txt(it("M"), sb("0"))))
            L.append("%d %d 0 %s T" % (c2 + 21600, r1, txt(it("M"), sb("1"))))
            L.append("%d %d 0 %s T" % (c2 + 21600, r2,
                                       txt(it("I"), th(), up("+"), th(), up("\\202"), isb("c"), th(),
                                           it("M"), isb("c"), th(), up("\\201"), th(), up("1"))))
        L.append("0 0 0 rgb 600 setlinewidth [] 0 setdash 0 setlinejoin")
        L.append("%d %d %d %d RS" % (x, Y0, PW, PH))
        L += xaxis(x, Y0, PW, [0, 1, 2, 3, 4, 5], 5, lab)
        if p == 0:
            L += yaxis(x, Y0, PH, [3, 23, 43, 63, 83, 103], 153, lab)
        else:
            L += yaxis(x, Y0, PH, [3, 23, 43, 63, 83, 103], 153, [])
        L.append("%d %d 1 %s T" % (x + PW // 2, top + 4200, ttl[p]))
        L.append("%d %d 1 %s T" % (x + PW // 2, Y0 - 22000, txt(it("x"), isb("j"))))
    L.append("%d %d 90 1 %s TR" % (14200, Y0 + PH // 2, txt(up("component value"))))
    save_eps(L, out, "Fig2")


# ------------------------------------------------------------------ figure 3
def scores():
    """The rows of Figure 3: panel (1 direct, 2 proposed), the selected k, the
    indeterminacy weight and the two other weights in tenths (T and F in the
    direct construction, M0 and M1 in the proposed one), and the
    cross-validated macro F1 on dipeptide composition at the selected k and
    class weighting, in units of 1 / K."""
    with open(os.path.join(RES, "weights_k.json")) as f:
        d = json.load(f)["DPC"]
    rows = []
    for p, key, order in ((1, "v1", (1, 0, 2)), (2, "v2", (0, 1, 2))):
        e = d[key]
        ik = e["ks"].index(e["best_k"])
        ic = int(e["best_cw"])
        for wt, s in zip(e["grid"], e["surface"]):
            t = [int(round(10 * wt[q])) for q in order]
            rows.append([p, e["best_k"], t[0], t[1], t[2], math.floor(s[ik][ic] * K + 0.5)])
    return rows


def fig3(prolog, out, rows):
    L = header("Fig3", 337, 187, prolog)
    PW = 142560
    PH = 138960
    X0 = [36000, 190080]
    Y0 = 28800
    MK = ["MO", "MS", "MT", "MD", "MV"]
    boxc = ["139 47 50 rgb", "47 79 143 rgb"]
    boxx = [37, 35]                  # centres of the two notes, at
    boxy = [41, 52]                  # (0.33, 0.56) and (0.31, 0.67)
    name = [txt(up("share on "), it("T")), txt(up("share on "), it("M"), sb("0"))]
    lab = ["0", "0.2", "0.4", "0.6", "0.8", "1"]
    labv = ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"]
    for p in range(2):
        x = X0[p]
        top = Y0 + PH
        sel = [r for r in rows if r[0] == p + 1]
        # A weight w sits at the fraction (w + 0.04) / 1.04 of the width, a
        # score s at the fraction (s - 0.15) / 0.65 of the height.
        L.append("220 220 220 rgb 500 setlinewidth [] 0 setdash 0 setlinecap")
        for t in range(6):
            X = x + rdiv((4 + 20 * t) * PW, 104)
            L.append("%d %d %d %d SG" % (X, Y0, X, top))
        for t in range(7):
            Y = Y0 + rdiv((5 + 10 * t) * PH, 65)
            L.append("%d %d %d %d SG" % (x, Y, x + PW, Y))
        # One curve for every share of the remaining weight that occurs at
        # three or more grid points; the share is kept in units of 1 / 10000.
        key = []
        for r in sel:
            if r[3] + r[4] > 0:
                key.append(rdiv(10000 * r[3], r[3] + r[4]))
            else:
                key.append(-1)
        shares = []
        for t in sorted(set(key)):
            if t >= 0 and key.count(t) >= 3:
                shares.append(t)
        nq = len(shares)
        flat = 0
        L.append("gsave %d %d %d %d rectclip" % (x, Y0, PW, PH))
        for q in range(nq):
            pts = sorted([r[2], r[5]] for r, t in zip(sel, key) if t == shares[q])
            xs = [rdiv((10 * a[0] + 4) * PW, 104) for a in pts]
            ys = [rdiv((a[1] - 150000) * PH, 650000) for a in pts]
            sc = [a[1] for a in pts]
            flat = max(flat, max(sc) - min(sc))
            L.append("%d vir 1100 setlinewidth [] 0 setdash 0 setlinecap 1 setlinejoin"
                     % (256 * (8 * (nq - 1) + 84 * q) // (100 * (nq - 1))))
            L += polyline(x, xs, Y0, ys)
            for i in range(len(xs)):
                L.append("%d %d 1600 %s" % (x + xs[i], Y0 + ys[i], MK[q % 5]))
        if p == 1:
            # the selected weights: the highest score of the whole grid
            best = sel[0]
            for r in sel:
                if r[5] > best[5]:
                    best = r
            L.append("232 38 42 rgb 1300 setlinewidth 0 setlinejoin")
            L.append("%d %d 5500 MSTAR" % (x + rdiv((10 * best[2] + 4) * PW, 104),
                                           Y0 + rdiv((best[5] - 150000) * PH, 650000)))
        L.append("grestore")
        t = rdiv(flat, 1000)
        L.append("%s 600 setlinewidth [] 0 setdash 0 setlinejoin" % boxc[p])
        L.append("%d %d 9000 2500 [%s %s] BT"
                 % (x + rdiv(boxx[p] * PW, 104), Y0 + rdiv(boxy[p] * PH, 65),
                    txt(up("largest variation")),
                    txt(up("along any curve: %d.%03d" % (t // 1000, t % 1000)))))
        # legend
        for q in range(nq):
            yb = Y0 + 7300 + (nq - 1 - q) * 9110
            h = rdiv(shares[q], 100)
            L.append("%d vir 1100 setlinewidth" % (256 * (8 * (nq - 1) + 84 * q) // (100 * (nq - 1))))
            L.append("%d %d %d %d SG" % (x + PW - 44000, yb + 2800, x + PW - 31200, yb + 2800))
            L.append("%d %d 1600 %s" % (x + PW - 37600, yb + 2800, MK[q % 5]))
            L.append("0 0 0 rgb")
            L.append("%d %d 0 %s T" % (x + PW - 24800, yb, txt(up("%d.%02d" % (h // 100, h % 100)))))
        L.append("%d %d 1 %s T" % (x + PW - 26600, Y0 + 7300 + nq * 9110 + 300, name[p]))
        L.append("0 0 0 rgb 600 setlinewidth")
        L.append("%d %d %d %d RS" % (x, Y0, PW, PH))
        L += xaxis(x, Y0, PW, [4, 24, 44, 64, 84, 104], 104, lab)
        if p == 0:
            L += yaxis(x, Y0, PH, [5, 15, 25, 35, 45, 55, 65], 65, labv)
            L.append("%d %d 1 %s T" % (x + PW // 2, top + 4200,
                                       txt(up("\\(a\\) direct, "), it("k"), th(), up("="), th(),
                                           up("%d" % sel[0][1]))))
        else:
            L += yaxis(x, Y0, PH, [5, 15, 25, 35, 45, 55, 65], 65, [])
            L.append("%d %d 1 %s T" % (x + PW // 2, top + 4200,
                                       txt(up("\\(b\\) proposed, "), it("k"), th(), up("="), th(),
                                           up("%d" % sel[0][1]))))
        L.append("%d %d 1 %s T" % (x + PW // 2, Y0 - 22000,
                                   txt(up("indeterminacy weight "), it("w"), isb("I"))))
    L.append("%d %d 90 1 %s TR" % (14200, Y0 + PH // 2, txt(up("cross-validated macro F1"))))
    save_eps(L, out, "Fig3")


# ------------------------------------------------------------------ figure 4
def tasks():
    """The rows of Figure 4: panel, source (1 to 4 for A to D), and the two
    coordinates x and y of a task in units of 1 / K.  Panel 1 is the fixed
    configuration, k = 5 with uniform weights and coupled coordinates; panel
    2 has both rules tuned and the coordinates cross-fitted, and adds the 53
    tasks of source D, which played no part in finding the relation."""
    def q(v):
        return math.floor(v * K + 0.5)
    code = {"A": 1, "B": 2, "C": 3}
    rows = []
    with open(os.path.join(RES, "regime.json")) as f:
        for t in json.load(f)["points"]:
            rows.append([1, code[t["source"][0]], q(t["x"]), q(t["y"])])
    with open(os.path.join(RES, "regime_crossfit.json")) as f:
        for t in json.load(f)["points"]:
            rows.append([2, code[t["source"][0]], q(t["crossfit_tuned"]["x"]), q(t["crossfit_tuned"]["y"])])
    with open(os.path.join(RES, "regime_pmlb.json")) as f:
        for t in json.load(f)["points"]:
            rows.append([2, 4, q(t["crossfit_tuned"]["x"]), q(t["crossfit_tuned"]["y"])])
    return rows


def fig4(prolog, out, rows):
    L = header("Fig4", 337, 318, prolog)
    AW = 285360                      # width and height of a panel
    PH = 112320
    X0 = 44640
    YB = [187200, 28800]             # lower edges of panels (a) and (b)
    w = 25                           # tasks in the moving average
    MK = ["MO", "MS", "MT", "MD"]
    sz = [1730, 1730, 1730, 1580]
    colour = ["76 114 176 rgb", "221 132 82 rgb", "85 168 104 rgb", "129 114 179 rgb"]
    lab = ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"]
    labv = ["\\2000.4", "\\2000.3", "\\2000.2", "\\2000.1", "0", "0.1", "0.2"]
    for p in range(2):
        Y0 = YB[p]
        top = Y0 + PH
        sel = [r for r in rows if r[0] == p + 1]
        # A value x sits at the fraction (x + 0.03) / 0.73 of the width, a
        # value y at the fraction (y + 0.50) / 0.76 of the height.
        X1 = X0 + rdiv(13 * AW, 73)              # x = 0.10
        X3 = X0 + rdiv(33 * AW, 73)              # x = 0.30
        L.append("237 242 249 rgb %d %d %d %d RF" % (X0, Y0, X1 - X0, PH))
        L.append("251 238 238 rgb %d %d %d %d RF" % (X3, Y0, X0 + AW - X3, PH))
        L.append("220 220 220 rgb 500 setlinewidth [] 0 setdash 0 setlinecap")
        for t in range(8):
            X = X0 + rdiv((3 + 10 * t) * AW, 73)
            L.append("%d %d %d %d SG" % (X, Y0, X, top))
        for t in range(7):
            Y = Y0 + rdiv((10 + 10 * t) * PH, 76)
            L.append("%d %d %d %d SG" % (X0, Y, X0 + AW, Y))
        L.append("140 140 140 rgb 600 setlinewidth [600 1200] 0 setdash")
        L.append("%d %d %d %d SG" % (X1, Y0, X1, top))
        L.append("%d %d %d %d SG" % (X3, Y0, X3, top))
        Y = Y0 + rdiv(50 * PH, 76)
        L.append("102 102 102 rgb 800 setlinewidth [] 0 setdash")
        L.append("%d %d %d %d SG" % (X0, Y, X0 + AW, Y))
        L.append("gsave %d %d %d %d rectclip" % (X0, Y0, AW, PH))
        for s in range(4):
            L.append(colour[s])
            for r in sel:
                if r[1] == s + 1:
                    L.append("%d %d %d %s" % (X0 + rdiv((r[2] + 30000) * AW, 730000),
                                              Y0 + rdiv((r[3] + 500000) * PH, 760000),
                                              sz[s], MK[s]))
        # moving average of y over w tasks of sources A to C in the order of x
        xa = [r[2] for r in sel if r[1] <= 3]
        ya = [r[3] for r in sel if r[1] <= 3]
        o = sorted(range(len(xa)), key=lambda i: xa[i])
        mx = []
        my = []
        for i in range(len(xa) - w + 1):
            tot = 0
            for t in range(w):
                tot = tot + ya[o[i + t]]
            mx.append(rdiv((xa[o[i + w // 2]] + 30000) * AW, 730000))
            my.append(rdiv((tot + 500000 * w) * PH, 760000 * w))
        L.append("0 0 0 rgb 1400 setlinewidth [] 0 setdash 0 setlinecap 1 setlinejoin")
        L += polyline(X0, mx, Y0, my)
        L.append("grestore")
        L.append("47 79 143 rgb")
        L.append("%d %d 0 %s T" % (X0 + rdiv(8 * AW, 730), Y0 + rdiv(75 * PH, 76) - 5800,
                                   txt(up("transformation helps"))))
        L.append("139 47 50 rgb")
        L.append("%d %d 2 %s T" % (X0 + rdiv(722 * AW, 730), Y0 + rdiv(PH, 76) + 1700,
                                   txt(up("transformation hurts"))))
        # legend: (a) names the symbols of sources A to C, (b) adds source D
        yb = Y0 + 7700
        if p == 0:
            for s in range(3):
                L.append(colour[s])
                L.append("%d %d %d %s" % (X0 + 12400 + 30950 * s, yb + 2800, sz[s], MK[s]))
            L.append("0 0 0 rgb")
            L.append("%d %d 0 %s T" % (X0 + 22000, yb, txt(up("A"))))
            L.append("%d %d 0 %s T" % (X0 + 52950, yb, txt(up("B"))))
            L.append("%d %d 0 %s T" % (X0 + 83900, yb, txt(up("C"))))
            xl = X0 + 99300
        else:
            L.append(colour[3])
            L.append("%d %d %d %s" % (X0 + 12400, yb + 2800, sz[3], MK[3]))
            L.append("0 0 0 rgb")
            L.append("%d %d 0 %s T" % (X0 + 22000, yb, txt(up("D"))))
            xl = X0 + 37400
        L.append("1400 setlinewidth")
        L.append("%d %d %d %d SG" % (xl, yb + 2800, xl + 12800, yb + 2800))
        L.append("%d %d 0 %s T" % (xl + 16000, yb, txt(up("moving average, A to C"))))
        L.append("600 setlinewidth 0 setlinejoin")
        L.append("%d %d %d %d RS" % (X0, Y0, AW, PH))
        L += xaxis(X0, Y0, AW, [3, 13, 23, 33, 43, 53, 63, 73], 73, lab)
        L += yaxis(X0, Y0, PH, [10, 20, 30, 40, 50, 60, 70], 76, labv)
        if p == 0:
            L.append("%d %d 1 %s T" % (X0 + AW // 2, top + 4200,
                                       txt(up("\\(a\\) "), it("k"), th(), up("="), th(),
                                           up("5, uniform weights"))))
            L.append("%d %d 1 %s T" % (X0 + AW // 2, Y0 - 22000,
                                       txt(it("x"), up(": macro F1 of plain "), it("k"), up("-NN \\("),
                                           it("k"), th(), up("="), th(),
                                           up("5\\) minus that of the majority classifier"))))
        else:
            L.append("%d %d 1 %s T" % (X0 + AW // 2, top + 4200,
                                       txt(up("\\(b\\) both rules tuned, cross-fitted"))))
            L.append("%d %d 1 %s T" % (X0 + AW // 2, Y0 - 22000,
                                       txt(it("x"), up(": macro F1 of plain "), it("k"), up("-NN \\("),
                                           it("k"),
                                           up(" tuned\\) minus that of the majority classifier"))))
        L.append("%d %d 90 1 %s TR" % (8300, Y0 + PH // 2, txt(it("y"), up(": macro F1 gained"))))
        L.append("%d %d 90 1 %s TR" % (17600, Y0 + PH // 2, txt(up("by the transformation"))))
    save_eps(L, out, "Fig4")


# --------------------------------------------------------------------- main
def main(out):
    with open(os.path.join(HERE, "figprolog.ps")) as f:
        prolog = f.read().replace("\r", "")
    while prolog.endswith("\n"):
        prolog = prolog[:-1]
    os.makedirs(out, exist_ok=True)
    fig1(prolog, out)
    fig2(prolog, out)
    fig3(prolog, out, scores())
    fig4(prolog, out, tasks())
    gs = shutil.which("gs")
    if gs:
        for i in range(1, 5):
            subprocess.run([gs, "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-dEPSCrop",
                            "-sDEVICE=pdfwrite", "-dAutoRotatePages=/None",
                            "-sOutputFile=" + os.path.join(out, "Fig%d.pdf" % i),
                            os.path.join(out, "Fig%d.eps" % i)], check=True)
        print("wrote Fig1.pdf ... Fig4.pdf")
    print("all figures written to", out)


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else OUT)
