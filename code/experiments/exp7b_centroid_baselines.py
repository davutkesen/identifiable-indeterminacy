"""Two baselines that ask where the gain comes from.

Does the success of the proposed construction come from the neutrosophic
framing, or simply from soft class assignment by distance to the class
centroids?  Two rules with no neighbour search at all
answer that directly:

  nearest centroid   assign the class whose centroid is nearest in the
                     standardised input space (sklearn NearestCentroid).

  share sum          compute the per-feature class shares s_{c,j}(x) of
                     equation (10) and assign the class maximising
                     sum_j s_{c,j}(x), i.e. the proposed construction's own
                     soft assignment, aggregated over features, with no
                     neighbours and no indeterminacy block.

If either matches the proposed nearest-neighbour rule, the neighbour search
and the indeterminacy block add nothing.  Both are evaluated on the
predefined test split; the share-sum rule has no hyper-parameter and the
nearest-centroid rule has none either.  The AUC of the nearest-centroid rule
is computed from the score |x - m_0| - |x - m_1|, the margin by which the
positive centroid is nearer.
"""
import json

import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

from common import REPS, blocks_v2, load_rep, scores


def main():
    out = {}
    for rep in REPS:
        Xtr, ytr = load_rep(rep, "train")
        Xte, yte = load_rep(rep, "test")
        out[rep] = {}
        sc = StandardScaler().fit(Xtr)
        A, B = sc.transform(Xtr), sc.transform(Xte)
        nc = NearestCentroid().fit(A, ytr)
        yp = nc.predict(B)
        # ranking score for the AUC: how much nearer the positive centroid is
        d0 = np.sqrt(((B - nc.centroids_[0]) ** 2).sum(1))
        d1 = np.sqrt(((B - nc.centroids_[1]) ** 2).sum(1))
        out[rep]["nearest centroid"] = scores(yte, yp, d0 - d1)
        # share sum: blocks are [I, M_0, M_1]; shares s_c = M_c / (1 - I)
        Btr, Bte = blocks_v2(Xtr, ytr, [Xtr, Xte])
        I, M0, M1 = Bte
        s1 = np.where(1 - I > 1e-12, M1 / np.maximum(1 - I, 1e-12), 0.5)
        s0 = np.where(1 - I > 1e-12, M0 / np.maximum(1 - I, 1e-12), 0.5)
        score1, score0 = s1.sum(1), s0.sum(1)
        yp = (score1 > score0).astype(int)
        out[rep]["share sum"] = scores(yte, yp, score1 / (score0 + score1))
        for k, v in out[rep].items():
            print(f"  {rep:5s} {k:18s} ACC={v['ACC']:.4f} F1={v['F1']:.4f} "
                  f"MCC={v['MCC']:.4f}", flush=True)
    json.dump(out, open("centroid_baselines.json", "w"), indent=1)
    print("written centroid_baselines.json")


if __name__ == "__main__":
    main()
