"""Train a small deterioration model on the PhysioNet/CinC 2019 sepsis challenge data (open, CC-BY 4.0).

    python model/train.py <dir with p*.psv files> [out.json]

Features are only what Vigil measures on a phone plus age and time since arrival:
    HR, RR, age, hours since arrival, HR change over the last 3 h, RR change over the last 3 h,
    HR rise since arrival (current minus first reading), RR rise since arrival.
Label: SepsisLabel, which the challenge already shifts 6 h early, so the model predicts sepsis onset
within the next 6 hours. Hourly ICU rows, forward-filled.

Prints AUROC / AUPRC for logistic regression, gradient boosting and the NEWS2 heart-rate + breathing
baseline on a patient-level 80/20 split, and writes the logistic model as plain JSON so the server can
score without scikit-learn.
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vitals import news_points_hr, news_points_rr  # noqa: E402

FEATS = ["hr", "rr", "age", "hours", "d_hr_3h", "d_rr_3h", "hr_rise", "rr_rise"]


def load(path):
    with open(path) as f:
        cols = f.readline().strip().split("|")
        rows = np.array([[float(x) if x != "NaN" else np.nan for x in line.strip().split("|")] for line in f])
    if len(rows) == 0:
        return None
    c = {k: i for i, k in enumerate(cols)}
    hr, rr = rows[:, c["HR"]], rows[:, c["Resp"]]
    # forward fill, then drop leading rows with nothing
    for x in (hr, rr):
        last = np.nan
        for i in range(len(x)):
            if np.isnan(x[i]):
                x[i] = last
            else:
                last = x[i]
    ok = ~np.isnan(hr) & ~np.isnan(rr)
    if ok.sum() < 4:
        return None
    hr, rr = hr[ok], rr[ok]
    age = rows[ok, c["Age"]]
    hours = rows[ok, c["ICULOS"]]
    y = rows[ok, c["SepsisLabel"]]
    n = len(hr)
    d_hr = np.array([hr[i] - hr[max(0, i - 3)] for i in range(n)])
    d_rr = np.array([rr[i] - rr[max(0, i - 3)] for i in range(n)])
    X = np.column_stack([hr, rr, age, hours, d_hr, d_rr, hr - hr[0], rr - rr[0]])
    return X, y


def news_baseline(X):
    return np.array([news_points_hr(h) + news_points_rr(r) for h, r in zip(X[:, 0], X[:, 1])], dtype=float)


def main():
    d = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "deterioration.json")
    files = sorted(glob.glob(os.path.join(d, "p*.psv")))
    pts = [r for r in (load(f) for f in files) if r is not None]
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(pts))
    cut = int(0.8 * len(pts))
    tr, te = [pts[i] for i in idx[:cut]], [pts[i] for i in idx[cut:]]
    Xtr, ytr = np.vstack([p[0] for p in tr]), np.concatenate([p[1] for p in tr])
    Xte, yte = np.vstack([p[0] for p in te]), np.concatenate([p[1] for p in te])
    sep = sum(int(p[1].max() > 0) for p in pts)
    print(f"patients {len(pts)} (train {len(tr)}, test {len(te)}), hourly rows {len(ytr) + len(yte)}, "
          f"sepsis patients {sep} ({100 * sep / len(pts):.1f}%), positive rows {100 * ytr.mean():.1f}%")

    def report(name, score):
        print(f"{name:28s} AUROC {roc_auc_score(yte, score):.3f}   AUPRC {average_precision_score(yte, score):.3f}")

    report("NEWS2 HR+RR points", news_baseline(Xte))
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(sc.transform(Xtr), ytr)
    report("logistic (8 features)", lr.predict_proba(sc.transform(Xte))[:, 1])
    for name, cols in [("logistic (HR, RR only)", [0, 1]), ("logistic (HR, RR, trends)", [0, 1, 4, 5, 6, 7]),
                       ("logistic (no hours)", [0, 1, 2, 4, 5, 6, 7])]:
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(sc.transform(Xtr)[:, cols], ytr)
        report(name, m.predict_proba(sc.transform(Xte)[:, cols])[:, 1])
    for name, cols in [("gradient boosting (8 feat)", list(range(8))), ("gradient boosting (no hours)", [0, 1, 2, 4, 5, 6, 7])]:
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(Xtr[:, cols], ytr)
        report(name, gb.predict_proba(Xte[:, cols])[:, 1])
    # Save the honest model: no "hours since ICU admission", which is an ICU artefact that does not exist
    # in a waiting room and carries most of the 8-feature gain.
    keep = [0, 1, 2, 4, 5, 6, 7]
    m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(sc.transform(Xtr)[:, keep], ytr)
    auroc = float(roc_auc_score(yte, m.predict_proba(sc.transform(Xte)[:, keep])[:, 1]))
    print("saved model coefficients (standardized):", dict(zip([FEATS[i] for i in keep], np.round(m.coef_[0], 3))))
    logit_tr = m.decision_function(sc.transform(Xtr)[:, keep])
    json.dump({"features": [FEATS[i] for i in keep], "mean": sc.mean_[keep].tolist(), "scale": sc.scale_[keep].tolist(),
               "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
               "logit_percentiles": np.percentile(logit_tr, np.arange(101)).tolist(),   # for "higher than X% of training hours"
               "trained_on": f"PhysioNet/CinC 2019 training set A, {len(pts)} patients, {len(ytr) + len(yte)} hourly ICU rows",
               "label": "sepsis onset within 6 h (Sepsis-3)", "test_auroc": auroc,
               "baseline_news2_hr_rr_auroc": float(roc_auc_score(yte, news_baseline(Xte)))}, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
