"""Classify tumours as malignant/benign, with the threshold tuned for recall."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (RocCurveDisplay, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import load

RESULTS = Path(__file__).resolve().parent.parent / "results"
TARGET_RECALL = 0.98


def main():
    RESULTS.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test = load()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=0),
        "svc_rbf": make_pipeline(StandardScaler(), SVC(probability=True, random_state=0)),
    }

    rows = []
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, model in models.items():
        cv_auc = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc").mean()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = proba >= 0.5
        rows.append((name, cv_auc, roc_auc_score(y_test, proba),
                     precision_score(y_test, pred), recall_score(y_test, pred)))
        RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=ax)
        print(f"{name:14s} cv_auc={cv_auc:.3f}  test_auc={rows[-1][2]:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_title("ROC on held-out test set")
    fig.tight_layout()
    fig.savefig(RESULTS / "roc.png", dpi=140)

    # choose the model by CV, then pick the highest threshold that still catches
    # TARGET_RECALL of malignant cases in the training folds
    best = max(rows, key=lambda r: r[1])[0]
    model = models[best]
    cv_proba = cross_val_predict(model, X_train, y_train, cv=skf, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, cv_proba)
    ok = np.where(rec[:-1] >= TARGET_RECALL)[0]
    threshold = float(thr[ok[-1]]) if len(ok) else 0.5

    proba = model.predict_proba(X_test)[:, 1]
    pred = proba >= threshold

    lines = ["Malignant is the positive class.", "",
             "| model | cv auc | test auc | precision @0.5 | recall @0.5 |", "|---|---|---|---|---|"]
    for name, cv_auc, auc, p, r in rows:
        lines.append(f"| {name} | {cv_auc:.3f} | {auc:.3f} | {p:.3f} | {r:.3f} |")
    lines += ["",
              f"Best by cross-validation: **{best}**. Threshold lowered to {threshold:.2f} "
              f"to aim for {TARGET_RECALL:.0%} recall on malignant cases: "
              f"test precision {precision_score(y_test, pred):.3f}, "
              f"test recall {recall_score(y_test, pred):.3f}."]
    (RESULTS / "metrics.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
