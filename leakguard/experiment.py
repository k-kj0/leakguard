"""Run the full experiment and write a reproducible report.

    python -m leakguard.experiment --n 60000 --horizon 50 --seed 7
"""
import argparse, hashlib, json, platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from .checks import (check_causality, check_data_integrity,
                     label_overlap_fraction, shuffled_label_auc)
from .cv import naive_kfold, purged_walk_forward
from .features import build_dataset, build_features, build_features_leaky
from .simulate import simulate_order_book


def make_model():
    return HistGradientBoostingClassifier(max_iter=80, max_depth=3, learning_rate=0.1, random_state=0)


def evaluate(X, y, splits):
    aucs, accs = [], []
    for tr, te in splits:
        m = make_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], p))
        accs.append(accuracy_score(y.iloc[te], p > 0.5))
    return {"auc": round(float(np.mean(aucs)), 4), "auc_std": round(float(np.std(aucs)), 4),
            "acc": round(float(np.mean(accs)), 4), "folds": len(aucs)}


def run(n=60_000, horizon=50, seed=7, out_dir="results"):
    df = simulate_order_book(n=n, seed=seed)
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:12]

    r = {"params": {"n": n, "horizon": horizon, "seed": seed},
         "data_hash": data_hash,
         "versions": {"python": platform.python_version(), "numpy": np.__version__,
                      "pandas": pd.__version__, "sklearn": sklearn.__version__}}

    r["integrity_issues"] = check_data_integrity(df)
    r["causality_clean_features"] = check_causality(build_features, df)
    r["causality_leaky_features"] = check_causality(build_features_leaky, df)

    X, y = build_dataset(df, horizon)
    N = len(X)
    naive = naive_kfold(N, 5, seed)
    purged = purged_walk_forward(N, horizon, 5, 0.4, embargo=horizon // 5)

    r["rows_used"] = N
    r["label_overlap_naive"] = round(label_overlap_fraction(*naive[0], horizon, N), 4)
    r["label_overlap_purged"] = round(label_overlap_fraction(*purged[0], horizon, N), 4)
    r["cv_naive_shuffled_kfold"] = evaluate(X, y, naive)
    r["cv_purged_walk_forward"] = evaluate(X, y, purged)
    r["shuffled_label_auc_purged"] = round(shuffled_label_auc(make_model, X, y, purged), 4)

    Xl, yl = build_dataset(df, horizon, build_features_leaky)
    pl = purged_walk_forward(len(Xl), horizon, 5, 0.4, embargo=horizon // 5)
    r["cv_purged_with_leaky_feature"] = evaluate(Xl, yl, pl)

    out = Path(out_dir); out.mkdir(exist_ok=True)
    (out / "latest.json").write_text(json.dumps(r, indent=2))
    (out / "latest.md").write_text(render_report(r))
    print(render_report(r))
    return r


def render_report(r):
    nv, pw, lk = r["cv_naive_shuffled_kfold"], r["cv_purged_walk_forward"], r["cv_purged_with_leaky_feature"]
    return f"""# Experiment report
Params: `{r['params']}` | data hash: `{r['data_hash']}` | rows used: {r['rows_used']}
Versions: {r['versions']}

## Data and leakage checks
- Data integrity issues: {r['integrity_issues'] or 'none'}
- Causality test on honest features, changed by truncation: {r['causality_clean_features'] or 'none (good)'}
- Causality test on deliberately leaky feature set, flagged: {r['causality_leaky_features']}
- Train rows whose label window overlaps test, naive K-fold: {r['label_overlap_naive']:.1%}
- Same, purged walk-forward: {r['label_overlap_purged']:.1%}
- AUC with shuffled labels (should be ~0.5): {r['shuffled_label_auc_purged']}

## Results (AUC, higher = better, 0.5 = coin flip)
| Setup | AUC | Accuracy | Verdict |
|---|---|---|---|
| Shuffled K-fold (WRONG for time series) | {nv['auc']} | {nv['acc']} | inflated by label overlap |
| Purged walk-forward (honest) | {pw['auc']} | {pw['acc']} | the number to trust |
| Purged walk-forward + leaky feature | {lk['auc']} | {lk['acc']} | too good to be true: caught by causality test |
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60_000)
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    run(a.n, a.horizon, a.seed)
