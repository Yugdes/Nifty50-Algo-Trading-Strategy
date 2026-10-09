"""Walk-forward machine-learning stock ranker.

Task: on each day, predict which stocks will beat the cross-sectional median
over the next `horizon` trading days. Each calendar year is predicted by models
trained only on earlier data, with a purge gap of `horizon` days so no training
label overlaps the test period (Lopez de Prado, 2018, ch. 7).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features as ft
from .strategies import eligible


def build_dataset(prices, volume, horizon=21, min_history=252):
    X = ft.ml_features(prices, volume)
    fwd = prices.shift(-horizon) / prices - 1
    mask = eligible(prices, min_history)
    fwd = fwd.where(mask)
    label = fwd.gt(fwd.median(axis=1), axis=0).astype(float).where(fwd.notna())
    y = pd.concat({"fwd_ret": fwd.stack(), "label": label.stack()}, axis=1)
    X.index.names = y.index.names = ["date", "ticker"]
    data = X.join(y, how="inner")
    keep = mask.stack()
    keep.index.names = ["date", "ticker"]
    data = data[keep.reindex(data.index).fillna(False).to_numpy()]
    data = data.dropna(subset=["ret_21", "mom_12_1", "vol_63"])
    return data


def models(seed=0):
    return {
        "Logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(C=0.1, max_iter=1000)),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=500,
            l2_regularization=1.0, random_state=seed),
    }


def walk_forward(data: pd.DataFrame, first_test_year: int, horizon: int = 21, seed: int = 0):
    """Return (predictions frame, per-year scores frame)."""
    feature_cols = [c for c in data.columns if c not in ("fwd_ret", "label")]
    dates = data.index.get_level_values(0)
    calendar = pd.DatetimeIndex(sorted(dates.unique()))
    preds, scores = [], []
    for year in range(first_test_year, calendar.year.max() + 1):
        test_mask = dates.year == year
        if not test_mask.any():
            continue
        test_start = calendar[calendar.year == year][0]
        purge_cut = calendar[max(0, calendar.get_loc(test_start) - horizon)]
        train = data[(dates < purge_cut)].dropna(subset=["label"])
        test = data[test_mask]
        out = pd.DataFrame(index=test.index)
        for name, model in models(seed).items():
            model.fit(train[feature_cols], train["label"])
            out[name] = model.predict_proba(test[feature_cols])[:, 1]
            scored = test["label"].notna()
            scores.append({
                "Year": year, "Model": name, "Train rows": len(train),
                "AUC": roc_auc_score(test.loc[scored, "label"], out.loc[scored, name]),
                "Rank IC": rank_ic(out.loc[scored, name], test.loc[scored, "fwd_ret"]).mean(),
            })
        preds.append(out)
    return pd.concat(preds), pd.DataFrame(scores)


def rank_ic(score: pd.Series, fwd: pd.Series) -> pd.Series:
    """Daily Spearman correlation between a score and forward returns."""
    df = pd.concat({"s": score, "f": fwd}, axis=1).dropna()
    return df.groupby(level=0).apply(lambda g: g["s"].rank().corr(g["f"].rank()) if len(g) > 5 else np.nan)


def feature_ic(data: pd.DataFrame, sample_every: int = 5) -> pd.DataFrame:
    """Mean rank IC and t-stat of each raw feature (non-overlapping-ish sampling)."""
    dates = pd.DatetimeIndex(sorted(data.index.get_level_values(0).unique()))[::sample_every]
    sub = data[data.index.get_level_values(0).isin(dates)]
    rows = []
    for col in [c for c in data.columns if not c.endswith("_rank") and c not in ("fwd_ret", "label")]:
        ic = rank_ic(sub[col], sub["fwd_ret"]).dropna()
        # Newey-West-free approximation: deflate t-stat for 21d label overlap at 5d sampling.
        n_eff = len(ic) * sample_every / 21
        rows.append({"Feature": col, "Mean IC": ic.mean(), "t-stat": ic.mean() / ic.std() * np.sqrt(n_eff)})
    return pd.DataFrame(rows).sort_values("Mean IC", key=abs, ascending=False)


def prediction_targets(preds: pd.Series, index, columns, quantile=0.2):
    """Monthly long-only portfolio of the top `quantile` of stocks by model score."""
    from .strategies import _as_targets, rebalance_dates, select_top
    wide = preds.unstack().reindex(index=index, columns=columns)
    dates = rebalance_dates(wide.dropna(how="all").index)
    w = select_top(wide, wide.notna(), dates, quantile)
    return _as_targets(w, dates, index)
