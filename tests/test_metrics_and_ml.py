import numpy as np
import pandas as pd
import pytest

from niftyquant import features as ft
from niftyquant import metrics as mx
from niftyquant import ml


def test_cagr_and_drawdown():
    idx = pd.bdate_range("2020-01-01", periods=252)
    r = pd.Series(0.0, idx)
    r.iloc[10] = -0.5
    r.iloc[20] = 1.0
    assert mx.cagr(r) == pytest.approx(0.0, abs=1e-12)
    assert mx.max_drawdown(r) == pytest.approx(-0.5)


def test_capm_recovers_beta_and_alpha():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=2000)
    m = pd.Series(rng.normal(0.0004, 0.01, len(idx)), idx)
    r = 0.0002 + 1.5 * m + pd.Series(rng.normal(0, 0.001, len(idx)), idx)
    alpha, t, beta = mx.capm(r, m)
    assert beta == pytest.approx(1.5, abs=0.02)
    assert alpha == pytest.approx(0.0002 * 252, abs=0.01)
    assert t > 3


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0006, 0.01, 2500), pd.bdate_range("2010-01-01", periods=2500))
    lo, hi = mx.block_bootstrap_sharpe(r, n=500)
    assert lo < mx.sharpe(r) < hi


def test_rsi_bounds():
    px = pd.DataFrame({"a": 100 + np.cumsum(np.random.default_rng(3).normal(0, 1, 300))})
    r = ft.rsi(px).dropna()
    assert ((r >= 0) & (r <= 100)).all().all()


def test_walk_forward_purges_training_window():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2015-01-01", "2019-12-31")
    cols = [f"S{i}" for i in range(15)]
    px = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.02, (len(idx), 15)), axis=0), idx, cols)
    vol = pd.DataFrame(rng.integers(1e5, 1e6, (len(idx), 15)), idx, cols).astype(float)
    data = ml.build_dataset(px, vol, horizon=21, min_history=252)
    preds, scores = ml.walk_forward(data, first_test_year=2018, horizon=21)
    assert set(scores["Year"]) == {2018, 2019}
    # random-walk prices: no model should look skilful out of sample
    assert scores["AUC"].between(0.42, 0.58).all()
    assert preds.index.get_level_values(0).min() >= pd.Timestamp("2018-01-01")
