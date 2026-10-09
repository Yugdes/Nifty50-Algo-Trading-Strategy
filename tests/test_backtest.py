import numpy as np
import pandas as pd
import pytest

from niftyquant import backtest as bt
from niftyquant import features as ft
from niftyquant import metrics as mx
from niftyquant import strategies as st


@pytest.fixture
def prices():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2018-01-01", periods=700)
    rets = rng.normal(0.0005, 0.02, size=(len(idx), 12))
    px = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), idx, [f"S{i}" for i in range(12)])
    px.iloc[:400, -1] = np.nan  # a stock that lists later
    return px


def test_buy_and_hold_matches_asset(prices):
    rets = ft.returns(prices)
    target = pd.DataFrame(np.nan, prices.index, prices.columns)
    target.iloc[0] = 0.0
    target.iloc[0, 0] = 1.0
    res = bt.run(target, rets, lag=0)
    expected = (1 + rets.iloc[1:, 0]).prod()
    assert (1 + res.returns).prod() == pytest.approx(expected, rel=1e-10)


def test_costs_equal_turnover_times_rate(prices):
    rets = ft.returns(prices)
    target = st.equal_weight(prices, min_history=50)
    gross = bt.run(target, rets, cost_bps=0)
    net = bt.run(target, rets, cost_bps=20)
    traded = gross.turnover > 0
    implied = 1 - (1 + net.returns[traded]) / (1 + gross.returns[traded])
    assert np.allclose(implied, gross.turnover[traded] * 0.002)


def test_cash_earns_risk_free(prices):
    rets = ft.returns(prices)
    empty = pd.DataFrame(0.0, prices.index, prices.columns)
    res = bt.run(empty, rets, risk_free=0.05)
    assert mx.cagr(res.returns) == pytest.approx(0.05, rel=1e-6)


@pytest.mark.parametrize("name", list(st.STRATEGIES))
def test_strategies_have_no_lookahead(prices, name):
    """Changing prices after date T must not change any decision made up to T."""
    bench = prices.mean(axis=1)
    cut = prices.index[550]
    base = st.STRATEGIES[name](prices, benchmark=bench, min_history=100)
    shocked = prices.copy()
    shocked.loc[shocked.index > cut] *= np.linspace(0.5, 2.0, shocked.shape[1])
    shocked_bench = bench.copy()
    shocked_bench[shocked_bench.index > cut] *= 0.3
    alt = st.STRATEGIES[name](shocked, benchmark=shocked_bench, min_history=100)
    pd.testing.assert_frame_equal(base.loc[:cut], alt.loc[:cut])


@pytest.mark.parametrize("name", list(st.STRATEGIES))
def test_weights_are_valid(prices, name):
    target = st.STRATEGIES[name](prices, benchmark=prices.mean(axis=1), min_history=100).dropna(how="all")
    gross = target.abs().sum(axis=1)
    assert (gross <= 1 + 1e-9).all()
    # never hold a stock before it has a price
    late = target.loc[target.index < prices.index[400], "S11"]
    assert (late == 0).all()


def test_execution_lag_shifts_returns(prices):
    rets = ft.returns(prices)
    target = st.momentum(prices, lookback=63, skip=5, min_history=100)
    r0 = bt.run(target, rets, lag=0).returns
    r1 = bt.run(target, rets, lag=1).returns
    assert not np.allclose(r0, r1)
