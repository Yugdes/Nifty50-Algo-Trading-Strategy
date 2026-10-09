"""Event-driven portfolio simulator with weight drift, execution lag and costs.

Conventions
-----------
* `target` holds the desired weights decided with information up to the close
  of each date (rows may be NaN = "no new decision").
* The order is executed `lag` trading days later at the close.
* Between trades, weights drift with prices; uninvested cash earns the risk-free
  rate (short proceeds are treated as cash, borrow costs are ignored).
* Costs are charged as `cost_bps` per unit of one-way turnover (sum |dw|).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    returns: pd.Series        # daily net portfolio returns
    gross_returns: pd.Series  # before transaction costs
    turnover: pd.Series       # one-way turnover on each trade date
    weights: pd.DataFrame     # end-of-day weights after trading


def run(
    target: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cost_bps: float = 0.0,
    risk_free: float = 0.0,
    lag: int = 1,
    start=None,
) -> BacktestResult:
    target = target.reindex(index=asset_returns.index, columns=asset_returns.columns)
    executed = target.shift(lag)
    decision = np.array(executed.notna().any(axis=1), dtype=bool)
    # Rows are complete portfolios; forward-fill so the first active day can
    # adopt the latest decision instead of waiting for the next rebalance.
    tgt = executed.ffill().fillna(0.0).to_numpy()
    rets = asset_returns.fillna(0.0).to_numpy()
    rf = (1 + risk_free) ** (1 / 252) - 1
    c = cost_bps / 1e4

    n_days, n_assets = rets.shape
    w = np.zeros(n_assets)
    net = np.zeros(n_days)
    gross = np.zeros(n_days)
    turn = np.zeros(n_days)
    w_hist = np.zeros((n_days, n_assets))
    active = np.ones(n_days, dtype=bool)
    if start is not None:
        active = np.asarray(asset_returns.index >= pd.Timestamp(start))
        if active.any():
            decision[np.argmax(active)] = True

    for t in range(n_days):
        # 1) mark to market with today's returns
        holdings = w * (1 + rets[t])
        cash = (1 - w.sum()) * (1 + rf)
        value = holdings.sum() + cash
        gross[t] = value - 1
        w = holdings / value
        # 2) trade at the close if there is a new decision
        if decision[t] and active[t]:
            new_w = tgt[t]
            turn[t] = np.abs(new_w - w).sum()
            w = new_w.copy()
        net[t] = (1 + gross[t]) * (1 - turn[t] * c) - 1
        w_hist[t] = w

    idx = asset_returns.index
    keep = active
    return BacktestResult(
        returns=pd.Series(net, idx)[keep],
        gross_returns=pd.Series(gross, idx)[keep],
        turnover=pd.Series(turn, idx)[keep],
        weights=pd.DataFrame(w_hist, idx, asset_returns.columns)[keep],
    )
