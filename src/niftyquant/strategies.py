"""Strategy definitions.

Each strategy maps price data to a frame of *target weights*: one complete row
per decision date (weights sum to gross exposure), NaN rows otherwise. Signals
at date t only use prices up to t; the backtester adds the execution lag.

All cross-sectional strategies are long-only. Single-stock shorting in Indian
cash equities is limited to intraday, so a realistic implementation would short
index futures to hedge instead; we report alpha/beta versus NIFTY 50 for that.
"""

import numpy as np
import pandas as pd

from . import features as ft


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "ME") -> pd.DatetimeIndex:
    """Last available trading day of each period ('ME' month, 'W-FRI' week)."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(pd.Grouper(freq=freq)).last().dropna())


def eligible(prices: pd.DataFrame, min_history: int = 252) -> pd.DataFrame:
    """A stock is tradable once it has `min_history` days of prices and a price today."""
    return prices.notna() & (prices.notna().cumsum() >= min_history)


def _as_targets(weights: pd.DataFrame, dates, index) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=index, columns=weights.columns)
    out.loc[dates] = weights.loc[dates].fillna(0.0).to_numpy()
    return out


def select_top(score: pd.DataFrame, mask: pd.DataFrame, dates, quantile=0.2, highest=True):
    """Equal-weight the top `quantile` of eligible stocks by `score` on each date."""
    w = pd.DataFrame(0.0, index=dates, columns=score.columns)
    for d in dates:
        s = score.loc[d][mask.loc[d]].dropna()
        if s.empty:
            continue
        n = max(1, int(round(len(s) * quantile)))
        picks = s.nlargest(n).index if highest else s.nsmallest(n).index
        w.loc[d, picks] = 1.0 / n
    return w


# --- strategies ------------------------------------------------------------

def equal_weight(prices, min_history=252, **_):
    dates = rebalance_dates(prices.index)
    mask = eligible(prices, min_history)
    w = mask.loc[dates].astype(float)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)
    return _as_targets(w, dates, prices.index)


def momentum(prices, lookback=252, skip=21, quantile=0.2, min_history=252, **_):
    """Cross-sectional 12-1 momentum (Jegadeesh & Titman, 1993)."""
    dates = rebalance_dates(prices.index)
    score = ft.momentum(prices, lookback, skip)
    w = select_top(score, eligible(prices, max(min_history, lookback)), dates, quantile)
    return _as_targets(w, dates, prices.index)


def low_volatility(prices, window=252, quantile=0.2, min_history=252, **_):
    """Hold the least volatile stocks (low-volatility anomaly, Baker et al., 2011)."""
    dates = rebalance_dates(prices.index)
    score = ft.volatility(prices, window)
    w = select_top(score, eligible(prices, max(min_history, window)), dates, quantile, highest=False)
    return _as_targets(w, dates, prices.index)


def short_term_reversal(prices, window=21, quantile=0.2, min_history=252, **_):
    """Buy last month's losers, rebalanced weekly (Lehmann, 1990; Jegadeesh, 1990)."""
    dates = rebalance_dates(prices.index, "W-FRI")
    score = prices / prices.shift(window) - 1
    w = select_top(score, eligible(prices, min_history), dates, quantile, highest=False)
    return _as_targets(w, dates, prices.index)


def trend_filtered(prices, benchmark, sma=200, min_history=252, **_):
    """Equal-weight portfolio when NIFTY is above its 200-day average, else cash (Faber, 2007)."""
    base = equal_weight(prices, min_history)
    dates = base.dropna(how="all").index
    risk_on = (benchmark > benchmark.rolling(sma).mean()).reindex(dates)
    base.loc[dates] = base.loc[dates].mul(risk_on.astype(float), axis=0)
    return base


def macd_rsi(prices, min_history=252, **_):
    """The original v1 rule, kept as a baseline.

    Long when MACD > signal and RSI < 30, short when MACD < signal and RSI > 70,
    otherwise keep the previous position. Equal risk budget per stock.
    """
    line, sig = ft.macd(prices)
    r = ft.rsi(prices)
    raw = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    raw[(line > sig) & (r < 30)] = 1.0
    raw[(line < sig) & (r > 70)] = -1.0
    mask = eligible(prices, min_history)
    pos = raw.ffill().fillna(0.0).where(mask, 0.0)
    w = pos.div(mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    changed = (pos.diff().abs().sum(axis=1) > 0) | pos.index.isin(rebalance_dates(prices.index))
    return w.where(changed)


STRATEGIES = {
    "Equal Weight": equal_weight,
    "Momentum 12-1": momentum,
    "Low Volatility": low_volatility,
    "ST Reversal": short_term_reversal,
    "Trend-Filtered EW": trend_filtered,
    "MACD+RSI (v1)": macd_rsi,
}
