"""Technical indicators computed column-wise on wide (date x ticker) frames.

Implemented directly in pandas so every formula is visible and testable.
All values at date t use only data up to and including t.
"""

import numpy as np
import pandas as pd


def returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None)


def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder's RSI."""
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, min_periods=window).mean()
    return 100 - 100 / (1 + gain / loss)


def macd(prices: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    line = prices.ewm(span=fast, min_periods=slow).mean() - prices.ewm(span=slow, min_periods=slow).mean()
    return line, line.ewm(span=signal, min_periods=signal).mean()


def momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Return from t-lookback to t-skip (skipping the most recent month)."""
    return prices.shift(skip) / prices.shift(lookback) - 1


def volatility(prices: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    return returns(prices).rolling(window, min_periods=int(window * 0.8)).std() * np.sqrt(252)


def ml_features(prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Scale-free features in long format (date, ticker) for the ML model.

    Each raw feature is also converted to a cross-sectional percentile rank so
    the model learns relative (stock-vs-stock) patterns rather than market timing.
    """
    ret = returns(prices)
    m_line, m_sig = macd(prices)
    raw = {
        "ret_5": prices / prices.shift(5) - 1,
        "ret_21": prices / prices.shift(21) - 1,
        "ret_63": prices / prices.shift(63) - 1,
        "mom_12_1": momentum(prices),
        "vol_21": ret.rolling(21).std(),
        "vol_63": ret.rolling(63).std(),
        "rsi_14": rsi(prices),
        "macd_hist": (m_line - m_sig) / prices,
        "dist_sma50": prices / prices.rolling(50).mean() - 1,
        "dist_sma200": prices / prices.rolling(200).mean() - 1,
        "max_ret_21": ret.rolling(21).max(),
        "volume_z": (np.log(volume.where(volume > 0)) - np.log(volume.where(volume > 0)).rolling(63).mean())
        / np.log(volume.where(volume > 0)).rolling(63).std(),
    }
    frames = {}
    for name, df in raw.items():
        frames[name] = df
        frames[name + "_rank"] = df.rank(axis=1, pct=True)
    out = pd.concat(frames, axis=1)
    out.columns.names = ["feature", "ticker"]
    return out.stack("ticker", future_stack=True)
