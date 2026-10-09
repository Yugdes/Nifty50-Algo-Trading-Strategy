"""Performance and risk statistics for daily return series."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _daily_rf(risk_free: float) -> float:
    return (1 + risk_free) ** (1 / TRADING_DAYS) - 1


def cagr(r: pd.Series) -> float:
    years = len(r) / TRADING_DAYS
    return (1 + r).prod() ** (1 / years) - 1


def ann_vol(r: pd.Series) -> float:
    return r.std() * np.sqrt(TRADING_DAYS)


def sharpe(r: pd.Series, risk_free: float = 0.0) -> float:
    ex = r - _daily_rf(risk_free)
    sd = ex.std()
    return np.nan if sd == 0 else ex.mean() / sd * np.sqrt(TRADING_DAYS)


def sortino(r: pd.Series, risk_free: float = 0.0) -> float:
    ex = r - _daily_rf(risk_free)
    downside = np.sqrt((ex.clip(upper=0) ** 2).mean())
    return np.nan if downside == 0 else ex.mean() / downside * np.sqrt(TRADING_DAYS)


def drawdown(r: pd.Series) -> pd.Series:
    wealth = (1 + r).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    return drawdown(r).min()


def capm(r: pd.Series, bench: pd.Series, risk_free: float = 0.0):
    """Annualised alpha, its t-stat, and beta from an OLS of excess returns."""
    rf = _daily_rf(risk_free)
    df = pd.concat([r - rf, bench - rf], axis=1).dropna()
    y, x = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    sigma2 = resid @ resid / (len(y) - 2)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    return coef[0] * TRADING_DAYS, coef[0] / se[0], coef[1]


def information_ratio(r: pd.Series, bench: pd.Series) -> float:
    active = (r - bench).dropna()
    sd = active.std()
    return np.nan if sd == 0 or np.isnan(sd) else active.mean() / sd * np.sqrt(TRADING_DAYS)


def block_bootstrap_sharpe(r: pd.Series, risk_free=0.0, n=2000, block=21, seed=0):
    """95% CI for the Sharpe ratio via a circular block bootstrap (keeps autocorrelation)."""
    rng = np.random.default_rng(seed)
    x = (r - _daily_rf(risk_free)).to_numpy()
    T = len(x)
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=(n, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :T] % T
    samples = x[idx]
    sr = samples.mean(axis=1) / samples.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)
    return np.percentile(sr, [2.5, 97.5])


def summary(r: pd.Series, bench: pd.Series, risk_free: float = 0.0, turnover: pd.Series = None,
            bootstrap: dict = None) -> dict:
    alpha, t_alpha, beta = capm(r, bench, risk_free)
    mdd = max_drawdown(r)
    out = {
        "CAGR": cagr(r),
        "Volatility": ann_vol(r),
        "Sharpe": sharpe(r, risk_free),
        "Sortino": sortino(r, risk_free),
        "Max Drawdown": mdd,
        "Calmar": cagr(r) / abs(mdd) if mdd < 0 else np.nan,
        "Beta": beta,
        "Alpha (ann.)": alpha,
        "Alpha t-stat": t_alpha,
        "Info Ratio": information_ratio(r, bench),
        "Monthly Hit Rate": ((1 + r).resample("ME").prod() - 1 > 0).mean(),
    }
    if turnover is not None:
        out["Turnover (ann.)"] = turnover.sum() / (len(r) / TRADING_DAYS)
    if bootstrap is not None:
        lo, hi = block_bootstrap_sharpe(r, risk_free, **bootstrap)
        out["Sharpe 95% CI low"], out["Sharpe 95% CI high"] = lo, hi
    return out


def calendar_returns(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.year).prod() - 1
