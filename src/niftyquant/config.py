"""Central configuration: universe, sample period and cost / financing assumptions."""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

BENCHMARK_TICKER = "^NSEI"

# NIFTY 50 constituents as of late 2025. Using today's constituents over the
# whole history introduces survivorship bias; see `analysis.survivorship_gap`
# and the README for how this is handled.
UNIVERSE = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDIGO", "INFY", "ITC", "JIOFIN", "JSWSTEEL",
    "KOTAKBANK", "LT", "M&M", "MARUTI", "MAXHEALTH", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN", "SUNPHARMA", "TCS",
    "TATACONSUM", "TMPV", "TATASTEEL", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]
TICKERS = [t + ".NS" for t in UNIVERSE]


@dataclass(frozen=True)
class Config:
    start: str = "2015-01-01"
    end: str = "2026-01-01"  # exclusive
    # First date strategies may trade: leaves a 1-year warm-up for 12-month signals.
    trade_start: str = "2016-01-01"
    # Development / hold-out split. Strategy parameters are taken from the
    # literature, not tuned, but results are still reported for both periods.
    oos_start: str = "2022-01-01"

    # One-way cost per unit of turnover, in basis points. Delivery trades in
    # India pay 0.1% STT on each side, ~0.015% stamp duty on buys and small
    # exchange/SEBI fees (~0.06% one-way on average) plus slippage.
    cost_bps: float = 15.0
    cost_grid_bps: tuple = (0.0, 5.0, 10.0, 15.0, 25.0, 40.0, 60.0)

    # Annual risk-free rate, approx. average 91-day T-bill yield over 2015-2025.
    risk_free: float = 0.06

    # Days between the signal close and the execution close. 1 = trade at the
    # next day's close, which is conservative versus same-close execution.
    execution_lag: int = 1

    top_quantile: float = 0.2
    min_history: int = 252
    bootstrap_samples: int = 2000
    bootstrap_block: int = 21
    seed: int = 7

    ml_horizon: int = 21
    ml_first_test_year: int = 2018
    tickers: list = field(default_factory=lambda: list(TICKERS))
