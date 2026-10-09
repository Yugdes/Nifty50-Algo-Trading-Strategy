"""Download, cache and quality-check daily price data."""

import logging

import numpy as np
import pandas as pd

from .config import BENCHMARK_TICKER, DATA_DIR, Config

log = logging.getLogger(__name__)

PRICES_FILE = DATA_DIR / "prices.csv"
VOLUME_FILE = DATA_DIR / "volume.csv"
BENCHMARK_FILE = DATA_DIR / "benchmark.csv"


def download(cfg: Config) -> None:
    """Fetch adjusted closes and volumes from Yahoo Finance and cache them as CSV."""
    import yfinance as yf

    raw = yf.download(
        cfg.tickers + [BENCHMARK_TICKER], start=cfg.start, end=cfg.end,
        auto_adjust=True, progress=False,
    )
    close, volume = raw["Close"], raw["Volume"]
    DATA_DIR.mkdir(exist_ok=True)
    close[cfg.tickers].round(4).to_csv(PRICES_FILE)
    volume[cfg.tickers].to_csv(VOLUME_FILE)
    close[[BENCHMARK_TICKER]].rename(columns={BENCHMARK_TICKER: "NIFTY50"}).round(4).to_csv(BENCHMARK_FILE)
    log.info("Downloaded %d tickers, %d rows", len(cfg.tickers), len(close))


def load(cfg: Config, refresh: bool = False):
    """Return (prices, volume, benchmark) aligned on the benchmark's trading calendar."""
    if refresh or not PRICES_FILE.exists():
        download(cfg)
    read = lambda f: pd.read_csv(f, index_col=0, parse_dates=True)
    prices, volume, bench = read(PRICES_FILE), read(VOLUME_FILE), read(BENCHMARK_FILE)["NIFTY50"]

    calendar = bench.dropna().index
    prices = prices.reindex(calendar)
    volume = volume.reindex(calendar)
    bench = bench.reindex(calendar)
    prices = clean_prices(prices)
    return prices, volume, bench


# Demerger ex-dates that Yahoo's adjusted series does not account for. The
# spun-off entity's value would otherwise show up as a large fake loss. We
# back-adjust earlier prices so the ex-date return is zero (a conservative
# approximation of holding the parent plus the spun-off shares).
UNADJUSTED_DEMERGERS = [
    ("ADANIENT.NS", "2015-06-03"),  # Adani Ports/Power/Transmission demerger
    ("TMPV.NS", "2025-10-14"),      # Tata Motors CV business demerger
]


def clean_prices(prices: pd.DataFrame, max_gap: int = 5) -> pd.DataFrame:
    """Fill short holiday/data gaps inside each series; never fill before listing."""
    filled = prices.ffill(limit=max_gap)
    first_valid = prices.apply(pd.Series.first_valid_index)
    for col, first in first_valid.items():
        filled.loc[filled.index < first, col] = np.nan
    return adjust_demergers(filled)


def adjust_demergers(prices: pd.DataFrame, events=UNADJUSTED_DEMERGERS) -> pd.DataFrame:
    prices = prices.copy()
    for ticker, date in events:
        if ticker not in prices or pd.Timestamp(date) not in prices.index:
            continue
        loc = prices.index.get_loc(pd.Timestamp(date))
        factor = prices[ticker].iloc[loc] / prices[ticker].iloc[loc - 1]
        prices.iloc[:loc, prices.columns.get_loc(ticker)] *= factor
    return prices


def quality_report(prices: pd.DataFrame, threshold: float = 0.35) -> pd.DataFrame:
    """List daily moves larger than `threshold`, which usually flag corporate-action errors."""
    rets = prices.pct_change(fill_method=None)
    big = rets.stack()
    big = big[big.abs() > threshold]
    return big.rename("return").reset_index().rename(columns={"level_0": "Date", "level_1": "Ticker"})
