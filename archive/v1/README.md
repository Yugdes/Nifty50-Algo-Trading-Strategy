# v1 (archived)

The first version of this project: a MACD + RSI rule backtested stock-by-stock on
20 hand-picked large caps (2018-2023), plus a classifier that initially reported
AUC 1.00 because of target leakage (fixed to 0.53 in the final v1 notebook).

It is kept for transparency. Its lessons drove the v2 redesign:

| v1 problem | v2 fix |
|---|---|
| Each stock backtested separately; 0-2 trades per stock | One portfolio across the whole universe |
| No costs, same-close execution | 15 bps one-way costs + cost grid, next-day execution |
| Single rule, compared only to buy-and-hold per stock | Six strategies vs NIFTY 50 and an equal-weight benchmark |
| Logic split across notebooks | Tested Python package, one command reproduces everything |
| Target leakage in ML; one train/test split | Purged walk-forward validation, cross-sectional target |
| Survivorship bias only mentioned | Quantified (EW-of-survivors vs index gap) |

The notebooks reference the old `data/` layout; the CSVs they used are in `archive/v1/data/`.
The MACD + RSI rule itself is re-implemented in v2 as the `MACD+RSI (v1)` baseline.
