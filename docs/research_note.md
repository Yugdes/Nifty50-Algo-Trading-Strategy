# Research Note: Systematic Signals in NIFTY 50 Equities, 2016-2025

**Yug Desai** · IIT Gandhinagar · Code and data: this repository (`python -m niftyquant`)

---

## Summary

We test five well-known systematic equity strategies, the MACD + RSI rule from v1 of this project, and a walk-forward ML stock ranker on the 50 current NIFTY 50 constituents. All are evaluated net of realistic Indian transaction costs, with next-day execution. **None beats a naive equal-weight portfolio of the same stocks on a risk-adjusted basis.** The most useful result is defensive: a simple 200-day trend filter cut the maximum drawdown from -37.5% to -13.8% and doubled the Calmar ratio, giving up about 5 pp of annual return. Short-horizon signals, whether rule-based or ML-based, have gross edges that are either statistically weak or erased by trading costs.

## 1. Data and set-up

- **Universe:** 50 NIFTY 50 constituents as of late 2025. Daily adjusted closes and volumes come from Yahoo Finance, Jan 2015 to Dec 2025. A stock becomes tradable after 252 days of history.
- **Cleaning:** an automated large-move check found two demergers missing from the adjusted series (Adani Enterprises, Jun 2015; Tata Motors, Oct 2025). Their ex-date returns are neutralised.
- **Simulation:** decisions use data up to the close of day *t* and execute at the close of *t+1*. Weights drift between rebalances, and cash earns 6% p.a.
- **Costs:** 15 bps per unit of one-way turnover, roughly 0.1% STT, stamp duty and exchange fees, plus slippage.
- **Benchmarks:** the NIFTY 50 price index, and an equal-weight (EW) portfolio of the universe, rebalanced monthly.

**The benchmark matters more than any signal.** EW of today's constituents compounded at 23.4% a year against 12.9% for the index. Most of that 10.4 pp gap is survivorship and selection bias: we are holding the stocks that went on to become index members. Because every strategy draws from the same biased universe, the fair test is whether it beats EW. Alpha against the index is not.

## 2. Strategies

| Strategy | Rule | Rebalance | Reference |
|---|---|---|---|
| Momentum 12-1 | Top 20% by return from t-12m to t-1m | Monthly | Jegadeesh & Titman (1993) |
| Low Volatility | Bottom 20% by 1-year volatility | Monthly | Baker, Bradley & Wurgler (2011) |
| Short-term Reversal | Bottom 20% by past-month return | Weekly | Jegadeesh (1990), Lehmann (1990) |
| Trend-Filtered EW | EW if NIFTY > 200-day SMA, else cash | Monthly | Faber (2007) |
| MACD + RSI (v1) | Long/short on MACD crossover + RSI 30/70 | Daily | — |
| ML Ranker | Top 20% by predicted probability of beating the median | Monthly | Gu, Kelly & Xiu (2020) |

Parameters are the literature defaults. Nothing was tuned on this sample.

## 3. Results (net of costs, 2016-2025)

| | CAGR | Vol | Sharpe [95% CI] | Max DD | Beta | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| Equal Weight | 23.4% | 16.4% | 1.01 [0.28, 1.79] | -37.5% | 0.96 | 0.8x |
| Momentum 12-1 | 23.9% | 21.0% | 0.85 [0.16, 1.58] | -44.9% | 1.03 | 6.5x |
| Low Volatility | 17.1% | 13.7% | 0.79 [0.14, 1.49] | -26.9% | 0.70 | 3.3x |
| ST Reversal | 19.6% | 21.0% | 0.68 [0.02, 1.41] | -43.8% | 1.06 | 43.5x |
| Trend-Filtered EW | 18.1% | 12.2% | 0.95 [0.31, 1.60] | -13.8% | 0.50 | 2.6x |
| MACD + RSI (v1) | -0.5% | 7.8% | -0.77 [-1.48, -0.13] | -36.6% | -0.37 | 1.2x |
| NIFTY 50 index | 12.9% | 16.2% | 0.47 [-0.18, 1.19] | -38.4% | 1.00 | – |

*CIs come from a circular block bootstrap (21-day blocks, 2,000 draws).*

**Momentum** is the only signal ahead of EW after costs (+0.6 pp CAGR), with higher volatility and a deeper drawdown. Its information ratio against EW is 0.12. The effect is robust across formation periods of 3-12 months and portfolio sizes of 10-50% (net Sharpe 0.63-1.05), and it held up in the 2022-25 hold-out (Sharpe 1.00). It is real but small, consistent with weaker momentum in a 50-stock universe.

**Short-term reversal** has the strongest gross signal: 27.7% gross CAGR, 4.2 pp above EW. Weekly rebalancing turns over ~44x a year, so the edge falls to zero at 8 bps one-way and the strategy loses money above ~55 bps. It is a capacity and execution problem, not a signal problem.

**Low volatility** delivered the expected risk reduction (beta 0.70, max DD -27%) but no excess return. In this bull-market sample, high-volatility stocks actually earned more (63-day volatility has the most positive single-feature IC, +0.04).

**The trend filter** sidesteps most of the March 2020 crash: NIFTY was already below its 200-day average at end-February. It also gave up returns in whipsaws (2016, 2018, 2019, 2022). Its Calmar of 1.31 is the best of all strategies, but the evidence rests largely on one crash.

**MACD + RSI** loses money. It trades rarely and its long/short positions are net short (beta -0.37) in a rising market.

## 4. Machine learning

A logistic regression and a gradient-boosted tree ensemble predict whether each stock beats the cross-sectional median over the next 21 trading days. They use 24 features: 12 price/volume signals plus their cross-sectional ranks. Each year from 2018 to 2025 is predicted by a model trained on all earlier data, with a 21-day purge so no training label overlaps the test period.

- Average out-of-sample AUC: **0.506** (boosting) and **0.508** (logistic). Annual AUCs range from 0.48 to 0.55, with no improvement as the training set grows.
- Average daily rank IC: about 0.01.
- Portfolio (2018-25): 23.0% CAGR vs 21.9% for EW, but a lower Sharpe (0.79 vs 0.90) and 14.5x turnover.

v1 of this project reported an AUC of 1.00 because the label was built from its own input features. The correctly specified test shows that price and volume features alone carry little forward-looking information for these heavily researched large caps.

## 5. Limitations

1. The universe uses today's constituents. A point-in-time membership history is the proper fix, and the EW benchmark only controls for the bias.
2. NIFTY 50 is a price index, while stock returns include dividends, which inflates measured alpha against the index by roughly 1-1.5 pp.
3. Ten years with one major crash give wide confidence intervals: no Sharpe difference here is statistically significant.
4. Costs are a flat rate; market impact and short-borrow costs (for the v1 long/short rule) are ignored.

## 6. Conclusion and next steps

For a long-only investor in NIFTY 50 stocks, the evidence favours a cheap, diversified equal-weight core. A trend overlay is an optional drawdown control, with the caveat that its benefit comes mostly from one crash. Higher-turnover signals need execution costs well below 10 bps to matter. Natural extensions:

- a point-in-time NIFTY 500 universe (more breadth for cross-sectional signals);
- fundamental factors (value, quality, earnings revisions);
- volatility-targeted position sizing;
- hedging market beta with NIFTY futures to isolate stock-selection alpha.

---

*References: Jegadeesh & Titman (1993) JF; Jegadeesh (1990) JF; Lehmann (1990) QJE; Baker, Bradley & Wurgler (2011) FAJ; Faber (2007) JWM; Gu, Kelly & Xiu (2020) RFS; Lopez de Prado (2018), Advances in Financial Machine Learning.*
