"""End-to-end research pipeline: data -> strategies -> backtests -> tables & figures."""

import logging

import numpy as np
import pandas as pd

from . import backtest as bt
from . import data as dt
from . import features as ft
from . import metrics as mx
from . import ml
from . import plots
from . import strategies as st
from .config import FIGURES_DIR, RESULTS_DIR, Config

log = logging.getLogger(__name__)


class Research:
    def __init__(self, cfg: Config, refresh: bool = False):
        self.cfg = cfg
        self.prices, self.volume, self.bench_px = dt.load(cfg, refresh)
        self.rets = ft.returns(self.prices)
        self.bench = self.bench_px.pct_change()
        self.targets = {name: f(self.prices, benchmark=self.bench_px, quantile=cfg.top_quantile,
                                min_history=cfg.min_history)
                        for name, f in st.STRATEGIES.items()}
        self.colours = plots.colour_map(list(self.targets) + ["ML Ranker"])

    def backtest(self, target, cost_bps=None, lag=None, start=None):
        c = self.cfg
        return bt.run(target, self.rets, cost_bps=c.cost_bps if cost_bps is None else cost_bps,
                      risk_free=c.risk_free, lag=c.execution_lag if lag is None else lag,
                      start=start or c.trade_start)

    def stats(self, res, bootstrap=False):
        c = self.cfg
        r = res.returns
        out = mx.summary(r, self.bench.reindex(r.index), c.risk_free, res.turnover,
                         dict(n=c.bootstrap_samples, block=c.bootstrap_block, seed=c.seed) if bootstrap else None)
        return out

    # --- analyses -----------------------------------------------------------

    def performance(self):
        res = {n: self.backtest(t) for n, t in self.targets.items()}
        rows = {n: self.stats(r, bootstrap=True) for n, r in res.items()}
        b = self.bench.loc[self.cfg.trade_start:]
        rows["NIFTY 50 index"] = mx.summary(b, b, self.cfg.risk_free,
                                            bootstrap=dict(n=self.cfg.bootstrap_samples,
                                                           block=self.cfg.bootstrap_block, seed=self.cfg.seed))
        table = pd.DataFrame(rows).T.rename_axis("Strategy")
        table.loc["NIFTY 50 index", ["Alpha (ann.)", "Alpha t-stat", "Info Ratio"]] = np.nan
        ew = res["Equal Weight"].returns
        table["Excess CAGR vs EW"] = table["CAGR"] - table.loc["Equal Weight", "CAGR"]
        table["IR vs EW"] = [mx.information_ratio(r.returns, ew) if n != "Equal Weight" else np.nan
                             for n, r in res.items()] + [mx.information_ratio(b, ew)]
        gross = {n: mx.cagr(r.gross_returns) for n, r in res.items()}
        table["Gross CAGR"] = pd.Series(gross)
        return res, table

    def cost_grid(self):
        rows = {}
        for cost in self.cfg.cost_grid_bps:
            rows[cost] = {n: mx.cagr(self.backtest(t, cost_bps=cost).returns) for n, t in self.targets.items()}
        return pd.DataFrame(rows).T.rename_axis("cost_bps")

    def breakeven_costs(self, grid: pd.DataFrame):
        """Cost (bps) at which each strategy's CAGR falls to the equal-weight CAGR."""
        out = {}
        for name in grid.columns.drop("Equal Weight"):
            gap = grid[name] - grid["Equal Weight"]
            if gap.iloc[0] <= 0:
                out[name] = 0.0
            elif (gap > 0).all():
                out[name] = np.inf
            else:
                out[name] = float(np.interp(0, -gap.to_numpy(), gap.index.to_numpy()))
        return pd.Series(out, name="Break-even cost vs EW (bps)")

    def subperiods(self, res):
        c = self.cfg
        rows = []
        for name, r in res.items():
            r = r.returns
            for label, s in (("Development 2016-2021", r[r.index < c.oos_start]),
                             ("Hold-out 2022-2025", r[r.index >= c.oos_start])):
                rows.append({"Strategy": name, "Period": label, "CAGR": mx.cagr(s),
                             "Sharpe": mx.sharpe(s, c.risk_free), "Max Drawdown": mx.max_drawdown(s)})
        b = self.bench.loc[c.trade_start:]
        for label, s in (("Development 2016-2021", b[b.index < c.oos_start]), ("Hold-out 2022-2025", b[b.index >= c.oos_start])):
            rows.append({"Strategy": "NIFTY 50 index", "Period": label, "CAGR": mx.cagr(s),
                         "Sharpe": mx.sharpe(s, c.risk_free), "Max Drawdown": mx.max_drawdown(s)})
        return pd.DataFrame(rows)

    def momentum_grid(self):
        """Net Sharpe across lookback and portfolio-size choices (robustness, not tuning)."""
        rows = {}
        for months in (3, 6, 9, 12):
            row = {}
            for q in (0.1, 0.2, 0.3, 0.5):
                tgt = st.momentum(self.prices, lookback=21 * months, skip=21, quantile=q,
                                  min_history=self.cfg.min_history)
                row[f"Top {q:.0%}"] = mx.sharpe(self.backtest(tgt).returns, self.cfg.risk_free)
            rows[f"{months}-1 months"] = row
        return pd.DataFrame(rows).T.rename_axis("Formation period")

    def lag_sensitivity(self):
        rows = {}
        for lag in (0, 1, 2):
            rows[f"lag {lag}"] = {n: mx.sharpe(self.backtest(t, lag=lag).returns, self.cfg.risk_free)
                                  for n, t in self.targets.items()}
        return pd.DataFrame(rows).rename_axis("Strategy")

    def survivorship_gap(self, ew_returns):
        b = self.bench.reindex(ew_returns.index)
        return {"EW of today's constituents CAGR": mx.cagr(ew_returns), "NIFTY 50 price index CAGR": mx.cagr(b),
                "Gap (pp per year)": mx.cagr(ew_returns) - mx.cagr(b)}

    def machine_learning(self):
        c = self.cfg
        data = ml.build_dataset(self.prices, self.volume, c.ml_horizon, c.min_history)
        preds, scores = ml.walk_forward(data, c.ml_first_test_year, c.ml_horizon, c.seed)
        ic = ml.feature_ic(data[data.index.get_level_values(0) >= c.trade_start])
        start = f"{c.ml_first_test_year}-01-01"
        tgt = ml.prediction_targets(preds["Gradient Boosting"], self.prices.index, self.prices.columns, c.top_quantile)
        compare = {"ML Ranker": self.backtest(tgt, start=start)}
        for name in ("Equal Weight", "Momentum 12-1"):
            compare[name] = self.backtest(self.targets[name], start=start)
        table = pd.DataFrame({n: self.stats(r) for n, r in compare.items()}).T.rename_axis("Strategy")
        table["Excess CAGR vs EW"] = table["CAGR"] - table.loc["Equal Weight", "CAGR"]
        return scores, ic, table, compare


def _md(df: pd.DataFrame, fmts: dict = None, index=True) -> str:
    fmts = fmts or {}
    df = df.copy()
    for col in df.columns:
        f = fmts.get(col)
        if f:
            df[col] = df[col].map(lambda v, f=f: "–" if pd.isna(v) else ("∞" if np.isinf(v) else f.format(v)))
    if index:
        df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" if i == 0 else "---:" for i in range(len(df.columns))) + "|"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in df.to_numpy()]
    return "\n".join([head, sep, *body])


def run(cfg: Config = None, refresh: bool = False, skip_ml: bool = False):
    cfg = cfg or Config()
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    R = Research(cfg, refresh)
    md = [f"# Results\n\nGenerated by `python -m niftyquant`. Sample: {cfg.trade_start} to "
          f"{R.prices.index[-1].date()}, {R.prices.shape[1]} stocks, costs {cfg.cost_bps:.0f} bps one-way, "
          f"execution lag {cfg.execution_lag} day, risk-free {cfg.risk_free:.0%}.\n"]
    P = "{:.1%}"
    F = "{:.2f}"

    log.info("Backtesting strategies")
    res, perf = R.performance()
    perf.to_csv(RESULTS_DIR / "performance_summary.csv")
    cols = ["CAGR", "Gross CAGR", "Volatility", "Sharpe", "Sharpe 95% CI low", "Sharpe 95% CI high", "Max Drawdown",
            "Beta", "Alpha (ann.)", "Alpha t-stat", "Excess CAGR vs EW", "IR vs EW", "Turnover (ann.)"]
    md += ["## Performance (net of costs)\n", _md(perf[cols], {
        "CAGR": P, "Gross CAGR": P, "Volatility": P, "Sharpe": F, "Sharpe 95% CI low": F, "Sharpe 95% CI high": F,
        "Max Drawdown": P, "Beta": F, "Alpha (ann.)": P, "Alpha t-stat": F, "Excess CAGR vs EW": P, "IR vs EW": F,
        "Turnover (ann.)": "{:.1f}x"}), ""]

    surv = R.survivorship_gap(res["Equal Weight"].returns)
    md += ["## Survivorship / selection bias\n", _md(pd.Series(surv).rename_axis("Measure").to_frame("Value"), {"Value": P}), ""]

    log.info("Cost sensitivity")
    grid = R.cost_grid()
    grid.to_csv(RESULTS_DIR / "cost_sensitivity.csv")
    be = R.breakeven_costs(grid)
    md += ["## CAGR vs one-way cost (bps)\n", _md(grid, {c: P for c in grid.columns}), "",
           "## Break-even cost versus Equal Weight\n", _md(be.rename_axis("Strategy").to_frame(), {be.name: "{:.0f}"}), ""]

    sub = R.subperiods(res)
    sub.to_csv(RESULTS_DIR / "subperiods.csv", index=False)
    pivot = sub.pivot(index="Strategy", columns="Period", values="Sharpe")
    md += ["## Sharpe: development vs hold-out\n", _md(pivot, {c: F for c in pivot.columns}), ""]

    log.info("Robustness grids")
    mg = R.momentum_grid()
    mg.to_csv(RESULTS_DIR / "momentum_robustness.csv")
    lag = R.lag_sensitivity()
    lag.to_csv(RESULTS_DIR / "execution_lag.csv")
    md += ["## Momentum robustness (net Sharpe)\n", _md(mg, {c: F for c in mg.columns}), "",
           "## Execution-lag sensitivity (net Sharpe)\n", _md(lag, {c: F for c in lag.columns}), ""]

    annual = pd.DataFrame({n: mx.calendar_returns(r.returns) for n, r in res.items()}).rename_axis("Year")
    annual["NIFTY 50 index"] = mx.calendar_returns(R.bench.loc[cfg.trade_start:])
    annual.to_csv(RESULTS_DIR / "annual_returns.csv")
    md += ["## Calendar-year returns\n", _md(annual, {c: P for c in annual.columns}), ""]

    log.info("Figures")
    c = R.colours
    main = ["Equal Weight", "Momentum 12-1", "Low Volatility", "Trend-Filtered EW", "MACD+RSI (v1)"]
    plots.equity_curves({n: res[n].returns for n in main}, R.bench.loc[cfg.trade_start:], c,
                        FIGURES_DIR / "equity_curves.png", "Growth of 1 rupee, net of costs (2016-2025)")
    dd = {n: mx.drawdown(res[n].returns) for n in ("Equal Weight", "Trend-Filtered EW")}
    dd["NIFTY 50 index"] = mx.drawdown(R.bench.loc[cfg.trade_start:])
    plots.drawdowns(dd, c, FIGURES_DIR / "drawdowns.png")
    plots.cost_sensitivity(grid.drop(columns="Equal Weight"), c, grid["Equal Weight"].iloc[0],
                           FIGURES_DIR / "cost_sensitivity.png")
    plots.heatmap(mg, FIGURES_DIR / "momentum_robustness.png", "Momentum: net Sharpe across parameter choices",
                  xlabel="Portfolio size (share of universe)", ylabel="Formation period")
    plots.heatmap(annual.T, FIGURES_DIR / "annual_returns.png", "Calendar-year returns, net of costs",
                  fmt="{:.0%}", center=0.0, figsize=(11, 4.2))

    if not skip_ml:
        log.info("Machine learning walk-forward")
        scores, ic, ml_table, _ = R.machine_learning()
        scores.to_csv(RESULTS_DIR / "ml_walkforward.csv", index=False)
        ic.to_csv(RESULTS_DIR / "feature_ic.csv", index=False)
        ml_table.to_csv(RESULTS_DIR / "ml_strategy.csv")
        avg = scores.groupby("Model")[["AUC", "Rank IC"]].mean()
        md += ["## ML walk-forward (out-of-sample, per year)\n",
               _md(scores.pivot(index="Year", columns="Model", values="AUC"), {m: "{:.3f}" for m in avg.index}), "",
               "Average:\n", _md(avg, {"AUC": "{:.3f}", "Rank IC": "{:.3f}"}), "",
               "## Single-feature rank IC\n", _md(ic, {"Mean IC": "{:.3f}", "t-stat": F}, index=False), "",
               f"## ML ranker portfolio ({cfg.ml_first_test_year}-2025)\n",
               _md(ml_table[["CAGR", "Sharpe", "Max Drawdown", "Alpha (ann.)", "Alpha t-stat", "Excess CAGR vs EW",
                             "Turnover (ann.)"]],
                   {"CAGR": P, "Sharpe": F, "Max Drawdown": P, "Alpha (ann.)": P, "Alpha t-stat": F,
                    "Excess CAGR vs EW": P, "Turnover (ann.)": "{:.1f}x"}), ""]
        plots.ml_scores(scores, FIGURES_DIR / "ml_walkforward.png")
        plots.feature_ic(ic, FIGURES_DIR / "feature_ic.png")

    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    log.info("Done. See %s", RESULTS_DIR / "RESULTS.md")
    return R
