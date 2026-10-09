"""Static figures for the README and research note."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8984"
# Categorical slots in fixed order; a strategy keeps its colour in every figure.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#a8312f", "#e34948", "#f3b3b2", "#f0efec", "#9ec5f4", "#2a78d6", "#104281"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#6da7ec", "#2a78d6", "#104281"])

pct = FuncFormatter(lambda x, _: f"{x:.0%}")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
    "legend.frameon": False, "legend.labelcolor": INK, "lines.linewidth": 1.8,
    "font.family": "DejaVu Sans",
})


def colour_map(names):
    return {n: SERIES[i % len(SERIES)] for i, n in enumerate(names)}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def equity_curves(returns: dict, benchmark: pd.Series, colours: dict, path, title):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot((1 + benchmark).cumprod(), color=NEUTRAL, ls="--", lw=1.6, label="NIFTY 50 index")
    for name, r in returns.items():
        ax.plot((1 + r).cumprod(), color=colours[name], label=name)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.75, 1, 1.5, 2, 3, 4, 6, 8, 12]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}x"))
    ax.set_title(title)
    ax.set_ylabel("Growth of 1 (log scale)")
    ax.legend(loc="upper left", ncol=2)
    _save(fig, path)


def drawdowns(dd: dict, colours: dict, path):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for name, s in dd.items():
        ax.plot(s, color=colours.get(name, NEUTRAL), lw=1.4, ls="--" if name.startswith("NIFTY") else "-",
                label=name)
    ax.yaxis.set_major_formatter(pct)
    ax.set_title("Drawdowns: the trend filter sidesteps most of the 2020 crash")
    ax.legend(loc="lower left", ncol=3)
    _save(fig, path)


def cost_sensitivity(table: pd.DataFrame, colours: dict, ew_value: float, path):
    """table: index = cost bps, columns = strategy, values = CAGR."""
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for name in table.columns:
        ax.plot(table.index, table[name], color=colours[name], marker="o", ms=4, label=name)
    ax.axhline(ew_value, color=NEUTRAL, ls=":", lw=1.2)
    ax.annotate("Equal Weight (no-skill benchmark)", (table.index.max(), ew_value), xytext=(0, 5),
                textcoords="offset points", ha="right", color=INK_2, fontsize=9)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("One-way transaction cost (bps)")
    ax.set_ylabel("CAGR, net of costs")
    ax.set_title("How much of each edge survives trading costs?")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    _save(fig, path)


def heatmap(df: pd.DataFrame, path, title, fmt="{:.2f}", center=None, xlabel="", ylabel="", cmap=None,
            figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    vals = df.to_numpy(dtype=float)
    if center is not None:
        span = np.nanmax(np.abs(vals - center))
        norm = TwoSlopeNorm(center, center - span, center + span)
        im = ax.imshow(vals, cmap=cmap or DIVERGING, norm=norm, aspect="auto")
    else:
        im = ax.imshow(vals, cmap=cmap or SEQUENTIAL, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(df.shape[1]), df.columns)
    ax.set_yticks(range(df.shape[0]), df.index)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    rgba = im.cmap(im.norm(vals))
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if np.isnan(vals[i, j]):
                continue
            lum = 0.2126 * rgba[i, j, 0] + 0.7152 * rgba[i, j, 1] + 0.0722 * rgba[i, j, 2]
            ax.text(j, i, fmt.format(vals[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if lum < 0.5 else INK)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _save(fig, path)


def ml_scores(scores: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    models = list(scores["Model"].unique())
    years = sorted(scores["Year"].unique())
    width = 0.38
    for k, mname in enumerate(models):
        s = scores[scores["Model"] == mname].set_index("Year").reindex(years)
        ax.bar(np.arange(len(years)) + (k - 0.5) * width, s["AUC"] - 0.5, width - 0.04, bottom=0.5,
               color=SERIES[k], label=mname)
    ax.axhline(0.5, color=INK_2, lw=1)
    ax.annotate("coin flip", (len(years) - 0.5, 0.5), xytext=(0, 4), textcoords="offset points",
                ha="right", color=INK_2, fontsize=9)
    ax.set_xticks(range(len(years)), years)
    ax.set_ylim(0.44, 0.58)
    ax.set_ylabel("Out-of-sample ROC AUC")
    ax.set_title("Walk-forward ML: out-of-sample AUC by year")
    ax.legend(loc="upper left")
    _save(fig, path)


def feature_ic(ic: pd.DataFrame, path):
    ic = ic.sort_values("Mean IC")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    colours = [SERIES[0] if v > 0 else SERIES[7] for v in ic["Mean IC"]]
    ax.barh(ic["Feature"], ic["Mean IC"], color=colours, height=0.6)
    ax.axvline(0, color=INK_2, lw=1)
    for y, (v, t) in enumerate(zip(ic["Mean IC"], ic["t-stat"])):
        ax.text(v + (0.002 if v >= 0 else -0.002), y, f"t={t:.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=INK_2)
    ax.set_xlabel("Mean rank IC vs. next-month return")
    ax.set_title("Single-feature predictive power (2016-2025)")
    lim = ic["Mean IC"].abs().max() * 1.6
    ax.set_xlim(-lim, lim)
    ax.grid(axis="y", visible=False)
    _save(fig, path)
