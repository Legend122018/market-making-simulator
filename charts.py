"""Draw the README charts from results/results.json. Run after run.py."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).parent / "results"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = {  # validated categorical slots 1 and 2
    "inventory_aware": ("Inventory-aware", "#2a78d6"),
    "symmetric": ("Symmetric baseline", "#eb6834"),
}


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def end_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=INK)


def legend(ax, loc="upper left"):
    leg = ax.legend(frameon=False, fontsize=8.5, loc=loc)
    for t in leg.get_texts():
        t.set_color(INK)


def equity_and_drawdown(data):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6.4), dpi=200, height_ratios=[3, 2],
                                      facecolor=SURFACE)
    for key, (label, colour) in SERIES.items():
        pnl = np.asarray(data["daily_pnl"][key])
        equity = np.cumsum(pnl)
        days = np.arange(1, len(pnl) + 1)
        drawdown = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:] - equity
        top.plot(days, equity, color=colour, linewidth=2, label=label)
        end_label(top, days[-1], equity[-1], f"{equity[-1]:,.0f}")
        bottom.plot(days, -drawdown, color=colour, linewidth=1.4, label=label)
    style(top, "Out-of-sample cumulative PnL, 1,000 unseen days", "", "Cumulative PnL")
    legend(top)
    top.set_xlim(0, 1060)
    style(bottom, "Drawdown from previous peak", "Trading day", "Drawdown")
    bottom.set_xlim(0, 1060)
    fig.tight_layout(h_pad=2.0)
    fig.savefig(RESULTS / "equity_and_drawdown.png", facecolor=SURFACE)
    plt.close(fig)


def toxicity_sweep(data):
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=200, facecolor=SURFACE)
    for key, (label, colour) in SERIES.items():
        rows = data["toxicity_sweep"][key]
        x = [r["toxicity"] for r in rows]
        y = [r["sharpe"] for r in rows]
        ax.plot(x, y, color=colour, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        end_label(ax, x[-1], y[-1], f"{y[-1]:.1f}")
    style(ax, "Sharpe ratio as order flow becomes more toxic",
          "Order-flow toxicity (0 = balanced flow, 0.5 = base case)", "Sharpe (annualised)")
    ax.set_xticks([r["toxicity"] for r in data["toxicity_sweep"]["symmetric"]])
    ax.set_xlim(-0.05, 0.85)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    legend(ax, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "toxicity_sweep.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    data = json.loads((RESULTS / "results.json").read_text())
    equity_and_drawdown(data)
    toxicity_sweep(data)
    print("charts written to", RESULTS)
