"""Performance metrics over a sequence of sessions (one session = one trading day)."""

from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def sharpe(pnl: np.ndarray) -> float:
    """Annualised Sharpe ratio of daily PnL (risk-free rate taken as zero)."""
    return float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(pnl: np.ndarray) -> float:
    """Largest peak-to-trough fall in cumulative PnL, sessions taken in order."""
    equity = np.concatenate([[0.0], np.cumsum(pnl)])
    return float((np.maximum.accumulate(equity) - equity).max())


def summarise(sessions, capital: float) -> dict:
    """Headline statistics. Returns are measured on `capital`, the notional
    the position limit allows (max inventory x starting price)."""
    pnl = sessions.pnl
    return {
        "mean_pnl": float(pnl.mean()),
        "std_pnl": float(pnl.std(ddof=1)),
        "annual_return_pct": float(pnl.mean() * TRADING_DAYS / capital * 100),
        "sharpe": sharpe(pnl),
        "max_drawdown": max_drawdown(pnl),
        "max_drawdown_pct": max_drawdown(pnl) / capital * 100,
        "losing_days_pct": float((pnl < 0).mean() * 100),
        "fees_per_day": float(sessions.fees.mean()),
        "fills_per_day": float(sessions.fills.mean()),
        "mean_abs_inventory": float(sessions.mean_abs_inventory.mean()),
    }
