"""Market-making simulator.

The mid-price follows arithmetic Brownian motion with occasional jumps. Each
step, a market maker posts a bid and an ask. Market orders arrive as Poisson
processes whose intensity decays exponentially with the quote's distance from
the mid, lambda(d) = A * exp(-k * d), as in Avellaneda & Stoikov (2008).

Real-market frictions that the original model leaves out:
  * latency: quotes are set from the current mid, but the price moves before
    they can be updated. A quote the price moves through is taken at once by
    faster traders, at the quote's own (now stale) price,
  * competition: the rest of the market quotes at a common half-spread, set to
    the level at which a naive market maker just breaks even (the zero-profit
    condition of a competitive market). Rivals have the same latency.
    Ordinary orders only reach quotes at or inside the market's best price,
  * toxic flow: hidden trending regimes in which order flow is one-sided and
    the price drifts with it,
  * news jumps, and volatility that changes from day to day,
  * a fee on every fill, a hard inventory limit, and a cost to flatten any
    inventory at the close.

Every fill happens at the quote's own price, never better. All sessions run in
parallel with NumPy, and every strategy sees the same random draws for the
same seed, so differences between strategies are not simulation noise.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Market:
    """The simulated market. One session is one trading day.

    Arrival parameters (A, k) follow Avellaneda & Stoikov (2008). Volatility,
    jumps and flow imbalance are set to plausible levels for a liquid stock;
    market_half_spread is normally replaced by the zero-profit level.
    """

    s0: float = 100.0              # starting mid-price
    sigma: float = 2.0             # average daily volatility
    vol_of_vol: float = 0.4        # day-to-day dispersion of volatility (lognormal)
    jump_rate: float = 2.0         # news jumps per day
    jump_size: float = 1.0         # standard deviation of a jump
    horizon: float = 1.0           # session length T
    n_steps: int = 200             # steps per session, dt = T / n_steps
    arrival_a: float = 140.0       # order-arrival scale A
    arrival_k: float = 1.5         # order-arrival decay k
    trend_drift: float = 5.0       # price drift per day while a trend is on
    toxicity: float = 0.5          # in a trend, buys x(1 + b) and sells x(1 - b)
    regime_switches: float = 5.0   # expected regime redraws per day
    market_half_spread: float = 0.2  # where the rest of the market quotes
    fee: float = 0.02              # cost per unit filled
    max_inventory: int = 25        # hard position limit, in units
    liquidation_cost: float = 0.25 # per-unit cost to flatten inventory at the close


@dataclass(frozen=True)
class Symmetric:
    """Baseline: quote a fraction of the market's half-spread either side of
    the mid, ignoring inventory. A ratio of 1 joins the market's best price;
    below 1 improves on it."""

    spread_ratio: float

    name = "Symmetric baseline"

    def quotes(self, s, q, tau, market_half_spread):
        h = self.spread_ratio * market_half_spread
        return s - h, s + h


@dataclass(frozen=True)
class InventoryAware:
    """Quote the same way, but centred on the Avellaneda-Stoikov reservation
    price r = s - q * gamma * sigma^2 * (T - t), which leans against
    inventory: a long book quotes lower, a short book higher.

    The paper's optimal-spread formula assumes no competition; against rivals
    quoting a competitive spread it prices itself out of the market. So, as in
    practice, the spread is set relative to the market and only the skew comes
    from the model. sigma is the volatility the strategy assumes, fixed at
    training time.
    """

    spread_ratio: float
    gamma: float                   # risk aversion
    sigma: float                   # volatility the strategy assumes

    name = "Inventory-aware (Avellaneda-Stoikov skew)"

    def quotes(self, s, q, tau, market_half_spread):
        reservation = s - q * self.gamma * self.sigma**2 * tau
        h = self.spread_ratio * market_half_spread
        return reservation - h, reservation + h


@dataclass
class Sessions:
    """Per-session outcomes, one entry per simulated session."""

    pnl: np.ndarray                 # after fees, adverse selection and close-out
    fees: np.ndarray                # fees paid, including on close-out
    fills: np.ndarray               # units filled
    mean_abs_inventory: np.ndarray  # time-averaged |inventory|
    max_abs_inventory: np.ndarray


def simulate(strategy, market: Market, n_sessions: int, seed: int) -> Sessions:
    rng = np.random.default_rng(seed)
    n = n_sessions
    dt = market.horizon / market.n_steps
    p_switch = market.regime_switches * dt

    # Each day's volatility is drawn around the average; the strategy never sees it.
    day_sigma = market.sigma * np.exp(market.vol_of_vol * rng.standard_normal(n)
                                      - 0.5 * market.vol_of_vol**2)

    s = np.full(n, market.s0)
    regime = np.zeros(n)           # -1 down-trend, 0 flat, +1 up-trend (hidden)
    q = np.zeros(n)
    cash = np.zeros(n)
    fees = np.zeros(n)
    fills = np.zeros(n)
    abs_inv_total = np.zeros(n)
    max_abs_inv = np.zeros(n)

    for step in range(market.n_steps):
        tau = market.horizon - step * dt
        bid, ask = strategy.quotes(s, q, tau, market.market_half_spread)

        # Fixed number of draws per step regardless of strategy, so every
        # strategy sees the same random numbers for the same seed.
        u_bid, u_ask, u_switch, u_regime, u_jump = rng.random((5, n))
        z, z_jump = rng.standard_normal((2, n))

        # Latency: the price moves before anyone's quotes can be updated.
        jump = (u_jump < market.jump_rate * dt) * market.jump_size * z_jump
        s_new = s + market.trend_drift * regime * dt + day_sigma * np.sqrt(dt) * z + jump

        # Price priority: ordinary orders reach a quote only if it is at or
        # inside the market's best price (rivals' quotes are equally stale).
        bid_competitive = (s - bid) <= market.market_half_spread + 1e-12
        ask_competitive = (ask - s) <= market.market_half_spread + 1e-12

        # Sell orders hit our bid, buy orders lift our ask. In an up-trend
        # buyers dominate; in a down-trend sellers do.
        d_bid = s_new - bid
        d_ask = ask - s_new
        sell_flow = np.clip(1.0 - market.toxicity * regime, 0.0, None)
        buy_flow = np.clip(1.0 + market.toxicity * regime, 0.0, None)
        p_bid = bid_competitive * (1.0 - np.exp(
            -market.arrival_a * sell_flow * np.exp(-market.arrival_k * np.maximum(d_bid, 0.0)) * dt))
        p_ask = ask_competitive * (1.0 - np.exp(
            -market.arrival_a * buy_flow * np.exp(-market.arrival_k * np.maximum(d_ask, 0.0)) * dt))

        # A stale quote the price has moved through is taken at once.
        p_bid = np.where(d_bid < 0, 1.0, p_bid)
        p_ask = np.where(d_ask < 0, 1.0, p_ask)

        # Never trade beyond the inventory limit. Fills are at our own price.
        bought = (u_bid < p_bid) & (q < market.max_inventory)
        sold = (u_ask < p_ask) & (q > -market.max_inventory)
        q += bought
        q -= sold
        cash += sold * ask - bought * bid

        n_filled = bought.astype(float) + sold
        cash -= n_filled * market.fee
        fees += n_filled * market.fee
        fills += n_filled

        # Advance the price, then the hidden regime may be redrawn (half the
        # time to flat, a quarter each to up or down).
        s = s_new
        new_regime = np.where(u_regime < 0.25, -1.0, np.where(u_regime < 0.75, 0.0, 1.0))
        regime = np.where(u_switch < p_switch, new_regime, regime)

        abs_inv_total += np.abs(q)
        np.maximum(max_abs_inv, np.abs(q), out=max_abs_inv)

    # Flatten at the close: mark inventory to the final mid, then pay the
    # liquidation cost and the fee on every unit closed out.
    close_out = np.abs(q) * (market.liquidation_cost + market.fee)
    fees += np.abs(q) * market.fee
    pnl = cash + q * s - close_out

    return Sessions(
        pnl=pnl,
        fees=fees,
        fills=fills,
        mean_abs_inventory=abs_inv_total / market.n_steps,
        max_abs_inventory=max_abs_inv,
    )


def zero_profit_half_spread(market: Market, seed: int = 0, n_sessions: int = 2000,
                            iterations: int = 30) -> float:
    """The half-spread at which a naive (symmetric) market maker breaks even on
    average. Competition pushes the market's spread down to this level: any
    wider and new market makers would undercut it; any tighter and they would
    lose money. Found by bisection on a dedicated seed."""

    def mean_pnl(h: float) -> float:
        m = replace(market, market_half_spread=h)
        return float(simulate(Symmetric(1.0), m, n_sessions, seed).pnl.mean())

    lo, hi = 0.01, 5.0
    if not mean_pnl(lo) < 0 < mean_pnl(hi):
        raise ValueError("no zero-profit spread in range")
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if mean_pnl(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def competitive(market: Market, seed: int = 0) -> Market:
    """The same market with its spread set to the zero-profit level."""
    return replace(market, market_half_spread=zero_profit_half_spread(market, seed))
