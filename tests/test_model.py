from dataclasses import replace

import numpy as np

from mmsim import (InventoryAware, Market, Symmetric, max_drawdown, simulate,
                   zero_profit_half_spread)

M = Market()
# A market with no price moves of any kind, for testing the mechanics in isolation.
STILL = replace(M, sigma=0.0, vol_of_vol=0.0, jump_rate=0.0, trend_drift=0.0, toxicity=0.0,
                fee=0.0, liquidation_cost=0.0, market_half_spread=0.5)


def test_same_seed_reproduces_exactly():
    strat = InventoryAware(spread_ratio=1.0, gamma=0.02, sigma=M.sigma)
    a = simulate(strat, M, 50, seed=7)
    b = simulate(strat, M, 50, seed=7)
    assert np.array_equal(a.pnl, b.pnl)


def test_inventory_never_breaches_limit():
    tight = replace(M, max_inventory=3)
    for strat in (Symmetric(0.5), InventoryAware(0.5, 0.005, M.sigma)):
        res = simulate(strat, tight, 200, seed=3)
        assert res.max_abs_inventory.max() <= tight.max_inventory


def test_still_market_only_earns_the_spread():
    # Nothing moves the price, so every round trip earns the spread and no
    # session can lose money.
    res = simulate(Symmetric(1.0), STILL, 200, seed=5)
    assert (res.pnl >= -1e-9).all()
    assert res.pnl.mean() > 0


def test_quote_behind_the_market_gets_no_ordinary_fills():
    res = simulate(Symmetric(1.5), STILL, 200, seed=5)
    assert res.fills.sum() == 0


def test_stale_quotes_are_picked_off_by_jumps():
    # The same quoter, with and without news jumps: jumps through stale
    # quotes can only cost money.
    jumpy = replace(STILL, jump_rate=20.0, jump_size=1.0)
    calm = simulate(Symmetric(1.0), STILL, 400, seed=9).pnl.mean()
    shocked = simulate(Symmetric(1.0), jumpy, 400, seed=9).pnl.mean()
    assert shocked < calm


def test_zero_profit_spread_breaks_even():
    h = zero_profit_half_spread(M, seed=0)
    pnl = simulate(Symmetric(1.0), replace(M, market_half_spread=h), 2000, seed=0).pnl
    assert abs(pnl.mean()) < 0.05


def test_reservation_price_leans_against_inventory():
    strat = InventoryAware(spread_ratio=1.0, gamma=0.2, sigma=2.0)
    s = np.array([100.0, 100.0])
    bid, ask = strat.quotes(s, q=np.array([5.0, -5.0]), tau=1.0, market_half_spread=0.2)
    centre = (bid + ask) / 2
    assert centre[0] < 100.0 < centre[1]  # long quotes lower, short quotes higher


def test_max_drawdown():
    assert max_drawdown(np.array([5.0, -3.0, -4.0, 10.0, -2.0])) == 7.0


def test_sharing_the_queue_cuts_ordinary_fills():
    # In a still market every fill is an ordinary one, so sharing the best
    # price with three rivals should cut fills to roughly a quarter.
    alone = simulate(Symmetric(1.0), replace(STILL, queue_share=1.0, refresh_steps=1), 400, seed=11)
    shared = simulate(Symmetric(1.0), replace(STILL, queue_share=0.25, refresh_steps=1), 400, seed=11)
    ratio = shared.fills.mean() / alone.fills.mean()
    assert 0.2 < ratio < 0.35


def test_slower_quote_updates_cost_money():
    # Same market and spread: quotes refreshed less often go stale and are
    # picked off more, so the same quoter earns less.
    moving = replace(M, market_half_spread=0.4, queue_share=1.0)
    fast = simulate(Symmetric(1.0), replace(moving, refresh_steps=1), 1000, seed=13).pnl.mean()
    slow = simulate(Symmetric(1.0), replace(moving, refresh_steps=4), 1000, seed=13).pnl.mean()
    assert slow < fast
