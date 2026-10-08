"""Calibrate in-sample, then evaluate out-of-sample and under stress.

Usage: python run.py

Every number in the README comes from this script. Seeds are fixed, so the
results reproduce exactly:
  seed 0  market set-up       (finds the competitive, zero-profit spread)
  seed 1  training sessions   (strategy parameters are chosen here, nowhere else)
  seed 2  out-of-sample test  (never seen during calibration)
  seed 3  stress test         (volatility +50%: the market re-prices its spread,
                               the strategies keep their trained settings)

The assumption sweep re-runs the whole pipeline (market set-up, calibration and
out-of-sample test) for each number of market makers sharing the best price and
each quote-update speed, because those two assumptions drive the Sharpe ratio.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from mmsim import InventoryAware, Market, Symmetric, competitive, simulate, summarise

RESULTS = Path(__file__).parent / "results"
MARKET_SEED, TRAIN_SEED, TEST_SEED, STRESS_SEED = 0, 1, 2, 3
N_TRAIN, N_TEST = 500, 1000          # 500 training days; 1,000 test days, about 4 years

RATIOS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25]     # quote as a fraction of the market's half-spread
GAMMAS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]     # risk aversion for the inventory skew
TOXICITIES = [0.0, 0.25, 0.5, 0.75]
MAKERS = [1, 2, 3, 4, 5]          # market makers sharing the best price (queue share = 1 / makers)
REFRESH = [1, 2, 3, 4]            # steps between quote updates


def calibrate(market: Market, capital: float):
    """Pick each strategy's parameters by Sharpe ratio on the training days only."""
    sym_grid = [
        {"spread_ratio": r, **summarise(simulate(Symmetric(r), market, N_TRAIN, TRAIN_SEED), capital)}
        for r in RATIOS
    ]
    inv_grid = [
        {"spread_ratio": r, "gamma": g,
         **summarise(simulate(InventoryAware(r, g, market.sigma), market, N_TRAIN, TRAIN_SEED), capital)}
        for r in RATIOS for g in GAMMAS
    ]
    best_sym = max(sym_grid, key=lambda row: row["sharpe"])
    best_inv = max(inv_grid, key=lambda row: row["sharpe"])
    return best_sym, best_inv, sym_grid, inv_grid


def assumption_sweep(market: Market):
    """Out-of-sample Sharpe for each queue share and update speed, with the
    market re-priced and both strategies re-calibrated every time."""
    rows = []
    for makers in MAKERS:
        for refresh in REFRESH:
            m = competitive(replace(market, queue_share=1 / makers, refresh_steps=refresh), seed=MARKET_SEED)
            capital = m.max_inventory * m.s0
            best_sym, best_inv, _, _ = calibrate(m, capital)
            inv = summarise(simulate(InventoryAware(best_inv["spread_ratio"], best_inv["gamma"], m.sigma),
                                     m, N_TEST, TEST_SEED), capital)
            sym = summarise(simulate(Symmetric(best_sym["spread_ratio"]), m, N_TEST, TEST_SEED), capital)
            rows.append({"makers": makers, "refresh_steps": refresh,
                         "market_half_spread": m.market_half_spread,
                         "inventory_aware_sharpe": inv["sharpe"], "symmetric_sharpe": sym["sharpe"]})
    return rows


def main():
    RESULTS.mkdir(exist_ok=True)
    market = competitive(Market(), seed=MARKET_SEED)
    capital = market.max_inventory * market.s0

    best_sym, best_inv, sym_grid, inv_grid = calibrate(market, capital)
    # Strategy parameters are frozen here, including the volatility they assume.
    strategies = {
        "inventory_aware": InventoryAware(best_inv["spread_ratio"], best_inv["gamma"], market.sigma),
        "symmetric": Symmetric(best_sym["spread_ratio"]),
    }

    stressed = competitive(replace(market, sigma=market.sigma * 1.5), seed=MARKET_SEED)
    sweep = {b: competitive(replace(market, toxicity=b), seed=MARKET_SEED) for b in TOXICITIES}

    out = {
        "market": asdict(market),
        "capital": capital,
        "chosen": {
            "inventory_aware": {"spread_ratio": best_inv["spread_ratio"], "gamma": best_inv["gamma"]},
            "symmetric": {"spread_ratio": best_sym["spread_ratio"]},
        },
        "calibration": {"inventory_aware": inv_grid, "symmetric": sym_grid},
        "out_of_sample": {},
        "stress": {"market_half_spread": stressed.market_half_spread},
        "toxicity_sweep": {"market_half_spread": {str(b): m.market_half_spread for b, m in sweep.items()}},
        "daily_pnl": {},
    }

    for name, strat in strategies.items():
        test = simulate(strat, market, N_TEST, TEST_SEED)
        out["out_of_sample"][name] = summarise(test, capital)
        out["daily_pnl"][name] = test.pnl.round(4).tolist()
        out["stress"][name] = summarise(simulate(strat, stressed, N_TEST, STRESS_SEED), capital)
        out["toxicity_sweep"][name] = [
            {"toxicity": b, **summarise(simulate(strat, m, N_TEST, TEST_SEED), capital)}
            for b, m in sweep.items()
        ]

    out["assumption_sweep"] = assumption_sweep(market)
    (RESULTS / "results.json").write_text(json.dumps(out, indent=2))

    oos, st = out["out_of_sample"], out["stress"]
    print(f"Competitive market half-spread: {market.market_half_spread:.3f}")
    print(f"Chosen in-sample: inventory-aware ratio={best_inv['spread_ratio']}, gamma={best_inv['gamma']}; "
          f"symmetric ratio={best_sym['spread_ratio']}\n")
    print(f"{'':28}{'Inventory-aware':>18}{'Symmetric':>14}")
    for label, key, fmt in [
        ("Mean daily PnL", "mean_pnl", "{:.2f}"),
        ("Daily PnL std", "std_pnl", "{:.2f}"),
        ("Sharpe (annualised)", "sharpe", "{:.2f}"),
        ("Max drawdown", "max_drawdown", "{:.0f}"),
        ("Max drawdown, % capital", "max_drawdown_pct", "{:.1f}%"),
        ("Losing days", "losing_days_pct", "{:.1f}%"),
        ("Fees per day", "fees_per_day", "{:.2f}"),
        ("Fills per day", "fills_per_day", "{:.1f}"),
        ("Mean |inventory|", "mean_abs_inventory", "{:.2f}"),
    ]:
        print(f"{label:28}{fmt.format(oos['inventory_aware'][key]):>18}{fmt.format(oos['symmetric'][key]):>14}")
    print(f"\nStress (volatility +50%, market half-spread re-priced to {stressed.market_half_spread:.3f}):")
    for key in ["mean_pnl", "sharpe", "max_drawdown", "losing_days_pct"]:
        print(f"  {key:22}{st['inventory_aware'][key]:>12.2f}{st['symmetric'][key]:>12.2f}")
    print("\nToxicity sweep (Sharpe):")
    for a, s in zip(out["toxicity_sweep"]["inventory_aware"], out["toxicity_sweep"]["symmetric"]):
        print(f"  toxicity {a['toxicity']:<5}{a['sharpe']:>10.2f}{s['sharpe']:>10.2f}")
    print("\nAssumption sweep (inventory-aware Sharpe; rows = market makers at the best price,"
          " columns = steps between quote updates):")
    print("       " + "".join(f"{r:>8}" for r in REFRESH))
    for makers in MAKERS:
        cells = [row for row in out["assumption_sweep"] if row["makers"] == makers]
        print(f"  {makers:>4} " + "".join(f"{c['inventory_aware_sharpe']:>8.2f}" for c in cells))


if __name__ == "__main__":
    main()
