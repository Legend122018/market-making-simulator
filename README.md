# Market-Making Simulator

A Python simulation of a market maker trading a single instrument in a competitive market. The market includes latency, slow quote updates, queue priority, toxic order flow, news jumps, changing volatility and fees. It tests whether leaning quotes against inventory, using the Avellaneda–Stoikov reservation price, beats naive symmetric quoting. Parameters are tuned on training days only; results come from 1,000 unseen days plus a volatility stress test.

## Headline result

Out of sample, over 1,000 unseen simulated trading days and after fees and close-out costs:

| | Inventory-aware | Symmetric baseline |
|---|---:|---:|
| Mean daily PnL | 1.00 | -0.90 |
| Daily PnL standard deviation | 7.33 | 20.96 |
| Sharpe ratio (annualised) | 2.2 | -0.7 |
| Max drawdown | 109 | 944 |
| Max drawdown, % of capital | 4.4% | 37.8% |
| Losing days | 33.9% | 36.4% |
| Fees paid per day | 0.64 | 0.77 |
| Units filled per day | 30 | 32 |
| Mean absolute inventory | 1.02 | 4.10 |

The naive market maker roughly breaks even (its -0.90 a day is within 1.4 standard errors of zero), which is what a competitive market should produce. Leaning quotes against inventory turns that into a small, steady profit: a Sharpe ratio of 2.2 and about 10% a year on capital, with 88% smaller maximum drawdown and 75% less inventory carried. Capital is the notional the position limit allows: 25 units at a price of 100, so 2,500.

![Out-of-sample cumulative PnL and drawdown](results/equity_and_drawdown.png)

## The model

**Price.** The mid-price follows Brownian motion with an average daily volatility of 2. Each day's volatility is drawn around that average, spread lognormally by 0.4, and the strategy never sees it. News jumps arrive about twice a day, each with a standard deviation of 1. Each day has 200 time steps.

**Order arrival.** Market orders arrive as Poisson processes. The chance that a quote at distance *d* from the mid fills falls as `A·exp(−k·d)`, with A = 140 and k = 1.5, as in Avellaneda & Stoikov (2008).

**Latency.** Quotes are set from the current mid, but the price moves before they can be updated. If the price moves through a stale quote, faster traders take it at once, at the quote's own price. This is the main way real market makers lose money, and every fill in the model happens at the quote's own price, never better.

**Update speed.** Quotes are refreshed only every third step, while the price moves every step, so they spend longer stale and are picked off more often.

**Queue priority.** Four market makers quote at the best price, and ordinary orders are shared between them, so each gets a quarter of the harmless flow. A quote the price moves through is taken in full, whatever its place in the queue. That asymmetry is what makes market making hard: you get a share of the good trades and all of the bad ones.

**Competition.** The rest of the market quotes a common spread, with the same latency and update speed. Ordinary orders only reach quotes at or inside the market's best price. The market's spread is set by the **zero-profit condition**: it is the half-spread at which a naive market maker just breaks even, found by bisection, and comes out at 0.510. Any wider and new market makers would undercut it; any tighter and they would lose money.

**Toxic flow.** The market is always in one of three hidden regimes: up-trend, down-trend or flat. The regime is redrawn about five times a day: flat half the time, up or down a quarter each. In an up-trend the price drifts up by 5 per day, buy orders arrive 1.5 times as fast as usual and sell orders half as fast. A market maker who keeps selling into that flow ends up short just as the price rises.

**Costs and limits.** Every fill pays a fee of 0.02, about 4% of the half-spread. Inventory is capped at ±25 units. At the close, any inventory is flattened at the mid, paying 0.25 per unit plus the fee.

## The strategies

Both strategies quote a fraction of the market's half-spread either side of a centre price. A fraction of 1 joins the market's best price; less than 1 improves on it.

**Symmetric baseline.** Centred on the mid, ignoring inventory.

**Inventory-aware.** Centred on the Avellaneda–Stoikov reservation price instead of the mid:

```
r = s − q·γ·σ²·(T − t)
```

Here *s* is the mid, *q* the inventory, *γ* the risk aversion, *σ* the volatility the strategy assumes and *T − t* the time left in the day. When long, both quotes shift down, so the bid drops behind the market and stops buying while the ask becomes more competitive. When short, the reverse.

The original paper also gives an optimal *spread*, but that formula assumes no competition. Against rivals quoting a competitive spread it prices itself out of the market. So, as in practice, the spread is set relative to the market and only the skew comes from the model.

## Method

1. **Set up the market (seed 0).** Find the zero-profit spread.
2. **Calibrate (training data only, seed 1).** Each strategy's parameters are chosen by Sharpe ratio on 500 training days. The spread fraction was searched from 0.5 to 1.25 and γ from 0.002 to 0.2. Both strategies chose to join the market's best price (fraction 1.0); the inventory-aware strategy chose γ = 0.05, inside the search range.
3. **Freeze.** The strategies keep these settings, including the volatility they assume.
4. **Test out of sample (seed 2).** Both strategies run on 1,000 days they have never seen. They get identical random draws, so the comparison isn't simulation noise.
5. **Stress test (seed 3).** Volatility rises 50%. The market re-prices its spread to the new zero-profit level, 0.655, as a competitive market would. The strategies keep their trained settings and are not told volatility has changed.
6. **Sensitivity.** The out-of-sample test is repeated at four levels of order-flow toxicity, and the whole pipeline is re-run for each queue share and update speed (below).

## Why the Sharpe ratio is 2, not 20

Earlier versions of this model reported Sharpe ratios of 28.6 and then 20.2. That was too good to be true for a single instrument, and two assumptions were doing most of the work: the market maker received every ordinary order that reached the best price, and its quotes updated as fast as the price moved. On a real exchange it would share those orders with other market makers in the queue while still being picked off in full when the price jumps through its quote, and a Python system updates its quotes far more slowly than the fastest firms. Adding both brought the Sharpe ratio down to 2.2.

Because the headline number depends so much on these two assumptions, the table re-runs the whole pipeline (market set-up, calibration and out-of-sample test) for each combination. Each cell is the inventory-aware strategy's out-of-sample Sharpe ratio:

| Market makers at the best price | Quotes refreshed every step | Every 2 steps | Every 3 steps | Every 4 steps |
|---|---:|---:|---:|---:|
| 1 (all the flow) | 20.2 | 11.9 | 7.3 | 5.7 |
| 2 | 14.4 | 7.7 | 4.9 | 3.6 |
| 3 | 9.8 | 5.2 | 3.5 | 2.7 |
| 4 | 7.1 | 3.8 | **2.2** | 1.8 |
| 5 | 5.4 | 2.8 | 0.5 | 1.3 |

Across the settings I consider plausible (three to five market makers, quotes refreshed every two to four steps), the Sharpe ratio ranges from 0.5 to 5.2, and the base case sits in the middle. What holds in every one of the 20 cells is the comparison: the inventory-aware strategy beats the naive baseline, whose Sharpe ratio never rises above 0.3.

## Stress and sensitivity

| Stress: volatility +50% | Inventory-aware | Symmetric baseline |
|---|---:|---:|
| Mean daily PnL | -0.19 | -0.82 |
| Sharpe ratio (annualised) | -0.3 | -0.5 |
| Max drawdown | 387 | 1,167 |
| Losing days | 37.4% | 35.6% |

When volatility jumps 50% and the strategy keeps its trained settings, its small edge disappears: it loses slightly, though less than the baseline, and its maximum drawdown is a third of the baseline's. Its skew still assumes the old volatility. A real desk would re-estimate volatility every day, so frozen settings make this a deliberately harsh test.

![Sharpe ratio as order flow becomes more toxic](results/toxicity_sweep.png)

**The skew only pays when order flow is informative.** With balanced flow (toxicity 0), the inventory-aware strategy loses money: Sharpe −2.1 against −1.2 for the baseline. Skewing then costs edge for no benefit, because inventory carries no information about where the price is going. As flow becomes more one-sided, inventory becomes a signal of the hidden trend, and leaning against it pays more: at toxicity 0.75 the Sharpe ratio is 2.8.

## Limitations

- **Synthetic prices.** Nothing is fitted to real market data. The natural next step is to estimate A, k, volatility and the competitive spread from real order-book data.
- **Rivals don't skew.** The market's spread is set by naive market makers. Against rivals who also manage inventory, spreads would be tighter and the edge smaller.
- **Average queue position.** Each market maker gets an equal share of the flow at the best price. On a real exchange the share depends on when each quote joined the queue.
- **Poisson arrivals.** Real order flow clusters. A Hawkes process would be more realistic.
- **One instrument, one unit per fill.** No hedging across correlated assets, and no large orders that sweep several price levels.

## Running it

```
pip install -r requirements.txt
python run.py       # market set-up, calibration, out-of-sample, stress and sweeps -> results/results.json (about 90 seconds)
python charts.py    # charts -> results/
python -m pytest    # model tests
```

The seeds are fixed, so every number above reproduces exactly.
