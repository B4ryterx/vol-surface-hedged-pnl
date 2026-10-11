# Volatility surface and hedged options P&L

Two questions about bitcoin options on Deribit: is the quoted volatility surface free of arbitrage, and
what does a seller of volatility earn, day by day, once price risk is hedged away?

## Results

**Surface** (snapshot of 11 October 2026, 384 quoted out-of-the-money options in 10 expiries, 2 to 348 days)

| Check | Result |
|---|---|
| SVI fit error | 0.31 volatility points on average; worst expiry 0.73 |
| Butterfly arbitrage (negative implied density) | None in any expiry |
| Calendar arbitrage (total variance falling with expiry) | None in 9 adjacent pairs |
| Shape | At-the-money volatility rises from 27% at 2 days to 39% at a year; puts are dearer than calls at every expiry |

**Selling volatility** (30-day at-the-money straddles sold daily, delta-hedged daily, March 2021 to October 2026)

| | |
|---|---|
| Implied 30-day volatility, average | 60.0% |
| Volatility that followed, average | 51.1% |
| Days on which implied was higher | 73% |
| Profit before costs | +28.2% of notional a year, Sharpe 2.6 |
| Profit after costs | +23.8% a year, Sharpe 2.2 |
| Straddles that made money | 73% |
| Worst day in the sample | -5.4% (5 August 2024, bitcoin -12.7%) |

**Daily attribution** (% of notional a year)

| Theta (time decay earned) | Gamma (cost of price moves) | Vega (change in implied volatility) | Unexplained | Total |
|---|---|---|---|---|
| +87.8 | -66.1 | +2.5 | +4.0 | +28.2 |

The three Greeks explain each day's P&L to within 0.035% on average, against a typical daily move of 0.36%.
Gamma losses are 75% of theta income, which is the ratio of realised to implied variance, (51.1/60)².

**Stress** (one-day repricing of the whole book)

| Shock | Loss, average over dates | Worst date |
|---|---|---|
| Bitcoin -20%, implied volatility +30 points | -11.7% | -21.5% |
| Bitcoin -40%, implied volatility +60 points (March 2020 scale) | -31.6% | -56.8% |
| Bitcoin +25%, implied volatility +20 points | -13.2% | -24.8% |

A single day on the scale of March 2020 would cost more than a year of profit. The sample contains no such day,
so its Sharpe ratio overstates the trade.

## Method

- `fetch.py` pulls Deribit's public data: the DVOL index (30-day implied volatility), daily bitcoin prices, and the option chain.
- `surface.py` fits raw SVI to each expiry's total variance by least squares and evaluates Gatheral's butterfly
  condition and the calendar condition on a grid.
- `hedge.py` prices each straddle by Black-Scholes at DVOL with zero interest, hedges at each day's close with the
  Black-Scholes delta, and splits the daily P&L into theta, gamma and vega terms plus a residual. Costs: straddles are
  sold one volatility point below the mark, and each hedge trade pays 5 bps.

## Limits

- **One volatility number per day.** Every straddle is re-marked at DVOL, the at-the-money 30-day figure. A real option
  drifts away from the money and shortens, and its own implied volatility differs; after large moves this understates losses.
- **Daily hedging at the close.** Moves within the day are not hedged and do not appear.
- **Timestamps differ by eight hours.** Price bars close at 08:00 UTC and DVOL at 00:00 UTC.
- **No crash in the sample.** See the stress table.
- **Costs are flat assumptions**, and margin requirements and liquidation risk are ignored.

## Run

```
python3 fetch.py && python3 surface.py && python3 hedge.py
```

Needs Python 3 with numpy, pandas and scipy. Data files are not in the repository.
