"""Sell 30-day at-the-money bitcoin straddles every day, delta-hedge daily, and explain each day's P&L by the Greeks.

Each straddle is priced by Black-Scholes at Deribit's DVOL index (30-day implied volatility), with zero interest.
It is re-marked every day at that day's DVOL, which ignores skew and the shape of the term structure.
P&L is in percent of the bitcoin price when the straddle was sold. The book holds 30 straddles of 1/30 size each.
"""
import os
import numpy as np, pandas as pd
from scipy.stats import norm
HERE = os.path.dirname(os.path.abspath(__file__))
pd.set_option("display.width", 220)
DAYS, YEAR = 30, 365.0
ENTRY_COST_VOL, HEDGE_COST = 0.01, 0.0005          # sell 1 vol point below the mark; 5 bps on each hedge trade

d = pd.read_csv(f"{HERE}/data/dvol.csv", parse_dates=["date"]).set_index("date")
s = pd.read_csv(f"{HERE}/data/spot.csv", parse_dates=["date"]); s["date"] = s.date.dt.normalize(); s = s.set_index("date")
df = d.join(s.close.rename("S"), how="inner").dropna(); df["vol"] = df.dvol / 100
S, V, dates = df.S.values, df.vol.values, df.index
ret = np.diff(np.log(S))

def greeks(S, K, T, sig):
    """Value, delta, gamma, vega and theta (per year) of a long straddle."""
    T = np.maximum(T, 1e-9); sq = sig * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sig ** 2 * T) / sq; d2 = d1 - sq
    value = S * (2 * norm.cdf(d1) - 1) - K * (2 * norm.cdf(d2) - 1)
    gamma = 2 * norm.pdf(d1) / (S * sq)
    return value, 2 * norm.cdf(d1) - 1, gamma, 2 * S * norm.pdf(d1) * np.sqrt(T), -S * norm.pdf(d1) * sig / np.sqrt(T)

# ---- 1. implied against realised
realised = np.array([np.sqrt(YEAR * np.mean(ret[i:i + DAYS] ** 2)) if i + DAYS <= len(ret) else np.nan for i in range(len(S))])
ok = ~np.isnan(realised)
gap = V[ok] - realised[ok]
print(f"data: {dates[0].date()} to {dates[-1].date()}, {len(df)} days")
print(f"implied 30-day vol averaged {V[ok].mean():.1%}; the vol that followed averaged {realised[ok].mean():.1%}; implied was higher on {np.mean(gap > 0):.0%} of days (mean gap {gap.mean() * 100:+.1f} points)")

# ---- 2. short straddles, hedged daily, attributed daily
n = len(S); cols = ["total", "theta", "gamma", "vega", "residual", "cost"]
book = np.zeros((n, len(cols))); cycle = []
for i in range(n - DAYS):
    K, S0 = S[i], S[i]; tot = np.zeros(len(cols)); prev_delta = 0.0
    for j in range(DAYS):
        t, u = i + j, i + j + 1
        T_now, T_next = (DAYS - j) / YEAR, (DAYS - j - 1) / YEAR
        v0, delta, gamma, vega, theta = greeks(S[t], K, T_now, V[t])
        v1 = abs(S[u] - K) if j == DAYS - 1 else greeks(S[u], K, T_next, V[u])[0]
        dS, dvol = S[u] - S[t], (V[u] - V[t]) if j < DAYS - 1 else 0.0
        total = -(v1 - v0) + delta * dS                               # short the straddle, long delta shares
        th, ga, ve = -theta / YEAR, -0.5 * gamma * dS ** 2, -vega * dvol
        cost = abs(delta - prev_delta) * S[t] * HEDGE_COST + (vega * ENTRY_COST_VOL if j == 0 else 0.0)
        prev_delta = delta
        row = np.array([total, th, ga, ve, total - th - ga - ve, cost]) / S0 * 100
        book[u] += row / DAYS; tot += row
    cycle.append(tot)
B = pd.DataFrame(book, index=dates, columns=cols).iloc[DAYS:]
C = pd.DataFrame(cycle, columns=cols)
net = B.total - B.cost
print(f"\nper straddle (% of spot at sale), {len(C)} straddles: sold for {np.mean([greeks(S[i], S[i], DAYS / YEAR, V[i])[0] / S[i] * 100 for i in range(n - DAYS)]):.2f} on average")
print(C.mean().round(3).to_string())
print(f"   straddles that made money before costs: {np.mean(C.total > 0):.0%}; worst {C.total.min():.1f}, best {C.total.max():.1f}; attribution error: mean residual {C.residual.mean():+.3f}")

def stats(x, label):
    eq = x.cumsum(); dd = (eq - eq.cummax()).min()
    print(f"{label}: {x.mean() * YEAR:+.1f}% a year, volatility {x.std() * np.sqrt(YEAR):.1f}%, Sharpe {x.mean() / x.std() * np.sqrt(YEAR):.2f}, worst day {x.min():.2f}%, deepest drawdown {dd:.1f}%")
print("\nbook of 30 overlapping straddles, daily P&L in % of notional:")
stats(B.total, "before costs"); stats(net, "after costs ")
ann = B.mean() * YEAR
print(f"yearly attribution: theta {ann.theta:+.1f}, gamma {ann.gamma:+.1f}, vega {ann.vega:+.1f}, residual {ann.residual:+.1f}, total {ann.total:+.1f}  (costs {ann.cost:.1f})")
print(f"daily attribution check: |total - (theta + gamma + vega)| averages {B.residual.abs().mean():.3f}% against a typical daily move of {B.total.abs().mean():.3f}%")

worst = B.nsmallest(6, "total").copy()
worst["btc_move_%"] = [(S[dates.get_loc(t)] / S[dates.get_loc(t) - 1] - 1) * 100 for t in worst.index]
worst["dvol_change"] = [(V[dates.get_loc(t)] - V[dates.get_loc(t) - 1]) * 100 for t in worst.index]
print("\nsix worst days:"); print(worst[["btc_move_%", "dvol_change", "total", "theta", "gamma", "vega", "residual"]].round(2).to_string())
by_year = pd.DataFrame({"before_costs": B.total.groupby(B.index.year).sum(), "after_costs": net.groupby(net.index.year).sum(), "worst_day": B.total.groupby(B.index.year).min()}).round(1)
print("\nby year (% of notional):"); print(by_year.to_string())
B.round(4).to_csv(f"{HERE}/daily_attribution.csv"); by_year.to_csv(f"{HERE}/by_year.csv")

# ---- 3. stress: reprice the whole book for one-day shocks larger than any in the sample
def shock(move, vol_up):
    losses = []
    for t in range(DAYS, n, 7):                                      # the book as it stood on many different days
        pnl = 0.0
        for i in range(t - DAYS + 1, t + 1):                         # straddles sold in the last 30 days
            K, T = S[i], (DAYS - (t - i)) / YEAR
            v0, delta, *_ = greeks(S[t], K, T, V[t])
            v1 = greeks(S[t] * (1 + move), K, max(T - 1 / YEAR, 1e-6), V[t] + vol_up)[0]
            pnl += (-(v1 - v0) + delta * S[t] * move) / S[i] * 100 / DAYS
        losses.append(pnl)
    return np.mean(losses), np.min(losses)
print("\nstress test: one-day P&L of the book, % of notional (average over dates, worst date)")
for move, vol_up, label in ((-0.127, 0.047, "repeat of the worst day in the sample (5 Aug 2024)"), (-0.20, 0.30, "bitcoin -20%, implied vol +30 points"),
                            (-0.40, 0.60, "bitcoin -40%, implied vol +60 points (March 2020 scale)"), (0.25, 0.20, "bitcoin +25%, implied vol +20 points")):
    a, w = shock(move, vol_up); print(f"  {label}: {a:.1f}, worst {w:.1f}")
print(f"  for scale: average yearly profit after costs {net.mean() * YEAR:.1f}")
