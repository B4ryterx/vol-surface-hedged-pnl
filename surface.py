"""Fit an SVI curve to each expiry of the bitcoin option chain and check the surface for arbitrage.

Raw SVI gives total implied variance w (variance x time to expiry) as a function of log-moneyness k = ln(K / F):
    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + s^2) )
Checks:  butterfly  g(k) >= 0 (Gatheral's condition: the implied density is non-negative)
         calendar   w increases with expiry at every k (no negative forward variance)
"""
import json, os
import numpy as np, pandas as pd
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
pd.set_option("display.width", 200)

raw = json.load(open(f"{HERE}/data/chain.json"))
t0 = pd.Timestamp(raw["fetched_ms"], unit="ms")
rows = []
for o in raw["options"]:
    _, exp, strike, kind = o["instrument_name"].split("-")
    expiry = pd.to_datetime(exp, format="%d%b%y") + pd.Timedelta(hours=8)          # Deribit options expire at 08:00 UTC
    rows.append(dict(expiry=expiry, T=(expiry - t0).total_seconds() / (365 * 86400), K=float(strike), kind=kind,
                     F=o["underlying_price"], iv=(o["mark_iv"] or 0) / 100, bid=o["bid_price"] or 0, oi=o["open_interest"]))
c = pd.DataFrame(rows)
c["k"] = np.log(c.K / c.F)
c = c[(c.iv > 0) & (c.bid > 0) & (c["T"] > 2 / 365) & (((c.kind == "C") & (c.k >= 0)) | ((c.kind == "P") & (c.k < 0)))]   # quoted, out-of-the-money
c["w"] = c.iv ** 2 * c["T"]

def svi(p, k):
    a, b, rho, m, s = p
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))

def g(p, k):                                                                      # butterfly condition
    a, b, rho, m, s = p
    w = svi(p, k); r = np.sqrt((k - m) ** 2 + s ** 2)
    w1 = b * (rho + (k - m) / r); w2 = b * s ** 2 / r ** 3
    return (1 - k * w1 / (2 * w)) ** 2 - w1 ** 2 / 4 * (1 / w + 0.25) + w2 / 2

fits, out = {}, []
for expiry, d in c.groupby("expiry"):
    if len(d) < 6: continue
    k, w = d.k.values, d.w.values
    start = [w.min() * 0.9, 0.1, -0.3, 0.0, 0.1]
    res = least_squares(lambda p: svi(p, k) - w, start, bounds=([-1, 1e-6, -0.999, -1, 1e-4], [1, 5, 0.999, 1, 3]))
    p = res.x; T = d["T"].iloc[0]
    fit_iv = np.sqrt(np.maximum(svi(p, k), 1e-12) / T)
    grid = np.linspace(k.min(), k.max(), 400)
    atm = np.sqrt(svi(p, 0.0) / T)
    skew = (np.sqrt(svi(p, -0.1) / T) - np.sqrt(svi(p, 0.1) / T))
    fits[expiry] = (p, T, k.min(), k.max())
    out.append(dict(expiry=expiry.date(), days=round(T * 365, 1), strikes=len(d), atm_vol=round(atm * 100, 1), put_minus_call_skew=round(skew * 100, 1),
                    rmse_vol_points=round(np.sqrt(np.mean((fit_iv - d.iv.values) ** 2)) * 100, 2), max_error=round(np.abs(fit_iv - d.iv.values).max() * 100, 2),
                    butterfly_violations=int((g(p, grid) < 0).sum()), min_g=round(g(p, grid).min(), 3)))
tab = pd.DataFrame(out)
exps = sorted(fits); cal = []
for e1, e2 in zip(exps, exps[1:]):
    p1, T1, lo1, hi1 = fits[e1]; p2, T2, lo2, hi2 = fits[e2]
    grid = np.linspace(max(lo1, lo2), min(hi1, hi2), 300)
    cal.append(int((svi(p2, grid) < svi(p1, grid)).sum()))
tab["calendar_violations_vs_next"] = cal + [np.nan]
print(f"snapshot {t0:%Y-%m-%d %H:%M} UTC | underlying about {c.F.median():,.0f} | {len(c)} quoted out-of-the-money options in {len(tab)} expiries\n")
print(tab.to_string(index=False))
print(f"\nmean fit error {tab.rmse_vol_points.mean():.2f} vol points | expiries with any butterfly violation: {(tab.butterfly_violations > 0).sum()} | adjacent pairs with any calendar violation: {int((np.array(cal) > 0).sum())} of {len(cal)}")
tab.to_csv(f"{HERE}/surface_fit.csv", index=False)
