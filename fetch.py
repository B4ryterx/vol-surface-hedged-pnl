"""Fetch public Deribit data for bitcoin options:
  data/dvol.csv     daily DVOL, Deribit's 30-day implied volatility index (percent a year)
  data/spot.csv     daily BTC perpetual prices
  data/chain.json   a snapshot of every listed BTC option with its mark implied volatility
"""
import json, os, time, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://www.deribit.com/api/v2/public/"

def get(method, **params):
    url = API + method + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=30))["result"]

now = int(time.time() * 1000)
start = int(pd.Timestamp("2021-03-01").timestamp() * 1000)

rows, end = [], now
while True:
    r = get("get_volatility_index_data", currency="BTC", start_timestamp=start, end_timestamp=end, resolution="1D")
    rows = r["data"] + rows
    if not r.get("continuation") or not r["data"]: break
    end = r["continuation"]; time.sleep(0.2)
dvol = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"]).drop_duplicates("ts").sort_values("ts")
dvol["date"] = pd.to_datetime(dvol.ts, unit="ms"); dvol[["date", "close"]].rename(columns={"close": "dvol"}).to_csv(f"{HERE}/data/dvol.csv", index=False)

frames, s = [], start
while s < now:
    e = min(s + 400 * 86_400_000, now)
    r = get("get_tradingview_chart_data", instrument_name="BTC-PERPETUAL", start_timestamp=s, end_timestamp=e, resolution="1D")
    if r.get("ticks"): frames.append(pd.DataFrame({"ts": r["ticks"], "close": r["close"], "high": r["high"], "low": r["low"]}))
    s = e + 1; time.sleep(0.2)
spot = pd.concat(frames).drop_duplicates("ts").sort_values("ts"); spot["date"] = pd.to_datetime(spot.ts, unit="ms")
spot[["date", "close", "high", "low"]].to_csv(f"{HERE}/data/spot.csv", index=False)

chain = get("get_book_summary_by_currency", currency="BTC", kind="option")
json.dump({"fetched_ms": now, "options": chain}, open(f"{HERE}/data/chain.json", "w"))
print(f"DVOL: {len(dvol)} days from {dvol.date.min().date()} to {dvol.date.max().date()} | spot: {len(spot)} days | option chain: {len(chain)} contracts")
