"""Trade-journal analysis: what works and what does not.

Reads trades.json (written by the dashboard's "Took this trade" and the
tracker's "Close" buttons), adds price history, and writes
site/data/analysis.json for the tracker page.

For each trade it measures:
  return        closed: from the recorded entry and exit prices
  max gain      best price reached in your favour while the trade was open (MFE)
  max drawdown  worst price against you while the trade was open (MAE)
Then it compares groups of trades (by factor, score, count, confirmation colour,
entry timing, holding time) and writes plain-English findings. Findings are
marked as early reads until a group has enough trades to mean something.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
log = logging.getLogger("analysis")
MIN_GROUP = 5          # trades needed before a group is compared
SOLID = 15             # trades needed before a finding is called reliable
TARGETS = (5.0, 10.0)  # profit targets of interest (%)

KEY_NAMES = {
    "candle": "Reversal candle (tail / engulfing)", "candle_doji": "Doji",
    "count": "Time count 7+", "fib": "Fibonacci level", "sr_major": "Major pivot zone",
    "sr": "Minor pivot", "hvn": "High-volume node", "trendline": "Trend line",
    "rsi": "RSI extreme / divergence", "bollinger": "Bollinger band tag",
    "extension": "Extended from 20 EMA", "volume": "Volume spike", "carried": "Entered a day or two after the signal",
}


def _ts(s: str | None) -> pd.Timestamp | None:
    return pd.Timestamp(s).tz_convert(NY) if s else None


def excursions(trades: list[dict], prices: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Max favourable / adverse move (%) between entry and exit (or today)."""
    out = {}
    for t in trades:
        df = prices.get(t["ticker"])
        if df is None or not t.get("entry_price"):
            continue
        start = _ts(t["entry_time"]).date()
        end = _ts(t["exit_time"]).date() if t.get("exit_time") else dt.date.max
        w = df[(df.index.date >= start) & (df.index.date <= end)]
        if not len(w):
            continue
        e = float(t["entry_price"])
        hi, lo = float(w["High"].max()), float(w["Low"].min())
        if t.get("side") == "short":
            mfe, mae = (e - lo) / e * 100, (e - hi) / e * 100
        else:
            mfe, mae = (hi - e) / e * 100, (lo - e) / e * 100
        out[t["id"]] = dict(mfe=round(mfe, 2), mae=round(mae, 2))
    return out


def _group_stats(name: str, g: pd.DataFrame) -> dict:
    return dict(group=name, n=int(len(g)), win_rate=round(float((g.ret > 0).mean()), 3),
                avg_return=round(float(g.ret.mean()), 2),
                avg_mfe=round(float(g.mfe.mean()), 2) if g.mfe.notna().any() else None,
                avg_days=round(float(g.days.mean()), 1))


def analyse(trades: list[dict], prices: dict[str, pd.DataFrame], now: dt.datetime) -> dict:
    exc = excursions(trades, prices)
    last = {t: dict(price=round(float(df["Close"].iloc[-1]), 2), time=str(df.index[-1].date()))
            for t, df in prices.items() if len(df)}
    closed = [t for t in trades if t.get("status") == "closed" and t.get("return_pct") is not None]
    res = dict(generated=now.strftime("%Y-%m-%d %H:%M %Z"), n_closed=len(closed),
               n_open=sum(t.get("status") == "open" for t in trades),
               prices=last, excursions=exc, breakdown=[], insights=[])
    if not closed:
        res["insights"] = ["No closed trades yet. Each trade you take and close adds to this analysis."]
        return res

    rows = []
    for t in closed:
        s = t.get("signal") or {}
        et = _ts(t["entry_time"])
        rows.append(dict(
            id=t["id"], ticker=t["ticker"], ret=float(t["return_pct"]),
            days=(_ts(t["exit_time"]) - et).total_seconds() / 86400,
            mfe=exc.get(t["id"], {}).get("mfe", np.nan), mae=exc.get(t["id"], {}).get("mae", np.nan),
            score=s.get("score") or 0, count=s.get("count") or 0, keys=set(s.get("keys") or []),
            carried=bool(s.get("carried")), confirm=s.get("confirm_status") or "unknown",
            side=t.get("side", "long"), entry_hour=et.hour + et.minute / 60, tf=t.get("timeframe", "daily")))
    d = pd.DataFrame(rows)
    if d.carried.any():
        d["keys"] = [k | {"carried"} if c else k for k, c in zip(d["keys"], d["carried"])]
    base = _group_stats("All closed trades", d)
    groups = [base]

    def add(name, mask):
        g = d[mask]
        if len(g):
            groups.append(_group_stats(name, g))

    for key, label in KEY_NAMES.items():
        add(f"With: {label}", d["keys"].apply(lambda k: key in k))
    for lo, hi, lab in ((0, 4, "Score under 4"), (4, 5.5, "Score 4 to 5.5"), (5.5, 99, "Score 5.5+")):
        add(lab, (d.score >= lo) & (d.score < hi))
    for lo, hi, lab in ((0, 7, "Count under 7"), (7, 9, "Count 7-8"), (9, 99, "Count 9+")):
        add(lab, (d["count"] >= lo) & (d["count"] < hi))
    for c in ("green", "amber", "red"):
        add(f"Confirmation {c} at entry", d.confirm == c)
    add("Long trades", d.side == "long")
    add("Short trades", d.side == "short")
    add("Entered before 3:00 PM", d.entry_hour < 15)
    add("Entered 3:00 PM or later", d.entry_hour >= 15)
    add("Held under 2 days", d.days < 2)
    add("Held 2-5 days", (d.days >= 2) & (d.days < 5))
    add("Held 5+ days", d.days >= 5)
    res["breakdown"] = groups

    # ---- plain-English findings ----
    n = len(d)
    ins = [f"{n} closed trades: {base['win_rate']:.0%} winners, average {base['avg_return']:+.2f}% per trade, "
           f"total {d.ret.sum():+.2f}% (sum of trade returns)."]
    if n < SOLID:
        ins.append(f"Early read: findings firm up after about {SOLID} closed trades; groups need {MIN_GROUP}+ trades to be compared.")
    comp = [g for g in groups[1:] if g["n"] >= MIN_GROUP and g["n"] < n]
    if comp:
        best = max(comp, key=lambda g: g["avg_return"])
        worst = min(comp, key=lambda g: g["avg_return"])
        ins.append(f"Strongest group so far: {best['group']} ({best['n']} trades, {best['win_rate']:.0%} win rate, "
                   f"{best['avg_return']:+.2f}% average).")
        if worst["group"] != best["group"]:
            ins.append(f"Weakest group so far: {worst['group']} ({worst['n']} trades, {worst['win_rate']:.0%} win rate, "
                       f"{worst['avg_return']:+.2f}% average).")
        for g in comp:
            if g["group"].startswith("With:") and g["win_rate"] - base["win_rate"] >= 0.15:
                ins.append(f"{g['group'][6:]} lifts the win rate by {100 * (g['win_rate'] - base['win_rate']):.0f} points "
                           f"versus all trades; consider giving it more weight in config.py.")
            if g["group"].startswith("With:") and base["win_rate"] - g["win_rate"] >= 0.15:
                ins.append(f"{g['group'][6:]} lowers the win rate by {100 * (base['win_rate'] - g['win_rate']):.0f} points; "
                           f"consider giving it less weight.")
    m = d.dropna(subset=["mfe"])
    if len(m):
        for tgt in TARGETS:
            hit = (m.mfe >= tgt).mean()
            ins.append(f"{hit:.0%} of trades reached +{tgt:.0f}% at some point while open.")
        gave_back = m[(m.mfe >= TARGETS[0]) & (m.ret < TARGETS[0] / 2)]
        if len(gave_back):
            ins.append(f"{len(gave_back)} trade(s) were up {TARGETS[0]:.0f}%+ but closed below "
                       f"{TARGETS[0] / 2:.1f}%; a profit-taking order at +{TARGETS[0]:.0f}% would have locked those in.")
        winners = m[m.ret > 0]
        if len(winners) >= 3:
            ins.append(f"Winning trades moved {abs(winners.mae.mean()):.1f}% against you on average before working "
                       f"(worst {abs(winners.mae.min()):.1f}%): a guide for where to set stops.")
        losers = m[m.ret <= 0]
        if len(losers) >= 3 and len(winners) >= 3:
            ins.append(f"Losing trades were held {losers.days.mean():.1f} days on average versus "
                       f"{winners.days.mean():.1f} days for winners.")
    res["insights"] = ins
    return res


def run(root: Path, download, now: dt.datetime | None = None) -> None:
    """Called by scanner.py after each scan (and by --analyze)."""
    now = now or dt.datetime.now(NY)
    f = root / "trades.json"
    trades = json.loads(f.read_text()) if f.exists() else []
    out = root / "site" / "data"
    out.mkdir(parents=True, exist_ok=True)
    (out / "trades.json").write_text(json.dumps(trades))
    prices = {}
    if trades:
        tickers = sorted({t["ticker"] for t in trades})
        oldest = min(_ts(t["entry_time"]) for t in trades)
        days = (now - oldest).days + 10
        period = "1mo" if days <= 28 else "6mo" if days <= 180 else "2y"
        prices = download(tickers, period)
    res = analyse(trades, prices, now)
    (out / "analysis.json").write_text(json.dumps(res, default=float))
    log.info("analysis: %d closed, %d open trades", res["n_closed"], res["n_open"])
