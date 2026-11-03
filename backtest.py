"""Backtest: replay the scanner over the last N trading days.

For every stock and every past day, the setup engine (setup.py) is run on the
prices up to that day's close, exactly as the live scanner would have seen them
(using the day's closing bar; the live scanner sees the bar at 3:30 PM, so the
backtest is slightly kinder on tails that held into the close). Every setup that
would have been listed is recorded once (first day seen) and then followed forward
with suggestions.evaluate_one, the same rules as the live tracker:
  close entry and safer entry, half at +5% / stop to breakeven / rest at +10%,
  15-bar time stop, stop first on a bar that touches both.
Earnings: past earnings dates are looked up, so setups with a report within the
blackout window are flagged exactly as the bot would have skipped them.

Known limits: today's S&P 500 list is used for the whole period (survivorship
bias), no commissions or slippage, and 60 days is a short sample, so read the
results as a first indication, not proof.

Writes site/data/backtest_suggestions.json and site/data/backtest_review.json,
shown on suggestions.html under "Backtest".
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config as C
import suggestions as S
from setup import evaluate_level_touch, evaluate_setup

NY = ZoneInfo("America/New_York")
log = logging.getLogger("backtest")


def _runs(df: pd.DataFrame) -> np.ndarray:
    """Longest of the up / down time counts ending at each bar (cheap pre-filter)."""
    o, c = df["Open"].to_numpy(), df["Close"].to_numpy()
    h, l = df["High"].to_numpy(), df["Low"].to_numpy()
    tr = np.maximum(h - l, np.abs(np.r_[np.nan, c[:-1]] - h))
    a = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    body = C.PARAMS["count_body_atr"]
    up = dn = 0
    out = np.zeros(len(c), dtype=int)
    for i in range(1, len(c)):
        real_down = (o[i] - c[i]) >= body * a[i] and c[i] < c[i - 1]
        real_up = (c[i] - o[i]) >= body * a[i] and c[i] > c[i - 1]
        up = 0 if real_down else up + 1
        dn = 0 if real_up else dn + 1
        out[i] = max(up, dn)
    return out


def _past_earnings(tickers: list[str]) -> dict[str, list[dt.date]]:
    out = {}
    try:
        import yfinance as yf
    except Exception:
        return out
    for t in tickers:
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=16)
            if ed is not None and len(ed):
                out[t] = sorted({pd.Timestamp(x).tz_convert(NY).date() if pd.Timestamp(x).tzinfo
                                 else pd.Timestamp(x).date() for x in ed.index})
        except Exception:
            pass
    return out


def run(root: Path, download, load_universe, days: int = 60, tfs=("daily", "weekly")) -> dict:
    t0 = time.time()
    tickers = load_universe()
    data = download(tickers, "6y" if "weekly" in tfs else "2y")
    today = dt.datetime.now(NY).date()
    recs: dict[str, dict] = {}
    n_eval = 0
    for t, raw in data.items():
        if len(raw) < 320:
            continue
        raw = raw.astype("float64")
        for tf in tfs:
            tfc = C.TIMEFRAMES[tf]
            bars = raw if tf == "daily" else raw.resample(tfc["rule"]).agg(
                {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
            if len(bars) < 120:
                continue
            runs = _runs(bars)
            # signal bars in the window: last `days` trading days (weekly: the weeks they cover)
            start = raw.index[-days] if len(raw) > days else raw.index[0]
            idx = [i for i in range(len(bars)) if bars.index[i] >= start and i >= 100]
            if tf != "daily" and idx and bars.index[idx[-1]].date() > today:
                idx = idx[:-1]                       # the current week is not finished
            for i in idx:
                if runs[max(0, i - 3):i + 1].max() < C.PARAMS["count_min"]:
                    continue                         # no time count of 7+: nothing can qualify
                sub = bars.iloc[max(0, i - 300):i + 1]
                # liquidity filter as of that day
                tail = raw[raw.index <= sub.index[-1]].iloc[-20:]
                if tail["Close"].iloc[-1] < C.MIN_PRICE or (tail["Close"] * tail["Volume"]).mean() < C.MIN_AVG_DOLLAR_VOLUME:
                    continue
                n_eval += 1
                sig = str(sub.index[-1].date())
                # ---- alternative strategy: enter at the liquidity level, no tail needed
                try:
                    lv = evaluate_level_touch(sub, tfc, C.PARAMS)
                except Exception as e:
                    lv = None
                    log.debug("level %s %s: %s", t, sig, e)
                if lv:
                    k2 = f"{t}|{tf}|{lv['direction']}|{sig}|level"
                    if k2 not in recs:
                        recs[k2] = dict(
                            key=k2, id=f"{t}-{tf}-{sig}-{lv['direction'][:4]}-lvl", ticker=t, timeframe=tf,
                            strategy="level", direction=lv["direction"],
                            side="short" if lv["direction"] == "bearish" else "long", rev_date=sig, signal_date=sig,
                            first_seen=sig + "T15:30", stage="level", stages_seen=["level"],
                            scan_price=round(float(sub["Close"].iloc[-1]), 2), entry=lv["entry"], stop=lv["stop"],
                            level=lv["level"], level_label=lv["level_label"], score=lv["score"], count=lv["count"],
                            efficiency=lv["efficiency"], move_atr=lv["move_atr"], keys=lv["keys"],
                            factors=lv["factors"], idea=lv["idea"], valid_at_last_scan=True, last_reason="",
                            status="pending", outcome={}, backtest=True)
                try:
                    res = evaluate_setup(sub, tfc, C.PARAMS, C.WEIGHTS)
                except Exception as e:
                    log.debug("%s %s: %s", t, sub.index[-1].date(), e)
                    continue
                if not res or not res.get("qualified"):
                    continue
                rev = str(sub.index[res["ext_pos"]].date())
                key = f"{t}|{tf}|{res['direction']}|{rev}"
                if key in recs:
                    continue                         # first sighting only, like the live tracker
                recs[key] = dict(
                    key=key, id=f"{t}-{tf}-{rev}-{res['direction'][:4]}", ticker=t, timeframe=tf,
                    direction=res["direction"], side="short" if res["direction"] == "bearish" else "long",
                    rev_date=rev, signal_date=sig, first_seen=sig + "T15:30", stage=res.get("stage"),
                    stages_seen=[res.get("stage")], scan_price=round(float(sub["Close"].iloc[-1]), 2),
                    stop=res.get("stop"), score=res.get("score"), count=res.get("count"),
                    moved_pct=res.get("moved_pct"), tail_pct=res.get("tail_pct"), efficiency=res.get("efficiency"),
                    move_atr=res.get("move_atr"), rsi=res.get("rsi"), keys=res.get("keys", []),
                    factors=res.get("factors", []), idea=res.get("idea"), valid_at_last_scan=True, strategy="tail",
                    last_reason="", status="pending", outcome={}, backtest=True)
    log.info("backtest: %d evaluations, %d setups found (%.0fs)", n_eval, len(recs), time.time() - t0)

    # earnings flags from actual past report dates
    er = _past_earnings(sorted({r["ticker"] for r in recs.values()}))
    for r in recs.values():
        d0 = dt.date.fromisoformat(r["signal_date"])
        nxt = next((d for d in er.get(r["ticker"], []) if d >= d0), None)
        soon, n = S.earnings_flag(d0, nxt.isoformat() if nxt else None)
        r.update(earnings_next=nxt.isoformat() if nxt else None, earnings_blackout=soon, earnings_days=n)

    # follow every setup forward with real prices
    for r in recs.values():
        df = data.get(r["ticker"])
        try:
            if r.get("strategy") == "level":
                S.evaluate_level(r, df.astype("float64"), today)
            else:
                S.evaluate_one(r, df.astype("float64"), today)
        except Exception as e:
            log.debug("evaluate %s: %s", r["id"], e)
    out = sorted(recs.values(), key=lambda r: r["first_seen"])
    rev = S.review(out, dt.datetime.now(NY))
    rev["backtest"] = dict(days=days, timeframes=list(tfs), stocks=len(data), evaluations=n_eval,
                           setups=len(out), seconds=round(time.time() - t0),
                           limits="Uses each day's closing bar (live scans see the 3:30 PM bar), today's S&P 500 list "
                                  "(survivorship bias), no commissions or slippage. A 60-day sample is a first "
                                  "indication, not proof.")
    d = root / "site" / "data"
    d.mkdir(parents=True, exist_ok=True)
    (d / "backtest_suggestions.json").write_text(json.dumps(out, default=float))
    (d / "backtest_review.json").write_text(json.dumps(rev, default=float))
    (root / "backtest_suggestions.json").write_text(json.dumps(out, default=float))
    return rev
