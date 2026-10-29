"""Confirmation meter: how likely is today's reversal candle to still be valid
at the 4:00 PM close?

The forming bar is "confirmed" at the close if price finishes on the reversal
side of BOTH the bar's midpoint and the level it rejected. Between the scan and
the close, we estimate the odds of that holding with a simple volatility model
and add intraday tape checks (5-minute bars).

Status colours
  green  "Likely to confirm"   hold odds >= GREEN_ODDS and the tape agrees
  amber  "Developing"          hold odds >= AMBER_ODDS and the tape is mixed
  red    "At risk"             low hold odds or the tape is against the trade
  grey   "Signal faded"        the setup no longer meets the scanner's thresholds
After 4:00 PM the green state reads "Confirmed at close".

The odds are a heuristic (normal approximation of the remaining move), not a
calibrated probability. Back-test before relying on them.
"""
from __future__ import annotations

import datetime as dt
from math import erf, sqrt

import pandas as pd

GREEN_ODDS, AMBER_ODDS = 0.70, 0.45


def _phi(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def minutes_to_close(now: dt.datetime) -> float:
    close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if now.weekday() >= 5:
        return 0.0
    return float(min(max((close - now).total_seconds() / 60, 0.0), 390.0))


def assess(row: dict, bar: tuple[float, float, float], daily_atr: float,
           intraday: pd.DataFrame | None, now: dt.datetime, qualifies: bool = True,
           use_mid: bool = True) -> dict:
    d = row["direction"]
    h, l, c = bar
    mid = l + 0.5 * (h - l)   # for a carried setup (use_mid=False) only the stop level counts

    # Invalidation price: the nearer of the bar midpoint and the rejected level.
    if d == "bearish":
        inv = min([mid if use_mid else 1e18] + [x for x in row["levels"] if x >= c])
        cushion = inv - c
    else:
        inv = max([mid if use_mid else -1e18] + [x for x in row["levels"] if x <= c])
        cushion = c - inv

    mins = minutes_to_close(now) if row.get("bar_is_live", True) else 0.0
    if mins <= 0 or daily_atr <= 0:
        hold = 1.0 if cushion > 0 else 0.0
    else:
        sigma = (daily_atr / 1.6) * sqrt(mins / 390)   # ATR ~ 1.6x daily std. dev.
        hold = _phi(cushion / sigma)

    checks = [dict(name="Candle holds to the close", ok=hold >= 0.6,
                   detail=f"{hold:.0%} odds; fails above {inv:.2f}" if d == "bearish"
                   else f"{hold:.0%} odds; fails below {inv:.2f}")]

    if intraday is not None and len(intraday) >= 6:
        tp = (intraday["High"] + intraday["Low"] + intraday["Close"]) / 3
        vwap = float((tp * intraday["Volume"]).cumsum().iloc[-1] / max(intraday["Volume"].sum(), 1))
        last = float(intraday["Close"].iloc[-1])
        r30 = intraday.iloc[-6:]                        # last 30 minutes
        move = float(r30["Close"].iloc[-1] - r30["Open"].iloc[0])
        with_rev = move < 0 if d == "bearish" else move > 0
        vol_ratio = float(r30["Volume"].mean() / max(intraday["Volume"].mean(), 1))
        earlier = intraday.iloc[:-6]
        if d == "bearish":
            side = last < vwap
            no_new = len(earlier) == 0 or r30["High"].max() < earlier["High"].max()
        else:
            side = last > vwap
            no_new = len(earlier) == 0 or r30["Low"].min() > earlier["Low"].min()
        checks += [
            dict(name="Price on reversal side of VWAP", ok=bool(side), detail=f"VWAP {vwap:.2f}"),
            dict(name="Last 30 min moving with the reversal", ok=bool(with_rev), detail=f"{move:+.2f}"),
            dict(name="No new session extreme in last 30 min", ok=bool(no_new), detail=""),
            dict(name="Last-30-min volume confirms", ok=bool(with_rev and vol_ratio >= 1.0),
                 detail=f"{vol_ratio:.1f}x session average"),
        ]

    passed = sum(ch["ok"] for ch in checks)
    tape_ok = passed - checks[0]["ok"]
    tape_n = len(checks) - 1
    if not qualifies:
        status, label = "grey", "Signal faded"
    elif hold >= GREEN_ODDS and (tape_n == 0 or tape_ok >= tape_n - 1):
        status, label = "green", "Confirmed at close" if mins <= 0 else "Likely to confirm"
    elif hold >= AMBER_ODDS and (tape_n == 0 or tape_ok >= tape_n / 2):
        status, label = "amber", "Developing"
    else:
        status, label = "red", "Failed at close" if mins <= 0 else "At risk"

    return dict(status=status, label=label, hold_odds=round(hold, 3), invalidation=round(inv, 2),
                minutes_left=round(mins), checks=checks, passed=passed, total=len(checks))
