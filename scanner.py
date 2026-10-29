#!/usr/bin/env python3
"""Sleeper-hold reversal scanner.

Usage
  python scanner.py --auto                 # scheduler mode: picks timeframes, honours the ET window
  python scanner.py --timeframe daily      # run now, one timeframe
  python scanner.py --timeframe daily weekly monthly
  python scanner.py --tickers AAPL MSFT NVDA --timeframe daily
  python scanner.py --refresh quick        # re-check today's candidates (about 15 seconds)
  python scanner.py --refresh full         # rescan the whole universe
  python scanner.py --explain --tickers TPR --asof 2026-10-27   # why was / wasn't it flagged?
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config as C
from confirm import assess
from indicators import atr, evaluate
import analysis
from report import write_outputs, write_pages

NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent
log = logging.getLogger("scanner")


# --------------------------------------------------------------------------- #
# Universe and data
# --------------------------------------------------------------------------- #
def load_universe() -> list[str]:
    if C.UNIVERSE == "sp500":
        try:
            import requests
            html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                                headers={"User-Agent": "Mozilla/5.0 reversal-scanner"}, timeout=20).text
            tbl = pd.read_html(io.StringIO(html))[0]
            return sorted(tbl["Symbol"].str.replace(".", "-", regex=False).unique())
        except Exception as e:  # network or layout change
            log.warning("S&P 500 list unavailable (%s); using %s", e, C.FALLBACK_TICKERS_FILE)
            path = ROOT / C.FALLBACK_TICKERS_FILE
    else:
        path = ROOT / C.UNIVERSE
    return [t.strip().upper() for t in path.read_text().splitlines()
            if t.strip() and not t.startswith("#")]


def download(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        raw = None
        for attempt in range(3):          # Yahoo occasionally rate-limits; retry politely
            try:
                raw = yf.download(batch, period=period, interval="1d", group_by="ticker",
                                  auto_adjust=True, threads=True, progress=False)
                if raw is not None and not raw.empty:
                    break
            except Exception as e:
                log.warning("download attempt %d failed: %s", attempt + 1, e)
            time.sleep(10 * (attempt + 1))
        if raw is None or raw.empty:
            continue
        for t in batch:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().astype("float64")
                if len(df):
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    out[t] = df
            except KeyError:
                log.debug("no data for %s", t)
    return out


# --------------------------------------------------------------------------- #
# Session helpers
# --------------------------------------------------------------------------- #
def session_fraction(now: dt.datetime) -> float:
    """Share of the regular session elapsed (used to project today's volume)."""
    open_ = now.replace(hour=9, minute=30, second=0, microsecond=0)
    frac = (now - open_).total_seconds() / (390 * 60)
    return float(min(max(frac, 0.05), 1.0))


def _trading_days():
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay
    return CustomBusinessDay(calendar=USFederalHolidayCalendar())


def timeframes_due(today: dt.date) -> list[str]:
    nxt = (pd.Timestamp(today) + _trading_days()).date()
    due = ["daily"]
    if nxt.isocalendar()[1] != today.isocalendar()[1]:
        due.append("weekly")    # last trading day of the week (usually Friday)
    if nxt.month != today.month:
        due.append("monthly")   # last trading day of the month
    return due


def in_window(now: dt.datetime) -> bool:
    lo, hi = (dt.time.fromisoformat(x) for x in C.SCAN_WINDOW_ET)
    return now.weekday() < 5 and lo <= now.time() <= hi


def prepare(df: pd.DataFrame, rule: str | None, now: dt.datetime) -> pd.DataFrame:
    df = df.astype("float64")   # Yahoo returns whole-number volume; keep everything float
    # Project today's partial volume to a full-session estimate.
    if df.index[-1].date() == now.date() and now.time() < dt.time(16, 0):
        df.iloc[-1, df.columns.get_loc("Volume")] /= session_fraction(now)
    if rule:
        df = df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min",
                                    "Close": "last", "Volume": "sum"}).dropna()
    return df


# --------------------------------------------------------------------------- #
# Scan
# --------------------------------------------------------------------------- #
def download_intraday(tickers: list[str], day: dt.date, until: dt.time | None = None) -> dict[str, pd.DataFrame]:
    """5-minute bars for one session (today, or any day in the last ~60 days)."""
    if not tickers:
        return {}
    period = "1d" if day == dt.datetime.now(NY).date() else "60d"
    try:
        import yfinance as yf
        raw = yf.download(tickers, period=period, interval="5m", group_by="ticker",
                          auto_adjust=True, threads=True, progress=False)
    except Exception as e:
        log.warning("intraday download failed: %s", e)
        return {}
    out = {}
    for t in tickers:
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().astype("float64")
            idx = pd.to_datetime(df.index)
            idx = idx.tz_convert(NY) if idx.tz is not None else idx.tz_localize(NY)
            df.index = idx
            df = df[df.index.date == day]
            if until is not None:
                df = df[df.index.time < until]
            if len(df):
                out[t] = df
        except KeyError:
            pass
    return out


def qualifies(res: dict) -> bool:
    return ((res.get("has_candle") or not C.REQUIRE_CANDLE)
            and res["n_factors"] >= C.MIN_FACTORS and res["score"] >= C.MIN_SCORE)


def carried_setup(df: pd.DataFrame, tf: dict) -> dict | None:
    """A qualifying setup from 1-N bars ago that has not run away yet: price is
    still within `late_atr` ATRs of the signal bar's close and has not broken
    the signal bar's tail."""
    P = C.PARAMS
    for k in range(1, P["carry_days"] + 1):
        if len(df) <= k + 60:
            break
        res = evaluate(df.iloc[:-k], tf, P, C.WEIGHTS)
        if not res or not qualifies(res):
            continue
        sig = df.iloc[-k - 1]
        after = df.iloc[-k:]
        a = res["atr"]
        c = float(df["Close"].iloc[-1])
        if res["direction"] == "bullish":
            intact = float(after["Low"].min()) > float(sig.Low)
            moved = (c - float(sig.Close)) / a
            stop = float(sig.Low)
        else:
            intact = float(after["High"].max()) < float(sig.High)
            moved = (float(sig.Close) - c) / a
            stop = float(sig.High)
        if intact and moved <= P["late_atr"]:
            res = dict(res)
            res["factors"] = [f"Setup from {k} bar{'s' if k > 1 else ''} ago, still near entry "
                              f"({moved:+.1f} ATR since signal; stop beyond {stop:.2f})"] + res["factors"]
            res["levels"] = [round(stop, 4)]
            res["carried"] = k
            res["keys"] = ["carried"] + list(res.get("keys", []))
            return res
    return None


def build_row(t: str, tf_name: str, raw: pd.DataFrame, now: dt.datetime,
              intraday: dict | None, keep: set[str]) -> dict | None:
    tf = C.TIMEFRAMES[tf_name]
    df = prepare(raw, tf["rule"], now)
    res = evaluate(df, tf, C.PARAMS, C.WEIGHTS)
    ok = bool(res) and qualifies(res)
    if not ok:
        carried = carried_setup(df, tf)
        if carried:
            res, ok = carried, True
    if not res or (not ok and t not in keep):
        return None
    res.pop("diag", None)
    last, prev = df.iloc[-1], df.iloc[-2]
    row = dict(ticker=t, timeframe=tf_name, close=round(float(last.Close), 2),
               change_pct=round((float(last.Close) / float(prev.Close) - 1) * 100, 2),
               bar_date=str(df.index[-1].date()), qualifies=ok,
               bar_is_live=raw.index[-1].date() == now.date(),
               bars=df.iloc[-45:][["Open", "High", "Low", "Close"]].round(4).to_numpy().tolist(),
               **res)
    row["confirm"] = assess(row, (float(last.High), float(last.Low), float(last.Close)),
                            float(atr(raw).iloc[-1]), (intraday or {}).get(t), now, ok,
                            use_mid=not res.get("carried"))
    return row


def scan(tf_name: str, data: dict[str, pd.DataFrame], now: dt.datetime,
         keep: set[str] | None = None, intraday: dict[str, pd.DataFrame] | None = None) -> list[dict]:
    """Score every ticker. Tickers in `keep` (earlier candidates being refreshed)
    stay on the list even if they no longer qualify, flagged as faded."""
    keep = keep or set()
    rows = []
    for t, raw in data.items():
        if len(raw) < 80:
            continue
        tail = raw.iloc[-20:]
        if t not in keep and (raw["Close"].iloc[-1] < C.MIN_PRICE or
                              (tail["Close"] * tail["Volume"]).mean() < C.MIN_AVG_DOLLAR_VOLUME):
            continue
        try:
            row = build_row(t, tf_name, raw, now, intraday, keep)
        except Exception as e:
            log.warning("%s: %s", t, e)
            continue
        if row:
            rows.append(row)
    rows.sort(key=lambda r: (r["qualifies"], not r.get("carried"), r["score"], r["n_factors"]), reverse=True)
    out = []
    for d in ("bearish", "bullish"):
        side = [r for r in rows if r["direction"] == d]
        out += [r for r in side if r["qualifies"]][:C.MAX_RESULTS] + [r for r in side if not r["qualifies"]]
    return sorted(out, key=lambda r: (r["qualifies"], not r.get("carried"), r["score"]), reverse=True)


# --------------------------------------------------------------------------- #
# Explain mode: "would ticker X have been flagged on date D at 3:30 PM?"
# --------------------------------------------------------------------------- #
def explain(tickers: list[str], asof: dt.date, at: dt.time) -> int:
    now = dt.datetime.combine(asof, at, tzinfo=NY)
    data = download(tickers, "3y")
    intra = download_intraday(tickers, asof, until=at)
    out = [f"# Signal check for {asof} at {at.strftime('%H:%M')} ET\n"]
    tf = C.TIMEFRAMES["daily"]
    for t in tickers:
        raw = data.get(t)
        if raw is None:
            out.append(f"## {t}\nNo price data.\n")
            continue
        raw = raw[raw.index.date <= asof].copy()
        if not len(raw) or raw.index[-1].date() != asof:
            out.append(f"## {t}\nNo trading bar on {asof}.\n")
            continue
        note = "closing bar for that day (5-minute data not available)"
        if t in intra:
            b = intra[t]
            raw.iloc[-1] = [float(b.Open.iloc[0]), float(b.High.max()), float(b.Low.min()),
                            float(b.Close.iloc[-1]), float(b.Volume.sum())]
            note = f"the bar as it stood at {at.strftime('%H:%M')} (rebuilt from 5-minute data)"
        df = prepare(raw, None, now)
        res = evaluate(df, tf, C.PARAMS, C.WEIGHTS)
        row = build_row(t, "daily", raw, now, {}, keep={t})
        out.append(f"## {t}\nUsing {note}.\n")
        if not res:
            out.append("Not enough history to evaluate.\n")
            continue
        d = res["diag"]
        cs = d["candle"]
        out.append(f"**Verdict: {'FLAGGED' if row and row['qualifies'] else 'NOT flagged'}**"
                   f" | direction tested: {res['direction']} (prior trend {d['prior_trend']})"
                   f" | score {res['score']} (need {C.MIN_SCORE}) | factors {res['n_factors']} (need {C.MIN_FACTORS})"
                   f" | reversal candle: {'yes' if res['has_candle'] else 'NO'}\n")
        out.append("Factors found:\n" + "\n".join(f"- {f}" for f in res["factors"]) + "\n")
        out.append(f"Candle: O {cs['open']:.2f} H {cs['high']:.2f} L {cs['low']:.2f} C {cs['close']:.2f}"
                   f" | range {cs['range_atr']} ATR | body {cs['body_pct']:.0%} | upper wick {cs['upper_wick_pct']:.0%}"
                   f" | lower wick {cs['lower_wick_pct']:.0%} | close position {cs['close_position']:.0%}"
                   f" (tail rule: wick >= {C.PARAMS['tail_min_pct']:.0%} of range and >= {C.PARAMS['tail_min_atr']} ATR,"
                   f" close in the reversal half)\n")
        out.append("Time counts (bars before this one): " +
                   ", ".join(f"{k} {v}" for k, v in d["counts"].items()) + f" (need {C.PARAMS['count_min']})\n")
        out.append(f"Nearest pivot zones (tolerance {d['tolerance']}): " +
                   "; ".join(f"{z['price']:.2f} ({z['touches']} touches, {z['rv']:.1f}x vol"
                             f"{', major' if z['major'] else ''})" for z in d["nearest_zones"]) + "\n")
        if row and row.get("carried"):
            out.append(f"Listed as a carried setup from {row['carried']} bar(s) earlier.\n")
    text = "\n".join(out)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text)
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def previous(tf: str, today: dt.date) -> set[str]:
    """Tickers listed by an earlier scan TODAY for this timeframe."""
    f = ROOT / "site" / "data" / f"{tf}.json"
    if not f.exists():
        return set()
    d = json.loads(f.read_text())
    if (d.get("generated_iso") or "")[:10] != today.isoformat():
        return set()
    return {r["ticker"] for r in d.get("rows", [])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeframe", nargs="+", choices=list(C.TIMEFRAMES))
    ap.add_argument("--auto", action="store_true",
                    help="scheduler mode: full scan first, quick refreshes after that")
    ap.add_argument("--refresh", choices=["quick", "full"],
                    help="quick = re-check today's candidates only; full = rescan the universe")
    ap.add_argument("--force", action="store_true", help="ignore the ET time window in --auto mode")
    ap.add_argument("--tickers", nargs="+", help="override the universe")
    ap.add_argument("--analyze", action="store_true", help="only update the trade-tracker analysis")
    ap.add_argument("--explain", action="store_true",
                    help="explain why --tickers were or were not flagged (use with --asof)")
    ap.add_argument("--asof", help="date to check, YYYY-MM-DD (default today)")
    ap.add_argument("--at", default="15:30", help="time of day for --explain, HH:MM ET")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if a.explain:
        if not a.tickers:
            ap.error("--explain needs --tickers")
        day = dt.date.fromisoformat(a.asof) if a.asof else dt.datetime.now(NY).date()
        return explain([t.upper() for t in a.tickers], day, dt.time.fromisoformat(a.at))

    if a.analyze:
        analysis.run(ROOT, download)
        write_pages(ROOT)
        return 0

    now = dt.datetime.now(NY)
    mode = a.refresh or "full"
    if a.auto:
        if not (a.force or in_window(now)):
            log.info("Outside scan window (%s ET); nothing to do.", now.strftime("%H:%M"))
            return 0
        mode = "quick" if previous("daily", now.date()) else "full"
    if a.timeframe:
        tfs = a.timeframe
    elif a.auto or a.refresh:
        tfs = timeframes_due(now.date())
    else:
        tfs = ["daily"]

    keep = {tf: previous(tf, now.date()) for tf in tfs}
    if mode == "quick":
        tfs = [tf for tf in tfs if keep[tf]] or tfs
        tickers = sorted(set().union(*(keep[tf] for tf in tfs))) or None
        if not tickers:
            log.info("No earlier scan today; running a full scan instead.")
            mode = "full"
    if mode == "full":
        tickers = [t.upper() for t in a.tickers] if a.tickers else load_universe()

    period = max((C.TIMEFRAMES[t]["period"] for t in tfs), key=lambda p: int(p[:-1]))
    log.info("%s scan | %d tickers | timeframes=%s | history=%s", mode, len(tickers), tfs, period)
    data = download(tickers, period)
    log.info("Downloaded %d tickers", len(data))
    if not data:
        log.error("No price data received from Yahoo Finance. Try again in a few minutes.")
        return 1

    for tf in tfs:
        rows = scan(tf, data, now, keep=keep[tf])
        intra = download_intraday([r["ticker"] for r in rows], now.date())
        rows = scan(tf, {r["ticker"]: data[r["ticker"]] for r in rows}, now, keep=keep[tf] | {r["ticker"] for r in rows},
                    intraday=intra)
        log.info("%s: %d candidates (%d green)", tf, len(rows),
                 sum(r["confirm"]["status"] == "green" for r in rows))
        write_outputs(tf, rows, now, ROOT, mode)
    try:
        analysis.run(ROOT, download, now)
    except Exception as e:   # the scan results still publish if the analysis fails
        log.warning("trade analysis failed: %s", e)
    write_pages(ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
