#!/usr/bin/env python3
"""Sleeper-hold reversal scanner.

Usage
  python scanner.py --auto                 # scheduler mode: picks timeframes, honours the ET window
  python scanner.py --timeframe daily      # run now, one timeframe
  python scanner.py --timeframe daily weekly monthly
  python scanner.py --tickers AAPL MSFT NVDA --timeframe daily
  python scanner.py --refresh quick        # re-check today's candidates (about 15 seconds)
  python scanner.py --refresh full         # rescan the whole universe
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import logging
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config as C
from confirm import assess
from indicators import atr, evaluate
from report import write_outputs

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
def download_intraday(tickers: list[str], today: dt.date) -> dict[str, pd.DataFrame]:
    """Today's 5-minute bars, used by the confirmation meter."""
    if not tickers:
        return {}
    try:
        import yfinance as yf
        raw = yf.download(tickers, period="1d", interval="5m", group_by="ticker",
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
            df = df[df.index.date == today]
            if len(df):
                out[t] = df
        except KeyError:
            pass
    return out


def scan(tf_name: str, data: dict[str, pd.DataFrame], now: dt.datetime,
         keep: set[str] | None = None, intraday: dict[str, pd.DataFrame] | None = None) -> list[dict]:
    """Score every ticker. Tickers in `keep` (earlier candidates being refreshed)
    stay on the list even if they no longer qualify, flagged as faded."""
    tf = C.TIMEFRAMES[tf_name]
    keep = keep or set()
    rows = []
    for t, raw in data.items():
        if len(raw) < 30:
            continue
        tail = raw.iloc[-20:]
        if t not in keep and (raw["Close"].iloc[-1] < C.MIN_PRICE or
                              (tail["Close"] * tail["Volume"]).mean() < C.MIN_AVG_DOLLAR_VOLUME):
            continue
        try:
            df = prepare(raw, tf["rule"], now)
            res = evaluate(df, tf, C.PARAMS, C.WEIGHTS)
        except Exception as e:
            log.warning("%s: %s", t, e)
            continue
        if not res:
            continue
        qualifies = res["n_factors"] >= C.MIN_FACTORS and res["score"] >= C.MIN_SCORE
        if not qualifies and t not in keep:
            continue
        last, prev = df.iloc[-1], df.iloc[-2]
        row = dict(ticker=t, timeframe=tf_name, close=round(float(last.Close), 2),
                   change_pct=round((float(last.Close) / float(prev.Close) - 1) * 100, 2),
                   bar_date=str(df.index[-1].date()), qualifies=qualifies,
                   bar_is_live=raw.index[-1].date() == now.date(),
                   bars=df.iloc[-45:][["Open", "High", "Low", "Close"]].round(4).to_numpy().tolist(),
                   **res)
        row["confirm"] = assess(row, (float(last.High), float(last.Low), float(last.Close)),
                                float(atr(raw).iloc[-1]), (intraday or {}).get(t), now, qualifies)
        rows.append(row)
    rows.sort(key=lambda r: (r["qualifies"], r["score"], r["n_factors"]), reverse=True)
    out = []
    for d in ("bearish", "bullish"):
        side = [r for r in rows if r["direction"] == d]
        out += [r for r in side if r["qualifies"]][:C.MAX_RESULTS] + [r for r in side if not r["qualifies"]]
    return sorted(out, key=lambda r: (r["qualifies"], r["score"]), reverse=True)


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
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
