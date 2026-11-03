"""THE BOT, live (v26) - paper trading.

Approved rules (Ron, November 3, 2026):
  daily chart   bot trend + 4-6 bar pullback; a liquidity line where 2+ kinds of support/resistance
                meet, within 2 ATRs of yesterday's close; no earnings within 3 trading days
  longs         SPY above its 50-day average; strongest levels only (major swing, prior week/month)
  shorts        SPY below its 50-day average (looser short rules otherwise); PAUSED Nov 3 (v27): logged
                as watch only (bot_5m_watch) until they show an edge
  5-minute      price runs the stops through the line, then the first 5-minute close back across it
                (within 1 hour) is the entry, before noon ET; stop beyond the stop-run extreme and at
                least 1.5% away; no entry more than 2% from the line; swing exits, 1-5 days
  universe      S&P 500 + the top-110 most-traded list from the FVG journal

Each run (every ~10 minutes, 9:40 AM to noon ET):
  1. the first run of the day arms the lines from yesterday's daily bars (saved in bot/armed-DATE.json)
  2. 5-minute bars are pulled for the armed tickers only and checked for a stop run + reclaim
  3. new signals are recorded in suggestions.json (strategy "bot_5m", paper=True)
  4. open bot trades are followed with the same management as the backtest
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import config as C
import suggestions as S
from setup import armed_levels, evaluate_sweep_5m

NY = ZoneInfo("America/New_York")
log = logging.getLogger("bot")
WINDOW = ("09:40", "12:05")


def universe(load_universe) -> list[str]:
    return sorted(set(load_universe()) | set(getattr(C, "JOURNAL_UNIVERSE", [])))


def _spy_up(daily: dict[str, pd.DataFrame]) -> bool | None:
    spy = daily.get("SPY")
    if spy is None or len(spy) < 60:
        return None
    c = spy["Close"].astype("float64")
    return bool(c.iloc[-1] > c.rolling(50).mean().iloc[-1])


def arm(root: Path, download, load_universe, today: dt.date) -> dict:
    """Lines for today, from daily bars up to YESTERDAY, with the approved filters applied."""
    path = root / "bot" / f"armed-{today}.json"
    if path.exists():
        return json.loads(path.read_text())
    tickers = universe(load_universe)
    data = download(sorted(set(tickers) | {"SPY"}), "2y")
    daily = {t: df[df.index.date < today].astype("float64") for t, df in data.items()}
    spy_up = _spy_up(daily)
    tf = C.TIMEFRAMES["daily"]
    out = {}
    for t in tickers:
        df = daily.get(t)
        if df is None or len(df) < 260:
            continue
        tail = df.iloc[-20:]
        if tail["Close"].iloc[-1] < C.MIN_PRICE or (tail["Close"] * tail["Volume"]).mean() < C.MIN_AVG_DOLLAR_VOLUME:
            continue
        try:
            lines = armed_levels(df.iloc[-300:], tf, C.PARAMS, C.BOT, spy_up=spy_up)
        except Exception as e:
            log.debug("arm %s: %s", t, e)
            continue
        keep = []
        for x in lines:
            if x["side"] == "long":
                if spy_up is False or x["group"] not in C.BOT_TESTS["strong_levels"]:
                    continue
            elif spy_up is not False:
                continue                       # shorts only while SPY is below its 50-day average
            keep.append(x)
        if keep:
            out[t] = keep
    # earnings blackout
    try:
        nxt = S.next_earnings(sorted(out), root, today)
        for t in list(out):
            if S.earnings_flag(today, nxt.get(t))[0]:
                out.pop(t)
    except Exception as e:
        log.warning("earnings lookup failed: %s", e)
    res = dict(date=str(today), spy_up=spy_up, lines=out, n=sum(len(v) for v in out.values()))
    path.parent.mkdir(exist_ok=True)
    for old in path.parent.glob("armed-*.json"):
        if old != path:
            old.unlink()
    path.write_text(S.jdump(res))
    log.info("armed %d lines on %d tickers (SPY above 50-day: %s)", res["n"], len(out), spy_up)
    return res


def check(root: Path, armed: dict, download_intraday, now: dt.datetime) -> list[dict]:
    """5-minute stop run + reclaim on the armed lines; record new signals."""
    today = now.date()
    lines = armed.get("lines") or {}
    if not lines:
        return []
    recs = S.load(root)
    keys = {r["key"] for r in recs}
    busy = {r["ticker"] for r in recs if r.get("strategy") in ("bot_5m", "bot_5m_watch") and r.get("status") != "closed"}
    bars = download_intraday(sorted(lines), today)
    new = []
    for t, ls in lines.items():
        b = bars.get(t)
        if b is None or not len(b) or t in busy:
            continue
        it = evaluate_sweep_5m(ls, b, C.BOT, confirm="close")
        if not it or it["entry_time"] >= C.BOT_TESTS["before"]:
            continue
        short = it["side"] == "short"
        mr = C.BOT["fvg"]["min_risk_pct"]                     # stop at least 1.5% away (approved)
        far = it["entry"] * (1 + mr / 100) if short else it["entry"] * (1 - mr / 100)
        stop = max(it["stop"], far) if short else min(it["stop"], far)
        risk = abs(it["entry"] - stop) / it["entry"] * 100
        if risk > C.BOT["max_risk_pct"]:
            continue
        watch = short and not C.BOT_LIVE.get("trade_shorts", False)
        strat = "bot_5m_watch" if watch else "bot_5m"
        key = f"{t}|daily|{it['direction']}|{today}|{strat}"
        if key in keys:
            continue
        size = 0.0 if watch else (C.BOT["short_size"] if short else C.BOT["long_size"])
        rec = dict(key=key, id=f"{t}-{today}-{it['side']}-{strat}", ticker=t, timeframe="daily", strategy=strat,
                   direction=it["direction"], side=it["side"], rev_date=str(today), signal_date=str(today),
                   first_seen=f"{today}T{it['entry_time']}", entry_time=it["entry_time"], sweep_time=it["sweep_time"],
                   stage="sweep5m", stages_seen=["sweep5m"], scan_price=it["entry"], entry=it["entry"],
                   stop=round(stop, 4), risk_pct=round(risk, 2), atr=it["atr"], level=it["level"],
                   level_label=it["level_label"], level_group=it["level_group"], confluence=it["confluence"],
                   count=it["count"], chase_pct=it["chase_pct"], spy_up=armed.get("spy_up"), size=size,
                   journal=t in getattr(C, "JOURNAL_UNIVERSE", []), keys=it["keys"], factors=it["factors"],
                   idea=it["idea"].replace(f"stop {it['stop']:.2f}", f"stop {stop:.2f}")
                   .replace(f"({it['risk_pct']:.1f}% risk)", f"({risk:.1f}% risk)")
                   + (" Watch only (shorts are paused)." if watch else f" Size {size}x. Paper trade."),
                   valid_at_last_scan=True, last_reason="", status="open", outcome={}, paper=not watch,
                   watch=watch, live=True)
        recs.append(rec)
        keys.add(key)
        new.append(rec)
    if new:
        S.save(root, recs)
    return new


def follow(root: Path, download, download_intraday, today: dt.date) -> None:
    """Manage open bot trades exactly as the backtest does (entry-day remainder, then daily bars)."""
    recs = S.load(root)
    todo = [r for r in recs if r.get("strategy") in ("bot_5m", "bot_5m_watch") and r.get("status") != "closed"]
    if not todo:
        return
    daily = download(sorted({r["ticker"] for r in todo}), "3mo")
    for r in todo:
        d0 = dt.date.fromisoformat(r["signal_date"])
        m5 = download_intraday([r["ticker"]], d0).get(r["ticker"])
        df = daily.get(r["ticker"])
        if m5 is None or df is None:
            continue
        m5 = m5.copy()
        m5.index = m5.index.tz_localize(None) if m5.index.tz is not None else m5.index
        try:
            S.evaluate_bot_intraday(r, df.astype("float64"), m5, today, bar_min=5, key=r["strategy"])
        except Exception as e:
            log.warning("follow %s: %s", r["id"], e)
    S.save(root, recs)


def summary(new: list[dict], armed: dict) -> str:
    lines = [f"### Bot (paper) {armed.get('date')}: {armed.get('n', 0)} armed lines, SPY above 50-day: {armed.get('spy_up')}"]
    lines += [f"- **{r['ticker']}** {r['side'].upper()} {r['entry']:.2f} at {r['entry_time']} ET, stop {r['stop']:.2f} "
              f"({r['risk_pct']}% risk), line {r['level']:.2f} ({r['level_label']})"
              + (" - watch only, shorts paused" if r.get("watch") else "") for r in new] or ["- No new signals this run."]
    return "\n".join(lines)


def run(root: Path, download, download_intraday, load_universe, force: bool = False) -> int:
    now = dt.datetime.now(NY)
    if not force and not (now.weekday() < 5 and WINDOW[0] <= now.strftime("%H:%M") <= WINDOW[1]):
        log.info("Outside the bot window (%s ET).", now.strftime("%H:%M"))
        try:
            follow(root, download, download_intraday, now.date())
        except Exception as e:
            log.warning("follow failed: %s", e)
        return 0
    armed = arm(root, download, load_universe, now.date())
    new = check(root, armed, download_intraday, now)
    for r in new:
        log.info("BOT SIGNAL %s", r["idea"])
    try:
        follow(root, download, download_intraday, now.date())
    except Exception as e:
        log.warning("follow failed: %s", e)
    text = summary(new, armed)
    gh = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh:
        with open(gh, "a") as f:
            f.write(text + "\n")
    print(text)
    return 0
