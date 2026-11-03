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
from setup import evaluate_level_touch, evaluate_setup, evaluate_sweep, evaluate_sweep_intraday, armed_levels, evaluate_sweep_5m, evaluate_fvg_5m

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


def _download_hourly(tickers: list[str], days: int) -> dict[str, pd.DataFrame]:
    """Hourly bars (New York time, naive timestamps = bar START). Yahoo keeps ~730 days."""
    return _download_intraday(tickers, "730d" if days > 230 else "1y", "60m")


def _download_5m(tickers: list[str], days: int) -> dict[str, pd.DataFrame]:
    """5-minute bars (New York time, bar START). Yahoo keeps only the last 60 days."""
    return _download_intraday(tickers, "60d", "5m")


def _download_intraday(tickers: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    out = {}
    try:
        import yfinance as yf
    except Exception:
        return out
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        try:
            raw = yf.download(batch, period=period, interval=interval, group_by="ticker", auto_adjust=True,
                              threads=True, progress=False)
        except Exception as e:
            log.warning("%s download failed: %s", interval, e)
            continue
        for t in batch:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().astype("float64")
                if interval == "5m":
                    df = df.between_time("09:30", "15:55") if df.index.tz is None else df
                idx = pd.to_datetime(df.index)
                idx = idx.tz_convert(NY) if idx.tz is not None else idx.tz_localize("UTC").tz_convert(NY)
                df.index = idx.tz_localize(None)
                if len(df):
                    out[t] = df
            except Exception:
                pass
        time.sleep(2)
    return out


def run(root: Path, download, load_universe, days: int = 60, tfs=("daily", "weekly"),
        download_hourly=None, download_5m=None) -> dict:
    t0 = time.time()
    tickers = sorted(set(load_universe()) | set(getattr(C, "JOURNAL_UNIVERSE", [])))
    # ~300 bars of history are needed before the first replayed day
    data = download(sorted(set(tickers) | {"SPY"}), "6y" if "weekly" in tfs else ("2y" if days <= 180 else "5y"))
    hourly = (download_hourly or _download_hourly)(sorted(data), days)
    log.info("hourly data for %d tickers", len(hourly))
    m5 = (download_5m or _download_5m)(sorted(data), days)
    log.info("5-minute data for %d tickers", len(m5))
    spy = data.pop("SPY", None)
    spy_up = None
    if spy is not None and len(spy) > 60:
        sc = spy["Close"].astype("float64")
        spy_up = (sc > sc.rolling(50).mean())
        spy_up.index = spy_up.index.date
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
                sub = bars.iloc[max(0, i - 300):i + 1]
                # ---- THE BOT (daily only): trend pullback + liquidity sweep
                if tf == "daily":
                    tail_ = sub.iloc[-20:]
                    liquid = tail_["Close"].iloc[-1] >= C.MIN_PRICE and (tail_["Close"] * tail_["Volume"]).mean() >= C.MIN_AVG_DOLLAR_VOLUME
                    if liquid:
                        sig_d = sub.index[-1].date()
                        mkt = None if spy_up is None else bool(spy_up.get(sig_d, True))
                        try:
                            bt = evaluate_sweep(sub, tfc, C.PARAMS, C.BOT, spy_up=mkt)
                        except Exception as e:
                            bt = None
                            log.debug("bot %s %s: %s", t, sig_d, e)
                        if bt:
                            k3 = f"{t}|daily|{bt['direction']}|{sig_d}|bot"
                            recs.setdefault(k3, dict(
                                key=k3, id=f"{t}-{sig_d}-{bt['side']}-bot", ticker=t, timeframe="daily", strategy="bot",
                                direction=bt["direction"], side=bt["side"], rev_date=str(sig_d), signal_date=str(sig_d),
                                first_seen=f"{sig_d}T15:30", stage="sweep", stages_seen=["sweep"],
                                scan_price=bt["entry"], entry=bt["entry"], stop=bt["stop"], risk_pct=bt["risk_pct"],
                                atr=bt["atr"], level=bt["level"], level_label=bt["level_label"],
                                level_group=bt["level_group"], confluence=bt["confluence"], count=bt["count"],
                                vol_ratio=bt["vol_ratio"], spy_up=mkt, short_flags=bt.get("short_flags"),
                                short_strict=bt.get("short_strict"), score=bt["score"], keys=bt["keys"], factors=bt["factors"],
                                idea=bt["idea"], valid_at_last_scan=True, last_reason="", status="pending",
                                outcome={}, backtest=True))
                # ---- THE BOT, intraday: hourly checks during the day with yesterday's daily context
                hr = hourly.get(t)
                if tf == "daily" and hr is not None and i > 0 and liquid and \
                        runs[i - 1] >= min(C.BOT["long_pullback"], C.BOT["short_pullback"]):
                    d_day = bars.index[i].date()
                    today_h = hr[hr.index.date == d_day]
                    ctx = bars.iloc[max(0, i - 300):i]
                    mkt = None if spy_up is None else bool(spy_up.get(d_day, True))
                    for k in range(1, len(today_h) + 1):
                        tb = today_h.iloc[:k]
                        end_t = (tb.index[-1] + pd.Timedelta(hours=1)).strftime("%H:%M")
                        if end_t not in C.BOT["intraday_checks"]:
                            continue
                        try:
                            it = evaluate_sweep_intraday(ctx, tb, tfc, C.PARAMS, C.BOT, spy_up=mkt)
                        except Exception as e:
                            it = None
                            log.debug("intraday %s %s: %s", t, d_day, e)
                        if it:
                            k4 = f"{t}|daily|{it['direction']}|{d_day}|bot_intraday"
                            recs.setdefault(k4, dict(
                                key=k4, id=f"{t}-{d_day}-{it['side']}-botI", ticker=t, timeframe="daily",
                                strategy="bot_intraday", direction=it["direction"], side=it["side"],
                                rev_date=str(d_day), signal_date=str(d_day), first_seen=f"{d_day}T{it['entry_time']}",
                                entry_time=it["entry_time"], stage="sweep", stages_seen=["sweep"],
                                scan_price=it["entry"], entry=it["entry"], stop=it["stop"], risk_pct=it["risk_pct"],
                                atr=it["atr"], level=it["level"], level_label=it["level_label"],
                                level_group=it["level_group"], confluence=it["confluence"], count=it["count"],
                                chase_pct=it.get("chase_pct"), vol_ratio=it["vol_ratio"], score=it["score"],
                                keys=it["keys"], factors=it["factors"], idea=it["idea"], valid_at_last_scan=True,
                                last_reason="", status="pending", outcome={}, backtest=True))
                            break                    # first trigger of the day only
                # ---- THE BOT, 5-minute execution (v20): daily chart arms the line, 5-min times the entry
                f5 = m5.get(t)
                if tf == "daily" and f5 is not None and i > 0 and liquid:
                    d_day = bars.index[i].date()
                    today5 = f5[f5.index.date == d_day]
                    if len(today5):
                        ctx = bars.iloc[max(0, i - 300):i]
                        mkt = None if spy_up is None else bool(spy_up.get(d_day, True))
                        try:
                            armed = armed_levels(ctx, tfc, C.PARAMS, C.BOT, spy_up=mkt)
                        except Exception as e:
                            armed = []
                            log.debug("armed %s %s: %s", t, d_day, e)
                        lo_d, hi_d = float(bars["Low"].iloc[i]), float(bars["High"].iloc[i])
                        armed = [x for x in armed if (lo_d < x["price"]) if x["side"] == "long"] + \
                                [x for x in armed if x["side"] == "short" and hi_d > x["price"]]
                        tries = [(c_, "bot_5m" if c_ == "close" else "bot_5m_break") for c_ in C.BOT["m5"]["confirm"]]
                        tries += [("fvg_limit", "bot_fvg_limit"), ("fvg_confirm", "bot_fvg_confirm"),
                                  ("fvg_half", "bot_fvg_half"), ("fvg_ote", "bot_fvg_ote")]
                        for conf, strat in (tries if armed else []):
                            if conf.startswith("fvg_"):
                                if conf in ("fvg_half", "fvg_ote"):
                                    it = evaluate_fvg_5m(armed, today5, C.BOT, mode="limit", entry_at=conf[4:])
                                else:
                                    it = evaluate_fvg_5m(armed, today5, C.BOT, mode=conf[4:])
                            else:
                                it = evaluate_sweep_5m(armed, today5, C.BOT, confirm=conf)
                            if not it:
                                continue
                            k5 = f"{t}|daily|{it['direction']}|{d_day}|{strat}"
                            recs.setdefault(k5, dict(
                                key=k5, id=f"{t}-{d_day}-{it['side']}-{strat}", ticker=t, timeframe="daily",
                                strategy=strat, direction=it["direction"], side=it["side"],
                                rev_date=str(d_day), signal_date=str(d_day), first_seen=f"{d_day}T{it['entry_time']}",
                                entry_time=it["entry_time"], sweep_time=it["sweep_time"], stage="sweep5m",
                                stages_seen=["sweep5m"], scan_price=it["entry"], entry=it["entry"], stop=it["stop"],
                                risk_pct=it["risk_pct"], atr=it["atr"], level=it["level"],
                                level_label=it["level_label"], level_group=it["level_group"],
                                confluence=it["confluence"], count=it["count"], chase_pct=it["chase_pct"],
                                sweep_depth_atr=it["sweep_depth_atr"], bars_to_reclaim=it["bars_to_reclaim"],
                                fvg=it.get("fvg"), fvg_ote=it.get("fvg_ote"),
                                journal=t in getattr(C, "JOURNAL_UNIVERSE", []),
                                vol_ratio=it["vol_ratio"], spy_up=mkt, short_flags=it.get("short_flags"),
                                short_strict=it.get("short_strict"), score=it["score"], keys=it["keys"],
                                factors=it["factors"], idea=it["idea"], valid_at_last_scan=True,
                                last_reason="", status="pending", outcome={}, backtest=True))
                if runs[max(0, i - 3):i + 1].max() < C.PARAMS["count_min"]:
                    continue                         # no time count of 7+: the other strategies cannot qualify
                # liquidity filter as of that day
                tail = raw[raw.index <= sub.index[-1]].iloc[-20:]
                if tail["Close"].iloc[-1] < C.MIN_PRICE or (tail["Close"] * tail["Volume"]).mean() < C.MIN_AVG_DOLLAR_VOLUME:
                    continue
                n_eval += 1
                sig = str(sub.index[-1].date())
                # ---- alternative strategy: enter at the liquidity level, no tail needed
                try:
                    lv = evaluate_level_touch(sub, tfc, C.PARAMS, tf, C.SIDE_RULES)
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
                            level_group=lv["level_group"], confluence=lv["confluence"],
                            confluence_groups=lv["confluence_groups"], vol_ratio=lv["vol_ratio"],
                            passes_side_rules=lv["passes_side_rules"], side_rule_reason=lv["side_rule_reason"],
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
            elif r.get("strategy") == "bot":
                S.evaluate_bot(r, df.astype("float64"), today)
            elif r.get("strategy") in ("bot_5m", "bot_5m_break", "bot_fvg_limit", "bot_fvg_confirm", "bot_fvg_half", "bot_fvg_ote"):
                if m5.get(r["ticker"]) is not None:
                    S.evaluate_bot_intraday(r, df.astype("float64"), m5[r["ticker"]], today, bar_min=5,
                                            key=r["strategy"])
            elif r.get("strategy") == "bot_intraday":
                if hourly.get(r["ticker"]) is not None:
                    S.evaluate_bot_intraday(r, df.astype("float64"), hourly[r["ticker"]], today)
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
    (d / "backtest_suggestions.json").write_text(S.jdump(out))
    (d / "backtest_review.json").write_text(S.jdump(rev))
    (root / "backtest_suggestions.json").write_text(S.jdump(out))
    return rev
