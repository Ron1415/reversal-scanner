"""Suggestions tracker: every setup the scanner suggests, on every timeframe,
frozen when first seen and then followed forward with real prices.

For each suggestion two entry versions are tested:
  close   entered at the close of the signal bar (the 3:30-4:00 PM "sleeper hold")
  safer   entered only if price breaks the reversal bar's low (short) / high (long)
          within TRACK["trigger_bars"] bars; cancelled if the stop is hit first
Each entered trade is managed the same way (Ron's plan):
  half at T1 (+5%), stop moves to breakeven, the rest at T2 (+10%);
  the stop is the scanner's stop; anything open after TRACK["max_hold_bars"] bars
  exits at that bar's close. When one bar touches both the stop and a target, the
  stop is assumed to come first. Full-T1 and full-T2 results are kept for comparison.

Earnings: the next earnings date is stored with each suggestion. Suggestions with
earnings within TRACK["earnings_blackout_days"] trading days are flagged; the bot
skips them, and the review reports them separately.

Files (repository root, committed by the workflow):
  suggestions.json   every suggestion and its outcome
  site/data/suggestions.json, site/data/review.json   copies for the web page
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config as C

NY = ZoneInfo("America/New_York")
log = logging.getLogger("suggestions")
T = C.TRACK


# --------------------------------------------------------------------------- #
# Earnings dates
# --------------------------------------------------------------------------- #
def next_earnings(tickers: list[str], root: Path, today: dt.date) -> dict[str, str | None]:
    """Next earnings date per ticker (cached for a day in earnings_cache.json)."""
    f = root / "earnings_cache.json"
    cache = json.loads(f.read_text()) if f.exists() else {}
    out = {}
    todo = [t for t in tickers if cache.get(t, {}).get("checked") != today.isoformat()]
    if todo:
        try:
            import yfinance as yf
        except Exception:
            yf = None
        for t in todo:
            nxt = None
            if yf is not None:
                try:
                    ed = yf.Ticker(t).get_earnings_dates(limit=12)
                    if ed is not None and len(ed):
                        days = sorted({pd.Timestamp(x).tz_localize(None).date() if pd.Timestamp(x).tzinfo is None
                                       else pd.Timestamp(x).tz_convert(NY).date() for x in ed.index})
                        fut = [d for d in days if d >= today]
                        nxt = fut[0].isoformat() if fut else None
                except Exception as e:
                    log.debug("earnings lookup failed for %s: %s", t, e)
            cache[t] = dict(checked=today.isoformat(), next=nxt)
        f.write_text(json.dumps(cache))
    for t in tickers:
        out[t] = cache.get(t, {}).get("next")
    return out


def trading_days_until(d0: dt.date, d1: dt.date) -> int:
    return int(np.busday_count(d0, d1)) if d1 >= d0 else -1


def earnings_flag(signal: dt.date, nxt: str | None) -> tuple[bool, int | None]:
    if not nxt:
        return False, None
    n = trading_days_until(signal, dt.date.fromisoformat(nxt))
    return (0 <= n <= T["earnings_blackout_days"]), n


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
def load(root: Path) -> list[dict]:
    f = root / "suggestions.json"
    return json.loads(f.read_text()) if f.exists() else []


def save(root: Path, recs: list[dict]) -> None:
    (root / "suggestions.json").write_text(json.dumps(recs, indent=0, default=float))


def record(root: Path, tf: str, rows: list[dict], now: dt.datetime) -> int:
    """Add new suggestions and update ones already seen today. Key = ticker,
    timeframe, direction and the date of the reversal bar."""
    recs = load(root)
    by_key = {r["key"]: r for r in recs}
    added = 0
    stamp = now.isoformat(timespec="minutes")
    for r in rows:
        rev = r.get("rev_date") or r.get("bar_date")
        key = f"{r['ticker']}|{tf}|{r['direction']}|{rev}"
        rec = by_key.get(key)
        if rec is None:
            if not r.get("qualifies"):
                continue                          # only suggestions the scanner actually listed
            rec = dict(
                key=key, id=f"{r['ticker']}-{tf}-{rev}-{r['direction'][:4]}", ticker=r["ticker"], timeframe=tf,
                direction=r["direction"], side="short" if r["direction"] == "bearish" else "long",
                rev_date=rev, signal_date=r.get("bar_date"), first_seen=stamp, stage=r.get("stage"),
                stages_seen=[r.get("stage")], scan_price=r.get("close"), stop=r.get("stop"),
                score=r.get("score"), count=r.get("count"), moved_pct=r.get("moved_pct"),
                tail_pct=r.get("tail_pct"), efficiency=r.get("efficiency"), move_atr=r.get("move_atr"),
                rsi=r.get("rsi"), keys=r.get("keys", []), factors=r.get("factors", []), idea=r.get("idea"),
                levels=r.get("levels", []), earnings_next=r.get("earnings_next"),
                earnings_blackout=bool(r.get("earnings_soon")), earnings_days=r.get("earnings_days"),
                status="pending", outcome={})
            recs.append(rec)
            by_key[key] = rec
            added += 1
        rec["last_seen"] = stamp
        rec["valid_at_last_scan"] = bool(r.get("qualifies"))
        rec["last_reason"] = r.get("reason", "")
        if r.get("stage") and r.get("stage") not in rec["stages_seen"]:
            rec["stages_seen"].append(r.get("stage"))
    save(root, recs)
    return added


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def _bars(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = C.TIMEFRAMES[tf]["rule"]
    if rule:
        df = df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last",
                                    "Volume": "sum"}).dropna()
    return df


def simulate(bars: pd.DataFrame, short: bool, entry: float, stop: float, mode: str = "plan") -> dict:
    """Walk forward bar by bar from the bar AFTER entry. mode: plan | t1 | t2."""
    sgn = -1 if short else 1
    t1 = entry * (1 + sgn * T["t1_pct"] / 100)
    t2 = entry * (1 + sgn * T["t2_pct"] / 100)
    risk = abs(entry - stop) / entry * 100
    ret = lambda px: sgn * (px - entry) / entry * 100
    pos, realized, cur_stop, half = 1.0, 0.0, stop, False
    mfe = mae = 0.0
    for i, (ts, b) in enumerate(bars.iterrows(), 1):
        hi, lo, op, cl = float(b.High), float(b.Low), float(b.Open), float(b.Close)
        mfe = max(mfe, ret(lo if short else hi))
        mae = min(mae, ret(hi if short else lo))
        hit_stop = hi >= cur_stop if short else lo <= cur_stop
        if hit_stop:                                   # stop first on a conflicting bar
            px = max(op, cur_stop) if short else min(op, cur_stop)   # gap through the stop fills at the open
            realized += pos * ret(px)
            return dict(done=True, ret=round(realized, 2), r=round(realized / risk, 2) if risk else None,
                        bars=i, exit=("breakeven stop" if half and cur_stop == entry else "stop"),
                        exit_date=str(ts.date()), mfe=round(mfe, 2), mae=round(mae, 2))
        reach = lambda px: (lo <= px) if short else (hi >= px)
        if mode == "t1" and reach(t1):
            return dict(done=True, ret=round(ret(t1), 2), r=round(ret(t1) / risk, 2) if risk else None, bars=i,
                        exit="T1", exit_date=str(ts.date()), mfe=round(mfe, 2), mae=round(mae, 2))
        if mode == "t2" and reach(t2):
            return dict(done=True, ret=round(ret(t2), 2), r=round(ret(t2) / risk, 2) if risk else None, bars=i,
                        exit="T2", exit_date=str(ts.date()), mfe=round(mfe, 2), mae=round(mae, 2))
        if mode == "plan":
            if not half and reach(t1):
                realized += 0.5 * ret(t1)
                pos, half, cur_stop = 0.5, True, entry
            if half and reach(t2):
                realized += 0.5 * ret(t2)
                return dict(done=True, ret=round(realized, 2), r=round(realized / risk, 2) if risk else None,
                            bars=i, exit="T2", exit_date=str(ts.date()), mfe=round(mfe, 2), mae=round(mae, 2))
        if i >= T["max_hold_bars"]:
            realized += pos * ret(cl)
            return dict(done=True, ret=round(realized, 2), r=round(realized / risk, 2) if risk else None,
                        bars=i, exit="time stop", exit_date=str(ts.date()), mfe=round(mfe, 2), mae=round(mae, 2))
    last = float(bars["Close"].iloc[-1]) if len(bars) else entry
    open_ret = realized + pos * ret(last)
    return dict(done=False, ret=round(open_ret, 2), r=round(open_ret / risk, 2) if risk else None,
                bars=len(bars), exit="open", exit_date=None, mfe=round(mfe, 2), mae=round(mae, 2))


def evaluate_one(rec: dict, df: pd.DataFrame, today: dt.date) -> None:
    bars = _bars(df, rec["timeframe"])
    short = rec["side"] == "short"
    stop = float(rec["stop"])
    sig = pd.Timestamp(rec["signal_date"])
    rev = pd.Timestamp(rec["rev_date"])
    # the signal / reversal bars in this timeframe
    # bar containing a date: daily bars are labelled by day, weekly/monthly by the period's end
    sig_i = int(bars.index.searchsorted(sig))
    rev_i = int(bars.index.searchsorted(rev))
    if sig_i >= len(bars) or rev_i >= len(bars):
        return
    if sig_i == len(bars) - 1:
        still_open = (bars.index[sig_i].date() > today) or (
            bars.index[sig_i].date() == today and dt.datetime.now(NY).time() < dt.time(16, 5))
        if still_open:
            return                                 # signal bar still forming; wait for its close
    sb, rb = bars.iloc[sig_i], bars.iloc[rev_i]
    out = {}

    # ---- version 1: entered at the close of the signal bar
    v = dict(entry_rule="close of the signal bar")
    if not rec.get("valid_at_last_scan", True):
        v.update(triggered=False, why=f"not entered: setup no longer valid at the last scan ({rec.get('last_reason') or 'changed'})")
    elif (short and float(sb.High) >= stop) or (not short and float(sb.Low) <= stop):
        v.update(triggered=False, why="not entered: price went through the stop before the close")
    else:
        entry = float(sb.Close)
        v.update(triggered=True, entry=round(entry, 2), entry_date=str(bars.index[sig_i].date()),
                 why="entered at the close", **{"risk_pct": round(abs(entry - stop) / entry * 100, 2)})
        after = bars.iloc[sig_i + 1:]
        v["plan"] = simulate(after, short, entry, stop, "plan")
        v["full_t1"] = simulate(after, short, entry, stop, "t1")
        v["full_t2"] = simulate(after, short, entry, stop, "t2")
    out["close"] = v

    # ---- version 2: safer entry on a break of the reversal bar
    trig = float(rb.Low) if short else float(rb.High)
    w = dict(entry_rule=f"break {'below' if short else 'above'} the reversal bar ({trig:.2f}) within {T['trigger_bars']} bars",
             trigger=round(trig, 2))
    window = bars.iloc[rev_i + 1: rev_i + 1 + T["trigger_bars"]]
    filled = None
    for j, (ts, b) in enumerate(window.iterrows()):
        hi, lo, op = float(b.High), float(b.Low), float(b.Open)
        stop_hit = hi >= stop if short else lo <= stop
        broke = lo < trig if short else hi > trig
        if broke:
            px = min(op, trig) if short else max(op, trig)          # gap past the trigger fills at the open
            if stop_hit and ((short and op >= stop) or (not short and op <= stop)):
                w.update(triggered=False, why="cancelled: opened beyond the stop")
                break
            filled = (bars.index.get_loc(ts), px, ts)
            break
        if stop_hit:
            w.update(triggered=False, why="cancelled: the stop was hit before the trigger")
            break
    if filled:
        k, px, ts = filled
        w.update(triggered=True, entry=round(px, 2), entry_date=str(ts.date()), why="entered on the break",
                 risk_pct=round(abs(px - stop) / px * 100, 2))
        # the fill bar itself can only stop out (conservative)
        fb = bars.iloc[k]
        if (short and float(fb.High) >= stop) or (not short and float(fb.Low) <= stop):
            w["plan"] = w["full_t1"] = w["full_t2"] = dict(done=True, ret=round(-w["risk_pct"], 2), r=-1.0, bars=0,
                                                           exit="stop (same bar as entry)", exit_date=str(ts.date()),
                                                           mfe=0.0, mae=round(-w["risk_pct"], 2))
        else:
            after = bars.iloc[k + 1:]
            w["plan"] = simulate(after, short, px, stop, "plan")
            w["full_t1"] = simulate(after, short, px, stop, "t1")
            w["full_t2"] = simulate(after, short, px, stop, "t2")
    elif "triggered" not in w:
        if len(window) < T["trigger_bars"]:
            w.update(triggered=None, why="waiting for the trigger")
        else:
            w.update(triggered=False, why=f"not triggered: price never broke {trig:.2f} within {T['trigger_bars']} bars")
    out["safer"] = w

    rec["outcome"] = out
    done = all((x.get("triggered") is False) or (x.get("triggered") and x["plan"]["done"]) for x in out.values())
    rec["status"] = "closed" if done else "open"


def sim_scaled(bars: pd.DataFrame, short: bool, entry: float, stop: float, add_px: float | None,
               first: float = 1.0, add: float = 0.0, atr_v: float = 0.0, mgmt: dict | None = None,
               max_hold: int | None = None, first_is_fill: bool = True) -> dict:
    """Level entry with an optional add-on and staged exits, bar by bar starting
    WITH the fill bar.

    mgmt (default = T["mgmt"]):
      be_trigger_atr  move the stop to breakeven once price has gone this many ATRs
                      in favour (None = only after the first target)
      targets         [(pct, share), ...] e.g. 5% half, 7.5% quarter, 10% the rest
      lock_pct        after the FIRST target the stop moves to lock this % profit
    Conservative bar order: adverse first (add-on fill, then the stop), then targets;
    a stop moved on a bar only takes effect from the next bar; the fill bar cannot
    reach a target. Profits are measured against the average price; targets are %
    from the first entry. R = profit / maximum planned risk (all entries filled)."""
    mg = mgmt or T["mgmt"]
    sgn = -1 if short else 1
    tgts = [(entry * (1 + sgn * pct / 100), share) for pct, share in mg["targets"]]
    max_risk = first * abs(entry - stop) + (add * abs(add_px - stop) if add_px is not None and add else 0.0)
    pos, cost, realized = first, first * entry, 0.0
    added, cur, next_stop = False, stop, stop
    hit = 0                                   # targets taken so far
    be_done = False
    start_size = first
    mfe = mae = 0.0
    adverse = lambda b, px: (b.High >= px) if short else (b.Low <= px)
    favour = lambda b, px: (b.Low <= px) if short else (b.High >= px)
    label = lambda: ("locked-profit stop" if hit else "breakeven stop" if be_done else "stop")
    def out(done, exitn, i, ts):
        return dict(done=done, ret=round(realized / entry * 100, 2), r=round(realized / max_risk, 2) if max_risk else None,
                    bars=i, exit=exitn, exit_date=str(ts.date()) if ts is not None else None, added=added,
                    targets_hit=hit, mfe=round(mfe, 2), mae=round(mae, 2))
    ts = None
    for i, (ts, b) in enumerate(bars.iterrows()):
        cur = next_stop                        # stops moved on the previous bar apply now
        mfe = max(mfe, sgn * ((b.Low if short else b.High) - entry) / entry * 100)
        mae = min(mae, sgn * ((b.High if short else b.Low) - entry) / entry * 100)
        if add_px is not None and add and not added and hit == 0 and not be_done and adverse(b, add_px):
            px = add_px if (first_is_fill and i == 0) else (max(b.Open, add_px) if short else min(b.Open, add_px))
            pos += add
            cost += add * px
            start_size += add
            added = True
        if adverse(b, cur):
            px = cur if (first_is_fill and i == 0) else (max(b.Open, cur) if short else min(b.Open, cur))
            realized += sgn * (px * pos - cost)
            return out(True, "stop (same bar as the fill)" if (first_is_fill and i == 0) else label(), i + (0 if first_is_fill else 1), ts)
        if first_is_fill and i == 0:
            continue
        avg = cost / pos
        # staged targets (share of the position size at the time of the first target)
        while hit < len(tgts) and favour(b, tgts[hit][0]):
            px, share = tgts[hit]
            qty = pos if hit == len(tgts) - 1 else min(pos, start_size * share)
            realized += sgn * (px - avg) * qty
            pos -= qty
            cost = avg * pos
            hit += 1
            if pos <= 1e-12:
                return out(True, f"final target {mg['targets'][-1][0]:g}%", i + (0 if first_is_fill else 1), ts)
        # stop management for the NEXT bar
        if hit >= 1:
            lock = avg * (1 + sgn * mg["lock_pct"] / 100)
            next_stop = max(next_stop, lock) if not short else min(next_stop, lock)
        elif mg.get("be_trigger_atr") is not None and atr_v > 0 and not be_done and \
                favour(b, avg + sgn * mg["be_trigger_atr"] * atr_v):
            be_done = True
            next_stop = max(next_stop, avg) if not short else min(next_stop, avg)
        if mg.get("be_trigger_atr") is None and hit >= 1 and not be_done:
            be_done = True
        if mg.get("trail_prev_bar") and (be_done or hit >= 1) and atr_v > 0:
            trail = (b.High + 0.1 * atr_v) if short else (b.Low - 0.1 * atr_v)
            next_stop = min(next_stop, trail) if short else max(next_stop, trail)
        if (i + (0 if first_is_fill else 1)) >= (max_hold or T["max_hold_bars"]):
            realized += sgn * (b.Close * pos - cost)
            return out(True, "time stop", i + (0 if first_is_fill else 1), ts)
    if len(bars):
        last = float(bars["Close"].iloc[-1])
        r = out(False, "open", len(bars) - 1, None)
        ro = realized + sgn * (last * pos - cost)
        r["ret"], r["r"] = round(ro / entry * 100, 2), (round(ro / max_risk, 2) if max_risk else None)
        return r
    return out(False, "open", 0, None)


def evaluate_level(rec: dict, df: pd.DataFrame, today: dt.date) -> None:
    """LEVEL-ENTRY test: a limit order at the level, filled on the signal bar.
    The fill bar can only stop out (we cannot know whether the stop came after
    the fill inside that bar, so the conservative answer is used)."""
    bars = _bars(df, rec["timeframe"])
    short = rec["side"] == "short"
    sig_i = int(bars.index.searchsorted(pd.Timestamp(rec["signal_date"])))
    if sig_i >= len(bars):
        return
    if sig_i == len(bars) - 1 and (bars.index[sig_i].date() > today or (
            bars.index[sig_i].date() == today and dt.datetime.now(NY).time() < dt.time(16, 5))):
        return
    entry, stop = float(rec["entry"]), float(rec["stop"])
    fb = bars.iloc[sig_i]
    risk = abs(entry - stop) / entry * 100
    v = dict(entry_rule=f"limit at the level ({rec['level']:.2f}); stop {T['level_stop_label']}", triggered=True,
             entry=round(entry, 2), entry_date=str(bars.index[sig_i].date()), why="filled at the level",
             risk_pct=round(risk, 2))
    if (short and float(fb.High) >= stop) or (not short and float(fb.Low) <= stop):
        v["plan"] = v["full_t1"] = v["full_t2"] = dict(done=True, ret=round(-risk, 2), r=-1.0, bars=0,
                                                       exit="stop (same bar as the fill)", exit_date=v["entry_date"],
                                                       mfe=0.0, mae=round(-risk, 2))
    else:
        after = bars.iloc[sig_i + 1:]
        v["full_t1"] = simulate(after, short, entry, stop, "t1")
        v["full_t2"] = simulate(after, short, entry, stop, "t2")
    # headline result = Ron's management (breakeven after a sharp move, 1/2 at 5% then lock 2.5%,
    # 1/4 at 7.5%, rest at 10%), with the default 1-ATR stop
    mg0 = dict(T["mgmt"], **({"be_trigger_atr": C.SHORT_MONITORING["be_trigger_atr"]} if short else {}))
    v["plan"] = sim_scaled(bars.iloc[sig_i:], short, entry, stop, None, 1.0, 0.0,
                           abs(stop - float(rec["level"])) / C.PARAMS["level_stop_atr"], mg0,
                           C.SHORT_MONITORING["max_hold_bars"] if short else None)
    rec["outcome"] = {"level": v}
    # ---- variants: different stop distances and add-on entries
    short_ = short
    level = float(rec["level"])
    a = abs(stop - level) / C.PARAMS["level_stop_atr"]          # ATR at the signal
    path = bars.iloc[sig_i:]
    for var in T["level_variants"]:
        sgn = 1 if short_ else -1
        vstop = level + sgn * var["stop_atr"] * a
        add_px = level + sgn * var["add_atr"] * a if var.get("add") else None
        if (short_ and entry >= vstop) or (not short_ and entry <= vstop):
            continue
        mg = dict(T["mgmt"], **var.get("mgmt", {}))
        hold = None
        if short_:                                  # closer monitoring for shorts
            mg["be_trigger_atr"] = C.SHORT_MONITORING["be_trigger_atr"] if mg.get("be_trigger_atr") is not None else None
            hold = C.SHORT_MONITORING["max_hold_bars"]
        p = sim_scaled(path, short_, entry, vstop, add_px, var.get("first", 1.0), var.get("add", 0.0), a, mg, hold)
        rec["outcome"][f"level:{var['name']}"] = dict(triggered=True, entry=round(entry, 2), why=var["label"],
                                                       risk_pct=round(abs(entry - vstop) / entry * 100, 2), plan=p)
    rec["status"] = "closed" if v["plan"]["done"] else "open"


def evaluate_bot(rec: dict, df: pd.DataFrame, today: dt.date) -> None:
    """THE BOT: entered at the close of the sweep bar; managed from the next bar."""
    B = C.BOT
    bars = _bars(df, rec["timeframe"])
    short = rec["side"] == "short"
    sig_i = int(bars.index.searchsorted(pd.Timestamp(rec["signal_date"])))
    if sig_i >= len(bars):
        return
    entry, stop = float(rec["entry"]), float(rec["stop"])
    after = bars.iloc[sig_i + 1:]
    mg = B["short_mgmt"] if short else B["long_mgmt"]
    p = sim_scaled(after, short, entry, stop, None, 1.0, 0.0, float(rec.get("atr") or 0), mg,
                   B["short_max_hold"] if short else B["long_max_hold"], first_is_fill=False)
    v = dict(entry_rule="at the close of the sweep bar", triggered=True, entry=round(entry, 2),
             entry_date=rec["signal_date"], why="entered at the close", risk_pct=rec.get("risk_pct"), plan=p)
    rec["outcome"] = {"bot": v}
    rec["status"] = "closed" if p["done"] else "open"


def evaluate_bot_intraday(rec: dict, daily: pd.DataFrame, hourly: pd.DataFrame, today: dt.date,
                          bar_min: int = 60, key: str = "bot_intraday") -> None:
    """Intraday entry: the rest of the entry day (hourly bars after the entry hour, merged
    into one bar) and then daily bars, managed exactly like the close-entry bot."""
    B = C.BOT
    short = rec["side"] == "short"
    d0 = pd.Timestamp(rec["signal_date"])
    et = rec.get("entry_time", "15:30")
    hr = hourly[hourly.index.date == d0.date()]
    rest = hr[[(t + pd.Timedelta(minutes=bar_min)).strftime("%H:%M") > et for t in hr.index]]
    parts = []
    if len(rest):
        parts.append(pd.DataFrame(dict(Open=[float(rest.Open.iloc[0])], High=[float(rest.High.max())],
                                       Low=[float(rest.Low.min())], Close=[float(rest.Close.iloc[-1])]),
                                  index=[d0]))
    after = daily[daily.index > d0][["Open", "High", "Low", "Close"]]
    path = pd.concat(parts + [after]) if parts else after
    hold = B["short_max_hold"] if short else B["long_max_hold"]
    path = path.iloc[:hold + (1 if parts else 0)]                  # entry-day remainder + N days
    mg = B["short_mgmt"] if short else B["long_mgmt"]
    p = sim_scaled(path, short, float(rec["entry"]), float(rec["stop"]), None, 1.0, 0.0, float(rec.get("atr") or 0),
                   mg, len(path) if len(path) else None, first_is_fill=False)
    if len(path) < hold + (1 if parts else 0) and p.get("exit") == "time stop":
        p["done"], p["exit"] = False, "open"
    rec["outcome"] = {key: dict(entry_rule=f"intraday at {et} ET", triggered=True, entry=rec["entry"],
                                           entry_date=rec["signal_date"], why=f"entered at {et}",
                                           risk_pct=rec.get("risk_pct"), plan=p)}
    rec["status"] = "closed" if p["done"] else "open"


def evaluate(root: Path, download, today: dt.date) -> list[dict]:
    recs = load(root)
    todo = [r for r in recs if r["status"] != "closed"]
    if todo:
        oldest = min(dt.date.fromisoformat(r["rev_date"]) for r in todo)
        days = (today - oldest).days + 40
        period = "3mo" if days < 85 else "1y" if days < 360 else "5y"
        if any(r["timeframe"] != "daily" for r in todo):
            period = "5y"
        prices = download(sorted({r["ticker"] for r in todo}), period)
        for r in todo:
            df = prices.get(r["ticker"])
            if df is not None and len(df):
                try:
                    evaluate_one(r, df, today)
                except Exception as e:
                    log.warning("evaluate %s: %s", r["id"], e)
        save(root, recs)
    return recs


# --------------------------------------------------------------------------- #
# Review: statistics and proposed rule changes (propose only, never applied)
# --------------------------------------------------------------------------- #
def mark_overlaps(recs: list[dict]) -> None:
    """A bot holds one position per stock and direction at a time: a suggestion that
    appears while an earlier one in the same stock is still open (close entry) is
    marked overlap=True and left out of the statistics."""
    busy: dict[str, str] = {}
    for r in sorted(recs, key=lambda x: (x.get("signal_date") or "", x.get("first_seen") or "")):
        k = f"{r['ticker']}|{r['side']}|{r.get('strategy', 'tail')}"
        sig = r.get("signal_date") or ""
        r["overlap"] = bool(busy.get(k) and sig <= busy[k])
        if r["overlap"]:
            continue
        oc = r.get("outcome") or {}
        v = oc.get("close") or oc.get("level") or oc.get("bot") or oc.get("bot_intraday") or oc.get("bot_5m") or oc.get("bot_5m_break") or {}
        if v.get("triggered"):
            p = v.get("plan") or {}
            busy[k] = p.get("exit_date") or "9999-12-31"


def _frame(recs: list[dict], version: str) -> pd.DataFrame:
    mark_overlaps(recs)
    rows = []
    for r in recs:
        if r.get("overlap"):
            continue
        v = r.get("outcome", {}).get(version)
        if not v:
            continue
        p = v.get("plan") or {}
        rows.append(dict(id=r["id"], tf=r["timeframe"], stage=r.get("stage"), side=r["side"],
                         triggered=v.get("triggered"), done=bool(p.get("done")), R=p.get("r"), ret=p.get("ret"),
                         t1=(v.get("full_t1") or {}).get("r"), t2=(v.get("full_t2") or {}).get("r"),
                         mfe=p.get("mfe"), bars=p.get("bars"), blackout=r.get("earnings_blackout", False),
                         count=r.get("count"), score=r.get("score"), moved_pct=r.get("moved_pct"),
                         tail_pct=r.get("tail_pct"), efficiency=r.get("efficiency"), move_atr=r.get("move_atr"),
                         risk_pct=v.get("risk_pct"), keys=set(r.get("keys") or []),
                         passes=bool(r.get("passes_side_rules", True)), signal_date=r.get("signal_date"),
                         group=r.get("level_group"), confluence=r.get("confluence")))
    return pd.DataFrame(rows)


def _stats(name: str, g: pd.DataFrame) -> dict:
    trig = g[g.triggered == True]   # noqa: E712
    closed = trig[trig.done]
    return dict(group=name, suggested=int(len(g)), triggered=int(len(trig)),
                trigger_rate=round(len(trig) / len(g), 3) if len(g) else None, closed=int(len(closed)),
                win_rate=round(float((closed.R > 0).mean()), 3) if len(closed) else None,
                avg_R=round(float(closed.R.mean()), 2) if len(closed) else None,
                avg_ret=round(float(closed.ret.mean()), 2) if len(closed) else None,
                avg_R_full_t1=round(float(closed.t1.mean()), 2) if len(closed) else None,
                avg_R_full_t2=round(float(closed.t2.mean()), 2) if len(closed) else None,
                hit_5pct=round(float((closed.mfe >= T["t1_pct"]).mean()), 3) if len(closed) else None,
                avg_bars=round(float(closed.bars.mean()), 1) if len(closed) else None)


PARAM_FOR = dict(count=("count_min", ">="), tail_pct=("tail_min_pct", ">="), efficiency=("min_efficiency", ">="),
                 move_atr=("min_move_atr", ">="), moved_pct=("late_pct", "<="), score=("(ranking score)", ">="),
                 risk_pct=("(maximum risk %)", "<="))


def review(recs: list[dict], now: dt.datetime) -> dict:
    res = dict(generated=now.strftime("%Y-%m-%d %H:%M %Z"), versions={}, proposals=[], notes=[])
    versions = ["close", "safer"] + [k for k in ("level", "bot", "bot_intraday", "bot_5m", "bot_5m_break") if any(k in (r.get("outcome") or {}) for r in recs)]
    for version in versions:
        d = _frame(recs, version)
        if d.empty:
            res["versions"][version] = dict(summary=None, breakdown=[])
            continue
        tradable = d[~d.blackout]
        groups = [_stats("All suggestions (earnings blackout excluded)", tradable),
                  _stats("Earnings within the blackout (the bot skips these)", d[d.blackout])]
        for col, vals in (("tf", ("daily", "weekly", "monthly")), ("stage", ("forming", "day2", "watch")),
                          ("side", ("long", "short"))):
            for v in vals:
                g = tradable[tradable[col] == v]
                if len(g):
                    groups.append(_stats(f"{col.replace('tf', 'timeframe')}: {v}", g))
        for key in sorted({k for ks in tradable["keys"] for k in ks}):
            g = tradable[tradable["keys"].apply(lambda s: key in s)]
            if len(g) >= 3:
                groups.append(_stats(f"with {key.replace('liq_', 'level: ').replace('_', ' ')}", g))
        res["versions"][version] = dict(summary=groups[0], breakdown=groups)

        # ---- proposals: would a stricter threshold have improved results?
        closed = tradable[(tradable.triggered == True) & tradable.done & tradable.R.notna()]  # noqa: E712
        for col, (param, op) in PARAM_FOR.items():
            x = closed.dropna(subset=[col])
            if len(x) < 2 * T["min_group"]:
                continue
            for q in (0.25, 0.5, 0.75):
                cut = float(x[col].quantile(q))
                keep = x[x[col] >= cut] if op == ">=" else x[x[col] <= cut]
                drop = x.drop(keep.index)
                if len(keep) >= T["min_group"] and len(drop) >= T["min_group"]:
                    gain = keep.R.mean() - drop.R.mean()
                    if gain >= T["min_edge_r"]:
                        res["proposals"].append(dict(
                            version=version, param=param, change=f"{col} {op} {cut:.2f}",
                            evidence=f"{len(keep)} trades with {col} {op} {cut:.2f} averaged {keep.R.mean():+.2f}R "
                                     f"vs {drop.R.mean():+.2f}R for the {len(drop)} others",
                            gain_R=round(float(gain), 2)))
        for col, vals in (("stage", ("forming", "day2", "watch")), ("tf", ("daily", "weekly", "monthly"))):
            for v in vals:
                g, rest = closed[closed[col] == v], closed[closed[col] != v]
                if len(g) >= T["min_group"] and len(rest) >= T["min_group"] and rest.R.mean() - g.R.mean() >= T["min_edge_r"]:
                    res["proposals"].append(dict(
                        version=version, param=f"{col} = {v}", change=f"drop {col} '{v}' from the bot",
                        evidence=f"{len(g)} trades averaged {g.R.mean():+.2f}R vs {rest.R.mean():+.2f}R for the rest",
                        gain_R=round(float(rest.R.mean() - g.R.mean()), 2)))
    # ---- level-entry variants side by side (overall, long, short, daily, weekly)
    var_rows = []
    for var in T["level_variants"]:
        d = _frame(recs, f"level:{var['name']}")
        if d.empty:
            continue
        d = d[(~d.blackout) & (d.triggered == True) & d.done]   # noqa: E712
        row = dict(name=var["name"], label=var["label"])
        rules = d[d.passes]
        dates = sorted(d.signal_date.dropna())
        mid = dates[len(dates) // 2] if dates else ""
        for gname, g in (("all", d), ("long", d[d.side == "long"]), ("short", d[d.side == "short"]),
                         ("new rules", rules), ("new rules long", rules[rules.side == "long"]),
                         ("new rules short", rules[rules.side == "short"]),
                         ("new rules 1st half", rules[rules.signal_date < mid]),
                         ("new rules 2nd half", rules[rules.signal_date >= mid]),
                         ("daily", d[d.tf == "daily"]), ("weekly", d[d.tf == "weekly"]),
                         ("weekly long", d[(d.tf == "weekly") & (d.side == "long")]),
                         ("daily long", d[(d.tf == "daily") & (d.side == "long")])):
            row[gname] = dict(n=int(len(g)), win=round(float((g.R > 0).mean()), 3) if len(g) else None,
                              avg_R=round(float(g.R.mean()), 2) if len(g) else None,
                              avg_ret=round(float(g.ret.mean()), 2) if len(g) else None)
        # the bot policy: long bias (sizes per side, optional skip of daily shorts)
        bp = C.BOT_POLICY
        pol = d[d.passes] if bp.get("use_side_rules") else d
        pol = pol if bp["daily_shorts"] else pol[~((pol.tf == "daily") & (pol.side == "short"))]
        size = np.where(pol.side == "long", bp["long_size"], bp["short_size"])
        w = pol.ret * size
        row["policy"] = dict(n=int(len(pol)), win=round(float((pol.R > 0).mean()), 3) if len(pol) else None,
                             avg_R=round(float((pol.R * size).sum() / size.sum()), 2) if len(pol) else None,
                             avg_ret=round(float(w.mean()), 2) if len(pol) else None,
                             total_ret=round(float(w.sum()), 1))
        var_rows.append(row)
    res["level_variants"] = var_rows
    # ---- THE BOT: headline, long vs short, halves, trades per week (close entry and intraday entry)
    for bot_key in ("bot", "bot_intraday", "bot_5m", "bot_5m_break"):
        _bot_summary(res, recs, bot_key)
    res["bot_policy"] = C.BOT_POLICY
    res["side_rules"] = C.SIDE_RULES
    # level families by side (the evidence behind SIDE_RULES)
    d0 = _frame(recs, "level")
    if not d0.empty:
        d0 = d0[(~d0.blackout) & (d0.triggered == True) & d0.done & d0.group.notna()]   # noqa: E712
        res["level_groups"] = [dict(group=g, side=sd, n=int(len(x)), avg_R=round(float(x.R.mean()), 2),
                                    win=round(float((x.R > 0).mean()), 3))
                               for (g, sd), x in d0.groupby(["group", "side"])]
    res["proposals"].sort(key=lambda p: -p["gain_R"])
    n_closed = sum(1 for r in recs if r["status"] == "closed")
    if n_closed < 2 * T["min_group"]:
        res["notes"].append(f"Early days: {n_closed} suggestions finished so far. Proposals need at least "
                            f"{T['min_group']} finished trades on each side of a comparison and an edge of "
                            f"{T['min_edge_r']}R, so expect the first proposals after a few weeks.")
    res["notes"].append("Proposals are never applied automatically. Approve them and the scanner and the "
                        "TradingView script are updated together.")
    return res


def _bot_summary(res: dict, recs: list[dict], bot_key: str) -> None:
    d = _frame(recs, bot_key)
    if d.empty:
        return
    d = d[(~d.blackout) & (d.triggered == True) & d.done]   # noqa: E712
    if d.empty:
        return
    dates = sorted(d.signal_date.dropna())
    mid = dates[len(dates) // 2] if dates else ""
    weeks = max(1.0, (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 7) if dates else 1.0
    size = np.where(d.side == "long", C.BOT["long_size"], C.BOT["short_size"])

    def blk(g, sz=None):
        if not len(g):
            return dict(n=0)
        w = g.ret * (sz if sz is not None else 1)
        return dict(n=int(len(g)), per_week=round(len(g) / weeks, 1), win=round(float((g.R > 0).mean()), 3),
                    avg_R=round(float(g.R.mean()), 2), avg_ret=round(float(g.ret.mean()), 2),
                    avg_days=round(float(g.bars.mean()), 1), hit_5pct=round(float((g.mfe >= T["t1_pct"]).mean()), 3),
                    total_ret_sized=round(float(w.sum()), 1))
    res[bot_key] = dict(all=blk(d, size), long=blk(d[d.side == "long"]), short=blk(d[d.side == "short"]),
                        first_half=blk(d[d.signal_date < mid]), second_half=blk(d[d.signal_date >= mid]),
                        rules=C.BOT)


def run(root: Path, download, now: dt.datetime) -> None:
    recs = evaluate(root, download, now.date())
    rev = review(recs, now)
    out = root / "site" / "data"
    out.mkdir(parents=True, exist_ok=True)
    lean = [{k: v for k, v in r.items() if k not in ("levels",)} for r in recs[-3000:]]
    (out / "suggestions.json").write_text(json.dumps(lean, default=float))
    (out / "review.json").write_text(json.dumps(rev, default=float))
    log.info("suggestions: %d tracked, %d closed, %d proposals", len(recs),
             sum(r["status"] == "closed" for r in recs), len(rev["proposals"]))
