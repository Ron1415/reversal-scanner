"""Reversal-setup engine, built around three requirements (all must be true):

  1. REVERSAL BAR SHAPING UP, NOT FINISHED
     The bar at the extreme of the move (highest high / lowest low of the last
     `extreme_lookback` bars) is either
       FORMING  today: a tail (or doji) is forming at the extreme right now
       DAY 2    the tail formed 1-2 bars ago and price is still close to it
       WATCH    today is still pushing into a liquidity level with no tail yet
     Anything that has already moved more than `late_pct` % away from the
     extreme is "too late" and is dropped.

  2. AT LIQUIDITY
     The extreme tagged or swept (traded through, then closed back) a level
     where orders cluster: prior swing highs/lows, equal highs/lows (double
     tops/bottoms), the swing that started a big move, prior week/month
     highs/lows, Fibonacci levels, trend lines, volume-backed zones.

  3. TIME COUNT
     At least `count_min` bars moving in one direction into the extreme, by
     the "bars since the last real opposite bar" count. Small opposite bars
     (noise) do not reset the count. The move must also be orderly (not chop)
     and large enough to be worth fading.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import atr, fib_levels, pivots, rsi, sr_levels, td_setup, trendline, volume_nodes

STAGE_RANK = {"forming": 3, "day2": 2, "watch": 1}


# --------------------------------------------------------------------------- #
# Liquidity map
# --------------------------------------------------------------------------- #
def liquidity_levels(df: pd.DataFrame, a: float, tf: dict, P: dict, side: str) -> list[dict]:
    """Levels built ONLY from bars before the reversal bar (the last row of df
    is the reversal bar). side='high' for resistance above, 'low' for support."""
    prior = df.iloc[:-1]
    k, lb = tf["pivot_k"], tf["sr_lookback"]
    w = prior.iloc[-lb:]
    H, L = w["High"].to_numpy(), w["Low"].to_numpy()
    out: list[dict] = []

    # Swing highs / lows, with "major" (started a big move) and "equal" (double top/bottom)
    col = "High" if side == "high" else "Low"
    pts = pivots(w[col], k, side)
    for i, v in pts:
        nxt = slice(i + 1, i + 1 + 20)
        if side == "high":
            swing = (v - L[nxt].min()) / a if len(L[nxt]) else 0
        else:
            swing = (H[nxt].max() - v) / a if len(H[nxt]) else 0
        twins = sum(1 for j, u in pts if j != i and abs(u - v) <= 0.5 * a)
        major = swing >= P["major_swing_atr"]
        if twins:
            label, strength = ("Equal highs (double top)" if side == "high" else "Equal lows (double bottom)"), 3
        elif major:
            label, strength = ("Major swing high" if side == "high" else "Major swing low"), 3
        else:
            label, strength = ("Prior swing high" if side == "high" else "Prior swing low"), 2
        out.append(dict(label=label, price=float(v), strength=strength))

    # Prior completed week / month high or low (daily charts only)
    if tf["rule"] is None and len(prior) > 30:
        cur = df.index[-1]
        for rule, name in (("W-FRI", "week"), ("ME", "month")):
            per = "W-FRI" if rule == "W-FRI" else "M"
            g = prior.resample(rule).agg({"High": "max", "Low": "min"}).dropna()
            g = g[g.index.to_period(per) < cur.to_period(per)]
            if len(g):
                v = float(g["High" if side == "high" else "Low"].iloc[-1])
                out.append(dict(label=f"Prior {name}'s {'high' if side == 'high' else 'low'}", price=v, strength=2))

    # Fibonacci, trend line, volume-backed zones, high-volume nodes
    fibs = fib_levels(df, tf["lookback"])
    for lab, v in fibs["bearish" if side == "high" else "bullish"]:
        out.append(dict(label=lab, price=float(v), strength=2))
    tl = trendline(df, k, tf["lookback"], side)
    if tl:
        out.append(dict(label=tl[0], price=float(tl[1]), strength=2))
    for z in sr_levels(df, k, lb, a, P):
        if z["major"]:
            out.append(dict(label=z["label"], price=float(z["price"]), strength=2))
    for lab, v in volume_nodes(df, lb, P):
        out.append(dict(label=lab, price=float(v), strength=1))
    return [x for x in out if x["price"] > 0]


def _touch(direction: str, h: float, l: float, c: float, lvl: float, tol: float) -> str | None:
    """'sweep' = traded through the level and closed back; 'tag' = came within tol."""
    if direction == "bearish":
        if h > lvl > c:
            return "sweep"
        return "tag" if abs(h - lvl) <= tol else None
    if l < lvl < c:
        return "sweep"
    return "tag" if abs(l - lvl) <= tol else None


# --------------------------------------------------------------------------- #
# Count and move quality
# --------------------------------------------------------------------------- #
def time_count(df: pd.DataFrame, end: int, up: bool, P: dict, start: int = 0) -> int:
    """Bars ending at `end` (inclusive) since the last REAL opposite bar.
    In a rally a bar only resets the count if it is red with a body of at least
    `count_body_atr` ATRs AND closes below the prior close; small red bars are
    noise and still count. Mirror for a decline."""
    o, c = df["Open"].to_numpy(), df["Close"].to_numpy()
    a = atr(df).to_numpy()
    body = P["count_body_atr"]
    n = 0
    for i in range(end, max(start, 1) - 1, -1):
        x = a[i] if np.isfinite(a[i]) else 0.0
        real_opposite = ((o[i] - c[i]) >= body * x and c[i] < c[i - 1]) if up else \
                        ((c[i] - o[i]) >= body * x and c[i] > c[i - 1])
        if real_opposite:
            break
        n += 1
    return n


def efficiency(close: np.ndarray, i0: int, i1: int) -> float:
    """Net move / total distance travelled (1 = straight line, 0 = chop)."""
    seg = close[max(i0, 0):i1 + 1]
    if len(seg) < 3:
        return 0.0
    path = np.abs(np.diff(seg)).sum()
    return float(abs(seg[-1] - seg[0]) / path) if path > 0 else 0.0


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
def evaluate_setup(df: pd.DataFrame, tf: dict, P: dict, W: dict) -> dict | None:
    n = len(df)
    if n < 80:
        return None
    o, h, l, c = (df[x].to_numpy() for x in ("Open", "High", "Low", "Close"))
    A = atr(df).to_numpy()
    r = rsi(df["Close"]).to_numpy()
    td = td_setup(df["Close"]).to_numpy()
    N = P["extreme_lookback"]
    late_pct = tf.get("late_pct", P["late_pct"])
    candidates = []

    for direction in ("bearish", "bullish"):
        up = direction == "bearish"            # a bearish reversal ends a rally
        why: list[str] = []
        # Reversal bar = the most extreme bar of the last few bars (today, or up to
        # carry_days ago). It must also be the extreme of the whole run into it.
        recent = P["carry_days"] + 1
        ext_pos = n - recent + int(np.argmax(h[-recent:]) if up else np.argmin(l[-recent:]))
        ext = float(h[ext_pos] if up else l[ext_pos])
        age = n - 1 - ext_pos
        a = float(A[ext_pos]) if np.isfinite(A[ext_pos]) else float(A[-1])
        if not a or a <= 0:
            continue
        tol = P["proximity_atr"] * a
        moved = (ext - c[-1]) / ext * 100 if up else (c[-1] - ext) / ext * 100
        cand = dict(direction=direction, ext=ext, ext_pos=ext_pos, age=age, moved_pct=round(moved, 2), why=why,
                    qualified=False)
        candidates.append(cand)

        # ---- the move into the extreme: the current leg only, i.e. since price was last
        # beyond the extreme (so an older, higher high is the level, not a disqualifier)
        lo = max(0, ext_pos - P["move_lookback"])
        j = ext_pos - 1
        while j >= lo and not ((h[j] > ext + tol) if up else (l[j] < ext - tol)):
            j -= 1
        leg = j + 1
        start_pos = leg + int(np.argmin(l[leg:ext_pos + 1]) if up else np.argmax(h[leg:ext_pos + 1]))
        move = (ext - l[start_pos]) if up else (h[start_pos] - ext)
        move_atr = move / a
        cand["move_atr"] = round(move_atr, 1)

        # ---- reversal bar state
        rb = ext_pos
        rng = h[rb] - l[rb]
        if rng <= 0:
            why.append("no range on the reversal bar"); continue
        upper, lower = h[rb] - max(o[rb], c[rb]), min(o[rb], c[rb]) - l[rb]
        tail = upper if up else lower
        pos = (c[rb] - l[rb]) / rng
        in_rev_half = pos <= 0.5 if up else pos >= 0.5
        tail_ok = tail >= P["tail_min_pct"] * rng and tail >= P["tail_min_atr"] * a and in_rev_half
        doji = abs(c[rb] - o[rb]) <= 0.1 * rng and tail >= 0.3 * rng
        cand["tail_pct"] = round(tail / rng, 2)
        cand["close_pos"] = round(pos, 2)

        if age > P["carry_days"]:
            why.append(f"the {'high' if up else 'low'} was {age} bars ago (reversal already under way)")
        if moved > late_pct:
            why.append(f"already {moved:.1f}% from the {'high' if up else 'low'} (limit {late_pct:g}%): reversal under way")

        if age == 0 and (tail_ok or doji):
            stage = "forming"
        elif 0 < age <= P["carry_days"] and (tail_ok or doji):
            stage = "day2"
        elif age == 0 and (pos >= 0.65 if up else pos <= 0.35) and move_atr >= P["watch_move_atr"]:
            stage = "watch"        # still pushing hard into the level, near the day's extreme
        else:
            stage = None
            why.append("no tail on the bar that made the extreme" if age > 0 else
                       "no tail yet and not pushing strongly enough for an early heads-up")
        cand["stage"] = stage

        # ---- time count into the extreme (bars before the reversal bar; a WATCH bar counts itself)
        end = rb if stage == "watch" else rb - 1
        cnt = time_count(df, end, up, P, start=start_pos)
        eff = efficiency(c, end - cnt, end) if cnt else 0.0
        need = P["count_min"]
        cand.update(count=cnt, efficiency=round(eff, 2), td=int(td[end]))
        if cnt < need:
            why.append(f"time count {cnt} (needs {need})")
        if cnt and eff < P["min_efficiency"]:
            why.append("choppy, not a clean move into the level")
        if move_atr < P["min_move_atr"]:
            why.append(f"move into the level too small ({move_atr:.1f} average daily ranges, need {P['min_move_atr']:g})")

        # ---- liquidity at the extreme
        lv = liquidity_levels(df.iloc[:rb + 1], a, tf, P, "high" if up else "low")
        hits = []
        for x in lv:
            if stage == "watch":   # level just ahead of price, or being tagged now
                ahead = (x["price"] - h[rb]) if up else (l[rb] - x["price"])
                # only strong levels (swings, double tops/bottoms, prior week/month, Fibonacci,
                # trend lines, volume zones); not high-volume nodes alone
                kind = "ahead" if (-tol <= ahead <= P["approach_atr"] * a and x["strength"] >= 2) else None
            else:
                kind = _touch(direction, h[rb], l[rb], c[rb], x["price"], tol)
            if kind:
                hits.append(dict(x, kind=kind, strength=x["strength"] + (1 if kind == "sweep" else 0)))
        hits.sort(key=lambda x: (-x["strength"], abs(x["price"] - ext)))
        merged = []                       # one entry per price area (strongest label wins)
        for x in hits:
            if all(abs(x["price"] - m["price"]) > 0.5 * a for m in merged):
                merged.append(x)
        hits = merged
        cand["hits"] = hits[:4]
        if not hits:
            why.append("no support / resistance / liquidity level at the extreme")
        elif stage == "watch" and not (hits[0]["strength"] >= 3 or len(hits) >= 2):
            why.append("level ahead is not strong enough for an early heads-up (needs a major swing, "
                       "double top/bottom, or two levels together)")
        if stage == "watch" and cnt and eff < P["watch_efficiency"]:
            why.append("move not clean enough for an early heads-up")

        # ---- score (ranking only; qualification is the three requirements)
        score = 0.0
        factors = []
        if stage == "forming":
            score += W["stage_forming"]
            factors.append(("Topping" if up else "Bottoming") + f" tail forming at the {'high' if up else 'low'}"
                           f" (tail {tail / rng:.0%} of range)" if tail_ok else "Doji forming at the extreme")
        elif stage == "day2":
            score += W["stage_day2"]
            factors.append(f"{'Topping' if up else 'Bottoming'} tail {age} bar{'s' if age > 1 else ''} ago at the "
                           f"{'high' if up else 'low'}; price still within {moved:.1f}% of it")
        elif stage == "watch":
            score += W["stage_watch"]
            factors.append(f"Still pushing {'up' if up else 'down'} into the level; no tail yet")
        if cnt:
            score += W["count_9"] if cnt >= 9 else W["count"] if cnt >= P["count_min"] else 0
            factors.append(f"Time count {cnt}: {'up' if up else 'down'} bars since the last real "
                           f"{'down' if up else 'up'} bar")
        for x in hits[:3]:
            score += W["liq_strength"] * x["strength"] / 3
            verb = {"sweep": "swept", "tag": "tagged", "ahead": "just ahead:"}[x["kind"]]
            factors.append(f"{'Liquidity ' + verb if x['kind'] != 'ahead' else 'Level just ahead:'} "
                           f"{x['label']} @ {x['price']:.2f}")
        ri = float(r[rb])
        if (up and ri >= P["rsi_hi"]) or (not up and ri <= P["rsi_lo"]):
            score += W["rsi"]; factors.append(f"RSI {'overbought' if up else 'oversold'} ({ri:.0f})")

        stop = ext + 0.1 * a if up else ext - 0.1 * a
        if stage == "watch" and hits:     # no tail yet: the stop belongs beyond the level ahead
            lvl = max(x["price"] for x in hits) if up else min(x["price"] for x in hits)
            stop = max(stop, lvl + 0.25 * a) if up else min(stop, lvl - 0.25 * a)
        entry = float(c[-1])
        risk = abs(stop - entry) / entry * 100
        best = hits[0] if hits else None
        verb = "into" if stage == "watch" else "at"
        idea = (f"Expecting {'DOWN' if up else 'UP'}: {cnt}-bar {'rally' if up else 'decline'} {verb} "
                f"{best['label'].lower() if best else 'a level'}"
                f"{' ' + format(best['price'], '.2f') if best else ''}; "
                + {"forming": f"{'topping' if up else 'bottoming'} tail forming now.",
                   "day2": f"tail {age} bar{'s' if age > 1 else ''} ago, not yet run away.",
                   "watch": "watch for a tail in the next 1-2 days.", None: ""}[stage])
        cand.update(score=round(score, 2), factors=factors, idea=idea, entry=round(entry, 2),
                    stop=round(stop, 2), risk_pct=round(risk, 2), rsi=round(ri, 1),
                    levels=[round(x["price"], 4) for x in hits[:3]],
                    keys=[f"stage_{stage}"] + (["count_9"] if cnt >= 9 else ["count"]) +
                         [f"liq_{x['label'].split(' (')[0].lower().replace(' ', '_')}" for x in hits[:3]] +
                         (["sweep"] if any(x["kind"] == "sweep" for x in hits) else []),
                    has_candle=stage in ("forming", "day2"), n_factors=len(factors), atr=round(a, 4))
        cand["qualified"] = not why and stage is not None

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x["qualified"], STAGE_RANK.get(x.get("stage"), 0), x.get("score", 0)), reverse=True)
    best = candidates[0]
    best.setdefault("factors", []); best.setdefault("score", 0); best.setdefault("levels", [])
    best.setdefault("keys", []); best.setdefault("n_factors", 0); best.setdefault("has_candle", False)
    best["alternatives"] = [dict(direction=x["direction"], why=x["why"]) for x in candidates[1:]]
    return best


# --------------------------------------------------------------------------- #
# Alternative strategy: enter AT the liquidity level (no reversal bar needed)
# --------------------------------------------------------------------------- #
def level_group(label: str) -> str:
    """Level family used by the side rules (config.SIDE_RULES)."""
    for key, grp in (("38.2", "Fib 38.2"), ("50.0", "Fib 50"), ("61.8", "Fib 61.8"), ("78.6", "Fib 78.6"),
                     ("extension", "Fib ext"), ("Trendline", "Trendline"), ("Equal", "Double top/bottom"),
                     ("Major swing", "Major swing"), ("pivot zone", "Volume zone"), ("Prior swing", "Prior swing"),
                     ("Prior week", "Prior week/month"), ("Prior month", "Prior week/month")):
        if key in label:
            return grp
    return label


def side_rules_check(side: str, tf_name: str, group: str, confluence: int, count: int, rules: dict) -> tuple[bool, str]:
    """Stricter rules for shorts, a bit more room for longs (config.SIDE_RULES)."""
    r = rules[side]
    if tf_name not in r["timeframes"]:
        return False, f"{side}s are not traded on the {tf_name} chart"
    if count < r["min_count"]:
        return False, f"{side}s need a time count of {r['min_count']}+ (have {count})"
    if group in r["levels_alone"]:
        return True, f"{group} is allowed on its own for {side}s"
    if group in r["levels_with_confluence"] and confluence >= r["min_confluence"]:
        return True, f"{group} allowed for {side}s with {confluence} levels together"
    if group in r["levels_with_confluence"]:
        return False, f"{group} needs {r['min_confluence']}+ levels together for {side}s (have {confluence})"
    return False, f"{group} is not used for {side}s"


def evaluate_level_touch(df: pd.DataFrame, tf: dict, P: dict, tf_name: str = "daily", rules: dict | None = None) -> dict | None:
    """Same setup as evaluate_setup (time count, orderly move of min_move_atr,
    liquidity map) but the trade is entered the moment the newest bar FIRST
    reaches a strong level: a limit order at the level, no tail required.

    Rules:
      * the level is built from bars before today and was not reached yesterday
        (first touch only); today's high (rally) / low (decline) reaches it
      * strong levels only: strength >= 3 (double top/bottom, major swing) or a
        prior week/month high/low or Fibonacci / trend line / volume zone (>= 2)
      * time count >= count_min on the bars before today, orderliness and move
        size as in the main setup
      * entry = the level (or today's open if price gapped through it);
        stop = level +/- level_stop_atr x ATR
    """
    n = len(df)
    if n < 80:
        return None
    o, h, l, c = (df[x].to_numpy() for x in ("Open", "High", "Low", "Close"))
    A = atr(df).to_numpy()
    a = float(A[-2]) if np.isfinite(A[-2]) else float("nan")
    if not np.isfinite(a) or a <= 0:
        return None
    best = None
    for direction in ("bearish", "bullish"):
        up = direction == "bearish"
        end = n - 2                                   # count the bars before today
        lo = max(0, end - P["move_lookback"])
        start_pos = lo + int(np.argmin(l[lo:end + 1]) if up else np.argmax(h[lo:end + 1]))
        cnt = time_count(df.iloc[:n - 1], end, up, P, start=start_pos)
        if cnt < P["count_min"]:
            continue
        eff = efficiency(c, end - cnt, end)
        if eff < P["min_efficiency"]:
            continue
        lv = liquidity_levels(df, a, tf, P, "high" if up else "low")
        cands = []
        for x in lv:
            p = x["price"]
            if x["strength"] < 2:
                continue
            first_touch = (h[-2] < p and h[-1] >= p) if up else (l[-2] > p and l[-1] <= p)
            if not first_touch:
                continue
            move = (p - l[start_pos]) if up else (h[start_pos] - p)
            if move / a < P["min_move_atr"]:
                continue
            cands.append(x)
        if not cands:
            continue
        # the first level price meets on the way (nearest to yesterday's close), strongest on ties
        x = min(cands, key=lambda z: (abs(z["price"] - c[-2]), -z["strength"]))
        p = x["price"]
        # confluence: distinct levels within 0.5 ATR of the chosen one (all levels, any strength >= 2)
        near = []
        for z in lv:
            if z["strength"] >= 2 and abs(z["price"] - p) <= 0.5 * a and all(abs(z["price"] - q) > 0.1 * a for q in near):
                near.append(z["price"])
        confl = max(1, len(near))
        groups = sorted({level_group(z["label"]) for z in lv if z["strength"] >= 2 and abs(z["price"] - p) <= 0.5 * a})
        side = "short" if up else "long"
        ok_rules, why_rules = side_rules_check(side, tf_name, level_group(x["label"]), confl, cnt, rules) if rules else (True, "")
        vol = df["Volume"].to_numpy()
        vol_ratio = float(vol[-4:-1].mean() / vol[-24:-4].mean()) if vol[-24:-4].mean() > 0 else None
        entry = max(o[-1], p) if up else min(o[-1], p)          # gap through the level fills at the open
        stop = p + P["level_stop_atr"] * a if up else p - P["level_stop_atr"] * a
        if (up and entry >= stop) or (not up and entry <= stop):
            continue                                            # opened beyond the stop: no trade
        cand = dict(direction=direction, stage="level", level=round(p, 4), level_label=x["label"],
                    entry=round(float(entry), 4), stop=round(float(stop), 4), count=cnt, efficiency=round(eff, 2),
                    move_atr=round(((p - l[start_pos]) if up else (h[start_pos] - p)) / a, 1), atr=round(a, 4),
                    keys=["stage_level", "count_9" if cnt >= 9 else "count",
                          f"liq_{x['label'].split(' (')[0].lower().replace(' ', '_')}"],
                    factors=[f"Time count {cnt} into the level", f"{x['label']} @ {p:.2f} reached today"],
                    idea=(f"Level entry: {'sell' if up else 'buy'} at {x['label'].lower()} {p:.2f} after a "
                          f"{cnt}-bar {'rally' if up else 'decline'}; stop {stop:.2f}."),
                    score=round(x["strength"] + (1 if cnt >= 9 else 0) + 0.5 * (confl - 1), 2), qualified=True,
                    level_group=level_group(x["label"]), confluence=confl, confluence_groups=groups,
                    vol_ratio=round(vol_ratio, 2) if vol_ratio else None,
                    passes_side_rules=ok_rules, side_rule_reason=why_rules)
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


# --------------------------------------------------------------------------- #
# THE BOT: trend pullback + liquidity sweep (accumulation -> manipulation -> distribution)
# --------------------------------------------------------------------------- #
def evaluate_sweep(df: pd.DataFrame, tf: dict, P: dict, B: dict, spy_up: bool | None = None,
                   weekly_down: bool | None = None) -> dict | None:
    """Simple, trend-following reversal off a liquidity sweep, daily bars, hold 1-5 days.

    LONG (moderate)                               SHORT (strict)
    1 trend  rising 50-day SMA above the 200-day   close < falling 50-day SMA below the 200-day,
                                                   and the market (SPY) below its 50-day SMA
    2 pullback  >= long_pullback bars down         >= short_pullback bars up (time count rules)
    3 sweep  today trades through a liquidity      the same, at strong levels only (short_levels)
             level and closes back beyond it       and with confluence >= short_confluence
    4 rejection close in the upper half            close in the lower short_close_pos, volume on the
                                                   sweep bar >= short_vol_mult x the 20-day average
    entry at the close; stop beyond the sweep extreme (long_stop_atr / short_stop_atr);
    skip when the risk is above max_risk_pct.
    """
    n = len(df)
    if n < 80:
        return None
    o, h, l, c, v = (df[x].to_numpy() for x in ("Open", "High", "Low", "Close", "Volume"))
    A = atr(df).to_numpy()
    a = float(A[-1]) if np.isfinite(A[-1]) else float("nan")
    if not np.isfinite(a) or a <= 0:
        return None
    sma = pd.Series(c).rolling(50).mean().to_numpy()
    sma200 = pd.Series(c).rolling(200).mean().to_numpy()
    if not np.isfinite(sma[-1]) or not np.isfinite(sma[-11]):
        return None
    rng = h[-1] - l[-1]
    if rng <= 0:
        return None
    pos = (c[-1] - l[-1]) / rng
    best = None
    for side in ("long", "short"):
        up_trade = side == "long"
        # 1. trend: longs = rising 50-day above the 200-day (price may dip below the 50-day on the
        #    pullback); shorts = price below a falling 50-day that is below the 200-day
        if not np.isfinite(sma200[-1]):
            continue
        if up_trade:
            if not (sma[-1] > sma[-11] and sma[-1] > sma200[-1]):
                continue
        else:
            if not (c[-1] < sma[-1] and sma[-1] < sma[-11] and sma[-1] < sma200[-1]):
                continue
            if B["short_need_weekly_down"]:
                s100 = pd.Series(c).rolling(100).mean().to_numpy()        # ~20-week average
                if not (np.isfinite(s100[-1]) and np.isfinite(s100[-26]) and c[-1] < s100[-1] and s100[-1] < s100[-26]):
                    continue
            if B["short_need_market_down"] and spy_up is True:
                continue
        # 2. pullback count into today (bars before today, small opposite bars are noise)
        cnt = time_count(df.iloc[:n - 1], n - 2, not up_trade, P)
        if cnt < (B["long_pullback"] if up_trade else B["short_pullback"]):
            continue
        # 4. rejection (cheap checks before the level map)
        if up_trade and pos < 0.5:
            continue
        if not up_trade:
            if pos > B["short_close_pos"]:
                continue
            avg_v = v[-21:-1].mean()
            if avg_v <= 0 or v[-1] < B["short_vol_mult"] * avg_v:
                continue
        # 3. sweep of a liquidity level (levels built from bars before today)
        lv = liquidity_levels(df, a, tf, P, "low" if up_trade else "high")
        hits = []
        for x in lv:
            p = x["price"]
            if x["strength"] < 2:
                continue
            swept = (l[-1] < p - 0.05 * a and c[-1] > p) if up_trade else (h[-1] > p + 0.05 * a and c[-1] < p)
            if swept:
                hits.append(x)
        # don't chase: only lines that price is still within max_chase_pct of
        mc = B.get("max_chase_pct")
        if mc is not None:
            hits = [z for z in hits if abs(c[-1] - z["price"]) / z["price"] * 100 <= mc]
        if not hits:
            continue
        x = max(hits, key=lambda z: (z["strength"], -abs(z["price"] - c[-1])))
        grp = level_group(x["label"])
        near = []
        for z in lv:
            if z["strength"] >= 2 and abs(z["price"] - x["price"]) <= 0.5 * a and all(abs(z["price"] - q) > 0.1 * a for q in near):
                near.append(z["price"])
        confl = max(1, len(near))
        if not up_trade and (grp not in B["short_levels"] or confl < B["short_confluence"]):
            continue
        entry = float(c[-1])
        # don't chase: skip if price is already more than max_chase_pct away from the liquidity line
        chase = (entry - x["price"]) / x["price"] * 100 if up_trade else (x["price"] - entry) / x["price"] * 100
        stop = float(l[-1] - B["long_stop_atr"] * a) if up_trade else float(h[-1] + B["short_stop_atr"] * a)
        risk = abs(entry - stop) / entry * 100
        if risk > B["max_risk_pct"] or risk < 0.3:
            continue
        cand = dict(direction="bullish" if up_trade else "bearish", side=side, stage="sweep", level=round(x["price"], 4),
                    level_label=x["label"], level_group=grp, confluence=confl, entry=round(entry, 4), stop=round(stop, 4),
                    risk_pct=round(risk, 2), count=cnt, atr=round(a, 4), close_pos=round(pos, 2), chase_pct=round(chase, 2),
                    vol_ratio=round(float(v[-1] / v[-21:-1].mean()), 2) if v[-21:-1].mean() > 0 else None,
                    keys=["stage_sweep", f"liq_{x['label'].split(' (')[0].lower().replace(' ', '_')}"],
                    factors=[f"Trend {'up' if up_trade else 'down'} (50-day average)",
                             f"{cnt}-bar pullback", f"Swept {x['label']} @ {x['price']:.2f} and closed back",
                             f"{confl} level(s) together"],
                    idea=(f"{'BUY' if up_trade else 'SELL'} at the close ~{entry:.2f}: {'uptrend' if up_trade else 'downtrend'}, "
                          f"{cnt}-bar pullback swept {x['label'].lower()} {x['price']:.2f} and reclaimed it; stop {stop:.2f} "
                          f"({risk:.1f}% risk)."),
                    score=round(x["strength"] + 0.5 * (confl - 1) + (0.5 if cnt >= 7 else 0), 2), qualified=True)
        if best is None or (cand["side"] == "long" and best["side"] == "short") or cand["score"] > best["score"]:
            best = cand
    return best


def evaluate_sweep_intraday(daily: pd.DataFrame, today_bars: pd.DataFrame, tf: dict, P: dict, B: dict,
                            spy_up: bool | None = None) -> dict | None:
    """The same bot rules, checked DURING the day.

    daily       daily bars up to and including YESTERDAY (the context: trend, pullback, levels)
    today_bars  today's hourly bars so far (New York time)
    Today's bar is built from the hourly bars so far (open, high so far, low so far, last
    price, volume so far projected to a full day) and evaluate_sweep is run on it, so a
    sweep that has already reclaimed the line triggers an entry at the current price,
    provided price is still within max_chase_pct of the line."""
    if not len(today_bars):
        return None
    last_t = today_bars.index[-1] + pd.Timedelta(hours=1)          # bar END time
    mins = (last_t.hour * 60 + last_t.minute) - (9 * 60 + 30)
    frac = min(max(mins / 390, 0.1), 1.0)
    row = pd.DataFrame(dict(Open=[float(today_bars["Open"].iloc[0])], High=[float(today_bars["High"].max())],
                            Low=[float(today_bars["Low"].min())], Close=[float(today_bars["Close"].iloc[-1])],
                            Volume=[float(today_bars["Volume"].sum()) / frac]),
                       index=[pd.Timestamp(today_bars.index[-1].date())])
    res = evaluate_sweep(pd.concat([daily, row]), tf, P, B, spy_up=spy_up)
    if res:
        res["entry_time"] = last_t.strftime("%H:%M")
    return res


def armed_levels(ctx: pd.DataFrame, tf: dict, P: dict, B: dict, spy_up: bool | None = None) -> list[dict]:
    """Daily-chart context for the 5-minute execution (v20).

    ctx = daily bars up to and including YESTERDAY. Returns the liquidity lines that are
    'armed' for today: the trend and pullback rules of the bot pass, and the line is a
    strong liquidity level that lines up with other support/resistance (confluence)."""
    M = B["m5"]
    n = len(ctx)
    if n < 210:
        return []
    c = ctx["Close"].to_numpy()
    A = atr(ctx).to_numpy()
    a = float(A[-1]) if np.isfinite(A[-1]) else float("nan")
    if not np.isfinite(a) or a <= 0:
        return []
    sma = pd.Series(c).rolling(50).mean().to_numpy()
    sma200 = pd.Series(c).rolling(200).mean().to_numpy()
    if not (np.isfinite(sma[-1]) and np.isfinite(sma[-11]) and np.isfinite(sma200[-1])):
        return []
    dummy = pd.concat([ctx, ctx.iloc[[-1]]])          # liquidity_levels ignores the last row
    out = []
    for side in ("long", "short"):
        up = side == "long"
        if up:
            if not (sma[-1] > sma[-11] and sma[-1] > sma200[-1]):
                continue
        else:
            if not (c[-1] < sma[-1] and sma[-1] < sma[-11] and sma[-1] < sma200[-1]):
                continue
            if B["short_need_weekly_down"]:
                s100 = pd.Series(c).rolling(100).mean().to_numpy()
                if not (np.isfinite(s100[-26]) and c[-1] < s100[-1] and s100[-1] < s100[-26]):
                    continue
            if B["short_need_market_down"] and spy_up is True:
                continue
        cnt = time_count(ctx, n - 1, not up, P)
        lo_cnt = B["long_pullback"] if up else B["short_pullback"]
        hi_cnt = M.get("max_pullback") or 99
        if cnt < lo_cnt or cnt > hi_cnt:
            continue
        lv = [x for x in liquidity_levels(dummy, a, tf, P, "low" if up else "high") if x["strength"] >= 2]
        for x in lv:
            p = x["price"]
            dist = (c[-1] - p) / a if up else (p - c[-1]) / a
            if dist < 0 or dist > M["max_dist_atr"]:
                continue                                  # must be on the right side and reachable today
            # confluence = how many DIFFERENT kinds of support/resistance sit at this line
            # (e.g. a double bottom + last month's low + a Fib level within 0.5 ATR)
            kinds = {level_group(z["label"]) for z in lv if abs(z["price"] - p) <= 0.5 * a}
            confl = max(1, len(kinds))
            grp = level_group(x["label"])
            if confl < (M["min_confluence"] if up else max(M["min_confluence"], B["short_confluence"])):
                continue
            if not up and grp not in B["short_levels"]:
                continue
            if any(y["side"] == side and abs(y["price"] - p) <= 0.1 * a for y in out):
                continue
            out.append(dict(side=side, price=float(p), label=x["label"], group=grp, strength=x["strength"],
                            confluence=confl, atr=a, count=cnt))
    return out


def evaluate_sweep_5m(armed: list[dict], bars5: pd.DataFrame, B: dict, confirm: str = "close") -> dict | None:
    """5-minute execution (v20): wait for price to take out the liquidity line (trigger the
    stops resting beyond it), then enter when it moves back the other way.

    For a long line: a 5-minute low trades at least sweep_atr x daily ATR below the line (the
    stop run). Within reclaim_bars bars of the first break, a 5-minute bar must CLOSE back
    above the line ('close'), or ('break', a change of character) close above the line in a
    green candle that also clears the high of every bar since the sweep low. Entry at that close; stop sweep low - long_stop_atr x ATR.
    Shorts mirror this. The entry must be within max_chase_pct of the line and the risk within
    max_risk_pct. Returns the first trigger of the day."""
    M = B["m5"]
    if not armed or bars5 is None or not len(bars5):
        return None
    t0, t1 = M["start"], M["end"]
    o, h, l, c, v = (bars5[x].to_numpy() for x in ("Open", "High", "Low", "Close", "Volume"))
    times = [(t + pd.Timedelta(minutes=5)).strftime("%H:%M") for t in bars5.index]   # bar END times
    best = None
    for L in armed:
        up = L["side"] == "long"
        p, a = L["price"], L["atr"]
        depth = M["sweep_atr"] * a
        start = None                                      # index of the first bar through the line
        ext_i = None                                      # bar holding the sweep extreme
        for i in range(len(c)):
            if times[i] > t1:
                break
            through = (l[i] < p - depth) if up else (h[i] > p + depth)
            if start is None:
                if through:
                    start, ext_i = i, i
                else:
                    continue
            else:
                if (l[i] < l[ext_i]) if up else (h[i] > h[ext_i]):
                    ext_i = i
            if i - start > M["reclaim_bars"]:
                break                                     # no reclaim in time: a real breakdown, not a sweep
            if times[i] < t0:
                continue
            back = c[i] > p if up else c[i] < p
            if confirm == "break" and back:
                # stronger: a candle in the trade direction that closes beyond every bar since the
                # sweep extreme (the small 5-minute pullback after the stop run is broken)
                if i <= ext_i:
                    back = False
                elif up:
                    back = c[i] > o[i] and c[i] > h[ext_i:i].max()
                else:
                    back = c[i] < o[i] and c[i] < l[ext_i:i].min()
            if not back:
                continue
            entry = float(c[i])
            ext = float(l[ext_i] if up else h[ext_i])
            stop = ext - B["long_stop_atr"] * a if up else ext + B["short_stop_atr"] * a
            risk = abs(entry - stop) / entry * 100
            chase = abs(entry - p) / p * 100
            if chase > B["max_chase_pct"] or risk > B["max_risk_pct"] or risk < 0.2:
                break
            base = float(np.median(v[:start])) if start >= 3 else float(np.median(v[:i + 1]))
            svol = float(v[start:i + 1].max())
            cand = dict(direction="bullish" if up else "bearish", side=L["side"], stage="sweep5m",
                        level=round(p, 4), level_label=L["label"], level_group=L["group"], confluence=L["confluence"],
                        entry=round(entry, 4), stop=round(stop, 4), risk_pct=round(risk, 2), count=L["count"],
                        atr=round(a, 4), chase_pct=round(chase, 2), entry_time=times[i], sweep_time=times[start],
                        sweep_depth_atr=round(abs(ext - p) / a, 2), bars_to_reclaim=int(i - start),
                        vol_ratio=round(svol / base, 2) if base > 0 else None,
                        keys=["stage_sweep5m", f"liq_{L['label'].split(' (')[0].lower().replace(' ', '_')}"],
                        factors=[f"Trend {'up' if up else 'down'} (50-day average)", f"{L['count']}-bar pullback",
                                 f"{L['label']} @ {p:.2f} with {L['confluence']} levels together",
                                 f"5-min stop run to {ext:.2f} at {times[start]}, back {'above' if up else 'below'} the line at {times[i]}"],
                        idea=(f"{'BUY' if up else 'SELL'} ~{entry:.2f} at {times[i]}: 5-min swept {L['label'].lower()} "
                              f"{p:.2f} (low of the run {ext:.2f}) and reclaimed it; stop {stop:.2f} ({risk:.1f}% risk)."),
                        score=round(L["strength"] + 0.5 * (L["confluence"] - 1), 2), qualified=True)
            if best is None or cand["entry_time"] < best["entry_time"]:
                best = cand
            break
    return best
