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
            why.append(f"the extreme was {age} bars ago (reversal already under way)")
        if moved > late_pct:
            why.append(f"already {moved:.1f}% from the extreme (limit {late_pct}%)")

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
            why.append(f"time count {cnt} (need {need})")
        if cnt and eff < P["min_efficiency"]:
            why.append(f"choppy move into the level (orderliness {eff:.2f}, need {P['min_efficiency']})")
        if move_atr < P["min_move_atr"]:
            why.append(f"move into the extreme only {move_atr:.1f} ATR (need {P['min_move_atr']})")

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
            why.append(f"run-up not clean enough for an early heads-up (orderliness {eff:.2f}, "
                       f"need {P['watch_efficiency']})")

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
