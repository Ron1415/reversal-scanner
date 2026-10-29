"""Indicator math and the reversal confluence evaluator.

Every bar frame passed in has columns Open, High, Low, Close, Volume, oldest
first. The last row is the bar being judged (usually still forming at 3:30 PM).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FIB_RETRACE = (0.382, 0.5, 0.618, 0.786)
FIB_EXTEND = (1.272, 1.618)


# --------------------------------------------------------------------------- #
# Basic indicators
# --------------------------------------------------------------------------- #
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev).abs(),
                    (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def td_setup(close: pd.Series) -> pd.Series:
    """DeMark TD Setup: +n consecutive closes above the close 4 bars earlier,
    -n consecutive closes below it. The count keeps running past 9."""
    up = (close > close.shift(4)).to_numpy()
    dn = (close < close.shift(4)).to_numpy()
    out = np.zeros(len(close), dtype=int)
    for i in range(1, len(close)):
        if up[i]:
            out[i] = out[i - 1] + 1 if out[i - 1] > 0 else 1
        elif dn[i]:
            out[i] = out[i - 1] - 1 if out[i - 1] < 0 else -1
    return pd.Series(out, index=close.index)


def close_streak(close: pd.Series) -> pd.Series:
    """Consecutive higher (+) or lower (-) closes versus the prior close."""
    d = np.sign(close.diff().fillna(0)).to_numpy()
    out = np.zeros(len(close), dtype=int)
    for i in range(1, len(close)):
        if d[i] > 0:
            out[i] = out[i - 1] + 1 if out[i - 1] > 0 else 1
        elif d[i] < 0:
            out[i] = out[i - 1] - 1 if out[i - 1] < 0 else -1
    return pd.Series(out, index=close.index)


def pivots(s: pd.Series, k: int, kind: str) -> list[tuple[int, float]]:
    """Confirmed swing points: bar is the extreme of k bars on each side."""
    roll = s.rolling(2 * k + 1, center=True)
    ext = roll.max() if kind == "high" else roll.min()
    mask = (s == ext).to_numpy()
    vals = s.to_numpy()
    return [(i, float(vals[i])) for i in np.flatnonzero(mask)]


# --------------------------------------------------------------------------- #
# Level tests
# --------------------------------------------------------------------------- #
def _rejected(direction: str, h: float, l: float, c: float, lvl: float, tol: float) -> bool:
    """Bearish: high tagged the level (or pierced it and closed back below).
    Bullish: low tagged the level (or pierced it and closed back above)."""
    if direction == "bearish":
        return abs(h - lvl) <= tol or (h > lvl > c)
    return abs(l - lvl) <= tol or (l < lvl < c)


def _nearest(direction, h, l, c, tol, levels):
    hits = [(lab, lvl) for lab, lvl in levels if lvl > 0 and _rejected(direction, h, l, c, lvl, tol)]
    if not hits:
        return None
    ref = h if direction == "bearish" else l
    return min(hits, key=lambda x: abs(ref - x[1]))


def fib_levels(df: pd.DataFrame, lookback: int) -> dict[str, list[tuple[str, float]]]:
    w = df.iloc[-lookback - 1:-1]
    if len(w) < 20:
        return {"bearish": [], "bullish": []}
    hi_pos, lo_pos = int(np.argmax(w["High"].to_numpy())), int(np.argmin(w["Low"].to_numpy()))
    hi, lo = float(w["High"].iloc[hi_pos]), float(w["Low"].iloc[lo_pos])
    leg = hi - lo
    if leg <= 0:
        return {"bearish": [], "bullish": []}
    if hi_pos > lo_pos:   # most recent leg is UP: retracements = support, extensions = resistance
        return {"bullish": [(f"Fib {r:.1%} retrace", hi - r * leg) for r in FIB_RETRACE],
                "bearish": [(f"Fib {e:.3g} extension", lo + e * leg) for e in FIB_EXTEND]}
    # most recent leg is DOWN: retracements = resistance, extensions = support
    return {"bearish": [(f"Fib {r:.1%} retrace", lo + r * leg) for r in FIB_RETRACE],
            "bullish": [(f"Fib {e:.3g} extension", hi - e * leg) for e in FIB_EXTEND]}


def sr_levels(df: pd.DataFrame, k: int, lookback: int, atr_val: float, P: dict) -> list[dict]:
    """Support/resistance zones built from swing pivots and graded by volume.

    1. Find confirmed swing highs and lows.
    2. Measure relative volume at each pivot (peak of the pivot bar and its
       neighbours vs. the 50-bar average volume).
    3. Merge pivots within `cluster_atr` ATRs into one zone (volume-weighted
       price) and count touches.
    4. A zone is MAJOR if any pivot printed >= `pivot_volume_mult` x average
       volume or the zone has >= `major_touches` touches.
    """
    w = df.iloc[-lookback - 1:-1]
    if len(w) < 2 * k + 5:
        return []
    vol = w["Volume"].to_numpy()
    avg = w["Volume"].rolling(50, min_periods=10).mean().bfill().to_numpy()
    pts = []
    for kind, col in (("high", "High"), ("low", "Low")):
        for i, v in pivots(w[col], k, kind):
            rv = float(vol[max(0, i - 1):i + 2].max() / avg[i]) if avg[i] > 0 else 0.0
            pts.append(dict(price=v, rv=rv, vol=float(vol[i]), kind=kind))
    if not pts:
        return []
    pts.sort(key=lambda p: p["price"])
    zones, cur = [], [pts[0]]
    for p in pts[1:]:
        if p["price"] - cur[-1]["price"] <= P["cluster_atr"] * atr_val:
            cur.append(p)
        else:
            zones.append(cur)
            cur = [p]
    zones.append(cur)

    out = []
    for z in zones:
        wts = np.array([max(p["vol"], 1.0) for p in z])
        price = float(np.average([p["price"] for p in z], weights=wts))
        rv = max(p["rv"] for p in z)
        major = rv >= P["pivot_volume_mult"] or len(z) >= P["major_touches"]
        tag = "Major pivot zone" if major else "Prior pivot"
        out.append(dict(label=f"{tag} ({len(z)} touch{'es' if len(z) > 1 else ''}, {rv:.1f}x vol)",
                        price=price, major=major, rv=rv, touches=len(z)))
    return out


def volume_nodes(df: pd.DataFrame, lookback: int, P: dict) -> list[tuple[str, float]]:
    """Volume-by-price profile. Each bar's volume is spread evenly across the
    price bins its range covers. Returns the Point of Control and other
    high-volume nodes (local peaks >= `hvn_mult` x the mean bin)."""
    w = df.iloc[-lookback - 1:-1]
    lo, hi = float(w["Low"].min()), float(w["High"].max())
    nb = P["profile_bins"]
    if hi <= lo or len(w) < 20:
        return []
    edges = np.linspace(lo, hi, nb + 1)
    prof = np.zeros(nb)
    for l, h, v in w[["Low", "High", "Volume"]].itertuples(index=False):
        a = int(np.clip(np.searchsorted(edges, l, "right") - 1, 0, nb - 1))
        b = int(np.clip(np.searchsorted(edges, h, "right") - 1, 0, nb - 1))
        prof[a:b + 1] += v / (b - a + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    poc = int(prof.argmax())
    out = [("Volume POC", float(mids[poc]))]
    thresh = P["hvn_mult"] * prof.mean()
    for i in range(1, nb - 1):
        if i != poc and prof[i] >= thresh and prof[i] >= prof[i - 1] and prof[i] >= prof[i + 1]:
            out.append(("High-volume node", float(mids[i])))
    return out


def trendline(df: pd.DataFrame, k: int, lookback: int, kind: str) -> tuple[str, float] | None:
    """Line through the last two confirmed swing highs (kind='high') or lows,
    projected to the current bar."""
    n = len(df)
    start = max(0, n - 1 - lookback)
    w = df.iloc[start:n - 1]
    pts = pivots(w["High"] if kind == "high" else w["Low"], k, kind)
    if len(pts) < 2:
        return None
    (p1, v1), (p2, v2) = pts[-2], pts[-1]
    if p2 - p1 < k:
        return None
    slope = (v2 - v1) / (p2 - p1)
    proj = v2 + slope * ((n - 1 - start) - p2)
    word = "falling" if slope < 0 else "rising"
    side = "resistance" if kind == "high" else "support"
    return (f"Trendline ({word} {side})", proj)


# --------------------------------------------------------------------------- #
# Candles
# --------------------------------------------------------------------------- #
def candle_signal(df: pd.DataFrame, direction: str):
    o, h, l, c = (float(df[x].iloc[-1]) for x in ("Open", "High", "Low", "Close"))
    po, pc = float(df["Open"].iloc[-2]), float(df["Close"].iloc[-2])
    rng = h - l
    if rng <= 0:
        return None
    body = abs(c - o)
    upper, lower = h - max(o, c), min(o, c) - l
    if direction == "bearish":
        if upper >= 2 * body and upper >= 0.5 * rng and lower <= 0.25 * rng:
            return "Shooting star / topping tail", "candle"
        if pc > po and c < o and o >= pc and c <= po:
            return "Bearish engulfing", "candle"
    else:
        if lower >= 2 * body and lower >= 0.5 * rng and upper <= 0.25 * rng:
            return "Hammer / bottoming tail", "candle"
        if pc < po and c > o and o <= pc and c >= po:
            return "Bullish engulfing", "candle"
    if body <= 0.1 * rng:
        return "Doji (indecision at extreme)", "candle_doji"
    return None


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
def evaluate(df: pd.DataFrame, tf: dict, P: dict, W: dict) -> dict | None:
    if len(df) < max(60, tf["pivot_k"] * 4 + 25):
        return None
    df = df.copy()
    df["ATR"] = atr(df)
    df["RSI"] = rsi(df["Close"])
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    mid = df["Close"].rolling(20).mean()
    sd = df["Close"].rolling(20).std()
    df["BBU"], df["BBL"] = mid + 2 * sd, mid - 2 * sd
    df["TD"] = td_setup(df["Close"])
    df["STREAK"] = close_streak(df["Close"])

    last = df.iloc[-1]
    h, l, c = float(last.High), float(last.Low), float(last.Close)
    a = float(last.ATR)
    if not np.isfinite(a) or a <= 0:
        return None
    tol = P["proximity_atr"] * a
    k, lb = tf["pivot_k"], tf["lookback"]

    # Exhaustion counts are measured on the bar BEFORE the reversal candle as
    # well, since a reversal bar often breaks the streak.
    td_now, td_prev = int(last.TD), int(df["TD"].iloc[-2])
    st_now, st_prev = int(last.STREAK), int(df["STREAK"].iloc[-2])

    tb = P["trend_lookback"]
    prior_up = df["Close"].iloc[-2] > df["Close"].iloc[-2 - tb]
    fibs = fib_levels(df, lb)
    sr = sr_levels(df, k, tf["sr_lookback"], a, P)
    hvn = volume_nodes(df, tf["sr_lookback"], P)
    best = None

    for direction in ("bearish", "bullish"):
        if (direction == "bearish") != bool(prior_up):
            continue   # only look for reversals of the trend that exists
        sgn = 1 if direction == "bearish" else -1
        factors, score, levels = [], 0.0, []

        def add(label, wkey, lvl=None):
            nonlocal score
            factors.append(label)
            score += W[wkey]
            if lvl is not None:
                levels.append(round(lvl, 4))

        cs = candle_signal(df, direction)
        if cs:
            add(*cs)

        td = max(sgn * td_now, sgn * td_prev)
        st = max(sgn * st_now, sgn * st_prev)
        if td >= P["td_min"] or st >= P["streak_min"]:
            parts = []
            if td >= P["td_min"]:
                parts.append(f"TD setup {td}")
            if st >= P["streak_min"]:
                parts.append(f"{st} straight {'up' if sgn > 0 else 'down'} closes")
            add("Exhaustion: " + ", ".join(parts), "exhaustion_9" if td >= 9 else "exhaustion")

        hit = _nearest(direction, h, l, c, tol, fibs[direction])
        if hit:
            add(f"{hit[0]} @ {hit[1]:.2f}", "fib", hit[1])

        # Pivot zones: prefer a volume-backed (major) zone over a minor pivot.
        zone_hits = [z for z in sr if _rejected(direction, h, l, c, z["price"], tol)]
        if zone_hits:
            ref = h if direction == "bearish" else l
            z = min(zone_hits, key=lambda z: (not z["major"], abs(ref - z["price"])))
            add(f"{z['label']} @ {z['price']:.2f}", "sr_major" if z["major"] else "sr", z["price"])

        # Volume profile: POC / high-volume node acting as the level.
        hit = _nearest(direction, h, l, c, tol, hvn)
        if hit:
            add(f"{hit[0]} @ {hit[1]:.2f}", "hvn", hit[1])

        tl = trendline(df, k, lb, "high" if direction == "bearish" else "low")
        if tl and _rejected(direction, h, l, c, tl[1], tol):
            add(f"{tl[0]} @ {tl[1]:.2f}", "trendline", tl[1])

        r = float(last.RSI)
        win = df.iloc[-21:-1]
        if direction == "bearish":
            div = h >= win["High"].max() and win["RSI"].max() >= 65 and r < win["RSI"].max() - 3
            if r >= P["rsi_hi"]:
                add(f"RSI overbought ({r:.0f})", "rsi")
            elif div:
                add(f"Bearish RSI divergence ({r:.0f})", "rsi")
            if h >= float(last.BBU):
                add("Tagged upper Bollinger band", "bollinger")
        else:
            div = l <= win["Low"].min() and win["RSI"].min() <= 35 and r > win["RSI"].min() + 3
            if r <= P["rsi_lo"]:
                add(f"RSI oversold ({r:.0f})", "rsi")
            elif div:
                add(f"Bullish RSI divergence ({r:.0f})", "rsi")
            if l <= float(last.BBL):
                add("Tagged lower Bollinger band", "bollinger")

        ext = sgn * (max(h, c) if sgn > 0 else min(l, c)) - sgn * float(last.EMA20)
        if ext / a >= P["extension_atr"]:
            add(f"Extended {ext / a:.1f} ATR from 20 EMA", "extension")

        avg_vol = float(df["Volume"].iloc[-21:-1].mean())
        if avg_vol > 0 and float(last.Volume) >= P["volume_mult"] * avg_vol:
            add(f"Volume {float(last.Volume) / avg_vol:.1f}x average", "volume")

        cand = dict(direction=direction, score=round(score, 2), n_factors=len(factors),
                    factors=factors, levels=levels, td=td_now, streak=st_now,
                    rsi=round(r, 1), atr=round(a, 4))
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best
