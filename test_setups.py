"""Synthetic charts shaped like Ron's examples. Run: python test_setups.py"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import config as C
from setup import evaluate_setup
from scanner import qualifies

rng = np.random.default_rng(5)

def bars(closes, wick=0.4, noise=0.15):
    c = np.array(closes, float); o = np.r_[c[0], c[:-1]] + rng.normal(0, noise, len(c))
    h = np.maximum(o, c) + abs(rng.normal(wick, wick / 3, len(c)))
    l = np.minimum(o, c) - abs(rng.normal(wick, wick / 3, len(c)))
    v = rng.integers(2e6, 4e6, len(c)).astype(float)
    return o, h, l, c, v

def frame(o, h, l, c, v):
    idx = pd.bdate_range(end="2026-10-29", periods=len(c))
    return pd.DataFrame(dict(Open=o, High=h, Low=l, Close=c, Volume=v), index=idx)

def base(n=150, level=100, sd=0.5):
    return list(level + np.cumsum(rng.normal(0, sd, n)) * 0.3)

def walk(a, b, n, noise=0.25, noise_bars=()):
    path = list(np.linspace(a, b, n + 1)[1:] + rng.normal(0, noise, n))
    for i in noise_bars:      # small counter-trend bars
        path[i] = path[i - 1] - np.sign(b - a) * 0.15
    return path

def set_bar(df, i, o, h, l, c):
    df.iloc[i, :4] = [o, h, l, c]

def run(name, df, expect):
    res = evaluate_setup(df, C.TIMEFRAMES["daily"], C.PARAMS, C.WEIGHTS)
    got = (res["stage"] if qualifies(res) else None) if res else None
    ok = (got == expect) if expect else got is None
    print(f"{'PASS' if ok else 'FAIL'}  {name:42s} expected {str(expect):8s} got {str(got):8s}"
          f"| {res['direction'] if res else ''} count {res.get('count') if res else ''} "
          f"| {'; '.join(res.get('why', [])) if res and not got else (res.get('idea') if res else '')}")
    return ok

results = []
# 1. LDOS: drop from a major high, long orderly rally back into it, topping tail sweeping the high today
c = base(120, 140) + walk(140, 120, 25) + walk(120, 140.6, 14, noise_bars=(5, 10))
o, h, l, c, v = bars(c); df = frame(o, h, l, c, v)
hi0 = float(df["High"].iloc[-60:-1].max())          # the major high on the left
set_bar(df, -1, hi0 - 0.6, hi0 + 0.5, hi0 - 1.3, hi0 - 0.9)
results.append(run("LDOS-like: topping tail at prior high", df, "forming"))

# 2. TPR: base lows ~110, rally to 122, 7-bar decline (with 2 small green bars), tail sweeping 110 today
c = base(130, 118) + walk(118, 110, 8) + walk(110, 122, 12) + walk(122, 111.5, 8, noise_bars=(3, 6))
o, h, l, c, v = bars(c); df = frame(o, h, l, c, v)
lo0 = float(df["Low"].iloc[-40:-1].min())           # the base lows
set_bar(df, -1, lo0 + 0.9, lo0 + 1.9, lo0 - 0.6, lo0 + 1.4)
results.append(run("TPR tail day (Oct 27)", df, "forming"))
d2 = pd.concat([df, frame(*[np.array([x]) for x in (lo0 + 1.4, lo0 + 1.9, lo0 + 0.9, lo0 + 1.5, 3e6)]).set_index(
    pd.DatetimeIndex([df.index[-1] + pd.offsets.BDay()]))])
results.append(run("TPR day after (small inside bar)", d2, "day2"))
d3 = pd.concat([d2, frame(*[np.array([x]) for x in (lo0 + 2, lo0 + 7.5, lo0 + 1.8, lo0 + 7.2, 6e6)]).set_index(
    pd.DatetimeIndex([d2.index[-1] + pd.offsets.BDay()]))])
results.append(run("TPR +5.7% day (too late)", d3, None))

# 3. PWR: tail at the low 5 bars ago, now +9%
c = base(120, 600) + walk(600, 640, 15) + walk(640, 584, 12) + walk(586, 634, 5)
o, h, l, c, v = bars(c, wick=2, noise=0.6); df = frame(o, h, l, c, v)
results.append(run("PWR 5 days after the low (+9%)", df, None))

# 4. Choppy ranges (SHW / DE / APTV / GNRC / CIEN): tail in the middle of a range
for k in range(4):
    c = list(330 + rng.normal(0, 4, 170))
    o, h, l, c, v = bars(c, wick=2, noise=1.5); df = frame(o, h, l, c, v)
    set_bar(df, -1, c[-2], c[-2] + 1, c[-2] - 6, c[-2] + 0.5)
    results.append(run(f"Choppy range, tail mid-range #{k + 1}", df, None))

# 5. EQT: rally, high yesterday, today -3.4%
c = base(140, 50) + walk(50, 56, 12)
o, h, l, c, v = bars(c, wick=0.3, noise=0.1); df = frame(o, h, l, c, v)
hi = float(df["High"].iloc[-1]); set_bar(df, -1, 55.8, hi + 0.4, 55.4, 55.6)
d2 = pd.concat([df, frame(*[np.array([x]) for x in (55.5, 55.9, 54.2, 54.4, 3e6)]).set_index(
    pd.DatetimeIndex([df.index[-1] + pd.offsets.BDay()]))])
results.append(run("EQT: -3.4% from yesterday's high", d2, None))

# 6. BAX: rally then a -6.8% engulfing bar today
c = base(140, 24) + walk(24, 25.6, 9)
o, h, l, c, v = bars(c, wick=0.12, noise=0.05); df = frame(o, h, l, c, v)
set_bar(df, -1, 25.6, 25.9, 23.6, 23.73)
results.append(run("BAX: -6.8% engulfing (already reversed)", df, None))

# 7. WATCH: orderly 8-bar rally still closing near the high, prior swing high just above
c = base(110, 90) + walk(90, 100, 6) + walk(100, 92, 10) + walk(92, 99.0, 8)
o, h, l, c, v = bars(c); df = frame(o, h, l, c, v)
top = float(df["High"].iloc[-25:-9].max())          # prior swing high just above
set_bar(df, -1, top - 1.4, top - 0.5, top - 1.5, top - 0.55)
results.append(run("Pushing into prior high, no tail yet", df, "watch"))

print(f"\n{sum(results)}/{len(results)} passed")
