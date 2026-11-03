"""Scanner settings. Adjust thresholds here; no code changes needed."""

# Universe: "sp500" pulls the current S&P 500 list; otherwise a path to a text
# file with one ticker per line (e.g. "watchlist.txt").
UNIVERSE = "sp500"
FALLBACK_TICKERS_FILE = "watchlist.txt"

# Liquidity filters
MIN_PRICE = 5.0
MIN_AVG_DOLLAR_VOLUME = 20_000_000  # 20-day average close * volume

# A ticker is listed only when it passes all three requirements in setup.py:
# reversal bar shaping up (not finished), at liquidity, and a time count.
# The score below only RANKS qualifying setups.
SHOW_WATCH = True     # also list WATCH setups (pushing into a level, no tail yet)
MAX_RESULTS = 40      # per direction, per timeframe

# Scan window in Eastern Time. The --auto mode exits outside this window so
# the scheduler can fire at both the EDT and EST clock times safely.
SCAN_WINDOW_ET = ("14:45", "15:59")

TIMEFRAMES = {
    #           history   resample   fib/trend lookback   pivot strength   S/R lookback
    "daily":   dict(period="2y",  rule=None,    lookback=120, pivot_k=5, sr_lookback=250, late_pct=3.0),
    "weekly":  dict(period="6y",  rule="W-FRI", lookback=104, pivot_k=3, sr_lookback=200, late_pct=5.0),
    "monthly": dict(period="20y", rule="ME",    lookback=60,  pivot_k=2, sr_lookback=120, late_pct=8.0),
}

PARAMS = dict(
    # ---- The three requirements (see setup.py) ----
    extreme_lookback=20,  # the reversal bar must be the highest high / lowest low of this many bars
    late_pct=3.0,         # dropped once price is more than this % away from the extreme
    move_lookback=40,     # bars searched for the start of the move into the extreme
    min_move_atr=3.0,     # the move into the extreme must be at least this many ATRs
    min_efficiency=0.35,  # orderliness of the counted bars (1 = straight line, 0 = chop)
    major_swing_atr=3.0,  # a swing high/low that started a move of this many ATRs is "major"
    watch_move_atr=4.0,   # WATCH stage needs a bigger move into the level
    watch_efficiency=0.5, # WATCH stage needs a cleaner run than FORMING
    approach_atr=0.6,     # WATCH stage: a strong level within this many ATRs ahead of price
    proximity_atr=0.50,   # how close (in ATRs) price must come to a level / zone
    # Reversal candle (bottoming / topping tail)
    tail_min_pct=0.45,    # tail must be at least this share of the bar's range
    tail_min_atr=0.30,    # ...and at least this many ATRs long
    # Time count: bars in the move before the reversal bar (largest of: bars since
    # the last opposite bar, bars without a new high/low, straight closes, TD setup)
    count_min=7,
    count_body_atr=0.30,  # a counter-trend bar only breaks the count if its body is
                          # at least this many ATRs (small green bars in a decline count)
    count_tol_atr=0.10,   # ignore new highs / closes smaller than this many ATRs
    # Keep a setup listed for this many days after the tail while price stays
    # within `late_atr` ATRs of the tail's close and has not broken the tail.
    level_stop_atr=1.0,   # LEVEL-ENTRY test: stop this many ATRs beyond the level (no tail to use)
    carry_days=2,         # DAY 2 stage: tail formed up to this many bars ago
    late_atr=1.0,
    rsi_hi=70, rsi_lo=30,
    extension_atr=2.5,    # distance from 20 EMA, in ATRs
    volume_mult=1.5,      # projected volume vs 20-bar average
    trend_lookback=10,    # bars used to define the prior trend being reversed
    # Volume-backed support / resistance
    pivot_volume_mult=1.8,  # pivot bar volume vs 50-bar average to count as "major"
    major_touches=3,        # or this many pivots clustered in one zone
    cluster_atr=0.5,        # pivots within this many ATRs merge into one zone
    profile_bins=50,        # price bins for the volume-by-price profile
    hvn_mult=1.5,           # bin volume vs mean bin to count as a high-volume node
)

# Factor weights. Candles and exhaustion counts are the "trigger"; the rest
# are location / context confluence.
# Bot policy (long bias): position size per side, and whether daily shorts are traded at all.
# The backtest table shows each level-entry variant under this policy.
BOT_POLICY = dict(
    strategy="level",      # enter at the liquidity line (no tail needed)
    long_size=1.25,        # longs: a bit more risk (1.25 x a normal position)
    short_size=0.5,        # shorts: half a position (markets drift up over time)
    daily_shorts=False,    # daily shorts were the weakest group in the 60-day backtest
    use_side_rules=True,   # apply SIDE_RULES below
)

# Which levels each side may trade (from the 60-day backtest, Aug-Oct 2026, level entry):
#   longs:  major swing lows (+0.08R), rising trend lines (+0.04R) and deeper Fibs (61.8/78.6, 50)
#           held up best; double bottoms, prior swings, prior week/month lows and volume zones
#           were weak alone, so they need another level in the same place (confluence).
#   shorts: almost everything lost; only swing highs and prior week/month highs held up, so shorts
#           need one of those AND confluence, weekly charts only, and a time count of 9+.
SIDE_RULES = dict(
    long=dict(timeframes=["daily", "weekly"], min_count=7, min_confluence=2,
              levels_alone=["Major swing", "Trendline", "Fib 61.8", "Fib 78.6", "Fib 50"],
              levels_with_confluence=["Double top/bottom", "Prior swing", "Prior week/month", "Volume zone",
                                      "Fib 38.2", "Fib ext"]),
    short=dict(timeframes=["weekly"], min_count=9, min_confluence=2,
               levels_alone=[],
               levels_with_confluence=["Major swing", "Prior swing", "Prior week/month"]),
)
# THE BOT (setup.evaluate_sweep): trend pullback + liquidity sweep, daily bars, 1-5 day holds.
BOT = dict(
    long_pullback=4,          # longs: at least this many bars down into the sweep
    long_stop_atr=0.2,        # longs: stop this far below the sweep low
    long_max_hold=5,          # longs: exit after 5 days
    long_mgmt=dict(be_trigger_atr=1.0, targets=[(5.0, 0.5), (7.5, 0.25), (10.0, 0.25)], lock_pct=2.5),
    short_pullback=5,         # shorts (strict): at least this many bars up into the sweep
    # v23: LOOSER shorts (Ron, Nov 3: 3 shorts in 6 months is too few). Entry needs only: price below a
    # falling 50-day average, a 4+ bar rally, 2+ levels together at the line, and (close entry) a close
    # in the lower half. The strict checks below are still recorded on every short so the backtest shows
    # what each one adds. Short management stays tight (breakeven after 0.5 ATR, trailing stop, 3 days, 0.5x size).
    short_loose=True,
    short_pullback_loose=4,
    short_need_weekly_down=True,
    short_need_market_down=True,   # SPY below its 50-day average
    short_levels=["Major swing", "Prior swing", "Prior week/month"],
    short_confluence=2,
    short_close_pos=0.4,      # close in the lower 40% of the bar
    short_vol_mult=1.2,       # sweep-bar volume vs the 20-day average
    short_stop_atr=0.1,       # shorts: stop this far above the sweep high
    short_max_hold=3,         # shorts: exit after 3 days
    short_mgmt=dict(be_trigger_atr=0.5, targets=[(5.0, 0.5), (7.5, 0.25), (10.0, 0.25)], lock_pct=2.5,
                    trail_prev_bar=True),   # after breakeven, trail the stop to the prior bar's high
    max_risk_pct=5.0,         # skip setups whose stop is further than this
    max_chase_pct=2.0,        # never enter if price is already more than 2% from the liquidity line
    intraday_checks=["10:30", "11:30", "12:30", "13:30", "14:30", "15:30"],   # hourly checks (New York time)
    long_size=1.25, short_size=0.5,
    # v20: 5-minute execution. The daily chart picks the line (bot trend + pullback rules, a strong
    # liquidity line where 2+ support/resistance levels meet); the 5-minute chart times the entry:
    # price runs the stops beyond the line, then closes back on the right side of it.
    m5=dict(
        min_confluence=2,     # liquidity line + at least one more support/resistance level together
        max_pullback=6,       # pullbacks longer than 6 bars lost money in the 6-month backtest
        max_dist_atr=2.0,     # only lines within 2 daily ATRs of yesterday's close (reachable today)
        sweep_atr=0.05,       # the stop run must trade at least 0.05 ATR through the line
        reclaim_bars=12,      # ...and get back within 12 five-minute bars (1 hour), or it is a breakdown
        start="09:40",        # no entries in the first 10 minutes
        end="15:55",
        confirm=["close", "break"],   # both entry triggers are tested side by side
    ),
    # v24: FVG timing for the swing entry (tested side by side; the FVG day-trading system stays separate)
    fvg=dict(
        form_bars=6,          # the gap must form within 30 minutes of the reclaim
        cutoff="12:00",       # unfilled pullback orders cancel at noon (matches the approved before-noon rule)
        min_risk_pct=1.5,     # same as the approved rule: stop at least 1.5% away
    ),
)

# v22: improvements tested SIDE BY SIDE on the bot's trades (close entry and both 5-minute entries).
# Every trade is re-run with each stop x target combination; the filters are applied on top.
BOT_TESTS = dict(
    stops=[  # (label, ATRs beyond the stop-run extreme (None = the bot's own setting), minimum risk %)
        ("Current stop (0.2 ATR beyond the stop run)", None, None),
        ("Wider stop (0.5 ATR beyond the stop run)", 0.5, None),
        ("Current stop, but at least 1.5% risk", None, 1.5),
    ],
    targets=[  # (label, None = the fixed % targets, or ATR multiples for 1/2, 1/4, 1/4 of the position)
        ("fixed +5 / +7.5 / +10%", None),
        ("ATR targets 1 / 1.5 / 2 ATR", [1.0, 1.5, 2.0]),
    ],
    spy_filter=True,                                   # longs only while SPY is above its 50-day average
    before="12:00",                                    # 5-minute entries only before noon
    strong_levels=["Major swing", "Prior week/month"],  # the level types that held up best
)

# v24: the top-110 most-traded list from the FVG journal (stocks + ETFs; futures left out). Added to the
# backtest so results can be shown for this list on its own.
JOURNAL_UNIVERSE = """SPY QQQ MU NVDA META AAPL TSLA AMD MSFT INTC GOOGL AMZN SOXL AVGO IWM MRVL SMH TSM SOXX PLTR ORCL TQQQ
HYG STX SOXS LQD WDC LLY DELL GLD AMAT TLT NFLX LRCX GEV MSTR EWZ V UNH WMT NOW CAT JPM CRWD XLE TXN COST XOM BAC
PANW GS KLAC MA JNJ HOOD XLF QCOM IBIT XLK SQQQ BA CSCO DIA XLV CVX CRM GDX PG VRT GE ADI KO ABBV XBI MRK SNOW HD
EEM EFA COIN SHOP PEP XLP VZ MCD WFC IBM XLI SMCI T ARM MS C XLU ANET RKLB IEF TMUS UBER RTX""".split()

# Closer monitoring for shorts: earlier breakeven and a shorter time stop
SHORT_MONITORING = dict(be_trigger_atr=0.5, max_hold_bars=8)

# Suggestions tracker (suggestions.py): how every suggestion is followed forward
TRACK = dict(
    trigger_bars=3,            # safer entry: reversal bar must be broken within this many bars
    max_hold_bars=15,          # time stop: exit at the close after this many bars
    t1_pct=5.0, t2_pct=10.0,   # half off at T1, stop to breakeven, rest at T2
    earnings_blackout_days=3,  # skip when earnings fall within this many trading days
    min_group=20,              # trades needed on each side before a change is proposed
    min_edge_r=0.15,           # improvement (in R) needed before a change is proposed
    level_stop_label="1 ATR beyond the level",
    # LEVEL-ENTRY variants tested side by side in the backtest (stop and add-on distances in ATRs
    # beyond the level; sizes are fractions of one normal position)
    # Trade management for LEVEL entries (Ron, Nov 3): stop to breakeven after a sharp move off the
    # level (be_trigger_atr ATRs in favour); 1/2 off at +5%, 1/4 at +7.5%, the rest at +10%;
    # after +5% the stop moves to lock in +2.5%.
    mgmt=dict(be_trigger_atr=1.0, targets=[(5.0, 0.5), (7.5, 0.25), (10.0, 0.25)], lock_pct=2.5),
    # LEVEL-ENTRY variants tested side by side (stop / add-on distances in ATRs beyond the level;
    # sizes are fractions of one normal position). All use `mgmt` unless they override it.
    level_variants=[
        dict(name="s1",        label="Stop 1 ATR",                                            stop_atr=1.0),
        dict(name="s1.5",      label="Stop 1.5 ATR",                                          stop_atr=1.5),
        dict(name="s2",        label="Stop 2 ATR",                                            stop_atr=2.0),
        dict(name="half_add",  label="Half at the level + half 1 ATR against; stop 2 ATR",    stop_atr=2.0, first=0.5, add_atr=1.0, add=0.5),
        dict(name="half_add2", label="Half at the level + half 0.75 ATR against; stop 1.5 ATR", stop_atr=1.5, first=0.5, add_atr=0.75, add=0.5),
        dict(name="double",    label="Full at the level + DOUBLE 1 ATR against; stop 2 ATR",  stop_atr=2.0, first=1.0, add_atr=1.0, add=1.0),
        dict(name="s2_be05",   label="Stop 2 ATR, breakeven after 0.5 ATR",                   stop_atr=2.0, mgmt=dict(be_trigger_atr=0.5)),
        dict(name="s2_be15",   label="Stop 2 ATR, breakeven after 1.5 ATR",                   stop_atr=2.0, mgmt=dict(be_trigger_atr=1.5)),
        dict(name="s2_old",    label="Stop 2 ATR, OLD exits (half at 5%, rest at 10%)",       stop_atr=2.0,
             mgmt=dict(be_trigger_atr=None, targets=[(5.0, 0.5), (10.0, 0.5)], lock_pct=0.0)),
    ],
)

WEIGHTS = dict(
    stage_forming=2.0, stage_day2=1.5, stage_watch=0.5,
    count=1.5, count_9=2.0,
    liq_strength=1.0,     # per level hit, scaled by level strength (sweeps and major swings score most)
    rsi=0.5,
)
