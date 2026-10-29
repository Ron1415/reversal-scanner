"""Scanner settings. Adjust thresholds here; no code changes needed."""

# Universe: "sp500" pulls the current S&P 500 list; otherwise a path to a text
# file with one ticker per line (e.g. "watchlist.txt").
UNIVERSE = "sp500"
FALLBACK_TICKERS_FILE = "watchlist.txt"

# Liquidity filters
MIN_PRICE = 5.0
MIN_AVG_DOLLAR_VOLUME = 20_000_000  # 20-day average close * volume

# A ticker is listed only when it meets ALL of these
REQUIRE_CANDLE = True # a reversal candle (tail / engulfing / doji) must be present today
MIN_FACTORS = 3       # at least three confluence factors (the candle counts as one)
MIN_SCORE = 3.0       # weighted score (see WEIGHTS)
MAX_RESULTS = 40      # per direction, per timeframe

# Scan window in Eastern Time. The --auto mode exits outside this window so
# the scheduler can fire at both the EDT and EST clock times safely.
SCAN_WINDOW_ET = ("14:45", "15:59")

TIMEFRAMES = {
    #           history   resample   fib/trend lookback   pivot strength   S/R lookback
    "daily":   dict(period="2y",  rule=None,    lookback=120, pivot_k=5, sr_lookback=250),
    "weekly":  dict(period="6y",  rule="W-FRI", lookback=104, pivot_k=3, sr_lookback=200),
    "monthly": dict(period="20y", rule="ME",    lookback=60,  pivot_k=2, sr_lookback=120),
}

PARAMS = dict(
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
    carry_days=2,
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
WEIGHTS = dict(
    candle=1.5, candle_doji=1.0,
    count=1.5,
    fib=1.0, sr=0.75, sr_major=1.5, hvn=1.0, trendline=1.0,
    rsi=1.0, bollinger=0.75, extension=0.75, volume=0.5,
)
