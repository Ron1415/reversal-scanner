"""Scanner settings. Adjust thresholds here; no code changes needed."""

# Universe: "sp500" pulls the current S&P 500 list; otherwise a path to a text
# file with one ticker per line (e.g. "watchlist.txt").
UNIVERSE = "sp500"
FALLBACK_TICKERS_FILE = "watchlist.txt"

# Liquidity filters
MIN_PRICE = 5.0
MIN_AVG_DOLLAR_VOLUME = 20_000_000  # 20-day average close * volume

# A ticker is listed only when it meets BOTH thresholds
MIN_FACTORS = 3       # at least three independent confluence factors
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
    proximity_atr=0.30,   # how close (in ATRs) price must come to a level
    td_min=7,             # TD Setup count that starts to matter (9 = classic signal)
    streak_min=6,         # consecutive closes in one direction
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
    exhaustion_9=1.5, exhaustion=1.0,
    fib=1.0, sr=0.75, sr_major=1.5, hvn=1.0, trendline=1.0,
    rsi=1.0, bollinger=0.75, extension=0.75, volume=0.5,
)
