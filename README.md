# Reversal Scanner

Scans the S&P 500 every weekday at about 3:05, 3:30 and 3:50 PM ET for reversal bars that are
shaping up but have not happened yet. A stock is listed only when all three are true:

1. A reversal bar at the high/low of the move: a tail forming now, a tail 1-2 days ago that has
   not run away (within 3%), or WATCH (still pushing into a strong level, no tail yet).
2. At liquidity: prior swing highs/lows, double tops/bottoms, the swing that started the move,
   prior week/month highs/lows, Fibonacci, trend lines, volume-backed zones.
3. A time count of 7+ bars in one direction (small opposite bars are treated as noise), in an
   orderly move of at least 3 ATRs.

The logic is in setup.py; every threshold is in config.py. test_setups.py checks the rules
against charts shaped like real examples. Weekly and monthly
scans run automatically on the last trading day of the week or month.

## Files

| File | Purpose |
|---|---|
| scanner.py | Main program: downloads prices, scores each stock, writes results |
| indicators.py | Indicator and pattern maths |
| setup.py | The three-requirement reversal engine |
| confirm.py | Intraday checks stored with each trade for the analysis |
| analysis.py | Trade-journal analysis for the tracker |
| suggestions.py, suggestions_template.html | Suggestions tracker: every suggestion followed forward (close entry and safer entry), earnings flags, weekly review and proposed rule changes |
| store.js, tracker_template.html | Trade journal and tracker page |
| report.py | Builds the web page and the Excel file |
| config.py | All settings and thresholds |
| dashboard_template.html | Web page design |
| watchlist.txt | Backup ticker list |
| requirements.txt | Python packages |
| scan.yml | GitHub automation (pasted into Actions during setup) |

The scanner creates two folders on its own: `site` (the published page) and `history` (a CSV per run).

## Settings

Edit `config.py` on GitHub (pencil icon) to change thresholds. To scan only your own list,
set `UNIVERSE = "watchlist.txt"` and edit that file.

## Notes

Prices come from Yahoo Finance via the free `yfinance` library. The 3:05 PM candle is still forming.
The confirmation odds are a heuristic, not a calibrated probability. This is a screening tool, not
investment advice.

## Suggestions tracker

Every listed setup (all timeframes) is saved to `suggestions.json` and followed forward:
- Entry versions: at the close of the signal bar, and the safer break of the reversal bar (3 bars).
- Management: half at +5%, stop to breakeven, rest at +10%; time stop after 15 bars; stop first on
  a bar that touches both.
- Earnings: next earnings date stored; within 3 trading days = flagged and skipped by the bot.
- Review (`suggestions.html`): results by timeframe, stage, side and level type, plus proposed rule
  changes (need 20+ finished trades on each side and a 0.15R edge). Proposals are never applied
  automatically.

## Backtest

Run the workflow with mode `backtest` to replay the scanner over the last 60 trading days (daily and
weekly). Every setup it would have listed is followed forward with the same rules as the live tracker.
Results: `suggestions.html#backtest`. Limits: uses each day's closing bar, today's S&P 500 list,
no commissions or slippage, and repeat signals while a trade in the same stock is open are excluded.

## Level entry (current choice for the bot)

Limit order at the liquidity line the first time price reaches it (same setup: time count, orderly
move, liquidity map). Exits (config.py `TRACK["mgmt"]`): stop to breakeven after a 1-ATR move in
favour; 1/2 off at +5% and the stop moves to lock +2.5%; 1/4 at +7.5%; the rest at +10%. The backtest
compares stop widths, add-on entries and breakeven triggers side by side (`TRACK["level_variants"]`).

## The bot: trend pullback + liquidity sweep (daily, 1-5 day holds)

Rules live in config.py `BOT`. Two entry versions are tested side by side:
- **Close entry** (`bot`): checked on the daily bar at 3:30 PM.
- **Intraday entry** (`bot_intraday`, v19): the same rules checked on the hourly chart at
  10:30, 11:30, 12:30, 1:30, 2:30 and 3:30 PM ET (`BOT["intraday_checks"]`), using yesterday's daily
  context and today's bar built from the hourly bars so far. The first check that sees a sweep and
  reclaim of the line enters at that price.
- **No chase** (both versions): the trade is skipped if price is already more than 2% from the
  liquidity line (`BOT["max_chase_pct"]`).
- **5-minute execution** (`bot_5m`, `bot_5m_break`, v20): the daily chart arms the line (trend,
  4-6 bar pullback, a strong liquidity line where at least 2 kinds of support/resistance meet, within
  2 ATRs of yesterday's close). The 5-minute chart times the entry: price must trade through the line
  (the stop run) and get back across it within 1 hour. `bot_5m` enters on the first 5-minute close back
  across the line; `bot_5m_break` waits for a candle that also clears every bar since the low of the
  stop run. Stop: beyond the stop-run extreme plus a small ATR buffer. Settings: `BOT["m5"]`.
  Yahoo keeps only about 60 days of 5-minute bars, so these two cover the last ~40 trading days.
The backtest downloads hourly bars (Yahoo keeps about 2 years) and reports both versions.

## TradingView

- `reversal_setups.pine`: indicator that draws the trade plan (separate download).
- `reversal_strategy.pine`: the same rules as a strategy for backtesting and, later, bot alerts.
