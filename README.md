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

## TradingView

- `reversal_setups.pine`: indicator that draws the trade plan (separate download).
- `reversal_strategy.pine`: the same rules as a strategy for backtesting and, later, bot alerts.
