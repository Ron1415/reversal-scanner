# Reversal Scanner

Scans the S&P 500 every weekday at about 3:05, 3:30 and 3:50 PM ET for potential reversals
(candles, TD Sequential, Fibonacci, volume-backed support and resistance, trend lines, RSI,
Bollinger bands) and publishes a ranked dashboard with an Excel download. Weekly and monthly
scans run automatically on the last trading day of the week or month.

## Files

| File | Purpose |
|---|---|
| scanner.py | Main program: downloads prices, scores each stock, writes results |
| indicators.py | Indicator and pattern maths |
| confirm.py | Green / amber / red confirmation meter for the last hour |
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
