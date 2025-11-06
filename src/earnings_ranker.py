# earnings_scorer_v2.py
# Flexible ticker scoring with multiple input modes
# Usage:
#   python earnings_scorer_v2.py              (interactive mode)
#   python earnings_scorer_v2.py AAPL         (single ticker)
#   python earnings_scorer_v2.py AAPL MSFT    (compare multiple)

import sys
import time
import math
import json
import warnings
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import yfinance as yf

# Sentiment (FinBERT)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Networking for Finnhub
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from urllib.error import URLError, HTTPError

warnings.filterwarnings("ignore", category=FutureWarning)

# ========= CONFIG =========
OUTPUT_CSV = "earnings_scores_log.csv"
PRICE_LOOKBACK_DAYS = 120
SENTENCE_PER_TICKER = 3
EARNINGS_DRIFT_EVENTS = 4
DRIFT_HOLD_DAYS = 3
NEWS_SLEEP_SEC = 0.15
NEWS_LOOKBACK_DAYS = 7

FINNHUB_API_KEY = "d40dd0hr01qqo3qhegc0d40dd0hr01qqo3qhegcg"


# ========= HELPERS =========
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).ewm(alpha=1 / length, adjust=False).mean()
    roll_down = pd.Series(loss, index=series.index).ewm(alpha=1 / length, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50.0)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def bollinger_bandwidth(close: pd.Series, length: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = close.rolling(length).mean()
    sd = close.rolling(length).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    with np.errstate(divide='ignore', invalid='ignore'):
        bbw = (upper - lower) / ma
    return bbw.replace([np.inf, -np.inf], np.nan).fillna(0)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    daily = (r.mean() - rf) / r.std()
    return float(daily * np.sqrt(252))


def trading_bdays_ahead(d0: pd.Timestamp, days: int) -> pd.Timestamp:
    return pd.bdate_range(d0, periods=days + 1)[-1]


def safe_pct_change(prices: pd.Series, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    try:
        p0 = prices.asof(start_date)
        p1 = prices.asof(end_date)
        if pd.isna(p0) or pd.isna(p1) or p0 == 0:
            return np.nan
        return (p1 / p0) - 1.0
    except Exception:
        return np.nan


def get_earnings_drift(ticker: str, close_series: pd.Series) -> float:
    try:
        tkr = yf.Ticker(ticker)
        ed = tkr.get_earnings_dates(limit=EARNINGS_DRIFT_EVENTS)
        if ed is None or ed.empty:
            return 0.0
        changes = []
        for dt in ed.index.to_pydatetime():
            d0 = pd.Timestamp(dt).tz_localize(None)
            d1 = trading_bdays_ahead(d0, DRIFT_HOLD_DAYS)
            pct = safe_pct_change(close_series, d0, d1)
            if not pd.isna(pct):
                changes.append(pct)
        return float(np.mean(changes)) if changes else 0.0
    except Exception:
        return 0.0


def check_earnings_date(ticker: str, days_ahead: int = 14, days_back: int = 7) -> dict:
    """
    Check if company has upcoming earnings or recently reported.
    Returns dict with:
        - has_upcoming: bool (earnings within days_ahead)
        - already_reported: bool (earnings within last days_back days)
        - next_earnings_date: date or None
        - days_until_earnings: int or None
        - status: str description
    """
    try:
        tkr = yf.Ticker(ticker)

        # Method 1: Try calendar attribute (future earnings)
        next_earnings = None
        try:
            calendar = tkr.calendar
            if calendar is not None and isinstance(calendar, dict):
                if 'Earnings Date' in calendar:
                    earnings_dates = calendar['Earnings Date']
                    if isinstance(earnings_dates, (list, tuple)) and len(earnings_dates) > 0:
                        next_earnings = pd.to_datetime(earnings_dates[0])
                    elif earnings_dates is not None:
                        next_earnings = pd.to_datetime(earnings_dates)
        except Exception:
            pass

        # Method 2: Try earnings_dates (historical + future)
        recent_earnings = None
        try:
            ed = tkr.get_earnings_dates(limit=15)
            if ed is not None and not ed.empty:
                today = pd.Timestamp.now().normalize()

                # Find most recent past earnings
                past_dates = [d for d in ed.index if pd.to_datetime(d).normalize() <= today]
                if past_dates:
                    recent_earnings = pd.to_datetime(past_dates[0])

                # Find next future earnings if not found yet
                if next_earnings is None:
                    future_dates = [d for d in ed.index if pd.to_datetime(d).normalize() > today]
                    if future_dates:
                        next_earnings = pd.to_datetime(future_dates[-1])  # Take earliest future date
        except Exception:
            pass

        # Method 3: Try info dict (backup)
        if next_earnings is None and recent_earnings is None:
            try:
                info = tkr.info
                if info and 'earningsTimestamp' in info and info['earningsTimestamp']:
                    timestamp = info['earningsTimestamp']
                    earnings_dt = pd.to_datetime(timestamp, unit='s')
                    today = pd.Timestamp.now().normalize()
                    if earnings_dt.normalize() > today:
                        next_earnings = earnings_dt
                    else:
                        recent_earnings = earnings_dt
            except Exception:
                pass

        today = pd.Timestamp.now().normalize()

        # Check if already reported recently
        already_reported = False
        days_since_report = None
        if recent_earnings is not None:
            days_since_report = (today - recent_earnings.normalize()).days
            if 0 <= days_since_report <= days_back:
                already_reported = True

        # Check if upcoming earnings
        has_upcoming = False
        days_until = None
        if next_earnings is not None:
            days_until = (next_earnings.normalize() - today).days
            if 0 <= days_until <= days_ahead:
                has_upcoming = True

        # Determine status
        if already_reported:
            status = f"  ALREADY REPORTED ({days_since_report} days ago)"
        elif has_upcoming:
            status = f"✓ UPCOMING EARNINGS (in {days_until} days)"
        elif next_earnings is not None and days_until is not None:
            if days_until < 0:
                status = f"  ALREADY REPORTED ({abs(days_until)} days ago)"
                already_reported = True
            else:
                status = f"Future earnings (in {days_until} days)"
        else:
            # No data found - default to scoring it anyway
            status = "  No earnings date found - scoring anyway"

        return {
            'has_upcoming': has_upcoming,
            'already_reported': already_reported,
            'next_earnings_date': next_earnings.date() if next_earnings else None,
            'days_until_earnings': days_until,
            'days_since_report': days_since_report,
            'status': status
        }

    except Exception as e:
        # On error, don't filter out - just warn
        return {
            'has_upcoming': False,
            'already_reported': False,
            'next_earnings_date': None,
            'days_until_earnings': None,
            'days_since_report': None,
            'status': f"  Could not fetch earnings date - scoring anyway"
        }


# ======== FINBERT ========
def load_finbert():
    model_name = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=mdl, tokenizer=tok, truncation=True)


def finbert_score(nlp, texts):
    if not texts:
        return 0.0
    results = nlp(texts)
    vals = []
    for r in results:
        label = r["label"].upper()
        conf = float(r["score"])
        if label == "POSITIVE":
            vals.append(+conf)
        elif label == "NEGATIVE":
            vals.append(-conf)
        else:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


# ======== FINNHUB NEWS ========
def fetch_finnhub_news_titles(ticker: str, max_items: int = 3, lookback_days: int = 7):
    if not FINNHUB_API_KEY or FINNHUB_API_KEY.strip() == "YOUR_API_KEY_HERE":
        return []
    to_dt = date.today()
    from_dt = to_dt - timedelta(days=lookback_days)
    params = {
        "symbol": ticker,
        "from": from_dt.isoformat(),
        "to": to_dt.isoformat(),
        "token": FINNHUB_API_KEY
    }
    url = f"https://finnhub.io/api/v1/company-news?{urlencode(params)}"
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        titles = []
        for item in data[:max_items]:
            h = item.get("headline")
            if h and isinstance(h, str):
                titles.append(h.strip())
        return titles
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return []


# ======== FEATURE COMPUTATION ========
def compute_features_for_ticker(ticker: str, finbert_pipe, verbose=True, skip_if_reported=True):
    try:
        if verbose:
            print(f"  Checking earnings date...", end=" ", flush=True)

        # Check earnings date first
        earnings_info = check_earnings_date(ticker, days_ahead=14, days_back=7)

        # Only skip if we successfully determined it was reported AND skip flag is on
        if skip_if_reported and earnings_info['already_reported']:
            if verbose:
                print(f" {earnings_info['status']}")
            return None

        if verbose:
            if earnings_info['has_upcoming']:
                print(f"✓", end=" ", flush=True)
            elif earnings_info['already_reported']:
                print(f" ", end=" ", flush=True)
            else:
                # No earnings date found or error - continue anyway
                print(f" ", end=" ", flush=True)

        if verbose:
            print(f"data...", end=" ", flush=True)

        hist = None
        for attempt in range(3):
            try:
                hist = yf.download(
                    ticker,
                    period=f"{PRICE_LOOKBACK_DAYS}d",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    timeout=10,
                    group_by='column'
                )
                if hist is not None and not hist.empty:
                    break
                time.sleep(1)
            except Exception as e:
                if attempt == 2:
                    if verbose:
                        print(f"Failed: {e}")
                time.sleep(1)

        if hist is None or hist.empty:
            if verbose:
                print("No data")
            return None

        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.droplevel(1)

        close = hist["Close"].squeeze().copy()
        high = hist["High"].squeeze().copy()
        low = hist["Low"].squeeze().copy()
        vol = hist["Volume"].squeeze().copy()

        if len(close) < 20:
            if verbose:
                print(f"Insufficient data ({len(close)} days)")
            return None

        # Momentum
        mom_10d = close.pct_change(10).iloc[-1] if len(close) >= 11 else np.nan
        rsi14 = rsi(close, 14).iloc[-1]
        ema10 = ema(close, 10).iloc[-1]
        ema50 = ema(close, 50).iloc[-1]
        ema_cross = 1.0 if (not np.isnan(ema10) and not np.isnan(ema50) and ema10 > ema50) else 0.0

        # Volatility
        atr14 = atr(high, low, close, 14).iloc[-1]
        price_now = close.iloc[-1]
        atr_pct = float(atr14 / price_now) if price_now else np.nan
        bbw = bollinger_bandwidth(close, 20, 2.0).iloc[-1]

        # Volume
        vol20 = vol.rolling(20).mean().iloc[-1]
        vol_ratio = float(vol.iloc[-1] / vol20) if (vol20 and not math.isclose(vol20, 0.0)) else 0.0

        # Sharpe
        ret = close.pct_change()
        sharpe30 = sharpe_ratio(ret.tail(30))

        # Earnings drift
        if verbose:
            print("drift...", end=" ", flush=True)
        drift = get_earnings_drift(ticker, close)

        # Sentiment
        if verbose:
            print("sentiment...", end=" ", flush=True)
        titles = fetch_finnhub_news_titles(ticker, max_items=SENTENCE_PER_TICKER, lookback_days=NEWS_LOOKBACK_DAYS)
        time.sleep(NEWS_SLEEP_SEC)
        sent = finbert_score(finbert_pipe, titles) if titles else 0.0

        rsi_scaled = (rsi14 - 50.0) / 20.0

        if verbose:
            print("✓")

        return {
            "Ticker": ticker,
            "mom_10d": mom_10d,
            "rsi_scaled": rsi_scaled,
            "ema_cross": ema_cross,
            "atr_pct": atr_pct,
            "bbw": bbw,
            "vol_ratio": vol_ratio,
            "sharpe30": sharpe30,
            "drift_3d": drift,
            "sent_finbert": sent,
            "earnings_status": earnings_info['status'],
            "earnings_date": earnings_info['next_earnings_date'],
            "days_until_earnings": earnings_info['days_until_earnings'],
            "has_upcoming": earnings_info['has_upcoming'],
            "already_reported": earnings_info['already_reported']
        }
    except Exception as e:
        if verbose:
            print(f"Error: {str(e)[:50]}")
        return None


def calculate_scores(features: dict) -> dict:
    momentum = (0.5 * features["mom_10d"] +
                0.3 * features["rsi_scaled"] +
                0.2 * features["ema_cross"])

    volatility = (-0.6 * features["atr_pct"] +
                  0.4 * features["bbw"])

    volume = features["vol_ratio"]
    sharpe = features["sharpe30"]
    earnings_drift = features["drift_3d"]
    sentiment = features["sent_finbert"]

    composite = (
            0.25 * momentum +
            0.20 * volatility +
            0.15 * volume +
            0.15 * sharpe +
            0.15 * earnings_drift +
            0.10 * sentiment
    )

    # Normalize to 0-10
    normalized_score = 5 * (1 + np.tanh(composite / 1.5))

    return {
        "Momentum": momentum,
        "Volatility": volatility,
        "Volume": volume,
        "Sharpe": sharpe,
        "EarningsDrift": earnings_drift,
        "Sentiment": sentiment,
        "Composite": composite,
        "Score_0_10": normalized_score
    }


def get_score_emoji(score: float) -> str:
    """Return emoji based on score."""
    if score >= 8:
        return "🟢"
    elif score >= 6:
        return "🟡"
    elif score >= 4:
        return "🟠"
    elif score >= 2:
        return "🔴"
    else:
        return "⛔"


def get_rating(score: float) -> str:
    """Return text rating based on score."""
    if score >= 8:
        return "EXCELLENT"
    elif score >= 6:
        return "GOOD"
    elif score >= 4:
        return "NEUTRAL"
    elif score >= 2:
        return "POOR"
    else:
        return "VERY POOR"


def save_to_csv(result: dict, filename: str = OUTPUT_CSV):
    """Save a scored ticker to CSV file."""
    import csv
    import os
    from datetime import datetime

    # Prepare row data
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ticker': result['Ticker'],
        'score': round(result['Score_0_10'], 2),
        'rating': get_rating(result['Score_0_10']),
        'momentum': round(result['Momentum'], 3),
        'volatility': round(result['Volatility'], 3),
        'volume': round(result['Volume'], 3),
        'sharpe': round(result['Sharpe'], 3),
        'earnings_drift': round(result['EarningsDrift'], 3),
        'sentiment': round(result['Sentiment'], 3),
        'earnings_status': result.get('earnings_status', 'Unknown'),
        'earnings_date': str(result.get('earnings_date', '')),
        'days_until_earnings': result.get('days_until_earnings', ''),
        'rsi': round(result['rsi_scaled'] * 20 + 50, 1),
        'volume_ratio': round(result['vol_ratio'], 2),
        'mom_10d': round(result['mom_10d'] * 100, 2) if not pd.isna(result['mom_10d']) else ''
    }

    # Check if file exists
    file_exists = os.path.isfile(filename)

    # Write to CSV
    with open(filename, 'a', newline='') as f:
        fieldnames = list(row.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header if new file
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    return filename


# ======== DISPLAY FUNCTIONS ========
def display_single_ticker_detailed(ticker: str, result: dict):
    """Detailed view for a single ticker."""
    scores = {k: v for k, v in result.items() if
              k in ["Momentum", "Volatility", "Volume", "Sharpe", "EarningsDrift", "Sentiment", "Composite",
                    "Score_0_10"]}

    print(f"\n{'=' * 60}")
    print(f"EARNINGS SCORE: {ticker.upper()}")
    print(f"{'=' * 60}\n")

    # Earnings status
    print(f"   {result.get('earnings_status', 'Unknown')}")
    if result.get('earnings_date'):
        print(f"     Next earnings: {result['earnings_date']}")

    score = scores['Score_0_10']
    emoji = get_score_emoji(score)
    rating = get_rating(score)

    print(f"\n  {emoji} SCORE: {score:.2f} / 10  ({rating})")
    print(f"\n{'=' * 60}")

    print("\nCOMPONENT BREAKDOWN:")
    print(f"  Momentum:        {scores['Momentum']:>7.3f}  (25% weight)")
    print(f"  Volatility:      {scores['Volatility']:>7.3f}  (20% weight)")
    print(f"  Volume:          {scores['Volume']:>7.3f}  (15% weight)")
    print(f"  Sharpe:          {scores['Sharpe']:>7.3f}  (15% weight)")
    print(f"  Earnings Drift:  {scores['EarningsDrift']:>7.3f}  (15% weight)")
    print(f"  Sentiment:       {scores['Sentiment']:>7.3f}  (10% weight)")

    print(f"\n{'=' * 60}")
    print("\nKEY INDICATORS:")
    print(f"  10-day momentum:   {result['mom_10d']:>7.2%}")
    print(f"  RSI:               {(result['rsi_scaled'] * 20 + 50):>7.1f}")
    print(f"  EMA trend:         {'Bullish' if result['ema_cross'] == 1 else 'Bearish'}")
    print(f"  Volume ratio:      {result['vol_ratio']:>7.2f}x")
    print(f"  30d Sharpe:        {result['sharpe30']:>7.2f}")
    print(f"  Earnings drift:    {result['drift_3d']:>7.2%}")
    print(f"  News sentiment:    {result['sent_finbert']:>7.2f}")
    print(f"\n{'=' * 60}\n")


def display_comparison_table(results: list):
    """Comparison table for multiple tickers."""
    if not results:
        return

    print(f"\n{'=' * 90}")
    print("EARNINGS SCORE COMPARISON")
    print(f"{'=' * 90}\n")

    # Sort by score
    results_sorted = sorted(results, key=lambda x: x['Score_0_10'], reverse=True)

    # Header
    print(f"{'Rank':<6}{'Ticker':<8}{'Score':<10}{'Rating':<12}{'Earnings Status':<25}{'Mom':>8}{'Sent':>8}")
    print(f"{'-' * 90}")

    # Rows
    for i, r in enumerate(results_sorted, 1):
        emoji = get_score_emoji(r['Score_0_10'])
        rating = get_rating(r['Score_0_10'])

        # Shorten earnings status for table
        earnings_short = r.get('earnings_status', 'Unknown')[:23]
        if r.get('days_until_earnings') is not None and r['days_until_earnings'] >= 0:
            earnings_short = f"In {r['days_until_earnings']}d"
        elif r.get('already_reported'):
            earnings_short = f" Reported"

        print(
            f"{i:<6}{r['Ticker']:<8}{r['Score_0_10']:<10.2f}{emoji + ' ' + rating:<12}{earnings_short:<25}{r['Momentum']:>8.3f}{r['Sentiment']:>8.3f}")

    print(f"{'-' * 90}")
    print(f"Legend:  Excellent (8-10) |  Good (6-8) |  Neutral (4-6) |  Poor (2-4) |  Very Poor (0-2)")
    print(f"         = Already reported (skip) | ✓ = Upcoming earnings")
    print(f"{'=' * 90}\n")


# ======== MAIN SCORING FUNCTION ========
def score_ticker(ticker: str, finbert_pipe, verbose=True, skip_if_reported=True, save_csv=True):
    """Score a single ticker."""
    if verbose:
        print(f"\n{ticker.upper()}:")

    features = compute_features_for_ticker(ticker.upper(), finbert_pipe, verbose, skip_if_reported)

    if features is None:
        return None

    scores = calculate_scores(features)
    result = {**features, **scores}

    # Save to CSV
    if save_csv:
        try:
            csv_file = save_to_csv(result)
            if verbose:
                print(f"  💾 Saved to {csv_file}")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Could not save to CSV: {e}")

    return result


# ======== INTERACTIVE MODE ========
def interactive_mode(finbert_pipe):
    """Interactive mode - keep asking for tickers."""
    print("\n" + "=" * 60)
    print("INTERACTIVE EARNINGS SCORER")
    print("=" * 60)
    print("\nEnter ticker symbols one at a time.")
    print("Type 'compare' to see comparison table.")
    print("Type 'all' to toggle showing already-reported companies.")
    print("Type 'q' or 'quit' to exit.")
    print(f"\n Results will be saved to: {OUTPUT_CSV}\n")

    results = []
    skip_reported = True

    while True:
        try:
            status = "Skip reported" if skip_reported else "Show all"
            ticker = input(f"[{status}] Enter ticker (or 'q' to quit): ").strip().upper()

            if ticker in ['Q', 'QUIT', 'EXIT']:
                if results:
                    print("\nFinal comparison:")
                    display_comparison_table(results)
                print(f"\n✓ All results saved to: {OUTPUT_CSV}")
                print("\nGoodbye!\n")
                break

            if ticker == 'COMPARE':
                if results:
                    display_comparison_table(results)
                else:
                    print("No tickers scored yet!\n")
                continue

            if ticker == 'ALL':
                skip_reported = not skip_reported
                status_msg = "Now SKIPPING" if skip_reported else "Now SHOWING"
                print(f"  {status_msg} already-reported companies\n")
                continue

            if not ticker:
                continue

            result = score_ticker(ticker, finbert_pipe, verbose=True, skip_if_reported=skip_reported, save_csv=True)

            if result:
                results.append(result)
                display_single_ticker_detailed(ticker, result)
            else:
                print(f"  ✗ Skipped or failed to score {ticker}\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!\n")
            break
        except EOFError:
            break


# ======== MAIN ========
def main():
    """Main entry point with flexible modes."""

    # Check for --all flag to include already-reported companies
    skip_reported = True
    args = sys.argv[1:]

    if '--all' in args:
        skip_reported = False
        args.remove('--all')
        print("Note: Including already-reported companies\n")

    # Load FinBERT once
    print("Loading FinBERT model...")
    finbert_pipe = load_finbert()
    print("✓ FinBERT loaded\n")

    # Determine mode based on arguments
    if len(args) == 0:
        # No arguments - interactive mode
        interactive_mode(finbert_pipe)

    elif len(args) == 1:
        # Single ticker - detailed view
        ticker = args[0]
        result = score_ticker(ticker, finbert_pipe, verbose=True, skip_if_reported=skip_reported, save_csv=True)
        if result:
            display_single_ticker_detailed(ticker, result)
            print(f" Saved to: {OUTPUT_CSV}\n")
        else:
            print(f"\n✗ Skipped or failed to score {ticker}")
            print(f"   (Use --all flag to include already-reported companies)\n")

    else:
        # Multiple tickers - comparison view
        tickers = args
        results = []

        print(f"Scoring {len(tickers)} tickers...")
        print(f"{'Skipping' if skip_reported else 'Including'} already-reported companies\n")

        for ticker in tickers:
            result = score_ticker(ticker, finbert_pipe, verbose=True, skip_if_reported=skip_reported, save_csv=True)
            if result:
                results.append(result)

        if results:
            display_comparison_table(results)
            print(f"✓ All results saved to: {OUTPUT_CSV}\n")
        else:
            print("\n✗ No tickers scored successfully")
            print("   (Use --all flag to include already-reported companies)\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted.\n")
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        raise
