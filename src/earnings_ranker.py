# earnings_ranker_fixed.py
# Earnings-week technical + sentiment ranking using FinBERT + Finnhub headlines
# Output: earnings_ranked.csv

import os
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

# Networking for Finnhub (use stdlib to avoid extra deps)
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from urllib.error import URLError, HTTPError

warnings.filterwarnings("ignore", category=FutureWarning)

# ========= CONFIG =========
OUTPUT_CSV = "earnings_ranked.csv"
PRICE_LOOKBACK_DAYS = 120
SENTENCE_PER_TICKER = 3
EARNINGS_DRIFT_EVENTS = 4
DRIFT_HOLD_DAYS = 3
NEWS_SLEEP_SEC = 0.15
NEWS_LOOKBACK_DAYS = 7  # as requested

# >>> Paste your Finnhub API key here <<<
FINNHUB_API_KEY = "d40dd0hr01qqo3qhegc0d40dd0hr01qqo3qhegcg"

# Default: top-50 from your provided list (edit/extend as needed)
TICKERS = [
    "MSFT", "AAPL", "GOOG", "AMZN", "META", "LLY", "V", "MA", "XOM", "ABBV",
    "UNH", "CVX", "CAT", "MRK", "LIN", "NOW", "NEE", "BKNG", "BA", "VZ",
    "KLAC", "SPGI", "GILD", "SYK", "WELL", "MELI", "ADP", "CMCSA", "SCCO", "MO",
    "CVS", "SO", "SBUX", "TT", "COIN", "RBLX", "CDNS", "ICE", "SHW", "AMT",
    "BMY", "UPS", "MSTR", "EQIX", "CI", "HWM", "WM", "RCL", "CVNA", "MDLZ"
]


# ========= HELPERS =========
def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu, sd = s.mean(skipna=True), s.std(skipna=True)
    if sd is None or sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=series.index)
    return (s - mu) / sd


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
    """Fetch up to `max_items` headlines for `ticker` from Finnhub within the last `lookback_days` days."""
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


# ======== PER-TICKER FEATURES ========
def compute_features_for_ticker(ticker: str, finbert_pipe):
    try:
        # Add retry logic and better error handling
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
                    group_by='column'  # Prevent MultiIndex issues
                )
                if hist is not None and not hist.empty:
                    break
                time.sleep(1)  # Wait before retry
            except Exception as e:
                if attempt == 2:  # Last attempt
                    print(f"  ⚠️  Failed after 3 attempts: {e}")
                time.sleep(1)

        if hist is None or hist.empty:
            print(f"  ✗ No data returned")
            return None

        # Handle MultiIndex columns from yfinance
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.droplevel(1)

        # Flatten and ensure we have the right columns
        close = hist["Close"].squeeze().copy()
        high = hist["High"].squeeze().copy()
        low = hist["Low"].squeeze().copy()
        vol = hist["Volume"].squeeze().copy()

        # Check if we have enough data
        if len(close) < 20:
            print(f"  ✗ Insufficient data (only {len(close)} days)")
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

        # Volume activity
        vol20 = vol.rolling(20).mean().iloc[-1]
        vol_ratio = float(vol.iloc[-1] / vol20) if (vol20 and not math.isclose(vol20, 0.0)) else 0.0

        # Sharpe (30d)
        ret = close.pct_change()
        sharpe30 = sharpe_ratio(ret.tail(30))

        # Earnings drift
        drift = get_earnings_drift(ticker, close)

        # Sentiment
        titles = fetch_finnhub_news_titles(ticker, max_items=SENTENCE_PER_TICKER, lookback_days=NEWS_LOOKBACK_DAYS)
        time.sleep(NEWS_SLEEP_SEC)
        sent = finbert_score(finbert_pipe, titles) if titles else 0.0

        # RSI centered
        rsi_scaled = (rsi14 - 50.0) / 20.0

        print(f"  ✓ Success")
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
        }
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:50]}")
        return None


def build_rank_table(tickers: list):
    print("Loading FinBERT (ProsusAI/finbert)...")
    finbert_pipe = load_finbert()
    print("✓ FinBERT loaded\n")

    rows = []
    failed = []
    for i, t in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {t}", end=" ")
        feat = compute_features_for_ticker(t, finbert_pipe)
        if feat is not None:
            rows.append(feat)
        else:
            failed.append(t)

    print(f"\n{'=' * 60}")
    print(f"✓ Successfully processed: {len(rows)}/{len(tickers)} tickers")
    if failed:
        print(f"✗ Failed tickers: {', '.join(failed)}")
    print(f"{'=' * 60}\n")

    if not rows:
        raise RuntimeError(
            "No data collected. Possible issues:\n"
            "  1. Network connectivity problems\n"
            "  2. Yahoo Finance is blocking requests\n"
            "  3. Invalid ticker symbols\n"
            "Try running with a smaller list of well-known tickers first (e.g., just AAPL, MSFT, GOOG)"
        )

    df = pd.DataFrame(rows)

    # Z-scores
    df["Z_mom"] = zscore(df["mom_10d"])
    df["Z_rsi"] = zscore(df["rsi_scaled"])
    df["Z_ema"] = zscore(df["ema_cross"])
    df["Z_atr"] = -zscore(df["atr_pct"])
    df["Z_bbw"] = zscore(df["bbw"])
    df["Z_vol"] = zscore(df["vol_ratio"])
    df["Z_shp"] = zscore(df["sharpe30"])
    df["Z_drift"] = zscore(df["drift_3d"])
    df["Z_sent"] = zscore(df["sent_finbert"])

    # Buckets
    df["Momentum"] = (0.5 * df["Z_mom"] + 0.3 * df["Z_rsi"] + 0.2 * df["Z_ema"])
    df["Volatility"] = (0.6 * df["Z_atr"] + 0.4 * df["Z_bbw"])
    df["Volume"] = df["Z_vol"]
    df["Sharpe"] = df["Z_shp"]
    df["EarningsDrift"] = df["Z_drift"]
    df["Sentiment"] = df["Z_sent"]

    # Composite
    df["Composite"] = (
            0.25 * df["Momentum"] +
            0.20 * df["Volatility"] +
            0.15 * df["Volume"] +
            0.15 * df["Sharpe"] +
            0.15 * df["EarningsDrift"] +
            0.10 * df["Sentiment"]
    )

    out_cols = [
        "Ticker",
        "Momentum", "Volatility", "Volume", "Sharpe", "EarningsDrift", "Sentiment",
        "mom_10d", "rsi_scaled", "ema_cross", "atr_pct", "bbw", "vol_ratio", "sharpe30", "drift_3d", "sent_finbert",
        "Composite"
    ]
    df_out = df[out_cols].sort_values("Composite", ascending=False).reset_index(drop=True)
    return df_out


def main():
    start = time.time()
    print("Starting earnings ranker...\n")

    try:
        table = build_rank_table(TICKERS)
        table.to_csv(OUTPUT_CSV, index=False)
        elapsed = time.time() - start

        print(f"\n{'=' * 60}")
        print(f"✓ SUCCESS - Saved: {OUTPUT_CSV}")
        print(f"  Rows: {len(table)} | Time: {elapsed:.1f}s")
        print(f"{'=' * 60}\n")
        print("Top 10 by Composite Score:")
        print(table[["Ticker", "Composite", "Momentum", "Sentiment"]].head(10).to_string(index=False))
    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"✗ ERROR: {str(e)}")
        print(f"{'=' * 60}\n")
        raise


if __name__ == "__main__":
    main()