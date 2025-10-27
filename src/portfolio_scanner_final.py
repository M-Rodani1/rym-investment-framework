import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests, time
from datetime import datetime
from io import StringIO

# ---------- Indicator helpers ----------
def compute_indicators(df):
    close = df["Close"].squeeze()

    # --- moving averages ---
    df["MA50"] = close.rolling(50, min_periods=25).mean()
    df["MA200"] = close.rolling(200, min_periods=100).mean()

    # --- RSI (14) ---
    delta = close.diff().squeeze()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).rolling(14, min_periods=14).mean()
    roll_down = pd.Series(down, index=close.index).rolling(14, min_periods=14).mean()
    rs = roll_up / roll_down
    df["RSI"] = 100 - (100 / (1 + rs))

    # --- MACD (12,26,9) ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = (macd_line - signal_line).squeeze()

    return df


def safe_float(x):
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)

# ---------- Single-ticker analysis ----------
def analyze_ticker(ticker, start="2020-01-01"):
    df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    if df.empty:
        return None
    df = compute_indicators(df)
    latest = df.iloc[-1]

    ma50 = safe_float(latest["MA50"])
    ma200 = safe_float(latest["MA200"])
    rsi = safe_float(latest["RSI"])
    macd = safe_float(latest["MACD_HIST"])
    trend = "Bullish" if ma50 > ma200 else "Bearish"
    macd_signal = "Bullish" if macd > 0 else "Bearish"

    ret = df["Close"].pct_change()
    vol = float(ret.std() * np.sqrt(252) * 100)
    sharpe = float((ret.mean() / ret.std()) * np.sqrt(252))

    return {
        "Ticker": ticker,
        "Trend": trend,
        "RSI": round(rsi, 2),
        "MACD": macd_signal,
        "Volatility_%": round(vol, 2),
        "Sharpe": round(sharpe, 2)
    }, df

# ---------- Plot ----------
def plot_ticker(df, ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"{ticker} Technical Chart — {datetime.today().strftime('%Y-%m-%d')}",
                 fontsize=13, fontweight="bold")
    ax1.plot(df.index, df["Close"], label="Close", color="black")
    ax1.plot(df.index, df["MA50"], label="MA50", color="blue")
    ax1.plot(df.index, df["MA200"], label="MA200", color="red")
    ax1.legend(); ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(df.index, df["RSI"], label="RSI", color="purple")
    ax2.axhline(70, color="red", linestyle="--")
    ax2.axhline(30, color="green", linestyle="--")
    ax2.legend(); ax2.grid(True, linestyle="--", alpha=0.5)

    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=25)
    plt.tight_layout()
    plt.show()

# ---------- Fetch S&P500 tickers safely ----------
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/120.0 Safari/537.36"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    html = StringIO(response.text)
    table = pd.read_html(html)[0]
    return table["Symbol"].tolist()

# ---------- Full S&P500 Scan ----------
def scan_sp500(start="2020-01-01"):
    tickers = get_sp500_tickers()
    print(f"Scanning {len(tickers)} S&P 500 tickers...\n")
    results = []
    for i, t in enumerate(tickers, 1):
        try:
            metrics, _ = analyze_ticker(t, start=start)
            if metrics:
                results.append(metrics)
                print(f"{i:>3}/{len(tickers)}  {t}")
        except Exception as e:
            print(f"{i:>3}/{len(tickers)}  {t}: {e}")
        time.sleep(0.3)
    df = pd.DataFrame(results)
    df.sort_values("Sharpe", ascending=False, inplace=True)
    df.to_csv("sp500_summary.csv", index=False)
    print("\n📁 Results saved to sp500_summary.csv")
    print(df.head(10))
    return df

# ---------- Main menu ----------
if __name__ == "__main__":
    print("\n--- S&P 500 Scanner ---")
    print("1.  Run full S&P 500 scan")
    print("2.  Plot a specific ticker (after scan)")
    choice = input("\nSelect option (1 or 2): ").strip()

    if choice == "1":
        scan_sp500()
    elif choice == "2":
        ticker = input("Enter ticker symbol (e.g. AAPL): ").strip().upper()
        df = yf.download(ticker, start="2020-01-01", progress=False, auto_adjust=False)
        if not df.empty:
            df = compute_indicators(df)
            plot_ticker(df, ticker)
        else:
            print(" No data for that ticker.")
    else:
        print("Invalid choice.")
