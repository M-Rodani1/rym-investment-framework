import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def analyze_ticker(ticker: str, start: str = "2020-01-01", show_chart: bool = True):
    print(f"\nFetching data for {ticker}...")
    df = yf.download(ticker, start=start, progress=False, auto_adjust=False)

    if df is None or df.empty:
        print(f" No data found for {ticker}. Check ticker symbol or date range.")
        return None

    # --- Flatten Close column if it's a DataFrame ---
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()

    # --- Indicators ---
    df["MA50"] = close.rolling(window=50, min_periods=25).mean()
    df["MA200"] = close.rolling(window=200, min_periods=100).mean()

    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).rolling(14, min_periods=14).mean()
    roll_down = pd.Series(down, index=close.index).rolling(14, min_periods=14).mean()
    rs = roll_up / roll_down
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = macd_line - signal_line

    # --- Mask for valid rows ---
    mask = df[["MA50", "MA200", "RSI"]].notna().all(axis=1)
    if not mask.any():
        print(f" Not enough data to compute indicators for {ticker}. Try an earlier start date.")
        return None

    latest = df.loc[mask].iloc[-1]

    ma50 = float(latest["MA50"])
    ma200 = float(latest["MA200"])
    rsi_val = float(latest["RSI"])
    macd_hist = float(latest["MACD_HIST"])

    trend = "Bullish" if ma50 > ma200 else "Bearish"
    macd_signal = "Bullish" if macd_hist > 0 else "Bearish"

    ret = close.pct_change()
    vol = ret.std(skipna=True) * np.sqrt(252) * 100.0
    sharpe = (ret.mean(skipna=True) / ret.std(skipna=True)) * np.sqrt(252)

    # --- Output summary ---
    summary = f"""
 Technical Summary for {ticker} ({datetime.today().strftime('%Y-%m-%d')})
────────────────────────────────────────────
Trend Direction     : {trend} (MA50: {ma50:.2f}, MA200: {ma200:.2f})
RSI (14-day)        : {rsi_val:.2f}
MACD Histogram      : {macd_signal}
Volatility (Ann.)   : {vol:.2f}%
Sharpe Ratio (1Y)   : {sharpe:.2f}

Interpretation:
- MA structure indicates a {trend.lower()} bias.
- RSI {rsi_val:.2f} suggests {'strong momentum' if rsi_val < 70 else 'overbought conditions'}.
- MACD histogram is {macd_signal.lower()}, confirming {'positive' if macd_signal == 'Bullish' else 'negative'} momentum.
- Volatility {vol:.2f}% implies {'high' if vol > 25 else 'moderate' if 15 < vol <= 25 else 'low'} risk.
- Sharpe {sharpe:.2f} indicates {'strong' if sharpe > 1 else 'moderate' if sharpe > 0.5 else 'weak'} risk-adjusted performance.

Suggested Documentation Line:
> **{ticker}** — {trend} setup with RSI {rsi_val:.1f}, volatility {vol:.1f}%, Sharpe {sharpe:.2f}.
────────────────────────────────────────────
"""
    print(summary)

    # --- Chart plotting ---
    if show_chart:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 7), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]}
        )
        fig.suptitle(f"{ticker} Technical Chart — {datetime.today().strftime('%Y-%m-%d')}",
                     fontsize=13, fontweight="bold")

        # Price + MAs
        ax1.plot(df.index, close, label="Close", color="black", linewidth=1.2)
        ax1.plot(df.index, df["MA50"], label="MA50", color="blue", linewidth=1)
        ax1.plot(df.index, df["MA200"], label="MA200", color="red", linewidth=1)
        ax1.set_ylabel("Price")
        ax1.legend(loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # RSI subplot
        ax2.plot(df.index, df["RSI"], label="RSI", color="purple", linewidth=1)
        ax2.axhline(70, color="red", linestyle="--", linewidth=0.8)
        ax2.axhline(30, color="green", linestyle="--", linewidth=0.8)
        ax2.set_ylabel("RSI")
        ax2.set_xlabel("Date")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper left")

        plt.tight_layout()
        plt.show()

    return summary


if __name__ == "__main__":
    ticker = input("Enter ticker (e.g. NVDA, GLD, BTC-USD): ").strip().upper()
    analyze_ticker(ticker)
