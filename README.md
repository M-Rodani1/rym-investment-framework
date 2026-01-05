# rym-investment-framework

Quantitative investment strategy framework built for the UK Investment Challenge 2025 securing 3rd place out of 28 QMUL teams. A dual-module Python system for systematic equity screening and earnings-driven momentum analysis, combining technical indicators, sentiment analysis, and post-earnings drift signals for competition-grade portfolio construction.

## Repository Structure
```
.
├── data/
│   ├── earnings_ranked.csv       # Output from earnings scorer
│   └── sp500_summary.csv         # Output from portfolio scanner
├── src/
│   ├── earnings_ranker.py        # Earnings momentum & sentiment scoring engine
│   └── portfolio_scanner_final.py # Technical scanner for S&P 500 constituents
└── README.md
```

## Project Overview

This framework was developed for systematic stock selection in the UK Investment Challenge 2025, where scoring heavily weights both performance and documentation quality. The system separates signal generation (ML-based screening) from conviction building (earnings timing, sentiment validation, fundamental context) to balance algorithmic efficiency with explainable decision-making required for competition submissions.

## Modules

### 1. Technical Portfolio Scanner (`portfolio_scanner_final.py`)

Implements a multi-indicator screening system for S&P 500 constituents, providing the initial universe of technically-strong candidates.

**Core Functionality:**
- Computes moving averages (MA50, MA200) for trend identification
- Calculates RSI (14-period) for momentum assessment
- Derives MACD histogram (12,26,9) for directional signals
- Computes annualized volatility and Sharpe ratios
- Batch processing of all S&P 500 tickers with rate limiting

**Technical Indicators:**
- Trend classification via moving average crossover (MA50 vs MA200)
- RSI overbought/oversold thresholds (30/70)
- MACD signal line crossover detection
- Rolling volatility (252-day annualized)
- Risk-adjusted returns (Sharpe ratio)

**Outputs:**
- `sp500_summary.csv`: Ranked constituent list sorted by Sharpe ratio
- Interactive matplotlib charts with dual-axis technical overlays

### 2. Earnings Momentum Scorer (`earnings_ranker.py`)

Multi-factor scoring system exploiting post-earnings announcement drift (PEAD) with integrated sentiment analysis. Designed to identify high-conviction opportunities around earnings events.

**Scoring Components (Weighted):**
- **Momentum (25%)**: 10-day returns, RSI scaling, EMA trend (8/21 crossover)
- **Volatility (20%)**: ATR normalization, Bollinger bandwidth, volume expansion
- **Volume (15%)**: Recent volume vs 50-day average
- **Sharpe (15%)**: 30-day risk-adjusted performance
- **Earnings Drift (15%)**: Average 3-day post-announcement returns (last 4 events)
- **Sentiment (10%)**: FinBERT analysis of recent news (7-day window via Finnhub API)

**Features:**
- Earnings calendar integration via yfinance (checks upcoming/recent reports)
- Automated filtering of already-reported companies (configurable)
- FinBERT transformer model for financial news sentiment classification
- 0-10 normalized scoring with categorical ratings (Excellent/Good/Neutral/Poor)
- CSV logging with timestamp for historical tracking and documentation

**Execution Modes:**
- Interactive CLI with persistent session
- Single ticker analysis with detailed breakdown
- Multi-ticker comparison with ranked output
- `--all` flag to include recently-reported companies

## Installation

### Requirements
```bash
pip install yfinance pandas numpy matplotlib transformers torch requests
```

### API Configuration
Set Finnhub API key in `earnings_ranker.py` (line 38):
```python
FINNHUB_API_KEY = "your_api_key_here"
```

## Usage

### Portfolio Scanner

**Full S&P 500 Scan:**
```bash
python portfolio_scanner_final.py
# Select option 1
```

**Plot Specific Ticker:**
```bash
python portfolio_scanner_final.py
# Select option 2, enter ticker (e.g., AAPL)
```

### Earnings Scorer

**Interactive Mode:**
```bash
python earnings_ranker.py
# Enter tickers one-by-one
# Type 'compare' for leaderboard
# Type 'all' to toggle reported-company filtering
```

**Single Ticker Analysis:**
```bash
python earnings_ranker.py AAPL
```

**Multi-Ticker Comparison:**
```bash
python earnings_ranker.py AAPL MSFT GOOGL NVDA
```

**Include Already-Reported Companies:**
```bash
python earnings_ranker.py --all AAPL MSFT
```

## Outputs

### Portfolio Scanner
- **CSV**: `sp500_summary.csv` with columns: `[Ticker, Trend, RSI, MACD, Volatility_%, Sharpe]`
- **Charts**: Dual-panel matplotlib figures (price/MA overlay + RSI subplot)

### Earnings Scorer
- **CSV**: `earnings_scores_log.csv` with timestamped entries containing:
  - All component scores and raw feature values
  - Earnings calendar status and days until/since report
  - Final 0-10 score and categorical rating
- **Console**: Detailed breakdowns or comparison tables depending on mode

## Competition Workflow

The framework supports a systematic weekly process for the UK Investment Challenge:

1. **Initial Screening**: Run `portfolio_scanner_final.py` to identify technically-strong S&P 500 constituents
2. **Earnings Filter**: Use `earnings_ranker.py` in interactive mode to score candidates with upcoming earnings
3. **Conviction Building**: Review detailed breakdowns (momentum, sentiment, drift patterns) for narrative documentation
4. **Position Sizing**: Select top-ranked picks with balanced risk metrics for portfolio allocation
5. **Documentation**: Export CSV logs for competition submission and performance tracking

## Implementation Notes

- **Rate Limiting**: 0.3s delay between S&P 500 ticker requests; 0.15s for news API calls
- **Data Lookback**: 120 days for price history, 4 events for earnings drift calculation
- **Sentiment**: Maximum 3 sentences per ticker analyzed via FinBERT
- **Earnings Window**: Flags companies with earnings within 14 days ahead or 7 days prior
- **Risk Management**: Sharpe calculation assumes zero risk-free rate; 30-day window for earnings scorer

## Potential Improvements

- Integrate live execution via broker API (Alpaca, Interactive Brokers)
- Add sector-neutral scoring to control for industry-wide moves
- Implement adaptive volatility weighting based on VIX regime
- Expand sentiment sources (Twitter/X, Reddit, earnings call transcripts)
- Add backtesting framework to validate scoring efficacy over historical earnings cycles
- Incorporate options flow data (put/call ratios, implied volatility skew)
- Build ensemble model combining both scanners for unified portfolio construction
- Automate weekly report generation for competition documentation requirements

## License

MIT License - Free for academic and commercial use.
