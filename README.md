Overview

QuantTrading is a Python-based quantitative trading model that predicts next-day market direction (UP/DOWN) for any stock or ETF using machine learning.
The project includes:
  Multi-ticker predictions
  Rich technical indicator features (RSI, MACD, Bollinger bands, momentum, volatility, etc.)
  Automatic threshold optimization
  Daily prediction logs stored per-ticker
  Auto-evaluation: logs whether past predictions were correct
  Clean output designed for spreadsheet use
  Rolling 7-day prediction history
  SPY/AAPL/QQQ tested and supported
This repository is built for ongoing experimentation, tracking real-time predictions, and improving accuracy over time.

How It Works

1. Train on historical data
The model downloads historical OHLCV data using Yahoo Finance and builds a feature set including:
  Moving averages (5, 20, 50-day)
  Volatility windows (10, 20-day)
  Lagged returns (1, 2, 5, 10-day)
  RSI-14
  MACD / MACD signal / MACD histogram
  Bollinger band position
  Volume MA20
  Day-of-week
  Close-to-close percentage returns

2. Diagnostic mode
Before generating a live prediction, the model:
  Splits the data 80/20 in time order
  Trains a RandomForestClassifier
  Tests thresholds from 0.40 → 0.60
  Selects the threshold that yields the best out-of-sample accuracy
      Reports:
          Diagnostic accuracy
          Best-performing threshold

3. Live prediction
After training on all but the most recent day, the model predicts whether tomorrow will be UP or DOWN, using:
  Tuned threshold
  Probability confidence
  Technical feature vector

Example output:
LIVE PREDICTION
    Ticker            : SPY
    As-of date        : 2025-11-26
    Latest close      : 680.19
    Predicting for    : 2025-11-27
    Prob market UP    : 0.576
    Decision thr      : 0.400
    Predicted class   : UP

Prediction Logging
Each run appends a new row to a ticker-specific CSV file:
live_predictions_SPY.csv
live_predictions_AAPL.csv
live_predictions_QQQ.csv

Usage
Run a prediction for SPY:

python ml_trading_model_v3.py -t SPY

Run for any ticker:

python ml_trading_model_v3.py -t AAPL
python ml_trading_model_v3.py -t TSLA
python ml_trading_model_v3.py -t QQQ
python ml_trading_model_v3.py -t MSFT

Requirements

Install dependencies inside a Python virtual environment:

pip install yfinance pandas numpy scikit-learn matplotlib

Project Structure

QuantTrading/
│
├── ml_trading_model_v1.py          # Basic version
├── ml_trading_model_v2.py          # SPY-only predictor
├── ml_trading_model_v3.py          # Multi-ticker, feature-rich predictor (main)
│
├── live_predictions_SPY.csv        # Auto-generated logs
├── live_predictions_AAPL.csv
├── live_predictions_QQQ.csv
│
├── .gitignore
└── README.md                       # <--- you are here

