import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==================== CONFIG ====================
DEFAULT_TICKER = "NVDA"   # <-- change this to whatever you want
DEFAULT_PERIOD = "2y"   # e.g. "5y", "2y", "1y"
# =================================================


def download_data(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Download historical daily data for the given ticker."""
    print(f"Requesting data from yfinance for {ticker}, period={period}...")
    df = yf.download(ticker, period=period, auto_adjust=True)

    # If yfinance returns MultiIndex columns (common in newer versions),
    # flatten them to just the first level: 'Open', 'High', 'Low', 'Close', 'Volume', etc.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"Raw download type: {type(df)}")
    print(f"Columns after flattening: {df.columns.tolist()}")
    print(f"Rows downloaded: {len(df)}")

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker} with period={period}")

    df.dropna(inplace=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df - df.copy()

    df["ret_1"] = df["Close"].pct_change()
    df["ret_3"] = df["Close"].pct_change(3)
    df["ret_5"] = df["Close"].pct_change(5)
    df["ret_10"] = df["Close"].pct_change(10)

    df["vol_5"] = df["ret_1"].rolling(window=5).std()
    df["vol_10"] = df["ret_1"].rolling(window=10).std()

    df["ma_5"] = df["Close"].rolling(window=5).mean()
    df["ma_10"] = df["Close"].rolling(window=10).mean()

    # RSI-ish momentum using pure pandas
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    roll_up = gain.rolling(window=14).mean()
    roll_down = loss.rolling(window=14).mean()
    rsi = 100.0 * roll_up / (roll_up + roll_down)
    df["rsi_14"] = rsi

    # Volume feature
    df["vol_rel"] = df["Volume"] / df["Volume"].rolling(20).mean()

    df["Target_Up"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(inplace=True)
    return df


def prepare_dataset(df: pd.DataFrame):
    # List of desired feature columns
    feature_cols_all = [
        "ret_1", "ret_3", "ret_5", "ret_10",
        "vol_5", "vol_10",
        "ma_5", "ma_10",
        "rsi_14",
        "vol_rel",
    ]

    # Keep only columns that actually exist in the dataframe
    available_cols = [col for col in feature_cols_all if col in df.columns]
    missing_cols = [col for col in feature_cols_all if col not in df.columns]

    print("Available feature columns:", available_cols)
    if missing_cols:
        print("WARNING - Missing feature columns (will be skipped):", missing_cols)

    if not available_cols:
        raise RuntimeError("No feature columns found in dataframe. Check feature engineering step.")

    X = df[available_cols].values
    y = df["Target_Up"].values

    return X, y, available_cols


def train_model(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    # v7-style: make the model care about both UP and DOWN days
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=10,
        class_weight="balanced",  # <-- key change
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    return model, acc, report, (X_train, X_test, y_train, y_test, y_proba)


def get_today_feature_row(df: pd.DataFrame, feature_cols):
    last_row = df.iloc[-1]
    return last_row[feature_cols].values.reshape(1, -1), last_row.name, float(last_row["Close"])


def log_prediction(
    ticker: str,
    trade_date: datetime,
    prob_up: float,
    prob_down: float,
    pred_up: int,
    close_price: float,
    log_dir: str = ".",
):
    filename = os.path.join(log_dir, f"live_predictions_{ticker}.csv")
    date_str = trade_date.strftime("%Y-%m-%d")

    row = {
        "date": date_str,
        "ticker": ticker,
        "close": float(close_price),
        "pred_up": int(pred_up),
        "prob_up": float(prob_up),
        "prob_down": float(prob_down),
        "Actual Up": "",
    }

    if os.path.exists(filename):
        df_log = pd.read_csv(filename)
        df_log = pd.concat([df_log, pd.DataFrame([row])], ignore_index=True)
    else:
        df_log = pd.DataFrame([row])

    df_log.to_csv(filename, index=False)
    print(f"Logged prediction to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Train trading model and log today's prediction.")
    parser.add_argument("-t", "--ticker", type=str, default=DEFAULT_TICKER)
    parser.add_argument("-p", "--period", type=str, default=DEFAULT_PERIOD)

    args = parser.parse_args()

    ticker = args.ticker.upper()

    print(f"=== Downloading data for {ticker} ({args.period}) ===")
    df_raw = download_data(ticker, period=args.period)

    print("=== Engineering features ===")
    df_feat = engineer_features(df_raw)

    print("Columns in feature dataframe:", list(df_feat.columns))

    print("=== Preparing dataset ===")
    X, y, feature_cols = prepare_dataset(df_feat)

    print("=== Training model ===")
    model, acc, report, data_splits = train_model(X, y)

    print("\n=== Evaluation (last 20% of history) ===")
    print(f"Accuracy: {acc:.3f}")
    print(report)

    X_today, date_index, close_price = get_today_feature_row(df_feat, feature_cols)
    prob_up = float(model.predict_proba(X_today)[0, 1])
    prob_down = float(1.0 - prob_up)
    pred_up = int(prob_up >= 0.5)

    trade_date = date_index.to_pydatetime() if hasattr(date_index, "to_pydatetime") else datetime.today()

    print("=== Today's Prediction ===")
    print(f"Date: {trade_date.strftime('%Y-%m-%d')}")
    print(f"Ticker: {ticker}")
    print(f"Last close: {close_price:.2f}")
    print(f"Prob UP tomorrow:   {prob_up:.3f}")
    print(f"Prob DOWN tomorrow: {prob_down:.3f}")
    print(f"Predicted direction: {'UP' if pred_up else 'DOWN'}")

    log_prediction(
        ticker=ticker,
        trade_date=trade_date,
        prob_up=prob_up,
        prob_down=prob_down,
        pred_up=pred_up,
        close_price=close_price,
    )


if __name__ == "__main__":
    main()