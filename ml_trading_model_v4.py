import os
import argparse
import traceback
from datetime import datetime

import numpy as np
import yfinance as yf
import pandas as pd
from pandas.tseries.offsets import BDay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# ========================= FEATURE ENGINEERING ========================= #

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical features to the price DataFrame."""
    df = df.copy()

    # Basic returns
    df["Return"] = df["Close"].pct_change()
    df["Target"] = (df["Return"].shift(-1) > 0).astype(int)
    df["ActualUp"] = (df["Close"].pct_change() > 0).astype(int)

    # Moving averages
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA_ratio_5_20"] = df["MA5"] / df["MA20"]

    # Volatility features
    df["Vol10"] = df["Return"].rolling(10).std()
    df["Vol20"] = df["Return"].rolling(20).std()

    # Lagged returns (momentum / reversal)
    df["Ret_1"] = df["Return"].shift(1)
    df["Ret_2"] = df["Return"].shift(2)
    df["Ret_5"] = df["Return"].shift(5)
    df["Ret_10"] = df["Return"].shift(10)

    # Day-of-week (0=Monday..4=Friday)
    df["DOW"] = df.index.dayofweek

    # ===== RSI-14 =====
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    df["RSI14"] = rsi.fillna(50)  # neutral when unknown

    # ===== MACD (12-26 EMA + 9 EMA signal) =====
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # ===== Bollinger band position (20-day) =====
    roll20 = df["Close"].rolling(20)
    ma20 = roll20.mean()
    std20 = roll20.std()
    df["Boll_pos"] = (df["Close"] - ma20) / (2 * std20.replace(0, np.nan))

    # ===== Volume features (simple) =====
    if "Volume" in df.columns:
        df["Vol_MA20"] = df["Volume"].rolling(20).mean()
    else:
        df["Vol_MA20"] = np.nan

    df = df.dropna()

    return df


def choose_threshold(probs: np.ndarray, y_true: np.ndarray, lo: float = 0.4, hi: float = 0.6) -> float:
    """Grid-search for the best classification threshold between lo and hi."""
    best_thr = 0.5
    best_acc = 0.0

    for thr in np.linspace(lo, hi, 17):
        preds = (probs >= thr).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    print(f">>> Best threshold in [{lo:.2f}, {hi:.2f}]: {best_thr:.3f} (acc={best_acc:.3f})")
    return best_thr


def get_model(model_name: str):
    """Return an sklearn model instance based on a simple name."""
    name = model_name.lower()
    if name == "gb":
        # Gradient Boosting - can capture more complex interactions than RF sometimes
        return GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=42,
        )
    else:
        # Default: Random Forest
        return RandomForestClassifier(
            n_estimators=600,
            max_depth=10,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )


# ============================== MAIN LOGIC ============================== #

def main():
    print(">>> Starting ML trading predictor v4...")

    parser = argparse.ArgumentParser(description="ML-based next-day direction predictor (multi-ticker).")
    parser.add_argument(
        "-t", "--ticker",
        type=str,
        default="SPY",
        help="Ticker symbol, e.g. SPY, AAPL, QQQ"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="gb",
        choices=["rf", "gb"],
        help="Model type: 'rf' (RandomForest) or 'gb' (GradientBoosting). Default: gb"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2018-01-01",
        help="Start date for historical data (YYYY-MM-DD). Default: 2018-01-01"
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    model_name = args.model
    start_date = args.start

    csv_path = f"live_predictions_{ticker}.csv"

    try:
        # =============== LOAD DATA ===============
        print(f">>> Downloading {ticker} data from Yahoo starting {start_date}...")
        df = yf.download(ticker, start=start_date, auto_adjust=False)
        if df.empty:
            print(">>> No data returned for this ticker. Check the symbol or start date.")
            return

        print(">>> Download done. Raw shape:", df.shape)

        # =============== FEATURES ===============
        print(">>> Building features...")
        df_feat = build_features(df)
        print(">>> After feature engineering, shape:", df_feat.shape)

        # Need a minimum amount of data
        if len(df_feat) < 300:
            print(">>> Not enough data after feature engineering to train a model. Try an earlier start date.")
            return

        feature_cols = [
            "MA5",
            "MA20",
            "MA50",
            "MA_ratio_5_20",
            "Vol10",
            "Vol20",
            "Return",
            "Ret_1",
            "Ret_2",
            "Ret_5",
            "Ret_10",
            "DOW",
            "RSI14",
            "MACD",
            "MACD_signal",
            "MACD_hist",
            "Boll_pos",
            "Vol_MA20",
        ]

        X = df_feat[feature_cols]
        y = df_feat["Target"]

        # =============== DIAGNOSTIC SPLIT ===============
        print(">>> Splitting train/test (time-ordered) for diagnostics...")
        split_idx = int(len(X) * 0.8)
        X_train_dbg, X_test_dbg = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train_dbg, y_test_dbg = y.iloc[:split_idx], y.iloc[split_idx:]

        model = get_model(model_name)
        print(f">>> Training {model_name.upper()} model on diagnostic split...")
        model.fit(X_train_dbg, y_train_dbg)
        probs_dbg = model.predict_proba(X_test_dbg)[:, 1]

        # Threshold tuning
        best_thr = choose_threshold(probs_dbg, y_test_dbg)

        # 0.5 reference
        plain_preds = (probs_dbg >= 0.5).astype(int)
        plain_acc = accuracy_score(y_test_dbg, plain_preds)
        print(f">>> Diagnostic accuracy at 0.5 threshold: {plain_acc:.3f}")

        # =============== FINAL TRAIN ON FULL HISTORY (EXCEPT LAST DAY) ===============
        print(">>> Training final model on full history (except last day)...")
        X_hist = X.iloc[:-1]
        y_hist = y.iloc[:-1]
        X_live = X.iloc[[-1]]

        model.fit(X_hist, y_hist)

        # =============== LIVE PREDICTION ===============
        latest_date = X_live.index[0]

        latest_close = df_feat.loc[latest_date, "Close"]
        if isinstance(latest_close, pd.Series):
            latest_close = float(latest_close.iloc[0])
        else:
            latest_close = float(latest_close)

        prob_up = float(model.predict_proba(X_live)[0, 1])
        pred_class = int(prob_up >= best_thr)

        next_trading_day = (latest_date + BDay(1)).date()

        print(">>> LIVE PREDICTION")
        print(f"    Ticker            : {ticker}")
        print(f"    Model             : {model_name.upper()}")
        print(f"    As-of date        : {latest_date.date()}")
        print(f"    Latest close      : {latest_close:.2f}")
        print(f"    Predicting for    : {next_trading_day}")
        print(f"    Prob market UP    : {prob_up:.3f}")
        print(f"    Decision thr      : {best_thr:.3f}")
        print(f"    Predicted class   : {'UP' if pred_class == 1 else 'DOWN'}")

        # =============== LOG TO CSV (CLEAN, LAST 7 ROWS) ===============
        print(">>> Logging prediction to CSV...")

        run_time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        confidence_pct = round(prob_up * 100.0, 1)
        signal_label = "UP" if pred_class == 1 else "DOWN"

        # Confidence-based color buckets
        if prob_up >= 0.7 or prob_up <= 0.3:
            signal_color = "Green"   # high confidence
        elif prob_up >= 0.55 or prob_up <= 0.45:
            signal_color = "Yellow"  # medium confidence
        else:
            signal_color = "Grey"    # low edge

        log_row = pd.DataFrame(
            {
                "Run Time": [run_time_str],
                "Ticker": [ticker],
                "Model": [model_name.upper()],
                "Date Checked": [latest_date.date().isoformat()],
                "Predicting For": [str(next_trading_day)],
                "Close": [round(latest_close, 2)],
                "Prob Up": [round(prob_up, 3)],
                "Confidence %": [confidence_pct],
                "Threshold": [round(best_thr, 3)],
                "Signal": [signal_label],
                "Actual Up": [""],
                "Correct": [""],
                "Signal Color": [signal_color],
            }
        )

        # Load existing history or start fresh
        if os.path.exists(csv_path):
            hist = pd.read_csv(csv_path)
            hist = pd.concat([hist, log_row], ignore_index=True)
        else:
            hist = log_row.copy()

        # Fill Actual Up and Correct when price data exists
        price_index = df_feat.index
        for idx, row in hist.iterrows():
            try:
                pred_date = pd.to_datetime(row["Predicting For"]).date()
            except Exception:
                continue

            ts = pd.Timestamp(pred_date)
            if ts in price_index:
                actual_val = int(df_feat.loc[ts, "ActualUp"])
                actual_label = "UP" if actual_val == 1 else "DOWN"
                hist.at[idx, "Actual Up"] = actual_label

                sig = str(row.get("Signal", ""))
                if sig in ["UP", "DOWN"]:
                    hist.at[idx, "Correct"] = "YES" if sig == actual_label else "NO"

        # Keep only last 7 rows
        hist = hist.tail(7)

        hist.to_csv(csv_path, index=False)
        print(f">>> Logged to {csv_path}")
        print(">>> Done.")

    except Exception:
        print(">>> ERROR OCCURRED:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
