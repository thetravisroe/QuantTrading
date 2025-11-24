import os
import traceback
from datetime import datetime

import yfinance as yf
import pandas as pd
from pandas.tseries.offsets import BDay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print(">>> Starting live ML predictor (v2 file)...")

try:
    TICKER = "SPY"
    CSV_PATH = "live_predictions_SPY.csv"

    # ===================== LOAD DATA =====================
    print(f">>> Downloading {TICKER} data from Yahoo...")
    df = yf.download(TICKER, start="2018-01-01", auto_adjust=False)
    print(">>> Download done. Raw shape:", df.shape)

    # Basic return + target (tomorrow up/down)
    df["Return"] = df["Close"].pct_change()
    df["Target"] = (df["Return"].shift(-1) > 0).astype(int)

    # ================== FEATURE ENGINEERING ==================
    print(">>> Creating features...")
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA_ratio"] = df["MA5"] / df["MA20"]
    df["Vol10"] = df["Return"].rolling(10).std()
    df["Ret_1"] = df["Return"].shift(1)
    df["Ret_2"] = df["Return"].shift(2)
    df["Ret_5"] = df["Return"].shift(5)
    df["DOW"] = df.index.dayofweek

    df = df.dropna()
    print(">>> After dropna, shape:", df.shape)

    feature_cols = [
        "MA5",
        "MA20",
        "MA_ratio",
        "Vol10",
        "Return",
        "Ret_1",
        "Ret_2",
        "Ret_5",
        "DOW",
    ]
    X = df[feature_cols]
    y = df["Target"]

    # ================== DIAGNOSTIC TRAIN/TEST SPLIT ==================
    print(">>> Splitting train/test (for diagnostics only)...")
    split_idx = int(len(X) * 0.8)
    X_train_dbg, X_test_dbg = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_dbg, y_test_dbg = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )

    print(">>> Training model on debug split...")
    model.fit(X_train_dbg, y_train_dbg)
    preds_dbg = model.predict(X_test_dbg)
    acc_dbg = accuracy_score(y_test_dbg, preds_dbg)
    print(f">>> Debug test accuracy (sanity check): {acc_dbg:.3f}")

    # ================== TRAIN FINAL MODEL ON ALL HISTORY (EXCEPT LAST DAY) ==================
    print(">>> Training final model on all history except last day...")
    X_hist = X.iloc[:-1]
    y_hist = y.iloc[:-1]
    X_live = X.iloc[[-1]]  # last row as DataFrame

    model.fit(X_hist, y_hist)

    # ================== MAKE LIVE PREDICTION ==================
    latest_date = X_live.index[0]

    # Safely get latest close as float
    latest_close = df.loc[latest_date, "Close"]
    if isinstance(latest_close, pd.Series):
        latest_close = float(latest_close.iloc[0])
    else:
        latest_close = float(latest_close)

    prob_up = model.predict_proba(X_live)[0, 1]
    pred_class = int(prob_up >= 0.5)  # 1 = UP, 0 = NOT UP

    # Next trading day = next business day
    next_trading_day = (latest_date + BDay(1)).date()

    print(">>> LIVE PREDICTION")
    print(f"    As-of date        : {latest_date.date()}")
    print(f"    Latest close      : {latest_close:.2f}")
    print(f"    Predicting for    : {next_trading_day}")
    print(f"    Prob market UP    : {prob_up:.3f}")
    print(f"    Predicted class   : {'UP' if pred_class == 1 else 'NOT UP'}")

    # ================== LOG TO CSV ==================
    print(">>> Logging prediction to CSV...")

    now_ts = datetime.now()

    log_row = pd.DataFrame(
        {
            "run_timestamp": [now_ts.isoformat(timespec="seconds")],
            "ticker": [TICKER],
            "as_of_date": [latest_date.date().isoformat()],
            "next_trading_day": [str(next_trading_day)],
            "latest_close": [latest_close],
            "prob_up": [float(prob_up)],
            "pred_class": [int(pred_class)],  # 1 = UP, 0 = NOT UP
        }
    )

    file_exists = os.path.exists(CSV_PATH)
    log_row.to_csv(CSV_PATH, mode="a", header=not file_exists, index=False)

    print(f">>> Logged to {CSV_PATH}")
    print(">>> Done.")

except Exception:
    print(">>> ERROR OCCURRED:")
    traceback.print_exc()
