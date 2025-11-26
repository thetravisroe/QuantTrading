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

    # Also precompute actual up/down for each day (for later)
    df["ActualUp"] = (df["Close"].pct_change() > 0).astype(int)

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
    print(f"    Predicted class   : {'UP' if pred_class == 1 else 'DOWN'}")

    # ================== BUILD NEW LOG ROW (CLEAN FORMAT) ==================
    print(">>> Logging prediction to CSV...")

    run_time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    confidence_pct = round(float(prob_up) * 100.0, 1)

    # Basic signal label
    signal_label = "UP" if pred_class == 1 else "DOWN"

    # Simple confidence-based color bucket
    # You can tweak these thresholds
    if prob_up >= 0.65 or prob_up <= 0.35:
        signal_color = "Green"  # high confidence either way
    else:
        signal_color = "Yellow"  # medium confidence

    log_row = pd.DataFrame(
        {
            "Run Time": [run_time_str],
            "Date Checked": [latest_date.date().isoformat()],
            "Predicting For": [str(next_trading_day)],
            "Close": [round(latest_close, 2)],
            "Prob Up": [round(float(prob_up), 2)],
            "Confidence %": [confidence_pct],
            "Signal": [signal_label],
            # These will be filled for past predictions
            "Actual Up": [""],
            "Correct": [""],
            "Signal Color": [signal_color],
        }
    )

    # ================== LOAD / APPEND / TRIM HISTORY ==================
    if os.path.exists(CSV_PATH):
        hist = pd.read_csv(CSV_PATH)
        hist = pd.concat([hist, log_row], ignore_index=True)
    else:
        hist = log_row.copy()

    # ================== FILL 'ACTUAL UP' AND 'CORRECT' WHERE POSSIBLE ==================
    # Make sure df index is Timestamp for lookup
    price_index = df.index

    # For each row in hist, if Predicting For date is in our price data,
    # set Actual Up and Correct
    for idx, row in hist.iterrows():
        try:
            pred_date = pd.to_datetime(row["Predicting For"]).date()
        except Exception:
            continue

        # if we have that date in the downloaded df
        if pd.Timestamp(pred_date) in price_index:
            actual_val = int(df.loc[pd.Timestamp(pred_date), "ActualUp"])
            actual_label = "UP" if actual_val == 1 else "DOWN"
            hist.at[idx, "Actual Up"] = actual_label

            # if we also have a Signal, compute Correct
            sig = str(row.get("Signal", ""))
            if sig in ["UP", "DOWN"]:
                hist.at[idx, "Correct"] = "YES" if sig == actual_label else "NO"

    # Keep only the last 7 rows
    hist = hist.tail(7)

    # Save back to CSV
    hist.to_csv(CSV_PATH, index=False)

    print(f">>> Logged to {CSV_PATH}")
    print(">>> Done.")

except Exception:
    print(">>> ERROR OCCURRED:")
    traceback.print_exc()

