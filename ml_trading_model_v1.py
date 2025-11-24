import traceback

print(">>> Starting ML trading script...")

try:
    import yfinance as yf
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    print(">>> Imports OK")

    # Load data (auto_adjust removed Adj Close)
    print(">>> Downloading SPY data from Yahoo...")
    df = yf.download("SPY", start="2015-01-01", end="2025-01-01", auto_adjust=False)
    print(">>> Download done. Raw shape:", df.shape)

    # Use Close instead of Adj Close
    df["Return"] = df["Close"].pct_change()
    df["Target"] = (df["Return"].shift(-1) > 0).astype(int)

    # Features (simple beginner-friendly)
    print(">>> Creating features...")
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["Vol10"] = df["Return"].rolling(10).std()
    df = df.dropna()
    print(">>> After dropna, shape:", df.shape)

    # Prepare features
    features = ["MA5", "MA20", "Vol10", "Return"]
    X = df[features]
    y = df["Target"]

    print(">>> Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    print(">>> Split sizes:",
          "X_train:", X_train.shape,
          "X_test:", X_test.shape)

    # ML Model
    print(">>> Training RandomForest...")
    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    print(">>> Model trained")

    # Predictions
    print(">>> Making predictions...")
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(">>> DONE")
    print("Accuracy:", acc)

except Exception as e:
    print(">>> ERROR OCCURRED:")
    traceback.print_exc()
