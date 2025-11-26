import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate live prediction logs for a ticker."
    )
    parser.add_argument(
        "-t",
        "--ticker",
        type=str,
        default="SPY",
        help="Ticker symbol, e.g. SPY, AAPL, QQQ. Default: SPY",
    )
    args = parser.parse_args()
    ticker = args.ticker.upper()

    csv_path = f"live_predictions_{ticker}.csv"

    if not os.path.exists(csv_path):
        print(f"Log file not found: {csv_path}")
        print("Run the predictor first to generate some logs.")
        return

    df = pd.read_csv(csv_path)

    if df.empty:
        print(f"{csv_path} is empty.")
        return

    print(f"=== Evaluation for {ticker} ===")
    print(f"Log file: {csv_path}")
    print(f"Total rows in log: {len(df)}")

    # Filter to rows where we know the actual outcome
    df_known = df[df["Actual Up"].isin(["UP", "DOWN"])].copy()

    if df_known.empty:
        print("No rows have 'Actual Up' filled yet.")
        print("Wait until some of the predicted days are in the past, then re-run.")
        return

    n_known = len(df_known)
    print(f"Rows with known outcome: {n_known}")

    # Overall accuracy
    df_known["is_correct"] = df_known["Correct"].str.upper().eq("YES")
    overall_acc = df_known["is_correct"].mean()

    print(f"Overall hit rate: {overall_acc:.3f} ({overall_acc * 100:.1f}%)")

    # Accuracy by Signal Color (confidence bucket)
    for color in ["Green", "Yellow", "Grey"]:
        df_color = df_known[df_known["Signal Color"] == color]
        if not df_color.empty:
            acc = df_color["is_correct"].mean()
            print(
                f"{color} signals: {len(df_color):2d} | "
                f"hit rate: {acc:.3f} ({acc * 100:.1f}%)"
            )

    # Accuracy by Signal direction (UP vs DOWN)
    for sig in ["UP", "DOWN"]:
        df_sig = df_known[df_known["Signal"] == sig]
        if not df_sig.empty:
            acc = df_sig["is_correct"].mean()
            print(
                f"Signal {sig:4s}: {len(df_sig):2d} | "
                f"hit rate: {acc:.3f} ({acc * 100:.1f}%)"
            )


if __name__ == "__main__":
    main()
