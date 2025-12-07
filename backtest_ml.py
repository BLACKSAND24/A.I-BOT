import logging
import joblib
import pandas as pd
from train_model import fetch_data, create_features
from config import TRADING_SYMBOL, MODEL_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

def backtest():
    try:
        model = joblib.load(MODEL_FILE)
    except Exception as e:
        logging.error("Load model failed: %s", e)
        return
    df = fetch_data(TRADING_SYMBOL, limit=2000)
    if df.empty:
        logging.error("No data")
        return
    df = create_features(df)
    X = df[["sma","volatility","return"]]
    df["preds"] = model.predict(X)
    df["strategy_return"] = df["preds"].shift(1) * df["return"]
    df["cumulative_return"] = (1 + df["strategy_return"]).cumprod()
    logging.info("Final cumulative: %s", df["cumulative_return"].iloc[-1])
    total_trades = int(df["preds"].sum())
    wins = int(df[df["preds"]==1]["strategy_return"].gt(0).sum())
    win_rate = (wins / total_trades * 100) if total_trades else 0
    logging.info("Trades: %d Wins: %d Win rate: %.2f%%", total_trades, wins, win_rate)

if __name__ == "__main__":
    backtest()