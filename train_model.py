import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from binance.client import Client
from config import API_KEY, API_SECRET, TRADING_SYMBOL, MODEL_FILE, INTERVAL, USE_TESTNET, BINANCE_TESTNET_API

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

client = Client(API_KEY, API_SECRET)
if USE_TESTNET:
    client.API_URL = BINANCE_TESTNET_API

def fetch_data(symbol=TRADING_SYMBOL, interval=INTERVAL, limit=1000):
    """Fetches historical kline data from Binance."""
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            "timestamp","open","high","low","close","volume",
            "close_time","qav","num_trades","taker_base","taker_quote","ignore"])

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    except Exception as e:
        logging.error(f"Failed to fetch data: {e}")
        return pd.DataFrame()

def create_features(df):
    """Generates technical features for the model."""
    df["return"] = df["close"].pct_change()
    df["sma"] = df["close"].rolling(window=20).mean()
    df["volatility"] = df["close"].rolling(window=20).std()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df.dropna()

def train_model():
    """Fetches data, creates features, trains a model, and saves it."""
    logging.info(f"[TRAIN] Fetching data for {TRADING_SYMBOL}...")
    df = fetch_data(TRADING_SYMBOL)

    if df.empty:
        logging.error("[TRAIN] No data fetched, aborting training.")
        return

    logging.info("[TRAIN] Creating features...")
    df = create_features(df)

    if df.empty:
        logging.error("[TRAIN] Not enough data to create features, aborting training.")
        return

    X = df[["sma", "volatility", "return"]]
    y = df["target"]

    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)
    logging.info(f"[TRAIN] Model trained and saved as {MODEL_FILE}")

if __name__ == "__main__":
    train_model()