import os
import json
from dotenv import load_dotenv

load_dotenv()

try:
    with open(os.path.join(os.path.dirname(__file__), "settings.json"), "r") as f:
        _settings = json.load(f)
except Exception:
    _settings = {}

API_KEY = _settings.get("binance_api_key") or os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_APIKEY") or ""
API_SECRET = _settings.get("binance_api_secret") or os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_SECRET") or ""
USE_TESTNET = bool(_settings.get("use_testnet", os.getenv("USE_TESTNET", "true").lower() in ("1","true","yes")))

# Multi-trade settings
TRADING_SYMBOLS = _settings.get("symbols", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT"])
NUM_CONCURRENT_TRADES = _settings.get("num_concurrent_trades", 5)
ALLOCATION_PER_TRADE = 1.0 / NUM_CONCURRENT_TRADES  # divide balance equally

TELEGRAM_TOKEN = _settings.get("telegram_token") or os.getenv("TELEGRAM_TOKEN") or ""
TELEGRAM_CHAT_ID = _settings.get("telegram_chat_id") or os.getenv("TELEGRAM_CHAT_ID") or ""
STOP_LOSS = float(_settings.get("stop_loss", os.getenv("STOP_LOSS", "0.02")))
TAKE_PROFIT = float(_settings.get("take_profit", os.getenv("TAKE_PROFIT", "0.03")))
MODEL_FILE = _settings.get("model_file", os.getenv("MODEL_FILE", "model.joblib"))
INTERVAL = _settings.get("kline_interval", os.getenv("KLINE_INTERVAL", "5m"))

# Testnet base URL
BINANCE_TESTNET_API = "https://testnet.binance.vision/api"

# Trading mode: "spot" or "futures"
TRADING_MODE = _settings.get("trading_mode", os.getenv("TRADING_MODE", "spot"))

# Futures settings
FUTURES_LEVERAGE = int(_settings.get("futures_leverage", os.getenv("FUTURES_LEVERAGE", "2")))
FUTURES_MODE = "ISOLATED" if _settings.get("futures_isolated", True) else "CROSS"

# Dynamic stop-loss
USE_TRAILING_SL = bool(_settings.get("use_trailing_sl", os.getenv("USE_TRAILING_SL", "true").lower() in ("1","true","yes")))
TRAILING_SL_PERCENT = float(_settings.get("trailing_sl_percent", os.getenv("TRAILING_SL_PERCENT", "0.015")))  # 1.5% below high

# Dashboard
DASHBOARD_PORT = int(_settings.get("dashboard_port", os.getenv("DASHBOARD_PORT", "5000")))
DASHBOARD_HOST = _settings.get("dashboard_host", os.getenv("DASHBOARD_HOST", "127.0.0.1"))

if not API_KEY or not API_SECRET:
    raise ValueError("Missing Binance API credentials in settings.json or environment")
MODEL_FILE = "models/model.pkl"  # or your actual path
TRADING_SYMBOL = "BTCUSDT"

