# trading.py
import os
import time
import threading
import logging
from typing import Dict, Any, List
import pandas as pd
import joblib
import ta

from exchange import BinanceExchange
from notifier import send_telegram_message
from config import MODEL_FILE, TRADING_SYMBOL

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TradingEngine:
    def __init__(self, load_model: bool = True):
        """
        High-level trading engine that wraps the exchange adapter and ML model.
        Methods exposed: fetch_ticker, fetch_ohlcv_df, compute_signal, place_order, get_trade_log
        """
        self.exchange = BinanceExchange()
        self.trade_log = []
        self.model = None

        if load_model:
            try:
                self.model = joblib.load(MODEL_FILE)
                logger.info("Loaded ML model from %s", MODEL_FILE)
            except Exception as e:
                logger.info("No ML model loaded (%s). Falling back to rule-based signals.", e)

    def fetch_ohlcv_df(self, symbol: str = TRADING_SYMBOL, timeframe: str = "1m", limit: int = 500) -> pd.DataFrame:
        """
        Return OHLCV DataFrame from the exchange. Columns: ts, open, high, low, close, vol
        """
        try:
            raw = self.exchange.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "vol"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            for c in ["open", "high", "low", "close", "vol"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        except Exception as e:
            logger.warning("fetch_ohlcv_df failed: %s", e)
            return pd.DataFrame()

    def fetch_ticker(self, symbol: str = TRADING_SYMBOL) -> Dict[str, Any]:
        """
        Return a simple ticker dict with last/bid/ask/timestamp.
        """
        try:
            t = self.exchange.exchange.fetch_ticker(symbol)
            return {
                "symbol": symbol,
                "timestamp": t.get("timestamp"),
                "last": t.get("last"),
                "bid": t.get("bid"),
                "ask": t.get("ask"),
            }
        except Exception as e:
            logger.warning("fetch_ticker error: %s", e)
            return {"symbol": symbol, "error": str(e)}

    def compute_signal(self, df: pd.DataFrame) -> str:
        """
        Compute a trading signal. Use ML model if available, otherwise simple RSI rule.
        Returns: "buy", "sell", or "hold"
        """
        try:
            if df is None or df.empty:
                return "hold"

            # ML path
            if self.model is not None:
                # build features consistent with train_model.py
                temp = df.copy()
                temp["return"] = temp["close"].pct_change()
                temp["sma"] = temp["close"].rolling(window=20).mean()
                temp["volatility"] = temp["close"].rolling(window=20).std()
                latest = temp.iloc[-1:][["sma", "volatility", "return"]].dropna()
                if not latest.empty:
                    pred = self.model.predict(latest)[0]
                    return "buy" if int(pred) == 1 else "sell"

            # Fallback: RSI rule
            series = pd.to_numeric(df["close"], errors="coerce").dropna()
            if len(series) < 15:
                return "hold"
            rsi = ta.momentum.rsi(series, window=14)
            last = rsi.iloc[-1] if not rsi.empty else None
            if last is None or pd.isna(last):
                return "hold"
            if last < 30:
                return "buy"
            if last > 70:
                return "sell"
            return "hold"
        except Exception as e:
            logger.error("compute_signal error: %s", e)
            return "hold"

    def place_order(self, symbol: str, side: str, size: float, mode: str = "paper") -> Dict[str, Any]:
        """
        Place an order. mode: 'paper' (default) or 'live'.
        For 'paper' a simulated trade entry is appended to trade_log.
        For 'live' uses exchange.market_buy/market_sell wrappers where available.
        """
        side_l = side.lower()
        entry = {"symbol": symbol, "side": side_l, "size": size, "mode": mode}
        try:
            if mode == "paper":
                entry["time"] = pd.Timestamp.now().isoformat()
                self.trade_log.append(entry)
                logger.info("Recorded paper trade: %s", entry)
                send_telegram_message(f"[PAPER] {side.upper()} {size} {symbol}")
                return {"status": "ok", "result": entry}

            # live trading branch
            if side_l == "buy":
                # try market_buy with quote_to_spend first (exchange wrapper may expect quote)
                try:
                    res = self.exchange.market_buy(symbol, quote_to_spend=size)
                except TypeError:
                    # fallback: treat size as base amount
                    res = self.exchange.exchange.create_market_order(symbol, "buy", size)
            else:
                # sell
                try:
                    res = self.exchange.market_sell(symbol, size)
                except Exception:
                    res = self.exchange.exchange.create_market_order(symbol, "sell", size)

            logger.info("Live order placed: %s", res)
            send_telegram_message(f"[LIVE] {side.upper()} {size} {symbol} - {res}")
            return {"status": "ok", "result": res}
        except Exception as e:
            logger.error("place_order failed: %s", e)
            send_telegram_message(f"[ERROR] Order failed: {e}")
            return {"status": "error", "message": str(e)}

    def get_trade_log(self):
        return self.trade_log

    def _map_tf(self, tf):
        return {
            '1m':'1m','3m':'3m','5m':'5m','15m':'15m','30m':'30m','1h':'1h','4h':'4h'
        }.get(tf, '1m')
