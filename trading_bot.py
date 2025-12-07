import time
import logging
import joblib
import pandas as pd
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, TIME_IN_FORCE_GTC
from config import (
    API_KEY, API_SECRET, TRADING_SYMBOLS, NUM_CONCURRENT_TRADES, ALLOCATION_PER_TRADE,
    STOP_LOSS, TAKE_PROFIT, MODEL_FILE, INTERVAL, USE_TESTNET, BINANCE_TESTNET_API
)
from notifier import send_telegram_message
from position_manager import PositionManager, Position
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

client = Client(API_KEY, API_SECRET)
if USE_TESTNET:
    client.API_URL = BINANCE_TESTNET_API

pos_mgr = PositionManager("positions.json")


def get_usdt_balance() -> float:
    """Get available USDT balance."""
    try:
        b = client.get_asset_balance(asset="USDT")
        if b and b.get("free") is not None:
            return float(b["free"])
    except Exception as e:
        logger.debug("get_usdt_balance failed: %s", e)
    return 0.0


def get_price(symbol: str) -> float:
    """Get current price for symbol."""
    try:
        tick = client.get_symbol_ticker(symbol=symbol)
        return float(tick.get("price", 0.0))
    except Exception as e:
        logger.debug("get_price failed for %s: %s", symbol, e)
        return 0.0


def get_latest_data(symbol: str, interval: str = INTERVAL, lookback: int = 30) -> pd.DataFrame:
    """Fetch OHLCV data for symbol."""
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=lookback)
        df = pd.DataFrame(klines, columns=[
            "timestamp","open","high","low","close","volume",
            "close_time","qav","num_trades","taker_base","taker_quote","ignore"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        return df
    except Exception as e:
        logger.error("get_latest_data failed for %s: %s", symbol, e)
        return pd.DataFrame()


def generate_signal_ml(df: pd.DataFrame, model) -> str:
    """Generate BUY/SELL/HOLD signal using ML model."""
    if df is None or df.empty or len(df) < 20:
        return "HOLD"
    
    try:
        df = df.copy()
        df["return"] = df["close"].pct_change()
        df["sma"] = df["close"].rolling(window=20).mean()
        df["volatility"] = df["close"].rolling(window=20).std()
        latest = df.iloc[-1:][["sma", "volatility", "return"]].dropna()
        
        if latest.empty:
            return "HOLD"
        
        pred = model.predict(latest)[0]
        return "BUY" if int(pred) == 1 else "SELL"
    except Exception as e:
        logger.error("generate_signal_ml failed: %s", e)
        return "HOLD"


def generate_signal_sma(df: pd.DataFrame) -> str:
    """Fallback: simple SMA crossover signal."""
    try:
        if df is None or df.empty or len(df) < 15:
            return "HOLD"
        sma_short = df["close"].rolling(window=5).mean().iloc[-1]
        sma_long = df["close"].rolling(window=15).mean().iloc[-1]
        if sma_short > sma_long:
            return "BUY"
        elif sma_short < sma_long:
            return "SELL"
        return "HOLD"
    except Exception as e:
        logger.error("generate_signal_sma failed: %s", e)
        return "HOLD"


def execute_trade(symbol: str, signal: str, usdt_balance: float, model=None) -> bool:
    """
    Execute a trade on a symbol if conditions are met.
    Returns True if trade executed, False otherwise.
    """
    # Skip if already have open position on this symbol
    if pos_mgr.has_open_position(symbol):
        logger.debug("Already have open position on %s, skipping.", symbol)
        return False
    
    # Only execute BUY signals for now (can expand to SELL/shorts later)
    if signal != "BUY":
        logger.debug("Signal for %s is %s, not executing.", symbol, signal)
        return False
    
    try:
        price = get_price(symbol)
        if price <= 0:
            logger.error("Invalid price for %s", symbol)
            return False
        
        # Calculate qty based on allocated USDT
        allocated_usdt = usdt_balance * ALLOCATION_PER_TRADE
        qty = allocated_usdt / price
        
        # Round to valid precision
        qty = math.floor(qty * 100000000) / 100000000
        
        if qty <= 0:
            logger.warning("Qty too small for %s: %.8f", symbol, qty)
            return False
        
        # Check if we have enough USDT
        required = qty * price
        if usdt_balance < required:
            logger.warning("Insufficient USDT for %s: need %.2f, have %.2f", symbol, required, usdt_balance)
            return False
        
        # Place market BUY order
        order = client.order_market_buy(symbol=symbol, quantity=round(qty, 8))
        logger.info("BUY executed on %s: qty=%.8f @ %.8f", symbol, qty, price)
        
        # Calculate stop-loss and take-profit
        stop_loss_price = round(price * (1 - STOP_LOSS), 8)
        take_profit_price = round(price * (1 + TAKE_PROFIT), 8)
        
        # Record position
        position = Position(
            symbol=symbol,
            side="BUY",
            qty=qty,
            entry_price=price,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price
        )
        pos_mgr.add_position(position)
        
        # Send telegram notification
        send_telegram_message(f"[TRADE] BUY {symbol}\nQty: {qty:.8f}\nPrice: {price:.2f}\nSL: {stop_loss_price:.2f}\nTP: {take_profit_price:.2f}")
        
        return True
    except Exception as e:
        logger.error("execute_trade failed for %s: %s", symbol, e)
        send_telegram_message(f"[ERROR] Trade execution failed for {symbol}: {e}")
        return False


def check_positions_sl_tp() -> None:
    """
    Monitor open positions against SL/TP levels.
    Close positions that hit SL or TP.
    """
    for pos in pos_mgr.get_open_positions():
        try:
            price = get_price(pos.symbol)
            if price <= 0:
                continue
            
            # Check for SL
            if price <= pos.stop_loss:
                pos_mgr.close_position(pos.symbol, price, "CLOSED_SL")
                pnl = (price - pos.entry_price) * pos.qty
                logger.info("Stop-loss hit on %s @ %.2f, PnL: %.8f", pos.symbol, price, pnl)
                send_telegram_message(f"[SL HIT] {pos.symbol}\nExit: {price:.2f}\nPnL: {pnl:.8f}")
            
            # Check for TP
            elif price >= pos.take_profit:
                pos_mgr.close_position(pos.symbol, price, "CLOSED_TP")
                pnl = (price - pos.entry_price) * pos.qty
                logger.info("Take-profit hit on %s @ %.2f, PnL: %.8f", pos.symbol, price, pnl)
                send_telegram_message(f"[TP HIT] {pos.symbol}\nExit: {price:.2f}\nPnL: {pnl:.8f}")
        except Exception as e:
            logger.error("Error monitoring position %s: %s", pos.symbol, e)


def run_bot() -> None:
    """Main trading loop."""
    model = None
    try:
        model = joblib.load(MODEL_FILE)
        logger.info("ML model loaded from %s", MODEL_FILE)
    except Exception as e:
        logger.warning("No ML model found: %s. Using SMA fallback.", e)
    
    logger.info("Starting multi-trade bot with %d symbols", len(TRADING_SYMBOLS))
    send_telegram_message(f"[BOT START] Multi-trade bot started\nSymbols: {', '.join(TRADING_SYMBOLS)}\nConcurrent: {NUM_CONCURRENT_TRADES}")
    
    cycle = 0
    while True:
        cycle += 1
        logger.info("--- Cycle %d ---", cycle)
        
        try:
            # Get available USDT
            usdt_balance = get_usdt_balance()
            logger.info("Available USDT: %.2f", usdt_balance)
            
            if usdt_balance < 10:
                logger.warning("Insufficient USDT balance (%.2f)", usdt_balance)
                time.sleep(300)
                continue
            
            # Check existing positions for SL/TP
            check_positions_sl_tp()
            
            # Generate signals for each symbol
            num_open = len(pos_mgr.get_open_positions())
            available_slots = NUM_CONCURRENT_TRADES - num_open
            
            if available_slots <= 0:
                logger.info("Max positions reached (%d), waiting...", NUM_CONCURRENT_TRADES)
                time.sleep(300)
                continue
            
            # Scan symbols for trading signals
            for symbol in TRADING_SYMBOLS:
                if not available_slots:
                    break
                
                if pos_mgr.has_open_position(symbol):
                    continue
                
                df = get_latest_data(symbol)
                signal = generate_signal_ml(df, model) if model else generate_signal_sma(df)
                logger.info("Signal for %s: %s", symbol, signal)
                
                if execute_trade(symbol, signal, usdt_balance, model):
                    available_slots -= 1
            
            # Log portfolio status
            summary = pos_mgr.summary()
            logger.info("Portfolio: %d open, %d closed, PnL: %.8f", 
                       summary["open_count"], summary["closed_count"], summary["total_pnl"])
            
            time.sleep(300)  # Wait 5 minutes before next cycle
            
        except Exception as e:
            logger.error("Unexpected error in bot loop: %s", e)
            send_telegram_message(f"[ERROR] Bot error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    try:
        client.get_account()
        logger.info("Binance API connection validated.")
    except Exception as e:
        logger.error("Binance API validation failed: %s", e)
        logger.error("Check API credentials in settings.json")
        exit(1)
    
    run_bot()