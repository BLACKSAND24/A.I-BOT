import logging
import math
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL
from config import API_KEY, API_SECRET, FUTURES_LEVERAGE, FUTURES_MODE, USE_TESTNET, BINANCE_TESTNET_API

logger = logging.getLogger(__name__)

class FuturesTrader:
    """Wrapper for Binance Futures trading."""
    
    def __init__(self):
        self.client = Client(API_KEY, API_SECRET)
        if USE_TESTNET:
            self.client.API_URL = BINANCE_TESTNET_API
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            logger.info("Set leverage %dx for %s", leverage, symbol)
            return True
        except Exception as e:
            logger.error("Failed to set leverage for %s: %s", symbol, e)
            return False
    
    def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> bool:
        """Set margin type (ISOLATED or CROSS)."""
        try:
            self.client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
            logger.info("Set margin type %s for %s", margin_type, symbol)
            return True
        except Exception as e:
            # May already be set; ignore error
            logger.debug("Margin type for %s: %s", symbol, e)
            return False
    
    def get_balance(self) -> float:
        """Get available USDT balance in futures account."""
        try:
            account = self.client.futures_account()
            for asset in account.get("assets", []):
                if asset["asset"] == "USDT":
                    return float(asset.get("availableBalance", 0))
        except Exception as e:
            logger.error("Failed to get futures balance: %s", e)
        return 0.0
    
    def get_price(self, symbol: str) -> float:
        """Get current mark price."""
        try:
            tick = self.client.futures_symbol_ticker(symbol=symbol)
            return float(tick.get("markPrice", 0.0))
        except Exception as e:
            logger.debug("get_price failed: %s", e)
            return 0.0
    
    def open_position(self, symbol: str, side: str, qty: float) -> dict:
        """Open a futures position (BUY/SELL)."""
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=round(qty, 8)
            )
            logger.info("Opened %s position on %s: qty=%.8f", side, symbol, qty)
            return order
        except Exception as e:
            logger.error("Failed to open position on %s: %s", symbol, e)
            return {}
    
    def close_position(self, symbol: str, qty: float) -> dict:
        """Close an open position."""
        try:
            # Get current position
            pos = self.client.futures_position_information(symbol=symbol)
            if not pos:
                return {}
            
            position = pos[0]
            current_side = position.get("positionSide", "LONG")
            close_side = SIDE_SELL if current_side == "LONG" else SIDE_BUY
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type="MARKET",
                quantity=round(qty, 8)
            )
            logger.info("Closed position on %s", symbol)
            return order
        except Exception as e:
            logger.error("Failed to close position on %s: %s", symbol, e)
            return {}
    
    def set_stop_loss(self, symbol: str, stop_price: float) -> bool:
        """Set stop-loss for open position."""
        try:
            pos = self.client.futures_position_information(symbol=symbol)
            if not pos or float(pos[0]["positionAmt"]) == 0:
                return False
            
            position = pos[0]
            current_side = position.get("positionSide", "LONG")
            close_side = SIDE_SELL if current_side == "LONG" else SIDE_BUY
            qty = abs(float(position.get("positionAmt", 0)))
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type="STOP_MARKET",
                stopPrice=round(stop_price, 8),
                closePosition=True,
                timeInForce="GTE_GTC"
            )
            logger.info("Set SL for %s @ %.2f", symbol, stop_price)
            return True
        except Exception as e:
            logger.debug("set_stop_loss failed: %s", e)
            return False
    
    def get_position_info(self, symbol: str) -> dict:
        """Get current position info."""
        try:
            pos = self.client.futures_position_information(symbol=symbol)
            if pos and float(pos[0]["positionAmt"]) != 0:
                p = pos[0]
                return {
                    "symbol": symbol,
                    "side": "LONG" if float(p.get("positionAmt", 0)) > 0 else "SHORT",
                    "qty": abs(float(p.get("positionAmt", 0))),
                    "entry_price": float(p.get("entryPrice", 0)),
                    "unrealized_pnl": float(p.get("unrealizedProfit", 0)),
                    "pnl_percent": float(p.get("percentage", 0)),
                }
            return {}
        except Exception as e:
            logger.error("get_position_info failed: %s", e)
            return {}
    
    def get_liquidation_price(self, symbol: str) -> float:
        """Get liquidation price for open position."""
        try:
            pos = self.client.futures_position_information(symbol=symbol)
            if pos:
                return float(pos[0].get("liquidationPrice", 0))
        except Exception as e:
            logger.debug("get_liquidation_price failed: %s", e)
        return 0.0