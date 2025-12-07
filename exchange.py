# exchange.py
import math
import logging
import os
from typing import Optional, Dict, Any, Tuple, List
import ccxt
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('BINANCE_APIKEY', '')
API_SECRET = os.getenv('BINANCE_SECRET', '')
USE_TESTNET = os.getenv('USE_TESTNET', 'true').lower() in ('1','true','yes')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinanceExchange:
    def __init__(self):
        params = {}
        if API_KEY:
            params['apiKey'] = API_KEY
            params['secret'] = API_SECRET
        if USE_TESTNET:
            params['sandbox'] = True
        self.exchange = ccxt.binance(params)
        self.ws_client = None
        # symbol rules cache used by normalize_qty_price; populate best-effort
        self.symbol_rules: Dict[str, Any] = {}
        try:
            markets = self.exchange.fetch_markets()
            for m in markets:
                # ccxt market object uses 'symbol' and may include 'limits' / 'precision'
                self.symbol_rules[m['symbol']] = m
        except Exception:
            # ignore if exchange doesn't support fetch_markets or network issues
            self.symbol_rules = {}
    
    # ======= Price and Balance =======
    def get_price(self, symbol: str) -> float:
        try:
            t = self.exchange.fetch_ticker(symbol)
            return float(t.get("last") or t.get("price") or 0.0)
        except Exception as e:
            logger.debug(f"get_price failed for {symbol}: {e}")
            return 0.0

    def get_bal(self, asset: str) -> float:
        try:
            b = self.exchange.get_asset_balance(asset=asset.upper())
            return float(b["free"]) if b and b.get("free") is not None else 0.0
        except Exception:
            return 0.0

    # ======= Find tradable symbol =======
    def find_symbol(self, base: str, quote_candidates: List[str]) -> Optional[str]:
        for q in quote_candidates:
            sym = f"{base.upper()}{q.upper()}"
            if sym in self.symbol_rules and self.symbol_rules[sym].get("status") == "TRADING":
                return sym
        return None

    # ======= Normalize qty & price =======
    def _get_filters(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        s = self.symbol_rules.get(symbol)
        filters = {}
        if s:
            for f in s.get("filters", []):
                filters[f["filterType"]] = f
        return filters

    def normalize_qty_price(self, symbol: str, qty: Optional[float], price: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        filters = self._get_filters(symbol)
        if price is not None and "PRICE_FILTER" in filters:
            tick = float(filters["PRICE_FILTER"]["tickSize"])
            price = math.floor(price / tick) * tick
        if qty is not None and "LOT_SIZE" in filters:
            step = float(filters["LOT_SIZE"]["stepSize"])
            min_qty = float(filters["LOT_SIZE"]["minQty"])
            max_qty = float(filters["LOT_SIZE"]["maxQty"])
            qty = math.floor(qty / step) * step
            if qty < min_qty:
                qty = 0.0
            if qty > max_qty:
                qty = max_qty
        return qty, price

    # ======= Market Orders =======
    def market_buy(self, symbol: str, quote_to_spend: float) -> Dict[str, Any]:
        price = self.get_price(symbol)
        if price <= 0:
            raise ValueError("Invalid price")
        qty = quote_to_spend / price
        qty, _ = self.normalize_qty_price(symbol, qty, None)
        if qty <= 0:
            raise ValueError("Quantity too small")
        # use ccxt market order
        return self.exchange.create_market_order(symbol, 'buy', qty)

    def market_sell(self, symbol: str, qty: float) -> Dict[str, Any]:
        qty, _ = self.normalize_qty_price(symbol, qty, None)
        if qty <= 0:
            raise ValueError("Quantity too small")
        return self.exchange.create_market_order(symbol, 'sell', qty)

    # ======= WebSocket =======
    def start_ws_ticker(self, symbol: str, callback):
        logger.info("Websocket ticker not implemented for ccxt wrapper; skipping start_ws_ticker")
        # If you need websockets, implement using exchange-specific SDK or aiohttp websockets.

    def stop_ws(self):
        if self.ws_client:
            try:
                self.ws_client.stop()
            except Exception:
                pass
            self.ws_client = None
