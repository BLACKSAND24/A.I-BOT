import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class Position:
    """Represents a single open trade position."""
    
    def __init__(self, symbol: str, side: str, qty: float, entry_price: float, stop_loss: float, take_profit: float, mode: str = "spot"):
        self.symbol = symbol
        self.side = side  # "BUY" or "SELL"
        self.qty = qty
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_time = datetime.now().isoformat()
        self.status = "OPEN"
        self.exit_price = None
        self.pnl = None
        self.mode = mode  # "spot" or "futures"
        self.highest_price = entry_price  # For trailing SL
        self.lowest_price = entry_price
        
    def update_price(self, current_price: float) -> None:
        """Update high/low for trailing SL."""
        if current_price > self.highest_price:
            self.highest_price = current_price
        if current_price < self.lowest_price:
            self.lowest_price = current_price
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "entry_time": self.entry_time,
            "status": self.status,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "mode": self.mode,
            "highest_price": self.highest_price,
            "lowest_price": self.lowest_price,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Position":
        pos = cls(data["symbol"], data["side"], data["qty"], data["entry_price"], 
                  data["stop_loss"], data["take_profit"], data.get("mode", "spot"))
        pos.entry_time = data.get("entry_time", pos.entry_time)
        pos.status = data.get("status", "OPEN")
        pos.exit_price = data.get("exit_price")
        pos.pnl = data.get("pnl")
        pos.highest_price = data.get("highest_price", pos.highest_price)
        pos.lowest_price = data.get("lowest_price", pos.lowest_price)
        return pos


class PositionManager:
    """Manages multiple open positions and persistence."""
    
    def __init__(self, filepath: str = "positions.json"):
        self.filepath = filepath
        self.positions: Dict[str, Position] = {}
        self.load()
    
    def add_position(self, position: Position) -> None:
        """Add a new position."""
        self.positions[position.symbol] = position
        logger.info("Added position: %s %s %.8f @ %.8f", position.side, position.symbol, position.qty, position.entry_price)
        self.save()
    
    def close_position(self, symbol: str, exit_price: float, status: str = "CLOSED_MANUAL") -> Optional[Position]:
        """Close a position and calculate PnL."""
        if symbol not in self.positions:
            logger.warning("Position not found: %s", symbol)
            return None
        
        pos = self.positions[symbol]
        pos.exit_price = exit_price
        pos.status = status
        
        # Calculate PnL
        if pos.side == "BUY":
            pos.pnl = (exit_price - pos.entry_price) * pos.qty
        else:  # SELL
            pos.pnl = (pos.entry_price - exit_price) * pos.qty
        
        logger.info("Closed position: %s status=%s pnl=%.8f", symbol, status, pos.pnl)
        self.save()
        return pos
    
    def get_open_positions(self) -> List[Position]:
        """Return list of open positions."""
        return [p for p in self.positions.values() if p.status == "OPEN"]
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol."""
        return self.positions.get(symbol)
    
    def has_open_position(self, symbol: str) -> bool:
        """Check if symbol has an open position."""
        pos = self.positions.get(symbol)
        return pos is not None and pos.status == "OPEN"
    
    def save(self) -> None:
        """Persist positions to JSON."""
        try:
            with open(self.filepath, 'w') as f:
                data = {k: v.to_dict() for k, v in self.positions.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save positions: %s", e)
    
    def load(self) -> None:
        """Load positions from JSON."""
        if not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                self.positions = {k: Position.from_dict(v) for k, v in data.items()}
                logger.info("Loaded %d positions from %s", len(self.positions), self.filepath)
        except Exception as e:
            logger.error("Failed to load positions: %s", e)
    
    def summary(self) -> dict:
        """Return summary of all positions (open + closed)."""
        open_pos = self.get_open_positions()
        closed_pos = [p for p in self.positions.values() if p.status != "OPEN"]
        total_pnl = sum(p.pnl for p in closed_pos if p.pnl is not None)
        return {
            "open_count": len(open_pos),
            "closed_count": len(closed_pos),
            "total_pnl": total_pnl,
            "open_positions": [p.to_dict() for p in open_pos],
            "closed_positions": [p.to_dict() for p in closed_pos],
        }
    
    def update_trailing_sl(self, symbol: str, current_price: float, trailing_sl_percent: float) -> float:
        """
        Update stop-loss for position using trailing SL logic.
        For BUY: move SL up if price rises
        For SELL: move SL down if price falls
        Returns new SL price, or original SL if no change.
        """
        pos = self.get_position(symbol)
        if not pos or pos.status != "OPEN":
            return None
        
        pos.update_price(current_price)
        
        if pos.side == "BUY":
            # For BUY: SL should be trailing_sl_percent below highest price
            new_sl = pos.highest_price * (1 - trailing_sl_percent)
            # Only move SL up, never down
            if new_sl > pos.stop_loss:
                pos.stop_loss = new_sl
                logger.info("Updated trailing SL for %s to %.2f", symbol, new_sl)
                self.save()
                return new_sl
        else:  # SELL
            # For SELL: SL should be trailing_sl_percent above lowest price
            new_sl = pos.lowest_price * (1 + trailing_sl_percent)
            if new_sl < pos.stop_loss:
                pos.stop_loss = new_sl
                logger.info("Updated trailing SL for %s to %.2f", symbol, new_sl)
                self.save()
                return new_sl
        
        return pos.stop_loss