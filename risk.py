# risk.py
import time
import json
import os

STORAGE_PATH = os.path.join(os.path.dirname(__file__), "risk_store.json")

def _read_store():
    if not os.path.exists(STORAGE_PATH):
        return {}
    try:
        with open(STORAGE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _write_store(d):
    with open(STORAGE_PATH, "w") as f:
        json.dump(d, f)

def today_key(prefix: str) -> str:
    d = time.gmtime()
    return f"{prefix}_{d.tm_year:04d}{d.tm_mon:02d}{d.tm_mday:02d}"

def record_pnl(pnl_quote: float):
    k = today_key("pnl")
    store = _read_store()
    cur = store.get(k, 0.0)
    store[k] = cur + pnl_quote
    _write_store(store)

def daily_pnl() -> float:
    store = _read_store()
    return float(store.get(today_key("pnl"), 0.0))

def hit_daily_loss_limit(limit_pct: float, starting_equity_quote: float) -> bool:
    if starting_equity_quote <= 0:
        return False
    pnl = daily_pnl()
    drawdown_pct = -pnl / starting_equity_quote * 100.0 if pnl < 0 else 0.0
    return drawdown_pct >= limit_pct
