# run.py
USE_TESTNET = os.getenv("USE_TESTNET", "false").lower() in ("1","true","yes")
FAKE_MODE = os.getenv("FAKE_MODE", "true").lower() in ("1","true","yes") # default to demo


# Choose which server to run: "api" or "dashboard" or "both"
MODE = os.getenv("RUN_MODE", "both").lower()


if "api" in MODE or "both" in MODE:
print(f"Starting API (testnet={USE_TESTNET}, fake_mode={FAKE_MODE})")
# Import here to ensure env variables loaded first
try:
from app import app as flask_app # make sure app.py exposes `app`
except Exception as e:
print("Could not import app from app.py:", e)
flask_app = None


if FAKE_MODE:
# If your app has a trading engine class, set it to fake mode.
try:
# exchange.py may define a TradingEngine or similar; attempt best-effort toggle
import exchange
if hasattr(exchange, 'TradingEngine'):
try:
exchange.TradingEngine.USE_FAKE_DATA = True
print("Enabled TradingEngine.USE_FAKE_DATA = True")
except Exception:
pass
except Exception:
pass


# Run flask app
if flask_app is not None:
port = int(os.getenv("PORT", 5000))
flask_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if "dashboard" in MODE or "both" in MODE:
print("Starting dashboard server")
# dashboard_server.py should expose a `run_dashboard(port, fake_mode)` or similar
try:
from dashboard_server import run_dashboard
port = int(os.getenv("DASH_PORT", 8050))
run_dashboard(port=port, fake_mode=FAKE_MODE)
except Exception as e:
print("Could not start dashboard via run_dashboard():", e)
print("Attempting to start dashboard_server.py as script")
import subprocess
subprocess.check_call(["python", "dashboard_server.py"])
# demo_data.py
import random
import datetime




def generate_price_series(start_price=30000.0, n=500, volatility=0.01, seed=42):
random.seed(seed)
prices = [start_price]
for i in range(1, n):
pct = random.gauss(0, volatility)
prices.append(max(0.01, prices[-1] * (1 + pct)))
timestamps = [ (datetime.datetime.utcnow() - datetime.timedelta(seconds=(n-i)*10)).isoformat() + "Z" for i in range(n)]
return list(zip(timestamps, prices))




def generate_signals(prices, seed=1):
random.seed(seed)
signals = []
for i,(t,p) in enumerate(prices):
s = "hold"
if (i % 50) < 10:
s = "buy"
elif (i % 50) > 40:
s = "sell"
signals.append({"time": t, "price": p, "signal": s})
return signals




if __name__ == "__main__":
p = generate_price_series()
s = generate_signals(p)
print("generated", len(p), "prices; sample:", p[-3:], s[-3:])