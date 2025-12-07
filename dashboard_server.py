import json
import logging
from flask import Flask, render_template, jsonify
from position_manager import PositionManager
from config import DASHBOARD_PORT, DASHBOARD_HOST

logger = logging.getLogger(__name__)
app = Flask(__name__)
pos_mgr = PositionManager("positions.json")

@app.route("/api/positions", methods=["GET"])
def get_positions():
    """Return all positions (open + closed)."""
    summary = pos_mgr.summary()
    return jsonify(summary)

@app.route("/api/positions/open", methods=["GET"])
def get_open_positions():
    """Return open positions only."""
    open_pos = pos_mgr.get_open_positions()
    return jsonify({"positions": [p.to_dict() for p in open_pos]})

@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Return portfolio stats."""
    summary = pos_mgr.summary()
    open_pos = summary["open_count"]
    closed_pos = summary["closed_count"]
    total_pnl = summary["total_pnl"]
    
    return jsonify({
        "open_positions": open_pos,
        "closed_positions": closed_pos,
        "total_pnl": total_pnl,
        "win_rate": (sum(1 for p in summary.get("closed_positions", []) if p.get("pnl", 0) > 0) / max(closed_pos, 1)) * 100 if closed_pos > 0 else 0,
    })

@app.route("/")
def dashboard():
    """Serve dashboard HTML."""
    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=True)
    # --- appended: run_dashboard wrapper ---


def run_dashboard(port=8050, fake_mode=False):
"""Run the dashboard. If fake_mode=True, try to inject demo data."""
try:
from demo_data import generate_price_series, generate_signals
data = generate_price_series()
try:
# if the dashboard uses a global `dashboard_data` dict, try to set it
if 'dashboard_data' in globals():
globals()['dashboard_data']['prices'] = data
except Exception:
pass
except Exception:
pass


# If the module defines `app` (Dash app), run it. Otherwise, fallback to running the script.
try:
# Most Dash apps expose `app` as the Dash instance
if 'app' in globals():
print(f"Running dash app on port {port}")
app.run_server(host='0.0.0.0', port=port, debug=False)
return
except Exception as e:
print("Error running Dash app directly:", e)


# fallback: execute the file as a script (so existing behavior remains)
import subprocess
subprocess.check_call(["python", "dashboard_server.py"])




if __name__ == "__main__":
run_dashboard()