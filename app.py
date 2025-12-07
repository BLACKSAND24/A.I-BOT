from flask import Flask, render_template, jsonify
from trading import TradingEngine
from fake_data import get_fake_price, get_fake_signal

app = Flask(__name__)
engine = TradingEngine()

@app.route("/")
def home():
    price = None
    error = None

    try:
        price = engine.get_price()
    except Exception as e:
        price = get_fake_price()
        error = "API connection error - showing simulated market data"

    return render_template("index.html",
                           price=price,
                           error=error)

@app.route("/price")
def price():
    try:
        result = engine.get_price()
        return jsonify({"price": result})
    except:
        result = get_fake_price()
        return jsonify({"price": result, "demo": True})

@app.route("/signal")
def signal():
        try:
            signal = engine.get_signal()
        except:
            signal = get_fake_signal()

        return jsonify({"signal": signal})

@app.route("/order/<side>")
def order(side):
    if side not in ["buy", "sell"]:
        return jsonify({"error": "Invalid side"})
    
    try:
        order = engine.place_order(side)
        return jsonify({"status": "success", "order": order})
    except:
        return jsonify({"status": "demo", "msg": f"Simulated {side} order executed"})

if __name__ == "__main__":
    app.run(debug=True)
