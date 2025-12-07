# Binance Trading Bot — Demo (Finals Project)

## What this delivers
- Flask web dashboard (demo)
- Trading engine using CCXT — supports Binance testnet via sandbox mode
- Simple LSTM + XGBoost model skeletons for price prediction
- Strategy using RSI, MACD, Bollinger bands
- Paper-trading mode

## Quick start (local)
1. Create `.env` from `.env.example` and fill keys (for demo, use Binance testnet keys or leave blank for paper mode).
2. Python environment:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Quick start (demo mode)


1. Create and activate virtualenv:
- linux/mac:
```
python3 -m venv venv
source venv/bin/activate
```
- windows:
```
python -m venv venv
venv\Scripts\activate
```


2. Install dependencies:


3. Edit `.env` (optional):
- To run in demo/fake mode (no API keys needed): set `FAKE_MODE=true`
- To run live: set `FAKE_MODE=false` and add exchange keys (BINANCE_KEY, BINANCE_SECRET) and set `USE_TESTNET=true` if you want testnet.


4. Run locally (both API + dashboard):

- API will run on http://0.0.0.0:5000
- Dashboard will run on http://0.0.0.0:8050 (if present)


5. To deploy to Render:
- Push to GitHub
- Sign into https://render.com and create a Web Service connected to your repo
- Render will detect Dockerfile and build image; set env vars (FAKE_MODE=true for demo)
- Deploy and copy the public URL for your demo