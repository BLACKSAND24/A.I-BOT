import logging
import requests
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

log = logging.getLogger(__name__)

# disable flag to avoid repeated failing HTTP calls
_TELEGRAM_DISABLED = False

def send_telegram_message(message: str) -> bool:
    global _TELEGRAM_DISABLED
    if _TELEGRAM_DISABLED:
        log.debug("Telegram disabled due to previous failures.")
        return False

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.debug("Telegram not configured.")
        return False

    # basic token sanity check (avoid leaking real token in logs)
    if TELEGRAM_TOKEN.startswith("YOUR_") or " " in TELEGRAM_TOKEN:
        log.debug("Telegram token looks invalid; skipping send.")
        _TELEGRAM_DISABLED = True
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
        r.raise_for_status()
        return True
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        log.error("Telegram send failed: %s (status=%s)", e, status)
        # disable further attempts on common client errors
        if status in (400, 401, 403, 404):
            _TELEGRAM_DISABLED = True
        return False
    except Exception as e:
        log.error("Telegram send failed: %s", e)
        return False
