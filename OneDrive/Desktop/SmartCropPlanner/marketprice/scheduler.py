import threading
import time
from config import FETCH_INTERVAL_MINUTES
from fetch_prices import fetch_market_prices
from models import insert_prices

_thread = None
_running = False

def _job():
    global _running
    print(f"[SCHEDULER] Started — fetching every {FETCH_INTERVAL_MINUTES} minutes.")
    while _running:
        try:
            prices = fetch_market_prices()
            if prices:
                insert_prices(prices)
                print(f"[SCHEDULER] Auto-fetched and stored {len(prices)} records.")
        except Exception as e:
            print(f"[SCHEDULER] Error: {e}")
        time.sleep(FETCH_INTERVAL_MINUTES * 60)

def start():
    global _thread, _running
    if _running:
        print("[SCHEDULER] Already running.")
        return
    _running = True
    _thread = threading.Thread(target=_job, daemon=True)
    _thread.start()

def stop():
    global _running
    _running = False
    print("[SCHEDULER] Stopped.")
