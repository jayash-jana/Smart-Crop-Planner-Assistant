import requests
import json
import random
from datetime import datetime
from config import AGMARKNET_BASE_URL, AGMARKNET_API_KEY, COMMODITIES, DEFAULT_STATE, DEFAULT_MARKET

def fetch_market_prices():
    """
    Fetch real-time market prices from data.gov.in Agmarknet API.
    Falls back to simulated prices if API is unavailable.
    """
    prices = []

    try:
        params = {
            "api-key": AGMARKNET_API_KEY,
            "format": "json",
            "filters[State]": DEFAULT_STATE,
            "limit": 50,
            "offset": 0
        }
        response = requests.get(AGMARKNET_BASE_URL, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            records = data.get("records", [])

            for record in records:
                commodity = record.get("Commodity", "Unknown")
                if commodity in COMMODITIES or not COMMODITIES:
                    prices.append({
                        "commodity":  commodity,
                        "market":     record.get("Market", DEFAULT_MARKET),
                        "state":      record.get("State", DEFAULT_STATE),
                        "min_price":  float(record.get("Min Price", 0)),
                        "max_price":  float(record.get("Max Price", 0)),
                        "modal_price": float(record.get("Modal Price", 0)),
                        "unit":       "per Quintal",
                        "fetched_at": datetime.now().isoformat(),
                        "source":     "Agmarknet API"
                    })

            if prices:
                print(f"[API] Fetched {len(prices)} price records from Agmarknet.")
                return prices

    except Exception as e:
        print(f"[API] Real API failed: {e}. Using simulated data.")

    # ── Fallback: Simulated realistic prices ──────────────────────────────────
    return _simulate_prices()


def _simulate_prices():
    """Generate realistic simulated market prices for demo / offline use."""
    base_prices = {
        "Tomato":  (800,  2500),
        "Onion":   (600,  1800),
        "Potato":  (700,  1500),
        "Rice":    (2000, 4000),
        "Wheat":   (1800, 3000),
        "Maize":   (1200, 2200),
        "Garlic":  (3000, 8000),
        "Ginger":  (2500, 6000),
        "Chilli":  (4000, 12000),
        "Brinjal": (500,  1500),
    }

    markets = ["Chennai", "Coimbatore", "Madurai", "Salem", "Trichy"]
    prices  = []

    for commodity, (low, high) in base_prices.items():
        modal = random.randint(low, high)
        prices.append({
            "commodity":   commodity,
            "market":      random.choice(markets),
            "state":       DEFAULT_STATE,
            "min_price":   round(modal * 0.85),
            "max_price":   round(modal * 1.15),
            "modal_price": modal,
            "unit":        "per Quintal",
            "fetched_at":  datetime.now().isoformat(),
            "source":      "Simulated (Demo)"
        })

    print(f"[SIM] Generated {len(prices)} simulated price records.")
    return prices
