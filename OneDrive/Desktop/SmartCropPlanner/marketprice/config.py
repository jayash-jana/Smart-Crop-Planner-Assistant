# config.py — API keys and constants

# ─── Data API ─────────────────────────────────────────────────────────────────
# Option 1: data.gov.in (India Agri Market Prices — free, no key needed)
AGMARKNET_BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
AGMARKNET_API_KEY  = "579b464db66ec23bdd000001cdd3946e44ce4aab7a29f0d40bdb9f7e"  # demo key

# Option 2: Open Food Facts or any commodity price API
# COMMODITY_API_URL = "https://your-api-endpoint.com"
# COMMODITY_API_KEY = "your-api-key-here"

# ─── Database ─────────────────────────────────────────────────────────────────
DB_PATH = "market.db"

# ─── Scheduler ────────────────────────────────────────────────────────────────
FETCH_INTERVAL_MINUTES = 30   # auto-fetch interval

# ─── Commodities to track ─────────────────────────────────────────────────────
COMMODITIES = [
    "Tomato", "Onion", "Potato", "Rice", "Wheat",
    "Maize", "Garlic", "Ginger", "Chilli", "Brinjal"
]

# ─── States / Markets ─────────────────────────────────────────────────────────
DEFAULT_STATE  = "Tamil Nadu"
DEFAULT_MARKET = "Chennai"
