from db import get_db
from datetime import datetime, timedelta

def insert_prices(prices: list):
    """Insert a list of price dicts into the DB."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.executemany("""
        INSERT INTO market_prices
            (commodity, market, state, min_price, max_price, modal_price, unit, source, fetched_at)
        VALUES
            (:commodity, :market, :state, :min_price, :max_price, :modal_price, :unit, :source, :fetched_at)
    """, prices)

    conn.commit()
    conn.close()
    print(f"[DB] Inserted {len(prices)} price records.")


def get_latest_prices():
    """
    Return the most recent price for each commodity
    (latest row per commodity ordered by created_at).
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT p.*
        FROM market_prices p
        INNER JOIN (
            SELECT commodity, MAX(created_at) AS max_date
            FROM market_prices
            GROUP BY commodity
        ) latest ON p.commodity = latest.commodity
                  AND p.created_at = latest.max_date
        ORDER BY p.commodity
    """)

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_price_history(commodity: str, limit: int = 30):
    """Return the last N price records for a given commodity."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT commodity, market, modal_price, min_price, max_price, unit, source, created_at
        FROM market_prices
        WHERE commodity = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (commodity, limit))

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows[::-1]   # chronological order


def get_price_comparison():
    """
    Compare today's modal prices with yesterday's for each commodity.
    Returns change amount and % change.
    """
    conn = get_db()
    cursor = conn.cursor()

    today     = datetime.now().date()
    yesterday = today - timedelta(days=1)

    def avg_for_date(date_str):
        cursor.execute("""
            SELECT commodity, AVG(modal_price) AS avg_price
            FROM market_prices
            WHERE DATE(created_at) = ?
            GROUP BY commodity
        """, (date_str,))
        return {row["commodity"]: row["avg_price"] for row in cursor.fetchall()}

    today_prices     = avg_for_date(str(today))
    yesterday_prices = avg_for_date(str(yesterday))

    comparison = []
    all_commodities = set(today_prices) | set(yesterday_prices)

    for commodity in sorted(all_commodities):
        today_val = today_prices.get(commodity)
        prev_val  = yesterday_prices.get(commodity)

        if today_val is not None and prev_val is not None:
            change    = today_val - prev_val
            pct       = (change / prev_val * 100) if prev_val else 0
        else:
            change = pct = 0

        comparison.append({
            "commodity":       commodity,
            "today_price":     round(today_val, 2) if today_val else None,
            "yesterday_price": round(prev_val, 2)  if prev_val  else None,
            "change":          round(change, 2),
            "pct_change":      round(pct, 2),
            "trend":           "up" if change > 0 else ("down" if change < 0 else "flat")
        })

    conn.close()
    return comparison
