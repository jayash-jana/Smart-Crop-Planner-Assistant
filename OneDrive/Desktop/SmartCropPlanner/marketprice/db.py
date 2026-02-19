import sqlite3
from config import DB_PATH

def get_db():
    """Return a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # allows dict-like row access
    return conn

def init_db():
    """Create tables if they don't exist."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_prices (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            commodity   TEXT    NOT NULL,
            market      TEXT,
            state       TEXT,
            min_price   REAL,
            max_price   REAL,
            modal_price REAL,
            unit        TEXT    DEFAULT 'per Quintal',
            source      TEXT,
            fetched_at  TEXT,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_commodity_date
        ON market_prices (commodity, created_at)
    """)

    conn.commit()
    conn.close()
    print("[DB] Database initialized.")
