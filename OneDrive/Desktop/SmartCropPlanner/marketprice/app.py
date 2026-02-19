from flask import Flask, render_template, jsonify, request
from db import init_db, get_db
from fetch_prices import fetch_market_prices
from models import insert_prices, get_latest_prices, get_price_history, get_price_comparison
import scheduler

app = Flask(__name__)

# Initialize DB on startup
init_db()

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/prices/fetch', methods=['POST'])
def fetch_and_store():
    """Fetch real-time prices from API and store in DB"""
    prices = fetch_market_prices()
    if prices:
        insert_prices(prices)
        return jsonify({"status": "success", "count": len(prices), "data": prices})
    return jsonify({"status": "error", "message": "Failed to fetch prices"}), 500

@app.route('/api/prices/latest', methods=['GET'])
def latest_prices():
    """Get the most recent prices from DB"""
    prices = get_latest_prices()
    return jsonify({"status": "success", "data": prices})

@app.route('/api/prices/history', methods=['GET'])
def price_history():
    """Get historical prices for a commodity"""
    commodity = request.args.get('commodity', 'Tomato')
    limit = int(request.args.get('limit', 30))
    history = get_price_history(commodity, limit)
    return jsonify({"status": "success", "commodity": commodity, "data": history})

@app.route('/api/prices/compare', methods=['GET'])
def compare_prices():
    """Compare today's prices with yesterday's prices"""
    comparison = get_price_comparison()
    return jsonify({"status": "success", "data": comparison})

@app.route('/api/scheduler/start', methods=['POST'])
def start_scheduler():
    scheduler.start()
    return jsonify({"status": "success", "message": "Scheduler started - fetching every 30 minutes"})

@app.route('/api/scheduler/stop', methods=['POST'])
def stop_scheduler():
    scheduler.stop()
    return jsonify({"status": "success", "message": "Scheduler stopped"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
