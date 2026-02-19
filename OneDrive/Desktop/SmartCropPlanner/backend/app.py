from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pulp import *
import joblib
import os
import sys

app = Flask(__name__)
CORS(app)

# ============================================
# PATH CONFIGURATION (No emojis)
# ============================================

# Get the backend directory
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to project root
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# Set correct paths
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'Crop_recommendation.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'crop_model.pkl')

print("="*50)
print("SMARTCROP PLANNER - Path Configuration")
print("="*50)
print(f"Project Root: {PROJECT_ROOT}")
print(f"Backend Dir: {BACKEND_DIR}")
print(f"Data Path: {DATA_PATH}")
print(f"Data file exists: {os.path.exists(DATA_PATH)}")
print("="*50)

def train_model():
    """Train Random Forest model on crop dataset"""
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print("\nERROR: Crop_recommendation.csv not found!")
        print(f"Expected at: {DATA_PATH}")
        print("\nQuick Fix:")
        print("1. Download dataset from Kaggle:")
        print("   https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
        print("2. Place the CSV file in: data/Crop_recommendation.csv")
        print("\nUsing sample data for now...")
        return create_sample_model()
    
    try:
        # Load dataset
        print(f"\nLoading dataset from: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        print(f"Dataset loaded! Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Prepare features and target
        feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        target_col = 'label'
        
        # Check if columns exist
        available_cols = df.columns.tolist()
        
        # Select features
        X = df[feature_cols]
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest
        print("Training Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        accuracy = model.score(X_test, y_test)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        
        # Save model
        joblib.dump(model, MODEL_PATH)
        print(f"Model saved to: {MODEL_PATH}")
        
        return model
        
    except Exception as e:
        print(f"Error training model: {e}")
        print("\nFalling back to sample model...")
        return create_sample_model()

def create_sample_model():
    """Create a sample model for demonstration"""
    from sklearn.datasets import make_classification
    
    print("\nCreating sample model for demonstration...")
    
    # Create synthetic dataset
    X, y = make_classification(
        n_samples=1000, 
        n_features=7,
        n_classes=8,
        n_informative=5,
        random_state=42
    )
    
    # Sample crop names
    crop_names = ['Rice', 'Maize', 'Groundnut', 'Cotton', 'Wheat', 
                  'Pulses', 'Sugarcane', 'Tomato']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    model.classes_ = np.array(crop_names)
    
    print("Sample model created")
    print("Note: This is demo data - not real crop recommendations")
    
    return model

# Load or train model
if os.path.exists(MODEL_PATH):
    try:
        print(f"\nLoading model from: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        print("Model loaded from disk")
    except Exception as e:
        print(f"Could not load model: {e}")
        model = train_model()
else:
    model = train_model()

# Crop database with profit data
CROP_DATABASE = {
    'Rice': {'profit_per_acre': 45000, 'cost_per_acre': 15000, 'water_need': 1200,
             'risk_factors': {'price_volatility': 0.3, 'rainfall_sensitivity': 0.4}},
    'Maize': {'profit_per_acre': 35000, 'cost_per_acre': 12000, 'water_need': 600,
              'risk_factors': {'price_volatility': 0.4, 'rainfall_sensitivity': 0.3}},
    'Groundnut': {'profit_per_acre': 40000, 'cost_per_acre': 13000, 'water_need': 500,
                  'risk_factors': {'price_volatility': 0.5, 'rainfall_sensitivity': 0.5}},
    'Cotton': {'profit_per_acre': 55000, 'cost_per_acre': 20000, 'water_need': 800,
               'risk_factors': {'price_volatility': 0.6, 'rainfall_sensitivity': 0.3}},
    'Wheat': {'profit_per_acre': 38000, 'cost_per_acre': 14000, 'water_need': 450,
              'risk_factors': {'price_volatility': 0.2, 'rainfall_sensitivity': 0.2}},
    'Pulses': {'profit_per_acre': 42000, 'cost_per_acre': 11000, 'water_need': 350,
               'risk_factors': {'price_volatility': 0.4, 'rainfall_sensitivity': 0.3}},
    'Sugarcane': {'profit_per_acre': 65000, 'cost_per_acre': 25000, 'water_need': 1500,
                  'risk_factors': {'price_volatility': 0.3, 'rainfall_sensitivity': 0.5}},
    'Tomato': {'profit_per_acre': 80000, 'cost_per_acre': 30000, 'water_need': 600,
               'risk_factors': {'price_volatility': 0.7, 'rainfall_sensitivity': 0.6}}
}

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/predict', methods=['POST'])
def predict_crops():
    try:
        data = request.json
        features = [[
            float(data['nitrogen']), 
            float(data['phosphorus']), 
            float(data['potassium']), 
            float(data['temperature']),
            float(data['humidity']), 
            float(data['ph']), 
            float(data['rainfall'])
        ]]
        
        probabilities = model.predict_proba(features)[0]
        crop_classes = model.classes_
        
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_crops = [str(crop_classes[i]) for i in top_indices]
        top_probs = [float(probabilities[i]) for i in top_indices]
        
        return jsonify({
            'recommended_crops': top_crops,
            'probabilities': top_probs,
            'message': 'ML prediction successful'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/optimize', methods=['POST'])
def optimize_allocation():
    try:
        data = request.json
        crops = data['crops']
        total_land = float(data['total_land'])
        budget = float(data['budget'])
        
        prob = LpProblem("Crop_Allocation", LpMaximize)
        areas = LpVariable.dicts("Area", crops, 0, total_land)
        
        # Objective: Maximize profit
        prob += lpSum([areas[c] * CROP_DATABASE.get(c, CROP_DATABASE['Rice'])['profit_per_acre'] for c in crops])
        
        # Constraints
        prob += lpSum([areas[c] for c in crops]) <= total_land
        prob += lpSum([areas[c] * CROP_DATABASE.get(c, CROP_DATABASE['Rice'])['cost_per_acre'] for c in crops]) <= budget
        
        prob.solve()
        
        allocation = {}
        total_profit = 0
        for c in crops:
            area = areas[c].varValue
            if area and area > 0.01:
                allocation[c] = round(area, 2)
                total_profit += area * CROP_DATABASE.get(c, CROP_DATABASE['Rice'])['profit_per_acre']
        
        return jsonify({
            'allocation': allocation,
            'total_profit': round(total_profit, 2),
            'status': 'Optimal'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/risk-analysis', methods=['POST'])
def analyze_risk():
    try:
        data = request.json
        allocation = data['allocation']
        total_land = float(data['total_land'])
        
        # Simple risk analysis
        if len(allocation) == 1:
            warning = "HIGH MONOCROP RISK DETECTED - Single crop covers entire land"
            risk_level = "High"
            risk_score = 0.8
        elif max(allocation.values()) / total_land > 0.7:
            warning = "MEDIUM MONOCROP RISK - Consider diversifying"
            risk_level = "Medium"
            risk_score = 0.5
        else:
            warning = "GOOD DIVERSIFICATION - Risk well distributed"
            risk_level = "Low"
            risk_score = 0.2
        
        return jsonify({
            'risk_score': risk_score,
            'risk_level': risk_level,
            'warning': warning
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare_profits():
    try:
        data = request.json
        allocation = data['allocation']
        total_land = float(data['total_land'])
        
        # Calculate multicrop profit
        multicrop_profit = 0
        for c, area in allocation.items():
            profit_per_acre = CROP_DATABASE.get(c, CROP_DATABASE['Rice'])['profit_per_acre']
            multicrop_profit += area * profit_per_acre
        
        # Simulate best monocrop (for demo)
        best_monocrop_profit = total_land * 45000  # Rice as example
        
        improvement = ((multicrop_profit - best_monocrop_profit) / best_monocrop_profit) * 100
        
        return jsonify({
            'multicrop_profit': round(multicrop_profit, 2),
            'best_monocrop': {'crop': 'Rice', 'profit': round(best_monocrop_profit, 2)},
            'improvement': round(improvement, 1),
            'message': f'Your diversified plan gives {improvement:.1f}% more profit than monocrop'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        'status': 'Server is running!',
        'project_root': PROJECT_ROOT,
        'data_path': DATA_PATH,
        'data_exists': os.path.exists(DATA_PATH),
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("SMARTCROP PLANNER API SERVER")
    print("="*50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data file: {DATA_PATH}")
    print(f"Data exists: {os.path.exists(DATA_PATH)}")
    print(f"Model: {MODEL_PATH}")
    print("="*50)
    print("\nServer running on http://localhost:5000")
    print("Test endpoint: http://localhost:5000/test")
    print("="*50)
    
    app.run(debug=True, port=5000)