from flask import Flask, request, jsonify
from flask_cors import CORS
import pymysql
from config import DB_CONFIG, PORT
from model import model
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)

def get_db_connection():
    """Get database connection"""
    return pymysql.connect(**DB_CONFIG)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'OK',
        'message': 'ML Service is running',
        'model_trained': model.is_trained
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict demand for bakery items"""
    try:
        data = request.json
        item_id = data.get('item_id')
        prediction_date = data.get('prediction_date', datetime.now().strftime('%Y-%m-%d'))
        historical_data = data.get('historical_data', [])
        
        # If no historical data provided, fetch from database
        if not historical_data and item_id:
            conn = get_db_connection()
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            cursor.execute(
                'SELECT * FROM sales WHERE item_id = %s ORDER BY sale_date DESC LIMIT 30',
                (item_id,)
            )
            historical_data = cursor.fetchall()
            conn.close()
            
            # Convert to list of dicts
            historical_data = [dict(row) for row in historical_data]
        
        # Make prediction
        prediction = model.predict(prediction_date, item_id, historical_data)
        
        return jsonify(prediction)
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            'error': str(e),
            'predicted_quantity': 50,
            'confidence_score': 0.50
        }), 500

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    """Train or retrain the model"""
    try:
        item_id = request.args.get('item_id') or (request.json.get('item_id') if request.is_json else None)
        
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        if item_id:
            # Train for specific item
            cursor.execute(
                'SELECT * FROM sales WHERE item_id = %s ORDER BY sale_date',
                (item_id,)
            )
        else:
            # Train on all data
            cursor.execute('SELECT * FROM sales ORDER BY sale_date')
        
        historical_data = cursor.fetchall()
        conn.close()
        
        if len(historical_data) < 10:
            return jsonify({
                'error': 'Insufficient data for training. Need at least 10 records.',
                'records_found': len(historical_data)
            }), 400
        
        # Convert to list of dicts
        historical_data = [dict(row) for row in historical_data]
        
        # Train model
        model.train(historical_data)
        
        return jsonify({
            'message': 'Model trained successfully',
            'records_used': len(historical_data),
            'model_trained': model.is_trained
        })
    
    except Exception as e:
        print(f"Training error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict demand for multiple items"""
    try:
        data = request.json
        items = data.get('items', [])
        prediction_date = data.get('prediction_date', datetime.now().strftime('%Y-%m-%d'))
        
        predictions = []
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        for item_id in items:
            cursor.execute(
                'SELECT * FROM sales WHERE item_id = %s ORDER BY sale_date DESC LIMIT 30',
                (item_id,)
            )
            historical_data = cursor.fetchall()
            historical_data = [dict(row) for row in historical_data]
            
            prediction = model.predict(prediction_date, item_id, historical_data)
            predictions.append({
                'item_id': item_id,
                **prediction
            })
        
        conn.close()
        return jsonify(predictions)
    
    except Exception as e:
        print(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"Starting ML Service on port {PORT}...")
    print("Training initial model...")
    
    # Try to train model on startup
    try:
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute('SELECT * FROM sales ORDER BY sale_date LIMIT 100')
        historical_data = cursor.fetchall()
        conn.close()
        
        if len(historical_data) >= 10:
            historical_data = [dict(row) for row in historical_data]
            model.train(historical_data)
            print("Model trained successfully!")
        else:
            print("Insufficient data for training. Model will use default predictions.")
    except Exception as e:
        print(f"Could not train model on startup: {str(e)}")
        print("Model will use default predictions until trained.")
    
    app.run(host='0.0.0.0', port=PORT, debug=True)
