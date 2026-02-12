import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

class DemandPredictionModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.label_encoders = {
            'day_of_week': LabelEncoder(),
            'season': LabelEncoder(),
            'weather': LabelEncoder()
        }
        self.is_trained = False
        self.model_path = 'models/demand_model.pkl'

    def prepare_features(self, df):
        """Prepare features for training/prediction"""
        df = df.copy()
        
        # Convert date to datetime if it's a string
        if 'sale_date' in df.columns:
            df['sale_date'] = pd.to_datetime(df['sale_date'])
            df['day_of_week'] = df['sale_date'].dt.day_name()
            df['month'] = df['sale_date'].dt.month
            df['year'] = df['sale_date'].dt.year
            df['day_of_month'] = df['sale_date'].dt.day
        
        # Encode categorical variables
        if 'day_of_week' in df.columns:
            if self.is_trained:
                df['day_of_week_encoded'] = self.label_encoders['day_of_week'].transform(df['day_of_week'].fillna('Monday'))
            else:
                df['day_of_week_encoded'] = self.label_encoders['day_of_week'].fit_transform(df['day_of_week'].fillna('Monday'))
        
        if 'season' in df.columns:
            if self.is_trained:
                df['season_encoded'] = self.label_encoders['season'].transform(df['season'].fillna('Spring'))
            else:
                df['season_encoded'] = self.label_encoders['season'].fit_transform(df['season'].fillna('Spring'))
        
        if 'weather' in df.columns:
            if self.is_trained:
                df['weather_encoded'] = self.label_encoders['weather'].transform(df['weather'].fillna('Sunny'))
            else:
                df['weather_encoded'] = self.label_encoders['weather'].fit_transform(df['weather'].fillna('Sunny'))
        
        # Select features
        feature_cols = ['month', 'year', 'day_of_month', 'day_of_week_encoded', 
                       'season_encoded', 'weather_encoded', 'is_holiday']
        
        # Add missing columns with default values
        for col in feature_cols:
            if col not in df.columns:
                if col == 'is_holiday':
                    df[col] = 0
                else:
                    df[col] = 0
        
        return df[feature_cols]

    def train(self, historical_data):
        """Train the model on historical data"""
        if len(historical_data) < 10:
            raise ValueError("Need at least 10 data points to train the model")
        
        df = pd.DataFrame(historical_data)
        
        # Prepare features
        X = self.prepare_features(df)
        y = df['quantity'].values
        
        # Train model
        self.model.fit(X, y)
        self.is_trained = True
        
        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump({
            'model': self.model,
            'label_encoders': self.label_encoders
        }, self.model_path)
        
        return True

    def predict(self, prediction_date, item_id=None, historical_data=None):
        """Predict demand for a given date"""
        if not self.is_trained and historical_data and len(historical_data) >= 10:
            self.train(historical_data)
        elif not self.is_trained:
            # Load saved model if exists
            if os.path.exists(self.model_path):
                saved = joblib.load(self.model_path)
                self.model = saved['model']
                self.label_encoders = saved['label_encoders']
                self.is_trained = True
            else:
                # Return default prediction
                return {
                    'predicted_quantity': 50,
                    'confidence_score': 0.50,
                    'model_version': 'default'
                }
        
        # Prepare prediction features
        pred_date = pd.to_datetime(prediction_date)
        
        # Determine season
        month = pred_date.month
        if month in [12, 1, 2]:
            season = 'Winter'
        elif month in [3, 4, 5]:
            season = 'Spring'
        elif month in [6, 7, 8]:
            season = 'Summer'
        else:
            season = 'Fall'
        
        # Create feature row
        features = pd.DataFrame([{
            'sale_date': pred_date,
            'day_of_week': pred_date.strftime('%A'),
            'month': month,
            'year': pred_date.year,
            'day_of_month': pred_date.day,
            'season': season,
            'weather': 'Sunny',  # Default, could be enhanced with weather API
            'is_holiday': 0  # Default, could be enhanced with holiday API
        }])
        
        X_pred = self.prepare_features(features)
        
        # Make prediction
        prediction = self.model.predict(X_pred)[0]
        predicted_quantity = max(0, int(round(prediction)))
        
        # Calculate confidence (simplified - could be enhanced)
        confidence = 0.85 if self.is_trained else 0.60
        
        return {
            'predicted_quantity': predicted_quantity,
            'confidence_score': confidence,
            'model_version': 'v1.0'
        }

# Global model instance
model = DemandPredictionModel()
