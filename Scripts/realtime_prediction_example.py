import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os

class RealtimeTrafficPredictor:
    def __init__(self, model_path, sequence_length=10, speed_mean=72.74, speed_std=19.04):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.sequence_length = sequence_length
        self.tunnel_encoder = LabelEncoder()
        self.direction_encoder = LabelEncoder()
        self.speed_mean = speed_mean
        self.speed_std = speed_std
        
    def fit_encoders(self, historical_data_path):
        """Fit encoders on historical data"""
        df = pd.read_parquet(historical_data_path) if historical_data_path.endswith('.parquet') else pd.read_csv(historical_data_path)
        self.tunnel_encoder.fit(df['tunnel'])
        self.direction_encoder.fit(df['direction'])
        
    def predict_next_speed(self, recent_data, tunnel, direction):
        """
        Predict speed for next 5 minutes
        
        Args:
            recent_data: DataFrame with last 10 observations (50 minutes)
            tunnel: str, tunnel name
            direction: str, direction
            
        Returns:
            predicted_speed: float
        """
        # Prepare features
        features = ['speed_mean', 'volume_total', 'occupancy_mean', 'speed_std', 'hour', 'weekday']
        X_num = recent_data[features].values.reshape(1, self.sequence_length, -1)
        X_tunnel = np.array([self.tunnel_encoder.transform([tunnel])[0]])
        X_direction = np.array([self.direction_encoder.transform([direction])[0]])
        
        # Predict
        predicted_speed = self.model.predict([X_num, X_tunnel, X_direction], verbose=0)[0][0]
        return predicted_speed
    
    def denormalize_speed(self, normalized_speed):
        """Convert normalized speed back to km/h"""
        return normalized_speed * self.speed_std + self.speed_mean


# Example usage
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Initialize predictor
    predictor = RealtimeTrafficPredictor(
        model_path=os.path.join(base_dir, 'best_model.h5'),
        sequence_length=10
    )
    
    # Fit encoders on historical data
    predictor.fit_encoders(os.path.join(base_dir, 'ML_Data', 'traffic_data_normalized.parquet'))
    
    # Test multiple scenarios
    scenarios = [
        {
            'name': 'Scenario 1: Building Congestion (Morning Rush)',
            'data': pd.DataFrame({
                'speed_mean': [0.5, 0.4, 0.3, 0.2, 0.0, -0.2, -0.5, -0.8, -1.2, -1.5],
                'volume_total': [-0.5, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                'occupancy_mean': [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                'speed_std': [-0.5, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
                'hour': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                'weekday': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            })
        },
        {
            'name': 'Scenario 2: Free Flow (Late Night)',
            'data': pd.DataFrame({
                'speed_mean': [0.6, 0.6, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8],
                'volume_total': [-0.8, -0.8, -0.9, -0.9, -0.9, -1.0, -1.0, -1.0, -1.0, -1.0],
                'occupancy_mean': [-0.7, -0.7, -0.8, -0.8, -0.8, -0.9, -0.9, -0.9, -0.9, -0.9],
                'speed_std': [-0.8, -0.8, -0.9, -0.9, -0.9, -1.0, -1.0, -1.0, -1.0, -1.0],
                'hour': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                'weekday': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            })
        },
        {
            'name': 'Scenario 3: Clearing Congestion',
            'data': pd.DataFrame({
                'speed_mean': [-1.5, -1.3, -1.0, -0.8, -0.5, -0.3, 0.0, 0.2, 0.4, 0.5],
                'volume_total': [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0, -0.1, -0.2, -0.3],
                'occupancy_mean': [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0, -0.1, -0.2, -0.3],
                'speed_std': [1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.5],
                'hour': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                'weekday': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            })
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(scenario['name'])
        print('='*60)
        
        predicted_speed = predictor.predict_next_speed(
            recent_data=scenario['data'],
            tunnel='CHT',
            direction='Northbound'
        )
        
        actual_speed = predictor.denormalize_speed(predicted_speed)
        
        print(f"Predicted speed (normalized): {predicted_speed:.4f}")
        print(f"Predicted speed (actual):     {actual_speed:.2f} km/h")
        
        if actual_speed < 30:
            print("⚠️  Severe congestion expected")
        elif actual_speed < 50:
            print("⚠️  Moderate congestion expected")
        else:
            print("✓ Free flow expected")
