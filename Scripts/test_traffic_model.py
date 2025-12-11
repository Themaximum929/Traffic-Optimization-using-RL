import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

class TrafficModelTester:
    def __init__(self, model_path, sequence_length=10):
        self.sequence_length = sequence_length
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.tunnel_encoder = LabelEncoder()
        self.direction_encoder = LabelEncoder()
        
    def prepare_test_data(self, data_path):
        print("Loading test data...")
        df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)
        
        df['tunnel_encoded'] = self.tunnel_encoder.fit_transform(df['tunnel'])
        df['direction_encoded'] = self.direction_encoder.fit_transform(df['direction'])
        df = df.sort_values(['detector_id', 'timestamp'])
        
        numerical_features = ['speed_mean', 'volume_total', 'occupancy_mean', 'speed_std', 'hour', 'weekday']
        X_numerical, X_tunnel, X_direction, y = [], [], [], []
        
        for detector in df['detector_id'].unique():
            detector_data = df[df['detector_id'] == detector].reset_index(drop=True)
            for i in range(len(detector_data) - self.sequence_length):
                X_numerical.append(detector_data[numerical_features].iloc[i:i+self.sequence_length].values)
                X_tunnel.append(detector_data['tunnel_encoded'].iloc[i+self.sequence_length-1])
                X_direction.append(detector_data['direction_encoded'].iloc[i+self.sequence_length-1])
                y.append(detector_data['speed_mean'].iloc[i+self.sequence_length])
        
        return np.array(X_numerical), np.array(X_tunnel), np.array(X_direction), np.array(y)
    
    def evaluate(self, X_numerical, X_tunnel, X_direction, y_true):
        print("Making predictions...")
        y_pred = self.model.predict([X_numerical, X_tunnel, X_direction], batch_size=256, verbose=1).flatten()
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\nMSE:  {mse:.4f}")
        print(f"RMSE: {np.sqrt(mse):.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")
        
        return y_pred
    
    def plot_results(self, y_true, y_pred):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Scatter
        idx = np.random.choice(len(y_true), min(1000, len(y_true)), replace=False)
        axes[0, 0].scatter(y_true[idx], y_pred[idx], alpha=0.5, s=10)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Speed (km/h)')
        axes[0, 0].set_ylabel('Predicted Speed (km/h)')
        axes[0, 0].set_title('Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_true[idx] - y_pred[idx]
        axes[0, 1].scatter(y_pred[idx], residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Speed (km/h)')
        axes[0, 1].set_ylabel('Residuals (km/h)')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error distribution
        errors = np.abs(y_true - y_pred)
        axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Absolute Error (km/h)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Error Distribution (Mean: {np.mean(errors):.2f})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Time series
        axes[1, 1].plot(y_true[:200], label='Actual', linewidth=2)
        axes[1, 1].plot(y_pred[:200], label='Predicted', linewidth=2, alpha=0.7)
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Speed (km/h)')
        axes[1, 1].set_title('Time Series (First 200 samples)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        print("\n✓ Plot saved: model_evaluation.png")
        plt.show()

def main():
    model_path = r'c:\Users\maxch\MaxProjects\NLP\best_model.h5'
    data_path = r'c:\Users\maxch\MaxProjects\NLP\ML_Data\traffic_data_normalized.parquet'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    tester = TrafficModelTester(model_path, sequence_length=10)
    X_numerical, X_tunnel, X_direction, y_true = tester.prepare_test_data(data_path)
    
    # Use last 20% as test set
    test_size = int(len(y_true) * 0.2)
    X_num_test = X_numerical[-test_size:]
    X_tun_test = X_tunnel[-test_size:]
    X_dir_test = X_direction[-test_size:]
    y_test = y_true[-test_size:]
    
    print(f"Test set size: {len(y_test)} samples")
    
    y_pred = tester.evaluate(X_num_test, X_tun_test, X_dir_test, y_test)
    tester.plot_results(y_test, y_pred)

if __name__ == "__main__":
    main()
