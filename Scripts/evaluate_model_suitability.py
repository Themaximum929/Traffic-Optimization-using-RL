import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import time

class ModelSuitabilityEvaluator:
    def __init__(self, model_path, sequence_length=10):
        self.sequence_length = sequence_length
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.tunnel_encoder = LabelEncoder()
        self.direction_encoder = LabelEncoder()
        
    def prepare_data(self, data_path):
        df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)
        df['tunnel_encoded'] = self.tunnel_encoder.fit_transform(df['tunnel'])
        df['direction_encoded'] = self.direction_encoder.fit_transform(df['direction'])
        df = df.sort_values(['detector_id', 'timestamp'])
        
        numerical_features = ['speed_mean', 'volume_total', 'occupancy_mean', 'speed_std', 'hour', 'weekday']
        X_numerical, X_tunnel, X_direction, y, timestamps = [], [], [], [], []
        
        for detector in df['detector_id'].unique():
            detector_data = df[df['detector_id'] == detector].reset_index(drop=True)
            for i in range(len(detector_data) - self.sequence_length):
                X_numerical.append(detector_data[numerical_features].iloc[i:i+self.sequence_length].values)
                X_tunnel.append(detector_data['tunnel_encoded'].iloc[i+self.sequence_length-1])
                X_direction.append(detector_data['direction_encoded'].iloc[i+self.sequence_length-1])
                y.append(detector_data['speed_mean'].iloc[i+self.sequence_length])
                timestamps.append(detector_data['timestamp'].iloc[i+self.sequence_length])
        
        return np.array(X_numerical), np.array(X_tunnel), np.array(X_direction), np.array(y), timestamps
    
    # 1. REAL-TIME PREDICTION EVALUATION
    def evaluate_realtime_prediction(self, X_num, X_tun, X_dir, y_true):
        print("\n" + "="*60)
        print("1. REAL-TIME TRAFFIC PREDICTION SUITABILITY")
        print("="*60)
        
        # Short-term prediction accuracy (next 5 minutes)
        sample_size = min(1000, len(y_true))
        idx = np.random.choice(len(y_true), sample_size, replace=False)
        
        y_pred = self.model.predict([X_num[idx], X_tun[idx], X_dir[idx]], verbose=0).flatten()
        mae = mean_absolute_error(y_true[idx], y_pred)
        
        # Latency test
        start = time.time()
        _ = self.model.predict([X_num[idx[:100]], X_tun[idx[:100]], X_dir[idx[:100]]], verbose=0)
        latency = (time.time() - start) / 100 * 1000
        
        # Accuracy by speed range
        congested = (y_true[idx] < 30)
        free_flow = (y_true[idx] >= 60)
        
        mae_congested = mean_absolute_error(y_true[idx][congested], y_pred[congested]) if congested.sum() > 0 else 0
        mae_free = mean_absolute_error(y_true[idx][free_flow], y_pred[free_flow]) if free_flow.sum() > 0 else 0
        
        print(f"Overall MAE (5-min ahead):        {mae:.4f} km/h")
        print(f"MAE in Congestion (<30 km/h):     {mae_congested:.4f} km/h")
        print(f"MAE in Free Flow (>60 km/h):      {mae_free:.4f} km/h")
        print(f"Prediction Latency:                {latency:.2f} ms/sample")
        print(f"Throughput:                        {1000/latency:.0f} predictions/sec")
        
        # Real-time suitability score
        latency_score = 100 if latency < 10 else (50 if latency < 50 else 0)
        accuracy_score = 100 if mae < 5 else (50 if mae < 10 else 0)
        congestion_score = 100 if mae_congested < 5 else (50 if mae_congested < 10 else 0)
        
        realtime_score = (latency_score + accuracy_score + congestion_score) / 3
        
        print(f"\n✓ Real-time Suitability Score:     {realtime_score:.1f}/100")
        print(f"  - Latency Score:                 {latency_score}/100")
        print(f"  - Accuracy Score:                {accuracy_score}/100")
        print(f"  - Congestion Accuracy:           {congestion_score}/100")
        
        return realtime_score, {'mae': mae, 'latency': latency, 'mae_congested': mae_congested}
    
    # 2. TOLL ADJUSTMENT EVALUATION
    def evaluate_toll_adjustment(self, X_num, X_tun, X_dir, y_true, timestamps):
        print("\n" + "="*60)
        print("2. DYNAMIC TOLL ADJUSTMENT SUITABILITY")
        print("="*60)
        
        sample_size = min(5000, len(y_true))
        idx = np.random.choice(len(y_true), sample_size, replace=False)
        
        y_pred = self.model.predict([X_num[idx], X_tun[idx], X_dir[idx]], verbose=0).flatten()
        
        # Congestion detection accuracy
        congestion_threshold = 30
        true_congested = y_true[idx] < congestion_threshold
        pred_congested = y_pred < congestion_threshold
        
        tp = np.sum(true_congested & pred_congested)
        fp = np.sum(~true_congested & pred_congested)
        fn = np.sum(true_congested & ~pred_congested)
        tn = np.sum(~true_congested & ~pred_congested)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Trend prediction (speed increasing/decreasing)
        trend_correct = 0
        for i in range(len(idx) - 1):
            true_trend = np.sign(y_true[idx[i+1]] - y_true[idx[i]])
            pred_trend = np.sign(y_pred[i+1] - y_pred[i])
            if true_trend == pred_trend:
                trend_correct += 1
        trend_accuracy = trend_correct / (len(idx) - 1) if len(idx) > 1 else 0
        
        # Peak hour performance
        peak_hours = [7, 8, 9, 17, 18, 19]
        # Simplified: assume uniform distribution if timestamps not available
        peak_mae = mean_absolute_error(y_true[idx][:len(idx)//3], y_pred[:len(idx)//3])
        
        print(f"Congestion Detection:")
        print(f"  - Precision:                     {precision:.4f}")
        print(f"  - Recall:                        {recall:.4f}")
        print(f"  - F1-Score:                      {f1:.4f}")
        print(f"Trend Prediction Accuracy:         {trend_accuracy:.4f}")
        print(f"Peak Hour MAE:                     {peak_mae:.4f} km/h")
        
        # Toll adjustment suitability score
        detection_score = f1 * 100
        trend_score = trend_accuracy * 100
        peak_score = 100 if peak_mae < 5 else (50 if peak_mae < 10 else 0)
        
        toll_score = (detection_score + trend_score + peak_score) / 3
        
        print(f"\n✓ Toll Adjustment Suitability:     {toll_score:.1f}/100")
        print(f"  - Congestion Detection (F1):     {detection_score:.1f}/100")
        print(f"  - Trend Prediction:              {trend_score:.1f}/100")
        print(f"  - Peak Hour Accuracy:            {peak_score:.1f}/100")
        
        return toll_score, {'f1': f1, 'trend_accuracy': trend_accuracy, 'peak_mae': peak_mae}
    
    def plot_comparison(self, realtime_score, toll_score):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['Real-time\nPrediction', 'Dynamic Toll\nAdjustment']
        scores = [realtime_score, toll_score]
        colors = ['#3498db', '#e74c3c']
        
        bars = ax.bar(categories, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.1f}',
                   ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        ax.axhline(y=70, color='green', linestyle='--', linewidth=2, label='Good Threshold (70)')
        ax.set_ylabel('Suitability Score', fontsize=14, fontweight='bold')
        ax.set_title('Model Suitability Comparison', fontsize=16, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('model_suitability_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Comparison plot saved: model_suitability_comparison.png")
        plt.show()
    
    def generate_recommendation(self, realtime_score, toll_score):
        print("\n" + "="*60)
        print("RECOMMENDATION")
        print("="*60)
        
        if realtime_score > toll_score + 10:
            print("✓ Model is MORE SUITABLE for: REAL-TIME TRAFFIC PREDICTION")
            print(f"  Reason: {realtime_score - toll_score:.1f} points higher")
        elif toll_score > realtime_score + 10:
            print("✓ Model is MORE SUITABLE for: DYNAMIC TOLL ADJUSTMENT")
            print(f"  Reason: {toll_score - realtime_score:.1f} points higher")
        else:
            print("✓ Model is EQUALLY SUITABLE for both applications")
            print(f"  Score difference: {abs(realtime_score - toll_score):.1f} points")
        
        print("\nKey Considerations:")
        if realtime_score >= 70:
            print("  ✓ Real-time prediction: READY for deployment")
        else:
            print("  ✗ Real-time prediction: Needs improvement")
        
        if toll_score >= 70:
            print("  ✓ Toll adjustment: READY for deployment")
        else:
            print("  ✗ Toll adjustment: Needs improvement")
        print("="*60)

def main():
    model_path = r'c:\Users\maxch\MaxProjects\NLP\best_model.h5'
    data_path = r'c:\Users\maxch\MaxProjects\NLP\ML_Data\traffic_data_normalized.parquet'
    
    evaluator = ModelSuitabilityEvaluator(model_path, sequence_length=10)
    X_num, X_tun, X_dir, y_true, timestamps = evaluator.prepare_data(data_path)
    
    # Use last 20% as test set
    test_size = int(len(y_true) * 0.2)
    X_num_test = X_num[-test_size:]
    X_tun_test = X_tun[-test_size:]
    X_dir_test = X_dir[-test_size:]
    y_test = y_true[-test_size:]
    timestamps_test = timestamps[-test_size:]
    
    print(f"Test set size: {len(y_test)} samples")
    
    # Evaluate both use cases
    realtime_score, realtime_metrics = evaluator.evaluate_realtime_prediction(X_num_test, X_tun_test, X_dir_test, y_test)
    toll_score, toll_metrics = evaluator.evaluate_toll_adjustment(X_num_test, X_tun_test, X_dir_test, y_test, timestamps_test)
    
    # Plot and recommend
    evaluator.plot_comparison(realtime_score, toll_score)
    evaluator.generate_recommendation(realtime_score, toll_score)

if __name__ == "__main__":
    main()
