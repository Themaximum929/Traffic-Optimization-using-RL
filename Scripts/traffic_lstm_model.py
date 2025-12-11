import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import datetime
import os

class TrafficLSTMModel:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.tunnel_encoder = LabelEncoder()
        self.direction_encoder = LabelEncoder()
        self.model = None
        
    def prepare_data(self, data_path):
        """Load and prepare data for training"""
        print("Loading data...")
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
        
        # Encode categorical features
        df['tunnel_encoded'] = self.tunnel_encoder.fit_transform(df['tunnel'])
        df['direction_encoded'] = self.direction_encoder.fit_transform(df['direction'])
        
        # Sort by detector and timestamp for sequence creation
        df = df.sort_values(['detector_id', 'timestamp'])
        
        # Numerical features
        numerical_features = ['speed_mean', 'volume_total', 'occupancy_mean', 'speed_std', 'hour', 'weekday']
        
        # Create sequences
        X_numerical, X_tunnel, X_direction, y = [], [], [], []
        
        for detector in df['detector_id'].unique():
            detector_data = df[df['detector_id'] == detector].reset_index(drop=True)
            
            for i in range(len(detector_data) - self.sequence_length):
                # Sequence of numerical features
                seq_numerical = detector_data[numerical_features].iloc[i:i+self.sequence_length].values
                # Categorical features (same for all timesteps in sequence)
                tunnel = detector_data['tunnel_encoded'].iloc[i+self.sequence_length-1]
                direction = detector_data['direction_encoded'].iloc[i+self.sequence_length-1]
                # Target (next speed_mean)
                target = detector_data['speed_mean'].iloc[i+self.sequence_length]
                
                X_numerical.append(seq_numerical)
                X_tunnel.append(tunnel)
                X_direction.append(direction)
                y.append(target)
        
        return np.array(X_numerical), np.array(X_tunnel), np.array(X_direction), np.array(y)
    
    def build_model(self, num_numerical_features, num_tunnels, num_directions):
        """Build the neural network model"""
        # Input layers
        numerical_input = Input(shape=(self.sequence_length, num_numerical_features), name='numerical_input')
        tunnel_input = Input(shape=(), name='tunnel_input')
        direction_input = Input(shape=(), name='direction_input')
        
        # Embedding layers with increased capacity
        tunnel_embedding = Embedding(num_tunnels, 8, name='tunnel_embedding')(tunnel_input)
        tunnel_embedding = tf.keras.layers.Flatten()(tunnel_embedding)
        
        direction_embedding = Embedding(num_directions, 4, name='direction_embedding')(direction_input)
        direction_embedding = tf.keras.layers.Flatten()(direction_embedding)
        
        # Enhanced LSTM layers with batch normalization
        lstm_out = LSTM(128, return_sequences=True, name='lstm_layer1')(numerical_input)
        lstm_out = tf.keras.layers.BatchNormalization()(lstm_out)
        lstm_out = Dropout(0.3)(lstm_out)
        
        lstm_out = LSTM(64, return_sequences=True, name='lstm_layer2')(lstm_out)
        lstm_out = tf.keras.layers.BatchNormalization()(lstm_out)
        lstm_out = Dropout(0.3)(lstm_out)
        
        lstm_out = LSTM(32, return_sequences=False, name='lstm_layer3')(lstm_out)
        lstm_out = tf.keras.layers.BatchNormalization()(lstm_out)
        
        # Concatenate all features
        concatenated = Concatenate(name='concatenate_features')([
            lstm_out, tunnel_embedding, direction_embedding
        ])
        
        # Deeper dense layers
        dense1 = Dense(64, activation='relu', name='dense1')(concatenated)
        dense1 = tf.keras.layers.BatchNormalization()(dense1)
        dropout1 = Dropout(0.3, name='dropout1')(dense1)
        
        dense2 = Dense(32, activation='relu', name='dense2')(dropout1)
        dense2 = tf.keras.layers.BatchNormalization()(dense2)
        dropout2 = Dropout(0.2, name='dropout2')(dense2)
        
        # Output layer
        output = Dense(1, activation='linear', name='output')(dropout2)
        
        # Create model
        model = Model(inputs=[numerical_input, tunnel_input, direction_input], outputs=output)
        
        return model
    
    def train(self, csv_path, epochs=50, batch_size=32, validation_split=0.2, use_tensorboard=True):
        """Train the model"""
        # Prepare data
        X_numerical, X_tunnel, X_direction, y = self.prepare_data(csv_path)
        
        print(f"Data shape: {X_numerical.shape}, {X_tunnel.shape}, {X_direction.shape}, {y.shape}")
        
        # Split data
        indices = np.arange(len(X_numerical))
        train_idx, val_idx = train_test_split(indices, test_size=validation_split, random_state=42)
        
        X_num_train, X_num_val = X_numerical[train_idx], X_numerical[val_idx]
        X_tun_train, X_tun_val = X_tunnel[train_idx], X_tunnel[val_idx]
        X_dir_train, X_dir_val = X_direction[train_idx], X_direction[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Build model
        num_numerical_features = X_numerical.shape[2]
        num_tunnels = len(self.tunnel_encoder.classes_)
        num_directions = len(self.direction_encoder.classes_)
        
        self.model = self.build_model(num_numerical_features, num_tunnels, num_directions)
        
        # Compile model with adjusted learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.002),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        print(self.model.summary())
        
        # Setup callbacks with adjusted patience
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1, min_delta=0.0001),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-7, verbose=1),
            ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
        ]
        if use_tensorboard:
            log_dir = f"logs/traffic_lstm_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            os.makedirs(log_dir, exist_ok=True)
            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
            callbacks.append(tensorboard_callback)
            print(f"TensorBoard logs: {log_dir}")
        
        # Train model
        history = self.model.fit(
            [X_num_train, X_tun_train, X_dir_train], y_train,
            validation_data=([X_num_val, X_tun_val, X_dir_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        val_pred = self.model.predict([X_num_val, X_tun_val, X_dir_val])
        mse = mean_squared_error(y_val, val_pred)
        mae = mean_absolute_error(y_val, val_pred)
        
        print(f"\nValidation Results:")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        return history
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Initialize model with longer sequences
    model = TrafficLSTMModel(sequence_length=10)
    
    # Train model with TensorBoard and adjusted batch size
    data_path = os.path.join(base_dir, 'ML_Data', 'traffic_data_normalized.parquet')
    history = model.train(data_path, epochs=150, batch_size=128, use_tensorboard=True)
    
    # Plot results
    model.plot_training_history(history)
    
    # Save model
    model.save_model(os.path.join(base_dir, 'Models', 'traffic_lstm_model.h5'))
    
    print("\nTo view TensorBoard: tensorboard --logdir=logs")

if __name__ == "__main__":
    main()