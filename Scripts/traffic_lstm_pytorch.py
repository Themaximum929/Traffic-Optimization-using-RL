import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import datetime
import os

class TrafficDataset(Dataset):
    def __init__(self, X_numerical, X_tunnel, X_direction, y):
        self.X_numerical = torch.FloatTensor(X_numerical)
        self.X_tunnel = torch.LongTensor(X_tunnel)
        self.X_direction = torch.LongTensor(X_direction)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X_numerical[idx], self.X_tunnel[idx], self.X_direction[idx], self.y[idx]

class TrafficLSTMModel(nn.Module):
    def __init__(self, num_numerical_features, num_tunnels, num_directions, sequence_length=10):
        super(TrafficLSTMModel, self).__init__()
        self.sequence_length = sequence_length
        
        # Enhanced embedding layers
        self.tunnel_embedding = nn.Embedding(num_tunnels, 8)
        self.direction_embedding = nn.Embedding(num_directions, 4)
        
        # Multi-layer LSTM with batch normalization
        self.lstm1 = nn.LSTM(num_numerical_features, 128, batch_first=True)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout_lstm1 = nn.Dropout(0.3)
        
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout_lstm2 = nn.Dropout(0.3)
        
        self.lstm3 = nn.LSTM(64, 32, batch_first=True)
        self.bn3 = nn.BatchNorm1d(32)
        
        # Deeper dense layers
        self.fc1 = nn.Linear(32 + 8 + 4, 64)  # LSTM output + embeddings
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(64, 32)
        self.bn_fc2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)
        
        self.output = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x_numerical, x_tunnel, x_direction):
        # Multi-layer LSTM processing
        lstm_out, _ = self.lstm1(x_numerical)
        lstm_out = self.bn1(lstm_out[:, -1, :])
        lstm_out = self.dropout_lstm1(lstm_out)
        lstm_out = lstm_out.unsqueeze(1).repeat(1, x_numerical.size(1), 1)
        
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.bn2(lstm_out[:, -1, :])
        lstm_out = self.dropout_lstm2(lstm_out)
        lstm_out = lstm_out.unsqueeze(1).repeat(1, x_numerical.size(1), 1)
        
        lstm_out, _ = self.lstm3(lstm_out)
        lstm_out = self.bn3(lstm_out[:, -1, :])
        
        # Embeddings
        tunnel_emb = self.tunnel_embedding(x_tunnel)
        direction_emb = self.direction_embedding(x_direction)
        
        # Concatenate features
        combined = torch.cat([lstm_out, tunnel_emb, direction_emb], dim=1)
        
        # Dense layers with batch normalization
        x = self.relu(self.fc1(combined))
        x = self.bn_fc1(x)
        x = self.dropout1(x)
        
        x = self.relu(self.fc2(x))
        x = self.bn_fc2(x)
        x = self.dropout2(x)
        
        x = self.output(x)
        
        return x

class TrafficLSTMTrainer:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.tunnel_encoder = LabelEncoder()
        self.direction_encoder = LabelEncoder()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
                seq_numerical = detector_data[numerical_features].iloc[i:i+self.sequence_length].values
                tunnel = detector_data['tunnel_encoded'].iloc[i+self.sequence_length-1]
                direction = detector_data['direction_encoded'].iloc[i+self.sequence_length-1]
                target = detector_data['speed_mean'].iloc[i+self.sequence_length]
                
                X_numerical.append(seq_numerical)
                X_tunnel.append(tunnel)
                X_direction.append(direction)
                y.append(target)
        
        return np.array(X_numerical), np.array(X_tunnel), np.array(X_direction), np.array(y)
    
    def train(self, csv_path, epochs=30, batch_size=64, validation_split=0.2, use_tensorboard=True):
        """Train the model"""
        # Prepare data
        X_numerical, X_tunnel, X_direction, y = self.prepare_data(csv_path)
        
        print(f"Data shape: {X_numerical.shape}, {X_tunnel.shape}, {X_direction.shape}, {y.shape}")
        
        # Split data
        indices = np.arange(len(X_numerical))
        train_idx, val_idx = train_test_split(indices, test_size=validation_split, random_state=42)
        
        # Create datasets
        train_dataset = TrafficDataset(X_numerical[train_idx], X_tunnel[train_idx], 
                                     X_direction[train_idx], y[train_idx])
        val_dataset = TrafficDataset(X_numerical[val_idx], X_tunnel[val_idx], 
                                   X_direction[val_idx], y[val_idx])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        num_numerical_features = X_numerical.shape[2]
        num_tunnels = len(self.tunnel_encoder.classes_)
        num_directions = len(self.direction_encoder.classes_)
        
        self.model = TrafficLSTMModel(num_numerical_features, num_tunnels, num_directions, self.sequence_length)
        self.model.to(self.device)
        
        # Loss and optimizer with adjusted learning rate
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, min_lr=1e-7, verbose=True)
        
        # TensorBoard
        writer = None
        if use_tensorboard:
            log_dir = f"logs/traffic_lstm_pytorch_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir)
            print(f"TensorBoard logs: {log_dir}")
        
        # Training loop
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 15
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_num, batch_tun, batch_dir, batch_y in train_loader:
                batch_num, batch_tun, batch_dir, batch_y = (
                    batch_num.to(self.device), batch_tun.to(self.device),
                    batch_dir.to(self.device), batch_y.to(self.device)
                )
                
                optimizer.zero_grad()
                outputs = self.model(batch_num, batch_tun, batch_dir)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_num, batch_tun, batch_dir, batch_y in val_loader:
                    batch_num, batch_tun, batch_dir, batch_y = (
                        batch_num.to(self.device), batch_tun.to(self.device),
                        batch_dir.to(self.device), batch_y.to(self.device)
                    )
                    
                    outputs = self.model(batch_num, batch_tun, batch_dir)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss - 0.0001:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model_pytorch.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    self.model.load_state_dict(torch.load('best_model_pytorch.pth'))
                    break
            
            # TensorBoard logging
            if writer:
                writer.add_scalar('Loss/Train', train_loss, epoch)
                writer.add_scalar('Loss/Validation', val_loss, epoch)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        if writer:
            writer.close()
        
        return {'train_loss': train_losses, 'val_loss': val_losses}
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'tunnel_encoder': self.tunnel_encoder,
                'direction_encoder': self.direction_encoder
            }, filepath)
            print(f"Model saved to {filepath}")
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 1, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Initialize trainer with longer sequences
    trainer = TrafficLSTMTrainer(sequence_length=10)
    
    # Train model with adjusted parameters
    data_path = os.path.join(base_dir, 'ML_Data', 'traffic_data_normalized.parquet')
    history = trainer.train(data_path, epochs=150, batch_size=128, use_tensorboard=True)
    
    # Plot results
    trainer.plot_training_history(history)
    
    # Save model
    trainer.save_model(os.path.join(base_dir, 'Models', 'traffic_lstm_pytorch.pth'))
    
    print("\nTo view TensorBoard: tensorboard --logdir=logs")

if __name__ == "__main__":
    main()