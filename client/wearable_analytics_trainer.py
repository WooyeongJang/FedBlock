import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from typing import Dict, List, Tuple, Optional
import json
import os

from .config import GLOBAL_TMP_PATH, GLOBAL_DATASETS


class WearableTimeSeriesNet(nn.Module):
    """LSTM-based neural network for wearable device time series analysis."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, 
                 output_dim: int = 1, dropout_rate: float = 0.2, sequence_length: int = 24):
        super(WearableTimeSeriesNet, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout_rate)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, 
                                             dropout=dropout_rate, batch_first=True)
        
        # Output layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last time step for prediction
        last_output = attn_out[:, -1, :]
        
        # Final prediction
        output = self.fc_layers(last_output)
        
        return output


class WearableAnalyticsTrainer:
    """
    Federated learning trainer for wearable device analytics.
    Supports analysis of:
    - Heart rate variability
    - Sleep pattern analysis
    - Activity level prediction
    - Stress level monitoring
    - Calorie burn estimation
    - Step count validation
    """
    
    def __init__(self, model_params, client_config):
        print('Initializing WearableAnalyticsTrainer...')
        self.client_config = client_config
        self.model_params = model_params
        self.model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Time series parameters
        self.sequence_length = getattr(client_config, 'sequence_length', 24)  # 24 hours
        self.prediction_horizon = getattr(client_config, 'prediction_horizon', 1)  # 1 hour ahead
        
        # Training metrics
        self.training_losses = []
        self.validation_losses = []
        self.validation_maes = []
        
        # Data paths
        current_directory = os.path.dirname(os.path.realpath(__file__))
        self.data_folder = current_directory + GLOBAL_TMP_PATH + '/wearable_data/'
        
        # Ensure data folder exists
        os.makedirs(self.data_folder, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def train_model(self) -> Dict:
        """
        Train the wearable analytics model using federated learning.
        Returns model parameters and training metrics.
        """
        try:
            # Load and preprocess time series data
            train_loader, val_loader, input_dim = self._load_datasets()
            
            # Initialize model
            self._initialize_model(input_dim)
            
            # Training loop
            optimizer = optim.AdamW(self.model.parameters(), 
                                  lr=self.client_config.learning_rate,
                                  weight_decay=1e-4)
            criterion = nn.MSELoss()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           T_max=self.client_config.epochs)
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.client_config.epochs):
                # Training phase
                train_loss = self._train_epoch(train_loader, optimizer, criterion)
                
                # Validation phase
                val_loss, val_mae = self._validate_epoch(val_loader, criterion)
                
                # Learning rate scheduling
                scheduler.step()
                
                # Store metrics
                self.training_losses.append(train_loss)
                self.validation_losses.append(val_loss)
                self.validation_maes.append(val_mae)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= 15:  # Early stopping patience
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                self.logger.info(f'Epoch {epoch + 1}/{self.client_config.epochs} - '
                               f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                               f'Val MAE: {val_mae:.4f}')
            
            # Return trained model parameters and metrics
            return {
                'model_params': [p.detach().cpu() for p in self.model.parameters()],
                'training_metrics': {
                    'training_losses': self.training_losses,
                    'validation_losses': self.validation_losses,
                    'validation_maes': self.validation_maes,
                    'best_val_loss': best_val_loss,
                    'final_mae': self.validation_maes[-1] if self.validation_maes else 0.0
                },
                'model_config': {
                    'sequence_length': self.sequence_length,
                    'prediction_horizon': self.prediction_horizon,
                    'input_dim': input_dim
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in wearable analytics training: {str(e)}")
            raise
    
    def _initialize_model(self, input_dim: int):
        """Initialize the time series neural network model."""
        self.model = WearableTimeSeriesNet(
            input_dim=input_dim,
            hidden_dim=getattr(self.client_config, 'hidden_dim', 128),
            num_layers=getattr(self.client_config, 'num_layers', 2),
            output_dim=getattr(self.client_config, 'output_dim', 1),
            dropout_rate=getattr(self.client_config, 'dropout_rate', 0.3),
            sequence_length=self.sequence_length
        ).to(self.device)
        
        # Load pre-trained weights if available
        if self.model_params is not None:
            self.logger.info('Loading model weights from federated server')
            state_dict = {}
            param_iter = iter(self.model_params)
            for name, param in self.model.named_parameters():
                state_dict[name] = next(param_iter)
            self.model.load_state_dict(state_dict)
        else:
            self.logger.info('Initializing model with random weights')
    
    def _train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                    criterion: nn.Module) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_sequences, batch_targets in train_loader:
            batch_sequences = batch_sequences.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            optimizer.zero_grad()
            predictions = self.model(batch_sequences)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            
            # Gradient clipping for LSTM stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate the model and return loss and MAE."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_sequences, batch_targets in val_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                predictions = self.model(batch_sequences)
                loss = criterion(predictions, batch_targets)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())
        
        # Calculate MAE
        mae = mean_absolute_error(all_targets, all_predictions) if len(all_targets) > 0 else 0.0
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        return avg_loss, mae
    
    def _load_datasets(self) -> Tuple[DataLoader, DataLoader, int]:
        """
        Load wearable device time series datasets for training and validation.
        This method generates synthetic wearable data for demonstration.
        """
        
        # Generate synthetic wearable time series data
        data = self._generate_synthetic_wearable_data()
        
        # Prepare time series sequences
        sequences, targets = self._create_sequences(data)
        
        # Split into train/validation
        n_samples = len(sequences)
        n_train = int(0.8 * n_samples)
        
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        train_sequences = sequences[train_indices]
        train_targets = targets[train_indices]
        val_sequences = sequences[val_indices]
        val_targets = targets[val_indices]
        
        # Convert to tensors
        train_sequences_tensor = torch.FloatTensor(train_sequences)
        train_targets_tensor = torch.FloatTensor(train_targets)
        val_sequences_tensor = torch.FloatTensor(val_sequences)
        val_targets_tensor = torch.FloatTensor(val_targets)
        
        # Create data loaders
        train_dataset = TensorDataset(train_sequences_tensor, train_targets_tensor)
        val_dataset = TensorDataset(val_sequences_tensor, val_targets_tensor)
        
        train_loader = DataLoader(train_dataset, 
                                batch_size=self.client_config.batch_size, 
                                shuffle=True)
        val_loader = DataLoader(val_dataset, 
                              batch_size=self.client_config.batch_size, 
                              shuffle=False)
        
        self.logger.info(f'Loaded {len(train_dataset)} training sequences and '
                        f'{len(val_dataset)} validation sequences')
        
        return train_loader, val_loader, sequences.shape[2]
    
    def _generate_synthetic_wearable_data(self) -> pd.DataFrame:
        """
        Generate synthetic wearable device data for demonstration purposes.
        Simulates realistic patterns in wearable sensor data.
        """
        np.random.seed(42)  # For reproducibility
        
        # Generate 30 days of hourly data
        n_hours = 30 * 24  # 30 days
        timestamps = pd.date_range(start='2024-01-01', periods=n_hours, freq='H')
        
        # Create synthetic wearable data with realistic patterns
        data = []
        
        for i, timestamp in enumerate(timestamps):
            hour_of_day = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Heart rate (varies by time of day and activity)
            base_hr = 70
            if 6 <= hour_of_day <= 22:  # Awake hours
                activity_boost = np.random.normal(20, 10)
                if 12 <= hour_of_day <= 14 or 18 <= hour_of_day <= 20:  # Meal times
                    activity_boost += 10
            else:  # Sleep hours
                activity_boost = -20
            
            heart_rate = max(50, base_hr + activity_boost + np.random.normal(0, 5))
            
            # Steps (higher during day, weekend variations)
            if 22 <= hour_of_day or hour_of_day <= 6:  # Sleep hours
                steps = np.random.poisson(5)
            else:
                base_steps = 300 if day_of_week < 5 else 200  # Weekday vs weekend
                steps = max(0, np.random.poisson(base_steps))
            
            # Calories burned (correlated with steps and heart rate)
            calories = 50 + 0.04 * steps + 0.5 * (heart_rate - 70) + np.random.normal(0, 10)
            calories = max(30, calories)
            
            # Sleep quality (only relevant during sleep hours)
            if 22 <= hour_of_day or hour_of_day <= 6:
                sleep_quality = np.random.normal(0.8, 0.15)
                sleep_quality = np.clip(sleep_quality, 0, 1)
            else:
                sleep_quality = 0  # Not sleeping
            
            # Stress level (higher during work hours)
            if 9 <= hour_of_day <= 17 and day_of_week < 5:  # Work hours
                stress_level = np.random.normal(0.6, 0.2)
            else:
                stress_level = np.random.normal(0.3, 0.15)
            stress_level = np.clip(stress_level, 0, 1)
            
            # Activity level (combination of steps and heart rate)
            activity_level = min(1.0, (steps / 500) * 0.7 + ((heart_rate - 60) / 60) * 0.3)
            
            data.append({
                'timestamp': timestamp,
                'hour_of_day': hour_of_day,
                'day_of_week': day_of_week,
                'heart_rate': heart_rate,
                'steps': steps,
                'calories_burned': calories,
                'sleep_quality': sleep_quality,
                'stress_level': stress_level,
                'activity_level': activity_level,
                'is_weekend': 1 if day_of_week >= 5 else 0,
                'is_sleeping': 1 if (22 <= hour_of_day or hour_of_day <= 6) else 0
            })
        
        df = pd.DataFrame(data)
        
        # Add some realistic noise and correlations
        df['heart_rate_variability'] = np.random.normal(30, 10, len(df))
        df['skin_temperature'] = np.random.normal(32.0, 1.5, len(df))
        
        self.logger.info(f'Generated synthetic wearable dataset with {len(df)} hourly samples')
        return df
    
    def _create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time series sequences for training.
        """
        # Select features for time series modeling
        feature_columns = [
            'hour_of_day', 'day_of_week', 'heart_rate', 'steps', 'calories_burned',
            'sleep_quality', 'stress_level', 'activity_level', 'is_weekend', 
            'is_sleeping', 'heart_rate_variability', 'skin_temperature'
        ]
        
        # Target variable (predict next hour's heart rate)
        target_column = 'heart_rate'
        
        features = data[feature_columns].values
        targets = data[target_column].values
        
        # Normalize features
        features_normalized = self.feature_scaler.fit_transform(features)
        targets_normalized = self.target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        
        # Create sequences
        sequences = []
        sequence_targets = []
        
        for i in range(len(features_normalized) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            seq = features_normalized[i:i + self.sequence_length]
            # Target (prediction horizon ahead)
            target = targets_normalized[i + self.sequence_length + self.prediction_horizon - 1]
            
            sequences.append(seq)
            sequence_targets.append(target)
        
        return np.array(sequences), np.array(sequence_targets).reshape(-1, 1)
    
    def predict_next_values(self, recent_data: np.ndarray, n_predictions: int = 24) -> np.ndarray:
        """
        Predict future values using the trained model.
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        predictions = []
        
        # Use the most recent sequence for prediction
        current_sequence = recent_data[-self.sequence_length:]
        
        with torch.no_grad():
            for _ in range(n_predictions):
                # Prepare input tensor
                input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                
                # Make prediction
                prediction = self.model(input_tensor)
                pred_value = prediction.cpu().numpy()[0, 0]
                predictions.append(pred_value)
                
                # Update sequence for next prediction (simplified approach)
                # In practice, you'd update with actual new sensor readings
                new_row = current_sequence[-1].copy()
                new_row[2] = pred_value  # Update heart rate feature
                current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # Denormalize predictions
        predictions_denormalized = self.target_scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        
        return predictions_denormalized
