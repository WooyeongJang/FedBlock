import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
from typing import Dict, List, Tuple, Optional
import json
import os

from .config import GLOBAL_TMP_PATH, GLOBAL_DATASETS


class BiomedicalRegressionNet(nn.Module):
    """Neural network for biomedical regression tasks."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], 
                 output_dim: int = 1, dropout_rate: float = 0.2):
        super(BiomedicalRegressionNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class BiomedicalRegressionTrainer:
    """
    Federated learning trainer for biomedical regression tasks.
    Supports various health-related prediction tasks including:
    - Lab value prediction
    - Risk assessment
    - Wearable device analytics
    - Vital sign monitoring
    - Medical outcome prediction
    """
    
    def __init__(self, model_params, client_config):
        print('Initializing BiomedicalRegressionTrainer...')
        self.client_config = client_config
        self.model_params = model_params
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training metrics
        self.training_losses = []
        self.validation_losses = []
        self.validation_r2_scores = []
        
        # Data paths
        current_directory = os.path.dirname(os.path.realpath(__file__))
        self.data_folder = current_directory + GLOBAL_TMP_PATH + '/biomedical_data/'
        
        # Ensure data folder exists
        os.makedirs(self.data_folder, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def train_model(self) -> Dict:
        """
        Train the biomedical regression model using federated learning.
        Returns model parameters and training metrics.
        """
        try:
            # Load and preprocess data
            train_loader, val_loader, input_dim = self._load_datasets()
            
            # Initialize model
            self._initialize_model(input_dim)
            
            # Training loop
            optimizer = optim.Adam(self.model.parameters(), 
                                 lr=self.client_config.learning_rate,
                                 weight_decay=1e-5)
            criterion = nn.MSELoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.client_config.epochs):
                # Training phase
                train_loss = self._train_epoch(train_loader, optimizer, criterion)
                
                # Validation phase
                val_loss, val_r2 = self._validate_epoch(val_loader, criterion)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Store metrics
                self.training_losses.append(train_loss)
                self.validation_losses.append(val_loss)
                self.validation_r2_scores.append(val_r2)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= 10:  # Early stopping patience
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                self.logger.info(f'Epoch {epoch + 1}/{self.client_config.epochs} - '
                               f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                               f'Val R²: {val_r2:.4f}')
            
            # Return trained model parameters and metrics
            return {
                'model_params': [p.detach().cpu() for p in self.model.parameters()],
                'training_metrics': {
                    'training_losses': self.training_losses,
                    'validation_losses': self.validation_losses,
                    'validation_r2_scores': self.validation_r2_scores,
                    'best_val_loss': best_val_loss,
                    'final_r2': self.validation_r2_scores[-1] if self.validation_r2_scores else 0.0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise
    
    def _initialize_model(self, input_dim: int):
        """Initialize the neural network model."""
        self.model = BiomedicalRegressionNet(
            input_dim=input_dim,
            hidden_dims=getattr(self.client_config, 'hidden_dims', [128, 64, 32]),
            output_dim=getattr(self.client_config, 'output_dim', 1),
            dropout_rate=getattr(self.client_config, 'dropout_rate', 0.3)
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
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            optimizer.zero_grad()
            predictions = self.model(batch_features)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate the model and return loss and R² score."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                predictions = self.model(batch_features)
                loss = criterion(predictions, batch_targets)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())
        
        # Calculate R² score
        r2 = r2_score(all_targets, all_predictions) if len(all_targets) > 0 else 0.0
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        return avg_loss, r2
    
    def _load_datasets(self) -> Tuple[DataLoader, DataLoader, int]:
        """
        Load biomedical datasets for training and validation.
        This method generates synthetic biomedical data for demonstration.
        In practice, this would load real patient data (with proper privacy safeguards).
        """
        
        # Generate synthetic biomedical data for demonstration
        # In real scenarios, this would load actual health data
        data = self._generate_synthetic_biomedical_data()
        
        # Split features and targets
        features = data.drop('target', axis=1).values
        targets = data['target'].values.reshape(-1, 1)
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features)
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features_normalized)
        targets_tensor = torch.FloatTensor(targets)
        
        # Split into train/validation
        n_samples = len(features_tensor)
        n_train = int(0.8 * n_samples)
        
        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        train_features = features_tensor[train_indices]
        train_targets = targets_tensor[train_indices]
        val_features = features_tensor[val_indices]
        val_targets = targets_tensor[val_indices]
        
        # Create data loaders
        train_dataset = TensorDataset(train_features, train_targets)
        val_dataset = TensorDataset(val_features, val_targets)
        
        train_loader = DataLoader(train_dataset, 
                                batch_size=self.client_config.batch_size, 
                                shuffle=True)
        val_loader = DataLoader(val_dataset, 
                              batch_size=self.client_config.batch_size, 
                              shuffle=False)
        
        self.logger.info(f'Loaded {len(train_dataset)} training samples and '
                        f'{len(val_dataset)} validation samples')
        
        return train_loader, val_loader, features_normalized.shape[1]
    
    def _generate_synthetic_biomedical_data(self) -> pd.DataFrame:
        """
        Generate synthetic biomedical data for demonstration purposes.
        This includes various health metrics and lab values.
        """
        np.random.seed(42)  # For reproducibility
        n_samples = getattr(self.client_config, 'dataset_size', 1000)
        
        # Generate synthetic patient data
        data = {
            # Vital signs
            'age': np.random.normal(45, 15, n_samples),
            'systolic_bp': np.random.normal(120, 20, n_samples),
            'diastolic_bp': np.random.normal(80, 15, n_samples),
            'heart_rate': np.random.normal(70, 12, n_samples),
            'temperature': np.random.normal(98.6, 1.5, n_samples),
            'respiratory_rate': np.random.normal(16, 4, n_samples),
            
            # Lab values
            'glucose': np.random.normal(100, 30, n_samples),
            'cholesterol_total': np.random.normal(200, 50, n_samples),
            'hdl_cholesterol': np.random.normal(50, 15, n_samples),
            'ldl_cholesterol': np.random.normal(120, 40, n_samples),
            'triglycerides': np.random.normal(150, 80, n_samples),
            'hemoglobin': np.random.normal(14, 2, n_samples),
            'white_blood_cells': np.random.normal(7000, 2000, n_samples),
            'platelets': np.random.normal(250000, 50000, n_samples),
            
            # Body measurements
            'bmi': np.random.normal(25, 5, n_samples),
            'weight': np.random.normal(70, 15, n_samples),
            'height': np.random.normal(170, 10, n_samples),
            
            # Lifestyle factors (binary/categorical encoded as numeric)
            'smoking': np.random.binomial(1, 0.2, n_samples),
            'alcohol_consumption': np.random.poisson(2, n_samples),
            'exercise_hours_per_week': np.random.exponential(3, n_samples),
            
            # Medical history indicators
            'diabetes_history': np.random.binomial(1, 0.1, n_samples),
            'hypertension_history': np.random.binomial(1, 0.15, n_samples),
            'heart_disease_history': np.random.binomial(1, 0.08, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Ensure realistic ranges
        df['age'] = np.clip(df['age'], 18, 100)
        df['systolic_bp'] = np.clip(df['systolic_bp'], 90, 200)
        df['diastolic_bp'] = np.clip(df['diastolic_bp'], 60, 120)
        df['heart_rate'] = np.clip(df['heart_rate'], 50, 120)
        df['glucose'] = np.clip(df['glucose'], 70, 300)
        df['bmi'] = np.clip(df['bmi'], 15, 45)
        
        # Create target variable (cardiovascular risk score)
        # This is a synthetic composite score for demonstration
        df['target'] = (
            0.1 * df['age'] +
            0.05 * df['systolic_bp'] +
            0.03 * df['diastolic_bp'] +
            0.02 * df['glucose'] +
            0.08 * df['bmi'] +
            2.0 * df['smoking'] +
            1.5 * df['diabetes_history'] +
            1.8 * df['hypertension_history'] +
            2.2 * df['heart_disease_history'] +
            np.random.normal(0, 2, n_samples)  # Add noise
        )
        
        # Normalize target to reasonable range
        df['target'] = np.clip(df['target'], 0, 100)
        
        self.logger.info(f'Generated synthetic biomedical dataset with {len(df)} samples')
        return df

    def save_model_checkpoint(self, checkpoint_path: str):
        """Save model checkpoint for later use."""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'training_metrics': {
                    'training_losses': self.training_losses,
                    'validation_losses': self.validation_losses,
                    'validation_r2_scores': self.validation_r2_scores
                }
            }, checkpoint_path)
            self.logger.info(f'Model checkpoint saved to {checkpoint_path}')

    def load_model_checkpoint(self, checkpoint_path: str, input_dim: int):
        """Load model checkpoint."""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self._initialize_model(input_dim)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler = checkpoint['scaler']
            
            if 'training_metrics' in checkpoint:
                metrics = checkpoint['training_metrics']
                self.training_losses = metrics.get('training_losses', [])
                self.validation_losses = metrics.get('validation_losses', [])
                self.validation_r2_scores = metrics.get('validation_r2_scores', [])
            
            self.logger.info(f'Model checkpoint loaded from {checkpoint_path}')
        else:
            self.logger.warning(f'Checkpoint file not found: {checkpoint_path}')
