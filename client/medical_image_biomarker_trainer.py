import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b0
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
from typing import Dict, List, Tuple, Optional
import json
import os
from PIL import Image
import cv2

from .config import GLOBAL_TMP_PATH, GLOBAL_DATASETS


class MedicalImageBiomarkerNet(nn.Module):
    """
    CNN-based neural network for extracting biomarkers from medical images.
    Supports regression tasks on medical imaging data.
    """
    
    def __init__(self, backbone: str = 'resnet50', num_biomarkers: int = 5, 
                 pretrained: bool = True, dropout_rate: float = 0.3):
        super(MedicalImageBiomarkerNet, self).__init__()
        
        # Initialize backbone
        if backbone == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove the final classification layer
        elif backbone == 'efficientnet_b0':
            self.backbone = efficientnet_b0(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Biomarker regression head
        self.biomarker_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_biomarkers)
        )
        
        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Predict biomarkers
        biomarkers = self.biomarker_head(attended_features)
        
        return biomarkers, attention_weights


class MedicalImageBiomarkerTrainer:
    """
    Federated learning trainer for medical image biomarker extraction.
    Supports analysis of:
    - Cardiac function parameters from echocardiograms
    - Lung capacity measurements from chest X-rays
    - Bone density assessment from DEXA scans
    - Retinal vessel analysis from fundus images
    - Brain volume measurements from MRI
    """
    
    def __init__(self, model_params, client_config):
        print('Initializing MedicalImageBiomarkerTrainer...')
        self.client_config = client_config
        self.model_params = model_params
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model configuration
        self.backbone = getattr(client_config, 'backbone', 'resnet50')
        self.num_biomarkers = getattr(client_config, 'num_biomarkers', 5)
        self.image_size = getattr(client_config, 'image_size', (224, 224))
        
        # Training metrics
        self.training_losses = []
        self.validation_losses = []
        self.validation_r2_scores = []
        self.validation_maes = []
        
        # Data paths
        current_directory = os.path.dirname(os.path.realpath(__file__))
        self.data_folder = current_directory + GLOBAL_TMP_PATH + '/medical_images/'
        
        # Ensure data folder exists
        os.makedirs(self.data_folder, exist_ok=True)
        
        # Image transformations
        self.train_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def train_model(self) -> Dict:
        """
        Train the medical image biomarker extraction model using federated learning.
        Returns model parameters and training metrics.
        """
        try:
            # Load and preprocess medical images
            train_loader, val_loader = self._load_datasets()
            
            # Initialize model
            self._initialize_model()
            
            # Training loop
            optimizer = optim.AdamW(self.model.parameters(), 
                                  lr=self.client_config.learning_rate,
                                  weight_decay=1e-4)
            criterion = nn.MSELoss()
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2)
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.client_config.epochs):
                # Training phase
                train_loss = self._train_epoch(train_loader, optimizer, criterion)
                
                # Validation phase
                val_loss, val_r2, val_mae = self._validate_epoch(val_loader, criterion)
                
                # Learning rate scheduling
                scheduler.step()
                
                # Store metrics
                self.training_losses.append(train_loss)
                self.validation_losses.append(val_loss)
                self.validation_r2_scores.append(val_r2)
                self.validation_maes.append(val_mae)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= 20:  # Early stopping patience
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                self.logger.info(f'Epoch {epoch + 1}/{self.client_config.epochs} - '
                               f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                               f'Val R²: {val_r2:.4f}, Val MAE: {val_mae:.4f}')
            
            # Return trained model parameters and metrics
            return {
                'model_params': [p.detach().cpu() for p in self.model.parameters()],
                'training_metrics': {
                    'training_losses': self.training_losses,
                    'validation_losses': self.validation_losses,
                    'validation_r2_scores': self.validation_r2_scores,
                    'validation_maes': self.validation_maes,
                    'best_val_loss': best_val_loss,
                    'final_r2': self.validation_r2_scores[-1] if self.validation_r2_scores else 0.0,
                    'final_mae': self.validation_maes[-1] if self.validation_maes else 0.0
                },
                'model_config': {
                    'backbone': self.backbone,
                    'num_biomarkers': self.num_biomarkers,
                    'image_size': self.image_size
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in medical image biomarker training: {str(e)}")
            raise
    
    def _initialize_model(self):
        """Initialize the medical image biomarker extraction model."""
        self.model = MedicalImageBiomarkerNet(
            backbone=self.backbone,
            num_biomarkers=self.num_biomarkers,
            pretrained=True,
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
            self.logger.info('Initializing model with ImageNet pre-trained weights')
    
    def _train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                    criterion: nn.Module) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_images, batch_biomarkers in train_loader:
            batch_images = batch_images.to(self.device)
            batch_biomarkers = batch_biomarkers.to(self.device)
            
            optimizer.zero_grad()
            predictions, attention_weights = self.model(batch_images)
            loss = criterion(predictions, batch_biomarkers)
            
            # Add attention regularization
            attention_reg = torch.mean(torch.var(attention_weights, dim=1))
            total_loss_with_reg = loss + 0.01 * attention_reg
            
            total_loss_with_reg.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float, float]:
        """Validate the model and return loss, R² score, and MAE."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_images, batch_biomarkers in val_loader:
                batch_images = batch_images.to(self.device)
                batch_biomarkers = batch_biomarkers.to(self.device)
                
                predictions, _ = self.model(batch_images)
                loss = criterion(predictions, batch_biomarkers)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_biomarkers.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Average across all biomarkers
        r2 = r2_score(all_targets, all_predictions, multioutput='uniform_average')
        mae = mean_absolute_error(all_targets, all_predictions)
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        return avg_loss, r2, mae
    
    def _load_datasets(self) -> Tuple[DataLoader, DataLoader]:
        """
        Load medical image datasets for training and validation.
        This method generates synthetic medical images for demonstration.
        """
        
        # Generate synthetic medical images and biomarkers
        train_images, train_biomarkers, val_images, val_biomarkers = self._generate_synthetic_medical_data()
        
        # Create datasets
        train_dataset = MedicalImageDataset(train_images, train_biomarkers, self.train_transform)
        val_dataset = MedicalImageDataset(val_images, val_biomarkers, self.val_transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, 
                                batch_size=self.client_config.batch_size, 
                                shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, 
                              batch_size=self.client_config.batch_size, 
                              shuffle=False, num_workers=2)
        
        self.logger.info(f'Loaded {len(train_dataset)} training images and '
                        f'{len(val_dataset)} validation images')
        
        return train_loader, val_loader
    
    def _generate_synthetic_medical_data(self) -> Tuple[List, np.ndarray, List, np.ndarray]:
        """
        Generate synthetic medical images and corresponding biomarkers.
        This simulates various medical imaging modalities.
        """
        np.random.seed(42)  # For reproducibility
        
        n_train_samples = getattr(self.client_config, 'train_samples', 800)
        n_val_samples = getattr(self.client_config, 'val_samples', 200)
        
        # Generate synthetic medical images
        train_images = []
        train_biomarkers = []
        
        for i in range(n_train_samples):
            # Create synthetic medical image (grayscale converted to RGB)
            image = self._create_synthetic_medical_image()
            train_images.append(image)
            
            # Generate corresponding biomarkers based on image characteristics
            biomarkers = self._extract_synthetic_biomarkers(image)
            train_biomarkers.append(biomarkers)
        
        val_images = []
        val_biomarkers = []
        
        for i in range(n_val_samples):
            image = self._create_synthetic_medical_image()
            val_images.append(image)
            
            biomarkers = self._extract_synthetic_biomarkers(image)
            val_biomarkers.append(biomarkers)
        
        # Convert biomarkers to numpy arrays and normalize
        train_biomarkers = np.array(train_biomarkers)
        val_biomarkers = np.array(val_biomarkers)
        
        # Normalize biomarkers
        train_biomarkers = self.scaler.fit_transform(train_biomarkers)
        val_biomarkers = self.scaler.transform(val_biomarkers)
        
        self.logger.info(f'Generated synthetic medical dataset with {len(train_images)} '
                        f'training and {len(val_images)} validation images')
        
        return train_images, train_biomarkers, val_images, val_biomarkers
    
    def _create_synthetic_medical_image(self) -> Image.Image:
        """
        Create a synthetic medical image with realistic patterns.
        This simulates various anatomical structures and pathologies.
        """
        # Create base image
        img_size = 256
        image = np.zeros((img_size, img_size), dtype=np.uint8)
        
        # Add anatomical structures
        # Simulate organ boundaries
        center_x, center_y = img_size // 2, img_size // 2
        
        # Main organ (circular/elliptical structure)
        for y in range(img_size):
            for x in range(img_size):
                dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                # Create organ boundaries with noise
                if dist_from_center < 80 + np.random.normal(0, 10):
                    image[y, x] = int(120 + np.random.normal(0, 20))
                
                # Add vessel-like structures
                if (abs(x - center_x) < 5 and abs(y - center_y) < 60) or \
                   (abs(y - center_y) < 5 and abs(x - center_x) < 60):
                    image[y, x] = min(255, image[y, x] + 50)
        
        # Add texture and noise
        noise = np.random.normal(0, 15, (img_size, img_size))
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # Apply Gaussian blur to simulate medical imaging artifacts
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Convert to RGB PIL Image
        image_rgb = np.stack([image, image, image], axis=2)
        pil_image = Image.fromarray(image_rgb)
        
        return pil_image
    
    def _extract_synthetic_biomarkers(self, image: Image.Image) -> List[float]:
        """
        Extract synthetic biomarkers from the generated medical image.
        These simulate real medical measurements.
        """
        # Convert to numpy for analysis
        img_array = np.array(image.convert('L'))
        
        # Biomarker 1: Average intensity (tissue density)
        avg_intensity = np.mean(img_array) / 255.0
        
        # Biomarker 2: Standard deviation (tissue heterogeneity)
        intensity_std = np.std(img_array) / 255.0
        
        # Biomarker 3: Edge density (structural complexity)
        edges = cv2.Canny(img_array, 50, 150)
        edge_density = np.sum(edges > 0) / (img_array.shape[0] * img_array.shape[1])
        
        # Biomarker 4: Symmetry measure
        left_half = img_array[:, :img_array.shape[1]//2]
        right_half = img_array[:, img_array.shape[1]//2:]
        right_half_flipped = np.fliplr(right_half)
        symmetry = 1.0 - np.mean(np.abs(left_half - right_half_flipped)) / 255.0
        
        # Biomarker 5: Central tendency (concentration of features)
        center_region = img_array[64:192, 64:192]  # Central 128x128 region
        central_intensity = np.mean(center_region) / 255.0
        
        # Add some realistic noise and correlations
        biomarkers = [
            avg_intensity + np.random.normal(0, 0.05),
            intensity_std + np.random.normal(0, 0.02),
            edge_density + np.random.normal(0, 0.01),
            symmetry + np.random.normal(0, 0.03),
            central_intensity + np.random.normal(0, 0.04)
        ]
        
        # Ensure biomarkers are in reasonable ranges
        biomarkers = [max(0.0, min(1.0, b)) for b in biomarkers]
        
        return biomarkers
    
    def predict_biomarkers(self, image_path: str) -> Dict[str, float]:
        """
        Predict biomarkers from a single medical image.
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.val_transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions, attention_weights = self.model(image_tensor)
            
        # Denormalize predictions
        predictions_np = predictions.cpu().numpy()
        predictions_denormalized = self.scaler.inverse_transform(predictions_np)[0]
        
        # Return biomarkers as named dictionary
        biomarker_names = [
            'tissue_density', 'tissue_heterogeneity', 'structural_complexity',
            'anatomical_symmetry', 'central_concentration'
        ]
        
        results = {name: float(value) for name, value in 
                  zip(biomarker_names[:len(predictions_denormalized)], 
                      predictions_denormalized)}
        
        return results


class MedicalImageDataset(torch.utils.data.Dataset):
    """Custom dataset for medical images and biomarkers."""
    
    def __init__(self, images: List, biomarkers: np.ndarray, transform=None):
        self.images = images
        self.biomarkers = biomarkers
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        biomarkers = self.biomarkers[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.FloatTensor(biomarkers)
