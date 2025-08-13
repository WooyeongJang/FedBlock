"""
Statistical biomedical analysis trainer for FedBlock federated learning.
Uses classical statistical methods instead of deep learning for lightweight,
mobile-friendly biomedical data analysis.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import os
import pickle
from datetime import datetime

from .config import GLOBAL_TMP_PATH, GLOBAL_DATASETS


class StatisticalBiomedicalAnalyzer:
    """
    Statistical analyzer for biomedical data using classical machine learning
    and statistical methods. Lightweight and mobile-friendly.
    """
    
    def __init__(self, model_params, client_config):
        print('Initializing StatisticalBiomedicalAnalyzer...')
        self.client_config = client_config
        self.model_params = model_params
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Configuration
        self.analysis_type = getattr(client_config, 'analysis_type', 'regression')
        self.model_type = getattr(client_config, 'model_type', 'linear_regression')
        
        # Data paths
        current_directory = os.path.dirname(os.path.realpath(__file__))
        self.data_folder = current_directory + GLOBAL_TMP_PATH + '/biomedical_data/'
        
        # Ensure data folder exists
        os.makedirs(self.data_folder, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def train_model(self) -> Dict:
        """
        Train statistical models for biomedical analysis.
        Returns model parameters and statistical metrics.
        """
        try:
            # Load and preprocess data
            data = self._load_biomedical_data()
            
            # Perform statistical analysis
            results = {}
            
            if self.analysis_type == 'regression':
                results = self._perform_regression_analysis(data)
            elif self.analysis_type == 'classification':
                results = self._perform_classification_analysis(data)
            elif self.analysis_type == 'survival':
                results = self._perform_survival_analysis(data)
            elif self.analysis_type == 'correlation':
                results = self._perform_correlation_analysis(data)
            else:
                results = self._perform_descriptive_analysis(data)
            
            return {
                'model_params': self._serialize_models(),
                'statistical_results': results,
                'analysis_type': self.analysis_type,
                'model_type': self.model_type,
                'feature_importance': self.feature_importance,
                'data_summary': self._get_data_summary(data)
            }
            
        except Exception as e:
            self.logger.error(f"Error in statistical analysis: {str(e)}")
            raise
    
    def _load_biomedical_data(self) -> pd.DataFrame:
        """Load or generate biomedical data for analysis."""
        
        # Generate synthetic biomedical data for demonstration
        data = self._generate_synthetic_biomedical_data()
        
        self.logger.info(f'Loaded biomedical dataset with {len(data)} samples')
        return data
    
    def _perform_regression_analysis(self, data: pd.DataFrame) -> Dict:
        """Perform regression analysis for continuous biomedical outcomes."""
        
        # Prepare features and target
        feature_columns = [col for col in data.columns if col != 'target']
        X = data[feature_columns]
        y = data['target']
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['features'] = scaler
        
        # Train multiple regression models
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                # Fit model
                if name == 'random_forest':
                    model.fit(X, y)  # Random Forest can handle unscaled data
                    predictions = model.predict(X)
                    
                    # Feature importance
                    self.feature_importance[name] = dict(zip(
                        feature_columns, model.feature_importances_
                    ))
                else:
                    model.fit(X_scaled, y)
                    predictions = model.predict(X_scaled)
                    
                    # Feature importance (coefficients)
                    if hasattr(model, 'coef_'):
                        self.feature_importance[name] = dict(zip(
                            feature_columns, np.abs(model.coef_)
                        ))
                
                # Calculate metrics
                mse = mean_squared_error(y, predictions)
                mae = mean_absolute_error(y, predictions)
                r2 = r2_score(y, predictions)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_scaled if name != 'random_forest' else X, 
                                          y, cv=5, scoring='r2')
                
                results[name] = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'cv_mean_r2': float(np.mean(cv_scores)),
                    'cv_std_r2': float(np.std(cv_scores))
                }
                
                # Store model
                self.models[name] = model
                
            except Exception as e:
                self.logger.error(f"Error training {name} model: {str(e)}")
                results[name] = {'error': str(e)}
        
        # Statistical tests
        results['statistical_tests'] = self._perform_regression_tests(X, y)
        
        return results
    
    def _perform_classification_analysis(self, data: pd.DataFrame) -> Dict:
        """Perform classification analysis for categorical biomedical outcomes."""
        
        # Convert target to binary classification for demonstration
        data['target_binary'] = (data['target'] > data['target'].median()).astype(int)
        
        feature_columns = [col for col in data.columns if col not in ['target', 'target_binary']]
        X = data[feature_columns]
        y = data['target_binary']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['features'] = scaler
        
        # Train classification models
        models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                # Fit model
                if name == 'random_forest':
                    model.fit(X, y)
                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)[:, 1]
                    
                    # Feature importance
                    self.feature_importance[name] = dict(zip(
                        feature_columns, model.feature_importances_
                    ))
                else:
                    model.fit(X_scaled, y)
                    predictions = model.predict(X_scaled)
                    probabilities = model.predict_proba(X_scaled)[:, 1]
                    
                    # Feature importance (coefficients)
                    if hasattr(model, 'coef_'):
                        self.feature_importance[name] = dict(zip(
                            feature_columns, np.abs(model.coef_[0])
                        ))
                
                # Calculate metrics
                accuracy = accuracy_score(y, predictions)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_scaled if name != 'random_forest' else X, 
                                          y, cv=5, scoring='accuracy')
                
                results[name] = {
                    'accuracy': float(accuracy),
                    'cv_mean_accuracy': float(np.mean(cv_scores)),
                    'cv_std_accuracy': float(np.std(cv_scores)),
                    'classification_report': classification_report(y, predictions, output_dict=True),
                    'confusion_matrix': confusion_matrix(y, predictions).tolist()
                }
                
                # Store model
                self.models[name] = model
                
            except Exception as e:
                self.logger.error(f"Error training {name} model: {str(e)}")
                results[name] = {'error': str(e)}
        
        return results
    
    def _perform_survival_analysis(self, data: pd.DataFrame) -> Dict:
        """Perform basic survival analysis using statistical methods."""
        
        # Simulate survival data
        data['time_to_event'] = np.random.exponential(scale=365, size=len(data))  # Days
        data['event_occurred'] = np.random.binomial(1, 0.3, size=len(data))  # 30% event rate
        
        results = {}
        
        # Kaplan-Meier estimation (simplified)
        times = np.sort(data['time_to_event'].unique())
        survival_probs = []
        
        for t in times:
            at_risk = len(data[data['time_to_event'] >= t])
            events = len(data[(data['time_to_event'] == t) & (data['event_occurred'] == 1)])
            if at_risk > 0:
                surv_prob = 1 - (events / at_risk)
                survival_probs.append(surv_prob)
            else:
                survival_probs.append(1.0)
        
        results['kaplan_meier'] = {
            'times': times.tolist(),
            'survival_probabilities': survival_probs,
            'median_survival_time': float(np.median(data['time_to_event']))
        }
        
        # Log-rank test simulation (simplified)
        # Split into two groups based on a key variable
        key_variable = data.columns[0]  # Use first feature
        median_split = data[key_variable].median()
        group1 = data[data[key_variable] <= median_split]
        group2 = data[data[key_variable] > median_split]
        
        # Simplified log-rank test
        group1_events = group1['event_occurred'].sum()
        group2_events = group2['event_occurred'].sum()
        total_events = group1_events + group2_events
        
        if total_events > 0:
            expected1 = len(group1) * total_events / len(data)
            chi2_stat = (group1_events - expected1) ** 2 / expected1
            p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
            
            results['log_rank_test'] = {
                'chi2_statistic': float(chi2_stat),
                'p_value': float(p_value),
                'group1_events': int(group1_events),
                'group2_events': int(group2_events)
            }
        
        return results
    
    def _perform_correlation_analysis(self, data: pd.DataFrame) -> Dict:
        """Perform correlation and association analysis."""
        
        # Numeric correlations
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        correlation_matrix = data[numeric_columns].corr()
        
        results = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'significant_correlations': []
        }
        
        # Find significant correlations
        for i, col1 in enumerate(numeric_columns):
            for j, col2 in enumerate(numeric_columns):
                if i < j:  # Avoid duplicates
                    corr_coef, p_value = pearsonr(data[col1], data[col2])
                    if p_value < 0.05:  # Significant correlation
                        results['significant_correlations'].append({
                            'variable1': col1,
                            'variable2': col2,
                            'correlation': float(corr_coef),
                            'p_value': float(p_value)
                        })
        
        # Spearman correlation for non-parametric relationships
        spearman_matrix = data[numeric_columns].corr(method='spearman')
        results['spearman_correlation'] = spearman_matrix.to_dict()
        
        return results
    
    def _perform_descriptive_analysis(self, data: pd.DataFrame) -> Dict:
        """Perform descriptive statistical analysis."""
        
        results = {
            'descriptive_statistics': data.describe().to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.astype(str).to_dict()
        }
        
        # Statistical tests for normality
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        normality_tests = {}
        
        for col in numeric_columns:
            if len(data[col].dropna()) > 3:  # Need at least 3 observations
                try:
                    statistic, p_value = stats.shapiro(data[col].dropna())
                    normality_tests[col] = {
                        'shapiro_statistic': float(statistic),
                        'shapiro_p_value': float(p_value),
                        'is_normal': p_value > 0.05
                    }
                except:
                    normality_tests[col] = {'error': 'Could not perform normality test'}
        
        results['normality_tests'] = normality_tests
        
        return results
    
    def _perform_regression_tests(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Perform statistical tests related to regression."""
        
        tests = {}
        
        # Test for multicollinearity (VIF)
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                              for i in range(len(X.columns))]
            tests['vif'] = vif_data.to_dict('records')
        except:
            tests['vif'] = {'error': 'Could not calculate VIF'}
        
        # Test for heteroscedasticity (Breusch-Pagan test)
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            model = LinearRegression().fit(X, y)
            predictions = model.predict(X)
            residuals = y - predictions
            
            # Simplified heteroscedasticity test
            residuals_squared = residuals ** 2
            corr_coef, p_value = pearsonr(predictions, residuals_squared)
            
            tests['heteroscedasticity'] = {
                'correlation_coef': float(corr_coef),
                'p_value': float(p_value),
                'heteroscedastic': p_value < 0.05
            }
        except:
            tests['heteroscedasticity'] = {'error': 'Could not perform test'}
        
        return tests
    
    def _generate_synthetic_biomedical_data(self) -> pd.DataFrame:
        """Generate synthetic biomedical data for statistical analysis."""
        
        np.random.seed(42)  # For reproducibility
        n_samples = getattr(self.client_config, 'dataset_size', 1000)
        
        # Generate realistic biomedical data with statistical relationships
        data = {
            # Demographics
            'age': np.random.normal(45, 15, n_samples),
            'gender': np.random.binomial(1, 0.5, n_samples),  # 0=Female, 1=Male
            
            # Vital signs
            'systolic_bp': np.random.normal(120, 20, n_samples),
            'diastolic_bp': np.random.normal(80, 15, n_samples),
            'heart_rate': np.random.normal(70, 12, n_samples),
            'bmi': np.random.normal(25, 5, n_samples),
            
            # Lab values
            'glucose': np.random.normal(100, 30, n_samples),
            'cholesterol': np.random.normal(200, 50, n_samples),
            'hemoglobin': np.random.normal(14, 2, n_samples),
            
            # Lifestyle factors
            'smoking': np.random.binomial(1, 0.2, n_samples),
            'exercise_hours': np.random.exponential(3, n_samples),
            
            # Medical history
            'diabetes': np.random.binomial(1, 0.1, n_samples),
            'hypertension': np.random.binomial(1, 0.15, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Ensure realistic ranges
        df['age'] = np.clip(df['age'], 18, 100)
        df['systolic_bp'] = np.clip(df['systolic_bp'], 90, 200)
        df['diastolic_bp'] = np.clip(df['diastolic_bp'], 60, 120)
        df['heart_rate'] = np.clip(df['heart_rate'], 50, 120)
        df['bmi'] = np.clip(df['bmi'], 15, 45)
        df['glucose'] = np.clip(df['glucose'], 70, 300)
        df['exercise_hours'] = np.clip(df['exercise_hours'], 0, 20)
        
        # Create target with realistic statistical relationships
        df['target'] = (
            0.3 * df['age'] +
            0.5 * df['systolic_bp'] +
            0.2 * df['bmi'] +
            0.1 * df['glucose'] +
            5.0 * df['smoking'] +
            3.0 * df['diabetes'] +
            2.0 * df['hypertension'] +
            np.random.normal(0, 5, n_samples)  # Add noise
        )
        
        # Normalize target to reasonable range (e.g., risk score 0-100)
        df['target'] = ((df['target'] - df['target'].min()) / 
                       (df['target'].max() - df['target'].min()) * 100)
        
        self.logger.info(f'Generated synthetic biomedical dataset with {len(df)} samples')
        return df
    
    def _serialize_models(self) -> Dict:
        """Serialize trained models for federated learning."""
        
        serialized = {}
        
        for name, model in self.models.items():
            try:
                # For scikit-learn models, extract key parameters
                if hasattr(model, 'coef_'):
                    serialized[name] = {
                        'type': type(model).__name__,
                        'coefficients': model.coef_.tolist() if hasattr(model.coef_, 'tolist') else model.coef_,
                        'intercept': float(model.intercept_) if hasattr(model, 'intercept_') else None
                    }
                elif hasattr(model, 'feature_importances_'):
                    serialized[name] = {
                        'type': type(model).__name__,
                        'feature_importances': model.feature_importances_.tolist(),
                        'n_estimators': getattr(model, 'n_estimators', None)
                    }
                else:
                    serialized[name] = {
                        'type': type(model).__name__,
                        'params': model.get_params()
                    }
            except Exception as e:
                self.logger.error(f"Error serializing model {name}: {str(e)}")
                serialized[name] = {'error': str(e)}
        
        return serialized
    
    def _get_data_summary(self, data: pd.DataFrame) -> Dict:
        """Get summary statistics of the dataset."""
        
        return {
            'n_samples': len(data),
            'n_features': len(data.columns) - 1,  # Exclude target
            'feature_names': [col for col in data.columns if col != 'target'],
            'target_stats': {
                'mean': float(data['target'].mean()),
                'std': float(data['target'].std()),
                'min': float(data['target'].min()),
                'max': float(data['target'].max())
            }
        }
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
