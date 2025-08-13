import asyncio
import sys
import aiohttp
import torch
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np

from .utils import model_params_to_request_params
from .federated_learning_config import FederatedLearningConfig
from .client_training_status import ClientTrainingStatus
from .server_status import ServerStatus
from .training_client import TrainingClient
from .training_type import TrainingType

# Import blockchain audit service
try:
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from blockchain.audit_service import get_blockchain_audit_service, TrainingMetrics
    BLOCKCHAIN_AVAILABLE = True
except ImportError:
    BLOCKCHAIN_AVAILABLE = False
    print("Warning: Blockchain audit service not available")


class Server:
    def __init__(self):
        self.mnist_model_params = None
        self.chest_x_ray_model_params = None
        self.biomedical_model_params = None
        self.wearable_model_params = None
        self.medical_image_model_params = None
        
        self.init_params()
        self.training_clients = {}
        self.status = ServerStatus.IDLE
        self.round = 0
        
        # Initialize blockchain audit service
        self.blockchain_audit = None
        if BLOCKCHAIN_AVAILABLE:
            try:
                self.blockchain_audit = get_blockchain_audit_service()
                if self.blockchain_audit.enabled:
                    logging.info("Blockchain audit service enabled")
                else:
                    logging.warning("Blockchain audit service disabled - missing configuration")
            except Exception as e:
                logging.error(f"Failed to initialize blockchain audit: {str(e)}")
        
        # Training session tracking
        self.active_sessions = {}  # client_url -> session_id
        self.training_metrics = {}  # session_id -> metrics

    def init_params(self):
        """Initialize model parameters for different training types."""
        if self.mnist_model_params is None:
            weights = torch.randn((28 * 28, 1), dtype=torch.float, requires_grad=True)
            bias = torch.randn(1, dtype=torch.float, requires_grad=True)
            self.mnist_model_params = weights, bias
        
        # Initialize parameters for biomedical models
        if self.biomedical_model_params is None:
            # For biomedical regression - these will be overridden by actual trained models
            self.biomedical_model_params = None
        
        if self.wearable_model_params is None:
            # For wearable analytics - LSTM-based model parameters
            self.wearable_model_params = None
        
        if self.medical_image_model_params is None:
            # For medical image biomarker extraction
            self.medical_image_model_params = None

    async def start_training(self, training_type):
        if self.status != ServerStatus.IDLE:
            print('Server is not ready for training yet, status:', self.status)
            for training_client in self.training_clients.values():
                print(training_client)
        elif len(self.training_clients) == 0:
            print("There aren't any clients registered in the system, nothing to do yet")
        else:
            # Increment training round
            # This is needed for deterministic MNIST training
            self.round += 1

            request_body = {}
            federated_learning_config = None
            if (
                    training_type == TrainingType.MNIST
                    or training_type == TrainingType.DETERMINISTIC_MNIST
            ):
                request_body = model_params_to_request_params(training_type, self.mnist_model_params)
                federated_learning_config = FederatedLearningConfig(learning_rate=1., epochs=20, batch_size=256)
            elif training_type == TrainingType.GOSSIP_MNIST:
                request_body = model_params_to_request_params(training_type, None)
                federated_learning_config = FederatedLearningConfig(learning_rate=1., epochs=20, batch_size=256)
            elif training_type == TrainingType.CHEST_X_RAY_PNEUMONIA:
                request_body = model_params_to_request_params(training_type, self.chest_x_ray_model_params)
                federated_learning_config = FederatedLearningConfig(learning_rate=0.0001, epochs=1, batch_size=2)
            # Biomedical training types
            elif training_type == TrainingType.BIOMEDICAL_REGRESSION:
                request_body = model_params_to_request_params(training_type, self.biomedical_model_params)
                federated_learning_config = FederatedLearningConfig(learning_rate=0.001, epochs=50, batch_size=32)
            elif training_type == TrainingType.WEARABLE_ANALYTICS:
                request_body = model_params_to_request_params(training_type, self.wearable_model_params)
                federated_learning_config = FederatedLearningConfig(learning_rate=0.0005, epochs=30, batch_size=16)
            elif training_type == TrainingType.MEDICAL_IMAGE_BIOMARKER:
                request_body = model_params_to_request_params(training_type, self.medical_image_model_params)
                federated_learning_config = FederatedLearningConfig(learning_rate=0.0001, epochs=25, batch_size=8)
            elif training_type in [TrainingType.CARDIOVASCULAR_RISK, TrainingType.LAB_VALUE_PREDICTION]:
                # Use biomedical regression for specialized health analytics
                request_body = model_params_to_request_params(training_type, self.biomedical_model_params)
                federated_learning_config = FederatedLearningConfig(learning_rate=0.001, epochs=40, batch_size=32)

            request_body['learning_rate'] = federated_learning_config.learning_rate
            request_body['epochs'] = federated_learning_config.epochs
            request_body['batch_size'] = federated_learning_config.batch_size
            request_body['training_type'] = training_type
            request_body['round'] = self.round

            if training_type == TrainingType.GOSSIP_MNIST:
                # Send all client urls and ids to each client for decentralized learning
                clients = [
                    {"client_id": client.client_id, "client_url": client.client_url}
                    for client in self.training_clients.values()
                ]
                request_body['clients'] = clients

            # Log training start on blockchain
            current_model_params = None
            if training_type in [TrainingType.MNIST, TrainingType.DETERMINISTIC_MNIST]:
                current_model_params = self.mnist_model_params
            elif training_type == TrainingType.CHEST_X_RAY_PNEUMONIA:
                current_model_params = self.chest_x_ray_model_params
            elif training_type == TrainingType.BIOMEDICAL_REGRESSION:
                current_model_params = self.biomedical_model_params
            elif training_type == TrainingType.WEARABLE_ANALYTICS:
                current_model_params = self.wearable_model_params
            elif training_type == TrainingType.MEDICAL_IMAGE_BIOMARKER:
                current_model_params = self.medical_image_model_params
            elif training_type in [TrainingType.CARDIOVASCULAR_RISK, TrainingType.LAB_VALUE_PREDICTION]:
                current_model_params = self.biomedical_model_params
            
            session_id = self._log_training_start_on_blockchain(training_type, current_model_params)
            if session_id:
                # Store session for all clients in this round
                for client_url in self.training_clients.keys():
                    self.active_sessions[client_url] = session_id

            print('There are', len(self.training_clients), 'clients registered')
            tasks = []
            for training_client in self.training_clients.values():
                if training_type == TrainingType.DETERMINISTIC_MNIST or training_type == TrainingType.GOSSIP_MNIST:
                    request_body['round_size'] = len(self.training_clients.values())
                tasks.append(
                    asyncio.ensure_future(self.do_training_client_request(training_type, training_client, request_body))
                )
            print('Requesting training to clients...')
            self.status = ServerStatus.CLIENTS_TRAINING
            await asyncio.gather(*tasks)
        sys.stdout.flush()

    async def do_training_client_request(self, training_type, training_client, request_body):
        request_url = training_client.client_url + '/training'
        print('Requesting training to client', request_url)
        async with aiohttp.ClientSession() as session:
            # Ensures individual client_ids are sent to each client
            request_body['client_id'] = training_client.client_id
            training_client.status = ClientTrainingStatus.TRAINING_REQUESTED
            async with session.post(request_url, json=request_body) as response:
                if response.status != 200:
                    print('Error requesting training to client', training_client.client_url)
                    training_client.status = ClientTrainingStatus.TRAINING_REQUEST_ERROR
                    self.update_server_model_params(training_type)
                else:
                    print('Client', training_client.client_url, 'started training')

    def update_client_model_params(self, training_type, training_client, client_model_params):
        print('New model params received from client', training_client.client_url)
        training_client.model_params = client_model_params
        training_client.status = ClientTrainingStatus.TRAINING_FINISHED
        self.update_server_model_params(training_type)

    # Forces the round to finish. This is used for Gossip training
    # since no parameters will be sent back to the server
    # so the server needs to know when the round is finished
    def finish_round(self, training_type, training_client):
        training_client.status = ClientTrainingStatus.TRAINING_FINISHED

        if self.can_update_central_model_params() and training_type == TrainingType.GOSSIP_MNIST:
            self.status = ServerStatus.IDLE
            for training_client in self.training_clients.values():
                training_client.status = ClientTrainingStatus.IDLE
        sys.stdout.flush()

    def update_server_model_params(self, training_type):
        if self.can_update_central_model_params():
            print('Updating global model params')
            self.status = ServerStatus.UPDATING_MODEL_PARAMS
            
            # Collect participant models for blockchain logging
            participant_models = []
            aggregated_model = None
            
            if training_type == TrainingType.MNIST or training_type == TrainingType.DETERMINISTIC_MNIST:
                received_weights = []
                received_biases = []
                for training_client in self.training_clients.values():
                    if training_client.status == ClientTrainingStatus.TRAINING_FINISHED:
                        received_weights.append(training_client.model_params[0])
                        received_biases.append(training_client.model_params[1])
                        participant_models.append(training_client.model_params)
                        training_client.status = ClientTrainingStatus.IDLE
                new_weights = torch.stack(received_weights).mean(0)
                new_bias = torch.stack(received_biases).mean(0)
                self.mnist_model_params = new_weights, new_bias
                aggregated_model = self.mnist_model_params
                print('Model weights for', training_type, 'updated in central model')
                
            elif training_type == TrainingType.CHEST_X_RAY_PNEUMONIA:
                received_weights = []
                for training_client in self.training_clients.values():
                    if training_client.status == ClientTrainingStatus.TRAINING_FINISHED:
                        training_client.status = ClientTrainingStatus.IDLE
                        received_weights.append(training_client.model_params)
                        participant_models.append(training_client.model_params)
                new_weights = np.stack(received_weights).mean(0)
                self.chest_x_ray_model_params = new_weights
                aggregated_model = self.chest_x_ray_model_params
                print('Model weights for', TrainingType.CHEST_X_RAY_PNEUMONIA, 'updated in central model')
                
            elif training_type == TrainingType.BIOMEDICAL_REGRESSION:
                received_params = []
                for training_client in self.training_clients.values():
                    if training_client.status == ClientTrainingStatus.TRAINING_FINISHED:
                        training_client.status = ClientTrainingStatus.IDLE
                        if hasattr(training_client.model_params, 'get') and 'model_params' in training_client.model_params:
                            received_params.append(training_client.model_params['model_params'])
                            participant_models.append(training_client.model_params['model_params'])
                
                if received_params:
                    # Federated averaging for biomedical regression
                    aggregated_params = self._federated_averaging(received_params)
                    self.biomedical_model_params = aggregated_params
                    aggregated_model = aggregated_params
                    print('Model weights for', TrainingType.BIOMEDICAL_REGRESSION, 'updated in central model')
                    
            elif training_type == TrainingType.WEARABLE_ANALYTICS:
                received_params = []
                for training_client in self.training_clients.values():
                    if training_client.status == ClientTrainingStatus.TRAINING_FINISHED:
                        training_client.status = ClientTrainingStatus.IDLE
                        if hasattr(training_client.model_params, 'get') and 'model_params' in training_client.model_params:
                            received_params.append(training_client.model_params['model_params'])
                            participant_models.append(training_client.model_params['model_params'])
                
                if received_params:
                    aggregated_params = self._federated_averaging(received_params)
                    self.wearable_model_params = aggregated_params
                    aggregated_model = aggregated_params
                    print('Model weights for', TrainingType.WEARABLE_ANALYTICS, 'updated in central model')
                    
            elif training_type == TrainingType.MEDICAL_IMAGE_BIOMARKER:
                received_params = []
                for training_client in self.training_clients.values():
                    if training_client.status == ClientTrainingStatus.TRAINING_FINISHED:
                        training_client.status = ClientTrainingStatus.IDLE
                        if hasattr(training_client.model_params, 'get') and 'model_params' in training_client.model_params:
                            received_params.append(training_client.model_params['model_params'])
                            participant_models.append(training_client.model_params['model_params'])
                
                if received_params:
                    aggregated_params = self._federated_averaging(received_params)
                    self.medical_image_model_params = aggregated_params
                    aggregated_model = aggregated_params
                    print('Model weights for', TrainingType.MEDICAL_IMAGE_BIOMARKER, 'updated in central model')
            
            # Log model aggregation on blockchain
            if participant_models and aggregated_model:
                self._log_model_aggregation_on_blockchain(training_type, aggregated_model, participant_models)
            
            # Log training completion for active sessions
            for client_url, session_id in self.active_sessions.items():
                if session_id and client_url in self.training_metrics:
                    self._log_training_completion_on_blockchain(
                        session_id, aggregated_model, self.training_metrics[client_url]
                    )
            
            # Clear active sessions
            self.active_sessions.clear()
            self.training_metrics.clear()
            
            self.status = ServerStatus.IDLE
        sys.stdout.flush()
    
    def _federated_averaging(self, model_params_list):
        """Perform federated averaging on model parameters."""
        if not model_params_list:
            return None
        
        try:
            # For PyTorch tensors
            if hasattr(model_params_list[0][0], 'clone'):
                averaged_params = []
                for i in range(len(model_params_list[0])):
                    param_sum = model_params_list[0][i].clone()
                    for j in range(1, len(model_params_list)):
                        param_sum += model_params_list[j][i]
                    averaged_params.append(param_sum / len(model_params_list))
                return averaged_params
            else:
                # For numpy arrays or other numeric types
                return np.mean(model_params_list, axis=0)
        except Exception as e:
            logging.error(f"Error in federated averaging: {str(e)}")
            return model_params_list[0] if model_params_list else None

    def can_update_central_model_params(self):
        for training_client in self.training_clients.values():
            if training_client.status != ClientTrainingStatus.TRAINING_FINISHED \
                    and training_client.status != ClientTrainingStatus.TRAINING_REQUEST_ERROR:
                return False
        return True

    def register_client(self, client_url):
        print('Registering new training client [', client_url, ']')
        if self.training_clients.get(client_url) is None:
            next_client_id = len(self.training_clients) + 1
            self.training_clients[client_url] = TrainingClient(client_url, next_client_id)
        else:
            print('Client [', client_url, '] was already registered in the system')
            self.training_clients.get(client_url).status = ClientTrainingStatus.IDLE
        sys.stdout.flush()

    def unregister_client(self, client_url):
        print('Unregistering client [', client_url, ']')
        try:
            self.training_clients.pop(client_url)
            print('Client [', client_url, '] unregistered successfully')
        except KeyError:
            print('Client [', client_url, '] is not registered yet')
        sys.stdout.flush()

    def can_do_training(self):
        for training_client in self.training_clients.values():
            if training_client.status != ClientTrainingStatus.IDLE \
                    and training_client.status != ClientTrainingStatus.TRAINING_REQUEST_ERROR:
                return False

        return True

    def _log_training_start_on_blockchain(self, training_type: str, model_params: Any) -> Optional[str]:
        """Log training start event on blockchain."""
        if not self.blockchain_audit or not self.blockchain_audit.enabled:
            return None
        
        try:
            session_id = self.blockchain_audit.log_training_start(training_type, model_params)
            if session_id:
                logging.info(f"Training start logged on blockchain: {session_id}")
            return session_id
        except Exception as e:
            logging.error(f"Failed to log training start on blockchain: {str(e)}")
            return None
    
    def _log_training_completion_on_blockchain(self, session_id: str, model_params: Any, 
                                             metrics: Dict[str, Any]) -> bool:
        """Log training completion event on blockchain."""
        if not self.blockchain_audit or not self.blockchain_audit.enabled or not session_id:
            return False
        
        try:
            training_metrics = TrainingMetrics(
                loss=metrics.get('loss', 0.0),
                accuracy=metrics.get('accuracy', 0.0),
                r2_score=metrics.get('r2_score'),
                mae=metrics.get('mae'),
                training_time=metrics.get('training_time'),
                epochs_completed=metrics.get('epochs_completed')
            )
            
            success = self.blockchain_audit.log_training_completion(
                session_id, model_params, training_metrics
            )
            if success:
                logging.info(f"Training completion logged on blockchain: {session_id}")
            return success
        except Exception as e:
            logging.error(f"Failed to log training completion on blockchain: {str(e)}")
            return False
    
    def _log_model_aggregation_on_blockchain(self, training_type: str, 
                                           aggregated_params: Any,
                                           participant_params: List[Any]) -> bool:
        """Log model aggregation event on blockchain."""
        if not self.blockchain_audit or not self.blockchain_audit.enabled:
            return False
        
        try:
            # Calculate hashes of participant models
            participant_hashes = []
            for params in participant_params:
                param_hash = self.blockchain_audit._calculate_model_hash(params)
                participant_hashes.append(bytes.fromhex(param_hash))
            
            aggregated_hash = self.blockchain_audit._calculate_model_hash(aggregated_params)
            
            # This would call a contract method for model aggregation
            # For now, just log locally
            logging.info(f"Model aggregation for {training_type} - Round {self.round}")
            logging.info(f"Participants: {len(participant_params)}, Aggregated hash: {aggregated_hash}")
            return True
        except Exception as e:
            logging.error(f"Failed to log model aggregation on blockchain: {str(e)}")
            return False
    
    def register_client_with_audit(self, client_url: str, institution: str = "Unknown"):
        """Register client with blockchain audit logging."""
        # Register client normally
        self.register_client(client_url)
        
        # Log registration on blockchain
        if self.blockchain_audit and self.blockchain_audit.enabled:
            client_id = str(self.training_clients[client_url].client_id)
            try:
                success = self.blockchain_audit.register_client(client_id, institution)
                if success:
                    logging.info(f"Client registration logged on blockchain: {client_id}")
                else:
                    logging.warning(f"Failed to log client registration on blockchain: {client_id}")
            except Exception as e:
                logging.error(f"Error logging client registration: {str(e)}")
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get blockchain audit summary."""
        if not self.blockchain_audit:
            return {
                'blockchain_enabled': False,
                'message': 'Blockchain audit service not available'
            }
        
        audit_summary = self.blockchain_audit.get_audit_summary()
        
        # Add server-specific statistics
        audit_summary.update({
            'server_status': self.status.name if hasattr(self.status, 'name') else str(self.status),
            'current_round': self.round,
            'registered_clients': len(self.training_clients),
            'active_sessions': len(self.active_sessions),
            'supported_training_types': [
                'BIOMEDICAL_REGRESSION',
                'WEARABLE_ANALYTICS', 
                'MEDICAL_IMAGE_BIOMARKER',
                'CARDIOVASCULAR_RISK',
                'LAB_VALUE_PREDICTION',
                'GENOMIC_ANALYSIS',
                'EPIDEMIOLOGICAL_MODELING',
                'CLINICAL_OUTCOME_PREDICTION'
            ]
        })
        
        return audit_summary

