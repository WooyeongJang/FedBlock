"""
Blockchain audit service for FedBlock biomedical federated learning.
Provides interface to interact with the FedBlockAudit smart contract.
"""

import json
import logging
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import os

try:
    from web3 import Web3
    from web3.contract import Contract
    from eth_account import Account
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    print("Warning: web3 not installed. Blockchain features will be disabled.")


@dataclass
class TrainingMetrics:
    """Data class for training performance metrics."""
    loss: float
    accuracy: float
    r2_score: Optional[float] = None
    mae: Optional[float] = None
    training_time: Optional[float] = None
    epochs_completed: Optional[int] = None
    
    def to_json(self) -> str:
        """Convert metrics to JSON string for blockchain storage."""
        return json.dumps(asdict(self))


@dataclass
class ClientInfo:
    """Data class for client information."""
    client_id: str
    institution: str
    address: str
    is_active: bool
    registration_time: datetime
    last_activity: datetime


class BlockchainAuditService:
    """
    Service for interacting with the FedBlockAudit smart contract.
    Provides methods for logging federated learning activities on the blockchain.
    """
    
    def __init__(self, web3_provider_url: str = None, 
                 contract_address: str = None,
                 private_key: str = None):
        """
        Initialize the blockchain audit service.
        
        Args:
            web3_provider_url: URL of the Web3 provider (e.g., Infura, local node)
            contract_address: Address of the deployed FedBlockAudit contract
            private_key: Private key for signing transactions
        """
        self.logger = logging.getLogger(__name__)
        
        if not WEB3_AVAILABLE:
            self.logger.warning("Web3 not available. Blockchain audit will be disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        
        # Initialize Web3 connection
        self.web3_provider_url = web3_provider_url or os.getenv('WEB3_PROVIDER_URL', 'http://localhost:8545')
        self.contract_address = contract_address or os.getenv('FEDBLOCK_CONTRACT_ADDRESS')
        self.private_key = private_key or os.getenv('FEDBLOCK_PRIVATE_KEY')
        
        try:
            self.w3 = Web3(Web3.HTTPProvider(self.web3_provider_url))
            self.account = Account.from_key(self.private_key) if self.private_key else None
            
            # Load contract ABI and initialize contract
            self.contract = self._load_contract()
            
            if self.w3.is_connected():
                self.logger.info(f"Connected to blockchain at {self.web3_provider_url}")
                self.logger.info(f"Using account: {self.account.address if self.account else 'None'}")
            else:
                self.logger.error("Failed to connect to blockchain")
                self.enabled = False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize blockchain service: {str(e)}")
            self.enabled = False
    
    def _load_contract(self) -> Optional[Contract]:
        """Load the FedBlockAudit smart contract."""
        if not self.contract_address:
            self.logger.warning("Contract address not provided")
            return None
        
        # Contract ABI (simplified - in production, load from JSON file)
        contract_abi = [
            {
                "inputs": [
                    {"internalType": "string", "name": "clientId", "type": "string"},
                    {"internalType": "string", "name": "institution", "type": "string"}
                ],
                "name": "registerClient",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "string", "name": "trainingType", "type": "string"},
                    {"internalType": "bytes32", "name": "modelHash", "type": "bytes32"}
                ],
                "name": "logTrainingStart",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "bytes32", "name": "sessionId", "type": "bytes32"},
                    {"internalType": "bytes32", "name": "updatedModelHash", "type": "bytes32"},
                    {"internalType": "string", "name": "performanceMetrics", "type": "string"}
                ],
                "name": "logTrainingCompletion",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "string", "name": "datasetType", "type": "string"},
                    {"internalType": "bytes32", "name": "datasetHash", "type": "bytes32"},
                    {"internalType": "string", "name": "accessType", "type": "string"}
                ],
                "name": "logDataAccess",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "string", "name": "complianceType", "type": "string"},
                    {"internalType": "bool", "name": "passed", "type": "bool"},
                    {"internalType": "string", "name": "details", "type": "string"}
                ],
                "name": "logPrivacyCompliance",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
        
        try:
            contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(self.contract_address),
                abi=contract_abi
            )
            return contract
        except Exception as e:
            self.logger.error(f"Failed to load contract: {str(e)}")
            return None
    
    def _calculate_model_hash(self, model_params: Any) -> str:
        """Calculate hash of model parameters for blockchain logging."""
        if isinstance(model_params, dict):
            # Convert dict to deterministic string
            model_str = json.dumps(model_params, sort_keys=True)
        elif hasattr(model_params, '__iter__'):
            # Handle list/tuple of tensors or parameters
            model_str = str([str(param) for param in model_params])
        else:
            model_str = str(model_params)
        
        return hashlib.sha256(model_str.encode()).hexdigest()
    
    def _send_transaction(self, function_call, gas_limit: int = 300000) -> Optional[str]:
        """Send a transaction to the blockchain."""
        if not self.enabled or not self.account:
            return None
        
        try:
            # Build transaction
            transaction = function_call.build_transaction({
                'from': self.account.address,
                'gas': gas_limit,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
            })
            
            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                self.logger.info(f"Transaction successful: {tx_hash.hex()}")
                return tx_hash.hex()
            else:
                self.logger.error(f"Transaction failed: {tx_hash.hex()}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to send transaction: {str(e)}")
            return None
    
    def register_client(self, client_id: str, institution: str) -> bool:
        """
        Register a new client on the blockchain.
        
        Args:
            client_id: Unique identifier for the client
            institution: Name of the institution
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.contract:
            self.logger.warning("Blockchain not enabled, skipping client registration")
            return False
        
        try:
            function_call = self.contract.functions.registerClient(client_id, institution)
            tx_hash = self._send_transaction(function_call)
            
            if tx_hash:
                self.logger.info(f"Client {client_id} registered successfully")
                return True
            else:
                self.logger.error(f"Failed to register client {client_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error registering client: {str(e)}")
            return False
    
    def log_training_start(self, training_type: str, model_params: Any) -> Optional[str]:
        """
        Log the start of a training session.
        
        Args:
            training_type: Type of biomedical training
            model_params: Initial model parameters
            
        Returns:
            Session ID if successful, None otherwise
        """
        if not self.enabled or not self.contract:
            self.logger.warning("Blockchain not enabled, skipping training start log")
            return None
        
        try:
            model_hash = self._calculate_model_hash(model_params)
            model_hash_bytes = Web3.keccak(text=model_hash)
            
            function_call = self.contract.functions.logTrainingStart(training_type, model_hash_bytes)
            tx_hash = self._send_transaction(function_call)
            
            if tx_hash:
                # Generate session ID (simplified - in practice, would be derived from contract event)
                session_id = Web3.keccak(text=f"{self.account.address}{training_type}{model_hash}{datetime.now().timestamp()}")
                self.logger.info(f"Training start logged for type {training_type}")
                return session_id.hex()
            else:
                self.logger.error(f"Failed to log training start for type {training_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error logging training start: {str(e)}")
            return None
    
    def log_training_completion(self, session_id: str, updated_model_params: Any, 
                              metrics: TrainingMetrics) -> bool:
        """
        Log the completion of a training session.
        
        Args:
            session_id: ID of the training session
            updated_model_params: Updated model parameters
            metrics: Training performance metrics
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.contract:
            self.logger.warning("Blockchain not enabled, skipping training completion log")
            return False
        
        try:
            updated_model_hash = self._calculate_model_hash(updated_model_params)
            updated_model_hash_bytes = Web3.keccak(text=updated_model_hash)
            session_id_bytes = bytes.fromhex(session_id.replace('0x', ''))
            
            function_call = self.contract.functions.logTrainingCompletion(
                session_id_bytes,
                updated_model_hash_bytes,
                metrics.to_json()
            )
            tx_hash = self._send_transaction(function_call)
            
            if tx_hash:
                self.logger.info(f"Training completion logged for session {session_id}")
                return True
            else:
                self.logger.error(f"Failed to log training completion for session {session_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error logging training completion: {str(e)}")
            return False
    
    def log_data_access(self, dataset_type: str, dataset_hash: str, access_type: str) -> bool:
        """
        Log data access for audit purposes.
        
        Args:
            dataset_type: Type of biomedical dataset
            dataset_hash: Hash of the dataset
            access_type: Type of access (read, write, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.contract:
            self.logger.warning("Blockchain not enabled, skipping data access log")
            return False
        
        try:
            dataset_hash_bytes = Web3.keccak(text=dataset_hash)
            
            function_call = self.contract.functions.logDataAccess(
                dataset_type,
                dataset_hash_bytes,
                access_type
            )
            tx_hash = self._send_transaction(function_call)
            
            if tx_hash:
                self.logger.info(f"Data access logged for {dataset_type}")
                return True
            else:
                self.logger.error(f"Failed to log data access for {dataset_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error logging data access: {str(e)}")
            return False
    
    def log_privacy_compliance(self, compliance_type: str, passed: bool, details: str) -> bool:
        """
        Log privacy compliance check results.
        
        Args:
            compliance_type: Type of compliance check (HIPAA, GDPR, etc.)
            passed: Whether the compliance check passed
            details: Additional details about the check
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.contract:
            self.logger.warning("Blockchain not enabled, skipping privacy compliance log")
            return False
        
        try:
            function_call = self.contract.functions.logPrivacyCompliance(
                compliance_type,
                passed,
                details
            )
            tx_hash = self._send_transaction(function_call)
            
            if tx_hash:
                self.logger.info(f"Privacy compliance logged: {compliance_type} - {'PASSED' if passed else 'FAILED'}")
                return True
            else:
                self.logger.error(f"Failed to log privacy compliance for {compliance_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error logging privacy compliance: {str(e)}")
            return False
    
    def get_client_training_history(self, client_address: str) -> List[Dict]:
        """
        Get training history for a specific client from blockchain events.
        This is a simplified implementation - in practice would parse contract events.
        
        Args:
            client_address: Address of the client
            
        Returns:
            List of training session records
        """
        if not self.enabled:
            return []
        
        # In a real implementation, this would:
        # 1. Query contract events for the client address
        # 2. Parse and return the training history
        # For now, return empty list
        self.logger.info(f"Training history requested for {client_address}")
        return []
    
    def verify_model_integrity(self, model_params: Any, claimed_hash: str) -> bool:
        """
        Verify the integrity of model parameters against a claimed hash.
        
        Args:
            model_params: Model parameters to verify
            claimed_hash: Claimed hash of the model
            
        Returns:
            True if hash matches, False otherwise
        """
        calculated_hash = self._calculate_model_hash(model_params)
        return calculated_hash == claimed_hash.replace('0x', '')
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """
        Get a summary of audit activities.
        
        Returns:
            Dictionary containing audit statistics
        """
        if not self.enabled or not self.contract:
            return {
                'enabled': False,
                'message': 'Blockchain audit not available'
            }
        
        try:
            # In a real implementation, this would query contract state
            return {
                'enabled': True,
                'contract_address': self.contract_address,
                'current_round': 0,  # Would query from contract
                'total_clients': 0,  # Would query from contract
                'total_training_sessions': 0,  # Would query from contract
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting audit summary: {str(e)}")
            return {
                'enabled': False,
                'error': str(e)
            }


# Singleton instance
blockchain_audit_service = None

def get_blockchain_audit_service(**kwargs) -> BlockchainAuditService:
    """Get the global blockchain audit service instance."""
    global blockchain_audit_service
    if blockchain_audit_service is None:
        blockchain_audit_service = BlockchainAuditService(**kwargs)
    return blockchain_audit_service
