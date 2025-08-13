// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "hardhat/console.sol";

/**
 * @title FederatedLearningAudit
 * @dev Smart contract for auditing federated learning activities
 * @notice This contract provides immutable logging for all federated learning events
 */
contract FederatedLearningAudit {
    
    // Event definitions for different audit activities
    event ClientRegistered(
        address indexed clientAddress,
        string clientUrl,
        uint256 timestamp,
        bytes32 sessionId
    );
    
    event ClientUnregistered(
        address indexed clientAddress,
        string clientUrl,
        uint256 timestamp,
        bytes32 sessionId
    );
    
    event TrainingRoundStarted(
        bytes32 indexed roundId,
        string trainingType,
        uint256 roundNumber,
        uint256 participantCount,
        address initiator,
        uint256 timestamp
    );
    
    event TrainingRoundCompleted(
        bytes32 indexed roundId,
        uint256 roundNumber,
        uint256 successfulParticipants,
        uint256 timestamp
    );
    
    event ModelParametersUpdated(
        address indexed clientAddress,
        bytes32 indexed roundId,
        bytes32 parametersHash,
        uint256 timestamp,
        string trainingType
    );
    
    event DataAccessLogged(
        address indexed clientAddress,
        string datasetType,
        bytes32 dataHash,
        uint256 timestamp,
        string accessType
    );
    
    event AuditTrailVerified(
        bytes32 indexed auditId,
        address verifier,
        bool isValid,
        uint256 timestamp
    );

    // Struct definitions
    struct ClientRecord {
        address clientAddress;
        string clientUrl;
        uint256 registrationTime;
        uint256 lastActivity;
        bool isActive;
        uint256 totalRoundsParticipated;
    }
    
    struct TrainingRound {
        bytes32 roundId;
        string trainingType;
        uint256 roundNumber;
        uint256 startTime;
        uint256 endTime;
        address[] participants;
        mapping(address => bytes32) participantParametersHash;
        bool isCompleted;
    }
    
    struct AuditRecord {
        bytes32 auditId;
        address entity;
        string eventType;
        bytes32 dataHash;
        uint256 timestamp;
        bytes32 blockHash;
    }

    // State variables
    mapping(address => ClientRecord) public clients;
    mapping(bytes32 => TrainingRound) public trainingRounds;
    mapping(bytes32 => AuditRecord) public auditRecords;
    
    address[] public registeredClients;
    bytes32[] public auditTrail;
    
    address public owner;
    uint256 public totalRounds;
    uint256 public totalAuditRecords;
    
    // Access control
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can perform this action");
        _;
    }
    
    modifier onlyRegisteredClient() {
        require(clients[msg.sender].isActive, "Only registered clients can perform this action");
        _;
    }

    constructor() {
        owner = msg.sender;
        totalRounds = 0;
        totalAuditRecords = 0;
    }

    /**
     * @dev Register a new client in the federated learning network
     * @param clientUrl The URL endpoint of the client
     * @param sessionId Unique session identifier
     */
    function registerClient(string memory clientUrl, bytes32 sessionId) external {
        require(!clients[msg.sender].isActive, "Client already registered");
        
        clients[msg.sender] = ClientRecord({
            clientAddress: msg.sender,
            clientUrl: clientUrl,
            registrationTime: block.timestamp,
            lastActivity: block.timestamp,
            isActive: true,
            totalRoundsParticipated: 0
        });
        
        registeredClients.push(msg.sender);
        
        _createAuditRecord(
            keccak256(abi.encodePacked("CLIENT_REGISTRATION", msg.sender, block.timestamp)),
            msg.sender,
            "CLIENT_REGISTRATION",
            keccak256(abi.encodePacked(clientUrl))
        );
        
        emit ClientRegistered(msg.sender, clientUrl, block.timestamp, sessionId);
    }

    /**
     * @dev Unregister a client from the federated learning network
     * @param sessionId Session identifier for verification
     */
    function unregisterClient(bytes32 sessionId) external onlyRegisteredClient {
        clients[msg.sender].isActive = false;
        
        _createAuditRecord(
            keccak256(abi.encodePacked("CLIENT_UNREGISTRATION", msg.sender, block.timestamp)),
            msg.sender,
            "CLIENT_UNREGISTRATION",
            sessionId
        );
        
        emit ClientUnregistered(
            msg.sender, 
            clients[msg.sender].clientUrl, 
            block.timestamp, 
            sessionId
        );
    }

    /**
     * @dev Start a new training round
     * @param trainingType Type of training (MNIST, GOSSIP_MNIST, etc.)
     * @param participants Array of participant addresses
     */
    function startTrainingRound(
        string memory trainingType,
        address[] memory participants
    ) external onlyOwner returns (bytes32) {
        totalRounds++;
        bytes32 roundId = keccak256(abi.encodePacked(
            "ROUND", 
            totalRounds, 
            block.timestamp, 
            trainingType
        ));
        
        TrainingRound storage newRound = trainingRounds[roundId];
        newRound.roundId = roundId;
        newRound.trainingType = trainingType;
        newRound.roundNumber = totalRounds;
        newRound.startTime = block.timestamp;
        newRound.participants = participants;
        newRound.isCompleted = false;
        
        // Update participant records
        for (uint256 i = 0; i < participants.length; i++) {
            if (clients[participants[i]].isActive) {
                clients[participants[i]].totalRoundsParticipated++;
                clients[participants[i]].lastActivity = block.timestamp;
            }
        }
        
        _createAuditRecord(
            roundId,
            msg.sender,
            "TRAINING_ROUND_STARTED",
            keccak256(abi.encodePacked(trainingType, participants.length))
        );
        
        emit TrainingRoundStarted(
            roundId,
            trainingType,
            totalRounds,
            participants.length,
            msg.sender,
            block.timestamp
        );
        
        return roundId;
    }

    /**
     * @dev Log model parameter updates from a client
     * @param roundId The training round identifier
     * @param parametersHash Hash of the model parameters
     * @param trainingType Type of training being performed
     */
    function logModelParametersUpdate(
        bytes32 roundId,
        bytes32 parametersHash,
        string memory trainingType
    ) external onlyRegisteredClient {
        require(trainingRounds[roundId].roundId != 0, "Invalid round ID");
        require(!trainingRounds[roundId].isCompleted, "Training round already completed");
        
        trainingRounds[roundId].participantParametersHash[msg.sender] = parametersHash;
        clients[msg.sender].lastActivity = block.timestamp;
        
        _createAuditRecord(
            keccak256(abi.encodePacked("MODEL_UPDATE", roundId, msg.sender)),
            msg.sender,
            "MODEL_PARAMETERS_UPDATED",
            parametersHash
        );
        
        emit ModelParametersUpdated(
            msg.sender,
            roundId,
            parametersHash,
            block.timestamp,
            trainingType
        );
    }

    /**
     * @dev Complete a training round
     * @param roundId The training round identifier
     */
    function completeTrainingRound(bytes32 roundId) external onlyOwner {
        require(trainingRounds[roundId].roundId != 0, "Invalid round ID");
        require(!trainingRounds[roundId].isCompleted, "Training round already completed");
        
        trainingRounds[roundId].endTime = block.timestamp;
        trainingRounds[roundId].isCompleted = true;
        
        // Count successful participants
        uint256 successfulParticipants = 0;
        for (uint256 i = 0; i < trainingRounds[roundId].participants.length; i++) {
            address participant = trainingRounds[roundId].participants[i];
            if (trainingRounds[roundId].participantParametersHash[participant] != 0) {
                successfulParticipants++;
            }
        }
        
        _createAuditRecord(
            roundId,
            msg.sender,
            "TRAINING_ROUND_COMPLETED",
            keccak256(abi.encodePacked(successfulParticipants, block.timestamp))
        );
        
        emit TrainingRoundCompleted(
            roundId,
            trainingRounds[roundId].roundNumber,
            successfulParticipants,
            block.timestamp
        );
    }

    /**
     * @dev Log data access events
     * @param datasetType Type of dataset being accessed
     * @param dataHash Hash of the data being accessed
     * @param accessType Type of access (READ, WRITE, TRAIN)
     */
    function logDataAccess(
        string memory datasetType,
        bytes32 dataHash,
        string memory accessType
    ) external onlyRegisteredClient {
        clients[msg.sender].lastActivity = block.timestamp;
        
        _createAuditRecord(
            keccak256(abi.encodePacked("DATA_ACCESS", msg.sender, block.timestamp)),
            msg.sender,
            "DATA_ACCESS",
            dataHash
        );
        
        emit DataAccessLogged(
            msg.sender,
            datasetType,
            dataHash,
            block.timestamp,
            accessType
        );
    }

    /**
     * @dev Internal function to create audit records
     * @param auditId Unique identifier for the audit record
     * @param entity Address of the entity performing the action
     * @param eventType Type of event being audited
     * @param dataHash Hash of relevant data
     */
    function _createAuditRecord(
        bytes32 auditId,
        address entity,
        string memory eventType,
        bytes32 dataHash
    ) internal {
        auditRecords[auditId] = AuditRecord({
            auditId: auditId,
            entity: entity,
            eventType: eventType,
            dataHash: dataHash,
            timestamp: block.timestamp,
            blockHash: blockhash(block.number - 1)
        });
        
        auditTrail.push(auditId);
        totalAuditRecords++;
    }

    /**
     * @dev Verify the integrity of the audit trail
     * @param auditId The audit record to verify
     */
    function verifyAuditTrail(bytes32 auditId) external returns (bool) {
        AuditRecord memory record = auditRecords[auditId];
        require(record.auditId != 0, "Audit record not found");
        
        bool isValid = (record.timestamp <= block.timestamp) && 
                      (record.entity != address(0));
        
        emit AuditTrailVerified(auditId, msg.sender, isValid, block.timestamp);
        
        return isValid;
    }

    // View functions for retrieving audit information
    
    /**
     * @dev Get client information
     * @param clientAddress Address of the client
     */
    function getClientInfo(address clientAddress) 
        external 
        view 
        returns (ClientRecord memory) 
    {
        return clients[clientAddress];
    }

    /**
     * @dev Get training round information
     * @param roundId Training round identifier
     */
    function getTrainingRoundInfo(bytes32 roundId) 
        external 
        view 
        returns (
            bytes32,
            string memory,
            uint256,
            uint256,
            uint256,
            address[] memory,
            bool
        ) 
    {
        TrainingRound storage round = trainingRounds[roundId];
        return (
            round.roundId,
            round.trainingType,
            round.roundNumber,
            round.startTime,
            round.endTime,
            round.participants,
            round.isCompleted
        );
    }

    /**
     * @dev Get audit record
     * @param auditId Audit record identifier
     */
    function getAuditRecord(bytes32 auditId) 
        external 
        view 
        returns (AuditRecord memory) 
    {
        return auditRecords[auditId];
    }

    /**
     * @dev Get total number of registered clients
     */
    function getTotalRegisteredClients() external view returns (uint256) {
        return registeredClients.length;
    }

    /**
     * @dev Get audit trail summary
     */
    function getAuditTrailSummary() 
        external 
        view 
        returns (uint256 totalRecords, uint256 totalRoundsCompleted) 
    {
        return (totalAuditRecords, totalRounds);
    }
}
