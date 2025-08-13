// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";

/**
 * @title FedBlockAudit
 * @dev Smart contract for auditing federated learning activities in biomedical research
 * Provides transparent, immutable logging of training activities, model updates, and data access
 */
contract FedBlockAudit is Ownable, ReentrancyGuard {
    using ECDSA for bytes32;

    // Events for different types of audit activities
    event ClientRegistered(
        address indexed clientAddress,
        string clientId,
        string institution,
        uint256 timestamp
    );

    event TrainingStarted(
        address indexed clientAddress,
        string clientId,
        string trainingType,
        bytes32 modelHash,
        uint256 round,
        uint256 timestamp
    );

    event TrainingCompleted(
        address indexed clientAddress,
        string clientId,
        string trainingType,
        bytes32 modelHash,
        bytes32 updatedModelHash,
        uint256 round,
        string performanceMetrics,
        uint256 timestamp
    );

    event ModelAggregated(
        bytes32 indexed aggregatedModelHash,
        bytes32[] participantModelHashes,
        uint256 round,
        uint256 participantCount,
        string aggregationMethod,
        uint256 timestamp
    );

    event DataAccessLogged(
        address indexed clientAddress,
        string clientId,
        string datasetType,
        bytes32 datasetHash,
        string accessType,
        uint256 timestamp
    );

    event PrivacyComplianceChecked(
        address indexed clientAddress,
        string clientId,
        string complianceType,
        bool passed,
        string details,
        uint256 timestamp
    );

    event AnomalyDetected(
        address indexed clientAddress,
        string clientId,
        string anomalyType,
        string severity,
        string description,
        uint256 timestamp
    );

    // Structs for storing audit data
    struct Client {
        address clientAddress;
        string clientId;
        string institution;
        bool isActive;
        uint256 registrationTime;
        uint256 lastActivity;
    }

    struct TrainingSession {
        address clientAddress;
        string clientId;
        string trainingType;
        bytes32 initialModelHash;
        bytes32 finalModelHash;
        uint256 round;
        uint256 startTime;
        uint256 endTime;
        string performanceMetrics;
        bool completed;
    }

    struct ModelVersion {
        bytes32 modelHash;
        uint256 round;
        address[] contributors;
        string aggregationMethod;
        uint256 timestamp;
        string metadata;
    }

    // Storage mappings
    mapping(address => Client) public clients;
    mapping(string => address) public clientIdToAddress;
    mapping(bytes32 => TrainingSession) public trainingSessions;
    mapping(uint256 => ModelVersion) public modelVersions;
    mapping(address => uint256) public clientTrainingCount;
    mapping(string => uint256) public datasetAccessCount;

    // Arrays for enumeration
    address[] public clientAddresses;
    bytes32[] public trainingSessionIds;
    uint256[] public rounds;

    // Contract state
    uint256 public currentRound;
    uint256 public totalTrainingSessions;
    uint256 public totalClients;
    bool public contractPaused;

    // Modifiers
    modifier onlyRegisteredClient() {
        require(clients[msg.sender].isActive, "Client not registered or inactive");
        _;
    }

    modifier whenNotPaused() {
        require(!contractPaused, "Contract is paused");
        _;
    }

    modifier validTrainingType(string memory trainingType) {
        bytes32 typeHash = keccak256(abi.encodePacked(trainingType));
        require(
            typeHash == keccak256("BIOMEDICAL_REGRESSION") ||
            typeHash == keccak256("WEARABLE_ANALYTICS") ||
            typeHash == keccak256("MEDICAL_IMAGE_BIOMARKER") ||
            typeHash == keccak256("CARDIOVASCULAR_RISK") ||
            typeHash == keccak256("LAB_VALUE_PREDICTION") ||
            typeHash == keccak256("GENOMIC_ANALYSIS") ||
            typeHash == keccak256("EPIDEMIOLOGICAL_MODELING") ||
            typeHash == keccak256("CLINICAL_OUTCOME_PREDICTION"),
            "Invalid training type"
        );
        _;
    }

    constructor() {
        currentRound = 1;
        totalTrainingSessions = 0;
        totalClients = 0;
        contractPaused = false;
    }

    /**
     * @dev Register a new client for federated learning
     * @param clientId Unique identifier for the client
     * @param institution Name of the institution
     */
    function registerClient(
        string memory clientId,
        string memory institution
    ) external whenNotPaused {
        require(bytes(clientId).length > 0, "Client ID cannot be empty");
        require(bytes(institution).length > 0, "Institution cannot be empty");
        require(clientIdToAddress[clientId] == address(0), "Client ID already exists");
        require(!clients[msg.sender].isActive, "Client already registered");

        clients[msg.sender] = Client({
            clientAddress: msg.sender,
            clientId: clientId,
            institution: institution,
            isActive: true,
            registrationTime: block.timestamp,
            lastActivity: block.timestamp
        });

        clientIdToAddress[clientId] = msg.sender;
        clientAddresses.push(msg.sender);
        totalClients++;

        emit ClientRegistered(msg.sender, clientId, institution, block.timestamp);
    }

    /**
     * @dev Log the start of a training session
     * @param trainingType Type of biomedical training being performed
     * @param modelHash Hash of the initial model
     */
    function logTrainingStart(
        string memory trainingType,
        bytes32 modelHash
    ) external onlyRegisteredClient whenNotPaused validTrainingType(trainingType) {
        bytes32 sessionId = keccak256(abi.encodePacked(
            msg.sender,
            trainingType,
            currentRound,
            block.timestamp
        ));

        trainingSessions[sessionId] = TrainingSession({
            clientAddress: msg.sender,
            clientId: clients[msg.sender].clientId,
            trainingType: trainingType,
            initialModelHash: modelHash,
            finalModelHash: bytes32(0),
            round: currentRound,
            startTime: block.timestamp,
            endTime: 0,
            performanceMetrics: "",
            completed: false
        });

        trainingSessionIds.push(sessionId);
        clients[msg.sender].lastActivity = block.timestamp;

        emit TrainingStarted(
            msg.sender,
            clients[msg.sender].clientId,
            trainingType,
            modelHash,
            currentRound,
            block.timestamp
        );
    }

    /**
     * @dev Log the completion of a training session
     * @param sessionId ID of the training session
     * @param updatedModelHash Hash of the updated model
     * @param performanceMetrics JSON string containing training metrics
     */
    function logTrainingCompletion(
        bytes32 sessionId,
        bytes32 updatedModelHash,
        string memory performanceMetrics
    ) external onlyRegisteredClient whenNotPaused {
        require(trainingSessions[sessionId].clientAddress == msg.sender, "Not authorized for this session");
        require(!trainingSessions[sessionId].completed, "Session already completed");

        trainingSessions[sessionId].finalModelHash = updatedModelHash;
        trainingSessions[sessionId].endTime = block.timestamp;
        trainingSessions[sessionId].performanceMetrics = performanceMetrics;
        trainingSessions[sessionId].completed = true;

        clientTrainingCount[msg.sender]++;
        totalTrainingSessions++;
        clients[msg.sender].lastActivity = block.timestamp;

        emit TrainingCompleted(
            msg.sender,
            clients[msg.sender].clientId,
            trainingSessions[sessionId].trainingType,
            trainingSessions[sessionId].initialModelHash,
            updatedModelHash,
            currentRound,
            performanceMetrics,
            block.timestamp
        );
    }

    /**
     * @dev Log model aggregation event
     * @param aggregatedModelHash Hash of the aggregated model
     * @param participantModelHashes Array of participant model hashes
     * @param aggregationMethod Method used for aggregation
     * @param metadata Additional metadata about the aggregation
     */
    function logModelAggregation(
        bytes32 aggregatedModelHash,
        bytes32[] memory participantModelHashes,
        string memory aggregationMethod,
        string memory metadata
    ) external onlyOwner whenNotPaused {
        require(participantModelHashes.length > 0, "No participant models");

        // Create contributor addresses array (simplified - in practice would map from model hashes)
        address[] memory contributors = new address[](participantModelHashes.length);
        for (uint i = 0; i < participantModelHashes.length; i++) {
            contributors[i] = clientAddresses[i % clientAddresses.length]; // Simplified mapping
        }

        modelVersions[currentRound] = ModelVersion({
            modelHash: aggregatedModelHash,
            round: currentRound,
            contributors: contributors,
            aggregationMethod: aggregationMethod,
            timestamp: block.timestamp,
            metadata: metadata
        });

        rounds.push(currentRound);

        emit ModelAggregated(
            aggregatedModelHash,
            participantModelHashes,
            currentRound,
            participantModelHashes.length,
            aggregationMethod,
            block.timestamp
        );

        currentRound++;
    }

    /**
     * @dev Log data access for audit purposes
     * @param datasetType Type of biomedical dataset accessed
     * @param datasetHash Hash of the dataset
     * @param accessType Type of access (read, write, delete, etc.)
     */
    function logDataAccess(
        string memory datasetType,
        bytes32 datasetHash,
        string memory accessType
    ) external onlyRegisteredClient whenNotPaused {
        require(bytes(datasetType).length > 0, "Dataset type cannot be empty");
        require(bytes(accessType).length > 0, "Access type cannot be empty");

        datasetAccessCount[datasetType]++;
        clients[msg.sender].lastActivity = block.timestamp;

        emit DataAccessLogged(
            msg.sender,
            clients[msg.sender].clientId,
            datasetType,
            datasetHash,
            accessType,
            block.timestamp
        );
    }

    /**
     * @dev Log privacy compliance check results
     * @param complianceType Type of compliance check (HIPAA, GDPR, etc.)
     * @param passed Whether the compliance check passed
     * @param details Additional details about the compliance check
     */
    function logPrivacyCompliance(
        string memory complianceType,
        bool passed,
        string memory details
    ) external onlyRegisteredClient whenNotPaused {
        require(bytes(complianceType).length > 0, "Compliance type cannot be empty");

        clients[msg.sender].lastActivity = block.timestamp;

        emit PrivacyComplianceChecked(
            msg.sender,
            clients[msg.sender].clientId,
            complianceType,
            passed,
            details,
            block.timestamp
        );
    }

    /**
     * @dev Log detected anomalies in training or data
     * @param anomalyType Type of anomaly detected
     * @param severity Severity level of the anomaly
     * @param description Description of the anomaly
     */
    function logAnomaly(
        string memory anomalyType,
        string memory severity,
        string memory description
    ) external onlyRegisteredClient whenNotPaused {
        require(bytes(anomalyType).length > 0, "Anomaly type cannot be empty");

        clients[msg.sender].lastActivity = block.timestamp;

        emit AnomalyDetected(
            msg.sender,
            clients[msg.sender].clientId,
            anomalyType,
            severity,
            description,
            block.timestamp
        );
    }

    // View functions for querying audit data

    /**
     * @dev Get client information by address
     */
    function getClient(address clientAddress) external view returns (Client memory) {
        return clients[clientAddress];
    }

    /**
     * @dev Get training session information
     */
    function getTrainingSession(bytes32 sessionId) external view returns (TrainingSession memory) {
        return trainingSessions[sessionId];
    }

    /**
     * @dev Get model version information
     */
    function getModelVersion(uint256 round) external view returns (ModelVersion memory) {
        return modelVersions[round];
    }

    /**
     * @dev Get total number of training sessions for a client
     */
    function getClientTrainingCount(address clientAddress) external view returns (uint256) {
        return clientTrainingCount[clientAddress];
    }

    /**
     * @dev Get access count for a dataset type
     */
    function getDatasetAccessCount(string memory datasetType) external view returns (uint256) {
        return datasetAccessCount[datasetType];
    }

    /**
     * @dev Get all client addresses
     */
    function getAllClients() external view returns (address[] memory) {
        return clientAddresses;
    }

    /**
     * @dev Get all training session IDs
     */
    function getAllTrainingSessions() external view returns (bytes32[] memory) {
        return trainingSessionIds;
    }

    /**
     * @dev Get all rounds
     */
    function getAllRounds() external view returns (uint256[] memory) {
        return rounds;
    }

    // Admin functions

    /**
     * @dev Deactivate a client (only owner)
     */
    function deactivateClient(address clientAddress) external onlyOwner {
        require(clients[clientAddress].isActive, "Client not active");
        clients[clientAddress].isActive = false;
    }

    /**
     * @dev Reactivate a client (only owner)
     */
    function reactivateClient(address clientAddress) external onlyOwner {
        require(!clients[clientAddress].isActive, "Client already active");
        clients[clientAddress].isActive = true;
    }

    /**
     * @dev Pause contract (only owner)
     */
    function pauseContract() external onlyOwner {
        contractPaused = true;
    }

    /**
     * @dev Unpause contract (only owner)
     */
    function unpauseContract() external onlyOwner {
        contractPaused = false;
    }

    /**
     * @dev Emergency function to update current round (only owner)
     */
    function setCurrentRound(uint256 newRound) external onlyOwner {
        require(newRound > 0, "Round must be positive");
        currentRound = newRound;
    }
}
