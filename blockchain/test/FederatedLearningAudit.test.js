const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("FederatedLearningAudit", function () {
  let federatedLearningAudit;
  let owner;
  let client1;
  let client2;
  let client3;

  beforeEach(async function () {
    // Get test accounts
    [owner, client1, client2, client3] = await ethers.getSigners();

    // Deploy the contract
    const FederatedLearningAudit = await ethers.getContractFactory("FederatedLearningAudit");
    federatedLearningAudit = await FederatedLearningAudit.deploy();
    await federatedLearningAudit.waitForDeployment();
  });

  describe("Deployment", function () {
    it("Should set the right owner", async function () {
      expect(await federatedLearningAudit.owner()).to.equal(owner.address);
    });

    it("Should initialize with zero rounds and records", async function () {
      expect(await federatedLearningAudit.totalRounds()).to.equal(0);
      expect(await federatedLearningAudit.totalAuditRecords()).to.equal(0);
    });
  });

  describe("Client Registration", function () {
    it("Should register a new client", async function () {
      const clientUrl = "http://127.0.0.1:5001";
      const sessionId = ethers.keccak256(ethers.toUtf8Bytes("session1"));

      await expect(federatedLearningAudit.connect(client1).registerClient(clientUrl, sessionId))
        .to.emit(federatedLearningAudit, "ClientRegistered")
        .withArgs(client1.address, clientUrl, await time.latest() + 1, sessionId);

      const clientInfo = await federatedLearningAudit.getClientInfo(client1.address);
      expect(clientInfo.clientAddress).to.equal(client1.address);
      expect(clientInfo.clientUrl).to.equal(clientUrl);
      expect(clientInfo.isActive).to.be.true;
    });

    it("Should not allow double registration", async function () {
      const clientUrl = "http://127.0.0.1:5001";
      const sessionId = ethers.keccak256(ethers.toUtf8Bytes("session1"));

      await federatedLearningAudit.connect(client1).registerClient(clientUrl, sessionId);
      
      await expect(
        federatedLearningAudit.connect(client1).registerClient(clientUrl, sessionId)
      ).to.be.revertedWith("Client already registered");
    });

    it("Should unregister a client", async function () {
      const clientUrl = "http://127.0.0.1:5001";
      const sessionId = ethers.keccak256(ethers.toUtf8Bytes("session1"));

      await federatedLearningAudit.connect(client1).registerClient(clientUrl, sessionId);
      
      await expect(federatedLearningAudit.connect(client1).unregisterClient(sessionId))
        .to.emit(federatedLearningAudit, "ClientUnregistered")
        .withArgs(client1.address, clientUrl, await time.latest() + 1, sessionId);

      const clientInfo = await federatedLearningAudit.getClientInfo(client1.address);
      expect(clientInfo.isActive).to.be.false;
    });
  });

  describe("Training Rounds", function () {
    beforeEach(async function () {
      // Register clients for testing
      await federatedLearningAudit.connect(client1).registerClient(
        "http://127.0.0.1:5001", 
        ethers.keccak256(ethers.toUtf8Bytes("session1"))
      );
      await federatedLearningAudit.connect(client2).registerClient(
        "http://127.0.0.1:5002", 
        ethers.keccak256(ethers.toUtf8Bytes("session2"))
      );
    });

    it("Should start a training round", async function () {
      const trainingType = "MNIST";
      const participants = [client1.address, client2.address];

      await expect(federatedLearningAudit.startTrainingRound(trainingType, participants))
        .to.emit(federatedLearningAudit, "TrainingRoundStarted");

      expect(await federatedLearningAudit.totalRounds()).to.equal(1);
    });

    it("Should log model parameter updates", async function () {
      const trainingType = "MNIST";
      const participants = [client1.address, client2.address];
      
      const tx = await federatedLearningAudit.startTrainingRound(trainingType, participants);
      const receipt = await tx.wait();
      
      // Extract roundId from event logs
      const event = receipt.logs.find(log => {
        try {
          const parsed = federatedLearningAudit.interface.parseLog(log);
          return parsed.name === "TrainingRoundStarted";
        } catch (e) {
          return false;
        }
      });
      
      const roundId = federatedLearningAudit.interface.parseLog(event).args.roundId;
      const parametersHash = ethers.keccak256(ethers.toUtf8Bytes("model_params_v1"));

      await expect(
        federatedLearningAudit.connect(client1).logModelParametersUpdate(
          roundId, 
          parametersHash, 
          trainingType
        )
      ).to.emit(federatedLearningAudit, "ModelParametersUpdated")
        .withArgs(client1.address, roundId, parametersHash, await time.latest() + 1, trainingType);
    });

    it("Should complete a training round", async function () {
      const trainingType = "MNIST";
      const participants = [client1.address, client2.address];
      
      const tx = await federatedLearningAudit.startTrainingRound(trainingType, participants);
      const receipt = await tx.wait();
      
      const event = receipt.logs.find(log => {
        try {
          const parsed = federatedLearningAudit.interface.parseLog(log);
          return parsed.name === "TrainingRoundStarted";
        } catch (e) {
          return false;
        }
      });
      
      const roundId = federatedLearningAudit.interface.parseLog(event).args.roundId;

      await expect(federatedLearningAudit.completeTrainingRound(roundId))
        .to.emit(federatedLearningAudit, "TrainingRoundCompleted");
    });
  });

  describe("Data Access Logging", function () {
    beforeEach(async function () {
      await federatedLearningAudit.connect(client1).registerClient(
        "http://127.0.0.1:5001", 
        ethers.keccak256(ethers.toUtf8Bytes("session1"))
      );
    });

    it("Should log data access events", async function () {
      const datasetType = "MNIST_TRAIN";
      const dataHash = ethers.keccak256(ethers.toUtf8Bytes("mnist_data_batch_1"));
      const accessType = "READ";

      await expect(
        federatedLearningAudit.connect(client1).logDataAccess(datasetType, dataHash, accessType)
      ).to.emit(federatedLearningAudit, "DataAccessLogged")
        .withArgs(client1.address, datasetType, dataHash, await time.latest() + 1, accessType);
    });
  });

  describe("Audit Trail Verification", function () {
    it("Should verify audit records", async function () {
      await federatedLearningAudit.connect(client1).registerClient(
        "http://127.0.0.1:5001", 
        ethers.keccak256(ethers.toUtf8Bytes("session1"))
      );

      const summary = await federatedLearningAudit.getAuditTrailSummary();
      expect(summary.totalRecords).to.be.greaterThan(0);
    });
  });

  describe("Access Control", function () {
    it("Should prevent non-registered clients from logging parameters", async function () {
      const roundId = ethers.keccak256(ethers.toUtf8Bytes("fake_round"));
      const parametersHash = ethers.keccak256(ethers.toUtf8Bytes("fake_params"));

      await expect(
        federatedLearningAudit.connect(client1).logModelParametersUpdate(
          roundId, 
          parametersHash, 
          "MNIST"
        )
      ).to.be.revertedWith("Only registered clients can perform this action");
    });

    it("Should prevent non-owners from starting training rounds", async function () {
      const trainingType = "MNIST";
      const participants = [client1.address];

      await expect(
        federatedLearningAudit.connect(client1).startTrainingRound(trainingType, participants)
      ).to.be.revertedWith("Only owner can perform this action");
    });
  });
});

// Helper to get current timestamp for testing
const time = {
  latest: async () => {
    const block = await ethers.provider.getBlock("latest");
    return block.timestamp;
  }
};
