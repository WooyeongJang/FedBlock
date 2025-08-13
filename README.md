# FedBlock: Biomedical Federated Learning with Blockchain Audit

FedBlock is an open-source framework specialized for **biomedical data analysis** using privacy-preserving federated learning with immutable blockchain audit logging. It enables healthcare institutions and research organizations to collaboratively train machine learning models on sensitive health data without sharing raw patient information, while maintaining complete transparency through blockchain-based audit trails.

## 🏥 Biomedical Focus

FedBlock is specifically designed for health-related analytics and statistical modeling:

- **Privacy-Preserving Health Analytics**: Train models on sensitive patient data without data leaving clinical sites
- **Regulatory Compliance**: Built-in audit logging for HIPAA, GDPR, and clinical research compliance  
- **Multi-Modal Health Data**: Support for clinical records, medical imaging, wearable devices, and genomic data
- **Statistical Analysis**: Advanced regression models and correlation analysis for biomedical research

## 🏗️ Architecture Overview

### Core Components
- **Python Client** (`client/`): Python-based federated learning participants
- **Android Client** (`client-android/`): Mobile federated learning client with native UI
- **Orchestration Server** (`server/`): Central coordination system
- **Blockchain Audit Layer** (`blockchain/`): Immutable logging system
- **Smart Contracts** (`contracts/`): On-chain audit trail management

## 🔐 Blockchain Audit Logging

FedBlock integrates blockchain technology to provide immutable audit trails for all federated learning activities:

### Audit Events Logged
- Client registration/deregistration
- Training round initiation
- Model parameter updates
- Training completion confirmations
- Access control events

### Smart Contract Features
- **Event Logging**: All training activities recorded as blockchain events
- **Data Integrity**: Cryptographic hashes of model parameters
- **Access Control**: Permission management for federated learning participants
- **Transparency**: Public verifiability of training history

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- Node.js 16+ (for blockchain features)
- Android Studio (for mobile client development)
- Docker (optional)

### Automated Setup

For **Windows**:
```bash
# Clone the repository
git clone https://github.com/your-org/fedblock.git
cd fedblock

# Run the setup script
setup_biomedical.bat
```

For **Linux/MacOS**:
```bash
# Clone the repository  
git clone https://github.com/your-org/fedblock.git
cd fedblock

# Make setup script executable and run
chmod +x setup_biomedical.sh
./setup_biomedical.sh
```

### Manual Setup

1. **Setup Python Environment**:
```bash
python -m venv fedblock_env
source fedblock_env/bin/activate  # On Windows: fedblock_env\Scripts\activate
pip install -r requirements.txt
```

2. **Setup Blockchain Environment** (optional):
```bash
cd blockchain
npm install
npx hardhat node &
npx hardhat run scripts/deploy.js --network localhost
```

3. **Configure Environment**:
```bash
cp .env.template .env
# Edit .env with your configuration
```

### Starting the Platform

**Quick Start**:
```bash
# Windows
start_fedblock.bat

# Linux/MacOS  
./start_fedblock.sh
```

**Manual Start**:
```bash
# Start the orchestration server
cd server
python -m flask run --host=localhost --port=5000

# In another terminal, start a Python client
cd client
export CLIENT_URL=http://127.0.0.1:5001
export SERVER_URL=http://127.0.0.1:5000
python -m flask run --port 5001
```

### Running Biomedical Analysis

Once the platform is running, you can start federated learning for various biomedical tasks:

```python
# Example: Start cardiovascular risk assessment
import requests

server_url = "http://localhost:5000"
requests.post(f"{server_url}/training", json={
    "training_type": "CARDIOVASCULAR_RISK",
    "epochs": 50,
    "learning_rate": 0.001
})
```

### Android Client Setup

1. **Open Android Studio**:
```bash
cd client-android
# Open this directory in Android Studio
```

2. **Build and Install**:
```bash
./gradlew assembleDebug
# Install the generated APK on your Android device
```

3. **Configure Connection**:
   - Set server URL in the app settings
   - Ensure your device is on the same network as the server

## 🩺 Supported Biomedical Analysis Types

FedBlock supports various biomedical federated learning tasks:

### 1. **Biomedical Regression Analysis**
- **Cardiovascular Risk Assessment**: Predict heart disease risk from clinical parameters
- **Lab Value Prediction**: Forecast laboratory test results from patient history  
- **Treatment Outcome Modeling**: Predict therapeutic response and side effects
- **Disease Progression Analysis**: Model progression of chronic diseases

### 2. **Wearable Device Analytics**
- **Heart Rate Variability Analysis**: Time-series analysis of cardiac patterns
- **Sleep Quality Assessment**: Multi-parameter sleep pattern analysis
- **Activity Level Prediction**: Behavioral pattern recognition from wearable sensors
- **Stress Level Monitoring**: Real-time stress assessment from physiological data

### 3. **Medical Image Biomarker Extraction**
- **Cardiac Function Analysis**: Echocardiogram parameter extraction
- **Pulmonary Assessment**: Lung capacity measurements from chest imaging
- **Bone Density Analysis**: DEXA scan biomarker extraction
- **Retinal Vessel Analysis**: Fundus image vascular assessment
- **Brain Volume Measurements**: MRI-based neurological biomarkers

### 4. **Genomic Analysis**
- **Polygenic Risk Scoring**: Multi-SNP disease risk assessment
- **Pharmacogenomics**: Drug response prediction from genetic variants
- **Population Genomics**: Ancestry and population structure analysis

### 5. **Epidemiological Modeling**
- **Disease Outbreak Prediction**: Population-level disease spread modeling
- **Public Health Analytics**: Community health trend analysis
- **Environmental Health**: Pollution and health outcome correlations

### 6. **Clinical Outcome Prediction**
- **Readmission Risk**: Hospital readmission probability assessment
- **Mortality Risk**: Survival analysis and risk stratification
- **Complication Prediction**: Post-operative and treatment complication forecasting

## 📱 Android Client Features

The Android client provides a mobile interface for federated learning participation:

### Key Features
- **Native Training UI**: Real-time training progress and statistics
- **Blockchain Integration**: Direct interaction with audit logging smart contracts
- **Model Management**: Local model storage and parameter synchronization
- **Peer Discovery**: Support for decentralized gossip learning protocols
- **Offline Capability**: Continue training when network connectivity is limited

### Technical Architecture
- **Kotlin/Compose**: Modern Android UI framework
- **TensorFlow Lite**: On-device machine learning inference
- **Web3j**: Ethereum blockchain interaction
- **Hilt**: Dependency injection for modular architecture
- **Retrofit**: HTTP API communication with federated learning server

### Supported Training Modes
1. **Centralized FL**: Traditional federated learning with server coordination
2. **Gossip Learning**: Peer-to-peer model parameter exchange
3. **Hybrid Mode**: Combination of centralized and decentralized approaches

## 🔗 Blockchain Integration

The blockchain audit layer provides:
- **Immutable Records**: All training events permanently recorded
- **Verifiable Computation**: Cryptographic proofs of training integrity
- **Decentralized Trust**: No single point of failure in audit trail
- **Compliance**: Meeting regulatory requirements for AI model provenance

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.