#!/bin/bash

# FedBlock Biomedical Setup Script
# Sets up the development environment for biomedical federated learning

echo "🏥 FedBlock Biomedical Federated Learning Setup"
echo "=============================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
echo "Python version: $python_version"

if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
    echo "❌ Python 3.8+ required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed"

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv fedblock_env
source fedblock_env/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install additional biomedical packages
echo "🧬 Installing biomedical analysis packages..."
pip install nibabel==5.1.0  # Medical imaging
pip install pydicom==2.3.1  # DICOM support
pip install SimpleITK==2.2.1  # Medical image processing
pip install bioinfokit==2.1.0  # Bioinformatics toolkit

# Setup blockchain development (optional)
if command -v npm &> /dev/null; then
    echo "🔗 Setting up blockchain environment..."
    cd blockchain
    npm init -y
    npm install --save-dev hardhat @openzeppelin/contracts
    npm install web3 ethers
    cd ..
else
    echo "⚠️  npm not found. Skipping blockchain setup."
fi

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p client/tmp/biomedical_data
mkdir -p client/tmp/wearable_data  
mkdir -p client/tmp/medical_images
mkdir -p server/static/models
mkdir -p logs/training
mkdir -p logs/audit

# Generate sample configuration
echo "⚙️  Generating configuration files..."

cat > config/biomedical_config.yaml << EOF
# FedBlock Biomedical Configuration

training:
  biomedical_regression:
    learning_rate: 0.001
    epochs: 50
    batch_size: 32
    hidden_dims: [128, 64, 32]
    dropout_rate: 0.3
    
  wearable_analytics:
    learning_rate: 0.0005
    epochs: 30
    batch_size: 16
    sequence_length: 24
    hidden_dim: 128
    num_layers: 2
    
  medical_image_biomarker:
    learning_rate: 0.0001
    epochs: 25
    batch_size: 8
    backbone: 'resnet50'
    num_biomarkers: 5
    image_size: [224, 224]

blockchain:
  enabled: true
  provider_url: "http://localhost:8545"
  contract_address: ""
  gas_limit: 300000
  
privacy:
  differential_privacy: true
  noise_multiplier: 1.1
  max_grad_norm: 1.0
  
audit:
  log_level: "INFO"
  enable_blockchain_logging: true
  compliance_checks: ["HIPAA", "GDPR"]
EOF

mkdir -p config

# Create environment file template
cat > .env.template << EOF
# FedBlock Environment Configuration
# Copy this file to .env and update the values

# Server Configuration
SERVER_HOST=localhost
SERVER_PORT=5000
DEBUG=true

# Blockchain Configuration  
WEB3_PROVIDER_URL=http://localhost:8545
FEDBLOCK_CONTRACT_ADDRESS=
FEDBLOCK_PRIVATE_KEY=

# Database (optional)
DATABASE_URL=sqlite:///fedblock.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/fedblock.log

# Security
SECRET_KEY=your-secret-key-here
ENABLE_HTTPS=false

# Biomedical Data
ENABLE_SYNTHETIC_DATA=true
DATA_PRIVACY_LEVEL=HIGH
EOF

# Install Android dependencies (if Android SDK is available)
if [ -d "$ANDROID_HOME" ]; then
    echo "📱 Android SDK found. Setting up Android client..."
    cd client-android
    ./gradlew clean build --warning-mode all
    cd ..
else
    echo "⚠️  Android SDK not found. Skipping Android setup."
fi

# Create startup script
cat > start_fedblock.sh << EOF
#!/bin/bash

# FedBlock Startup Script

echo "🚀 Starting FedBlock Biomedical Federated Learning Platform"

# Activate virtual environment
source fedblock_env/bin/activate

# Load environment variables
if [ -f .env ]; then
    export \$(cat .env | xargs)
fi

# Start blockchain node (if enabled)
if [ "\$ENABLE_BLOCKCHAIN" = "true" ]; then
    echo "🔗 Starting local blockchain node..."
    cd blockchain
    npx hardhat node &
    BLOCKCHAIN_PID=\$!
    cd ..
    sleep 5
fi

# Start the server
echo "🖥️  Starting FedBlock server..."
cd server
python -m flask run --host=\$SERVER_HOST --port=\$SERVER_PORT &
SERVER_PID=\$!
cd ..

echo "✅ FedBlock server started on http://\$SERVER_HOST:\$SERVER_PORT"
echo "📊 Dashboard available at http://\$SERVER_HOST:\$SERVER_PORT/dashboard"

# Handle shutdown
function cleanup {
    echo "🛑 Shutting down FedBlock..."
    kill \$SERVER_PID 2>/dev/null
    if [ ! -z "\$BLOCKCHAIN_PID" ]; then
        kill \$BLOCKCHAIN_PID 2>/dev/null
    fi
    echo "👋 FedBlock stopped"
}

trap cleanup EXIT INT TERM

# Wait for processes
wait
EOF

chmod +x start_fedblock.sh

# Create requirements for development
cat > requirements-dev.txt << EOF
# Development dependencies for FedBlock

# Testing
pytest==7.4.0
pytest-asyncio==0.21.1
pytest-cov==4.1.0
pytest-mock==3.11.1

# Code quality
black==23.7.0
isort==5.12.0
flake8==6.0.0
mypy==1.5.1

# Documentation
sphinx==7.1.2
sphinx-rtd-theme==1.3.0

# Jupyter for research
jupyter==1.0.0
ipykernel==6.25.0
matplotlib==3.7.2
seaborn==0.12.2

# Profiling and monitoring
memory-profiler==0.60.0
line-profiler==4.0.2
py-spy==0.3.14
EOF

echo ""
echo "🎉 FedBlock Biomedical Setup Complete!"
echo ""
echo "Next steps:"
echo "1. Copy .env.template to .env and configure your settings"
echo "2. Run './start_fedblock.sh' to start the platform"
echo "3. Visit http://localhost:5000 to access the dashboard"
echo ""
echo "For Android development:"
echo "1. Open client-android/ in Android Studio"
echo "2. Build and run on your Android device"
echo ""
echo "For blockchain features:"
echo "1. Install Node.js and npm"
echo "2. Deploy smart contracts with 'cd blockchain && npx hardhat run scripts/deploy.js'"
echo ""
echo "📚 Documentation: https://github.com/your-org/fedblock/wiki"
echo "🐛 Issues: https://github.com/your-org/fedblock/issues"
