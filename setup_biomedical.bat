@echo off
REM FedBlock Biomedical Setup Script for Windows
REM Sets up the development environment for biomedical federated learning

echo 🏥 FedBlock Biomedical Federated Learning Setup
echo ==============================================

REM Check Python version
python --version > nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found

REM Create virtual environment
echo 🔧 Creating virtual environment...
python -m venv fedblock_env
call fedblock_env\Scripts\activate.bat

REM Install Python dependencies
echo 📦 Installing Python dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

REM Install additional biomedical packages
echo 🧬 Installing biomedical analysis packages...
python -m pip install nibabel==5.1.0
python -m pip install pydicom==2.3.1
python -m pip install SimpleITK==2.2.1
python -m pip install bioinfokit==2.1.0

REM Setup blockchain development (optional)
where npm >nul 2>nul
if %errorlevel%==0 (
    echo 🔗 Setting up blockchain environment...
    cd blockchain
    call npm init -y
    call npm install --save-dev hardhat @openzeppelin/contracts
    call npm install web3 ethers
    cd ..
) else (
    echo ⚠️  npm not found. Skipping blockchain setup.
)

REM Create necessary directories
echo 📁 Creating directory structure...
mkdir client\tmp\biomedical_data 2>nul
mkdir client\tmp\wearable_data 2>nul
mkdir client\tmp\medical_images 2>nul
mkdir server\static\models 2>nul
mkdir logs\training 2>nul
mkdir logs\audit 2>nul
mkdir config 2>nul

REM Generate sample configuration
echo ⚙️  Generating configuration files...

(
echo # FedBlock Biomedical Configuration
echo.
echo training:
echo   biomedical_regression:
echo     learning_rate: 0.001
echo     epochs: 50
echo     batch_size: 32
echo     hidden_dims: [128, 64, 32]
echo     dropout_rate: 0.3
echo.    
echo   wearable_analytics:
echo     learning_rate: 0.0005
echo     epochs: 30
echo     batch_size: 16
echo     sequence_length: 24
echo     hidden_dim: 128
echo     num_layers: 2
echo.    
echo   medical_image_biomarker:
echo     learning_rate: 0.0001
echo     epochs: 25
echo     batch_size: 8
echo     backbone: 'resnet50'
echo     num_biomarkers: 5
echo     image_size: [224, 224]
echo.
echo blockchain:
echo   enabled: true
echo   provider_url: "http://localhost:8545"
echo   contract_address: ""
echo   gas_limit: 300000
echo.  
echo privacy:
echo   differential_privacy: true
echo   noise_multiplier: 1.1
echo   max_grad_norm: 1.0
echo.  
echo audit:
echo   log_level: "INFO"
echo   enable_blockchain_logging: true
echo   compliance_checks: ["HIPAA", "GDPR"]
) > config\biomedical_config.yaml

REM Create environment file template
(
echo # FedBlock Environment Configuration
echo # Copy this file to .env and update the values
echo.
echo # Server Configuration
echo SERVER_HOST=localhost
echo SERVER_PORT=5000
echo DEBUG=true
echo.
echo # Blockchain Configuration  
echo WEB3_PROVIDER_URL=http://localhost:8545
echo FEDBLOCK_CONTRACT_ADDRESS=
echo FEDBLOCK_PRIVATE_KEY=
echo.
echo # Database ^(optional^)
echo DATABASE_URL=sqlite:///fedblock.db
echo.
echo # Logging
echo LOG_LEVEL=INFO
echo LOG_FILE=logs/fedblock.log
echo.
echo # Security
echo SECRET_KEY=your-secret-key-here
echo ENABLE_HTTPS=false
echo.
echo # Biomedical Data
echo ENABLE_SYNTHETIC_DATA=true
echo DATA_PRIVACY_LEVEL=HIGH
) > .env.template

REM Create startup script for Windows
(
echo @echo off
echo REM FedBlock Startup Script for Windows
echo.
echo echo 🚀 Starting FedBlock Biomedical Federated Learning Platform
echo.
echo REM Activate virtual environment
echo call fedblock_env\Scripts\activate.bat
echo.
echo REM Start the server
echo echo 🖥️  Starting FedBlock server...
echo cd server
echo python -m flask run --host=localhost --port=5000
echo cd ..
echo.
echo echo ✅ FedBlock server started on http://localhost:5000
echo echo 📊 Dashboard available at http://localhost:5000/dashboard
echo.
echo pause
) > start_fedblock.bat

REM Create requirements for development
(
echo # Development dependencies for FedBlock
echo.
echo # Testing
echo pytest==7.4.0
echo pytest-asyncio==0.21.1
echo pytest-cov==4.1.0
echo pytest-mock==3.11.1
echo.
echo # Code quality
echo black==23.7.0
echo isort==5.12.0
echo flake8==6.0.0
echo mypy==1.5.1
echo.
echo # Documentation
echo sphinx==7.1.2
echo sphinx-rtd-theme==1.3.0
echo.
echo # Jupyter for research
echo jupyter==1.0.0
echo ipykernel==6.25.0
echo matplotlib==3.7.2
echo seaborn==0.12.2
echo.
echo # Profiling and monitoring
echo memory-profiler==0.60.0
echo line-profiler==4.0.2
) > requirements-dev.txt

echo.
echo 🎉 FedBlock Biomedical Setup Complete!
echo.
echo Next steps:
echo 1. Copy .env.template to .env and configure your settings
echo 2. Run 'start_fedblock.bat' to start the platform
echo 3. Visit http://localhost:5000 to access the dashboard
echo.
echo For Android development:
echo 1. Open client-android\ in Android Studio
echo 2. Build and run on your Android device
echo.
echo For blockchain features:
echo 1. Install Node.js and npm from https://nodejs.org
echo 2. Deploy smart contracts with 'cd blockchain && npx hardhat run scripts/deploy.js'
echo.
echo 📚 Documentation: https://github.com/your-org/fedblock/wiki
echo 🐛 Issues: https://github.com/your-org/fedblock/issues
echo.
pause
