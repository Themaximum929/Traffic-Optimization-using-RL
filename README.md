# Federated Edge Intelligence System for Traffic Optimization

## Project Overview
Complete federated learning system for smart city traffic optimization using heterogeneous edge devices (Arduino BLE 33 Sense + Python Clients + Android + Server).

## Quick Start

### Windows
```bash
# 1. Setup environment
cd NLP
python -m venv .venv
.venv\Scripts\activate
pip install -r federated_learning/requirements.txt

# 2. Run system (automated)
federated_learning\RUN_SYSTEM.bat

# OR manually:
python federated_learning/server/fl_server.py  # Terminal 1
python federated_learning/edge_client/wht_client.py  # Terminal 2
python federated_learning/edge_client/eht_client.py  # Terminal 3
```

### Linux/Mac
```bash
# 1. Setup environment
cd NLP
python3 -m venv .venv
source .venv/bin/activate
pip install -r federated_learning/requirements.txt

# 2. Run system
chmod +x federated_learning/run_system.sh
./federated_learning/run_system.sh
```

## Key Features
- ✅ **Real Federated Learning**: FedAvg aggregation with actual model.fit() training
- ✅ **Differential Privacy**: ε=1.0, δ=1e-5 with gradient clipping
- ✅ **Image Processing**: OpenCV vehicle detection from real traffic images
- ✅ **Heterogeneous Devices**: Python clients + Arduino + Android
- ✅ **Future Predictions**: 15min and 30min ahead forecasting
- ✅ **User Feedback Loop**: Android app sends real trip data for model improvement
- ✅ **Real-time Monitoring**: HTTP API + Android app with charts

## System Architecture
```
┌─────────────────┐
│   FL Server     │  Port 9000 (FL Socket)
│  (Aggregator)   │  Port 8888 (HTTP API)
└────────┬────────┘
         │
    ┌────┴────┬────────────┬──────────┐
    │         │            │          │
┌───▼───┐ ┌──▼───┐  ┌─────▼─────┐ ┌─▼────────┐
│  WHT  │ │ EHT  │  │  Arduino  │ │ Android  │
│Client │ │Client│  │  Gateway  │ │   App    │
└───────┘ └──────┘  └─────┬─────┘ └──────────┘
                          │
                    ┌─────▼──────┐
                    │  Arduino   │
                    │  BLE 33    │
                    │ + Camera   │
                    └────────────┘
```

## Documentation
- **[EXECUTION_PROOF.md](federated_learning/EXECUTION_PROOF.md)** - Logs, screenshots, test results
- **[REPRODUCIBILITY.md](federated_learning/REPRODUCIBILITY.md)** - Setup guide, pitfalls, solutions
- **[requirements.txt](federated_learning/requirements.txt)** - Python dependencies
- **[ARDUINO_CAMERA_SETUP.md](federated_learning/edge_client/ARDUINO_CAMERA_SETUP.md)** - Arduino setup
- **[ANDROID_SETUP.md](federated_learning/android_client/ANDROID_SETUP.md)** - Android app setup

## Model Architecture
- **Type**: Multi-input LSTM
- **Inputs**: 
  - Numerical: (batch, 10, 6) - Time series traffic data
  - Tunnel: (batch,) - Tunnel ID (CHT/WHT/EHT)
  - Direction: (batch,) - Direction (Northbound/Southbound)
- **Parameters**: 137,249 (536 KB)
- **Output**: (batch, 1) - Speed prediction
- **Training**: Real gradient descent with Adam optimizer
- **Privacy**: Differential privacy with ε=1.0

## Performance
- **FL Round Time**: ~30 seconds
- **Client Training**: 2-3 seconds (3 epochs, 50 samples)
- **Image Processing**: 100-200ms per image
- **Prediction Accuracy**: MAE ~5-10 km/h
- **Privacy Budget**: ε=1.0 per round, cumulative tracking

## Technical Artifacts

### ✅ Runnable Code
- Python FL server and clients
- Arduino simulation with TFLite
- Android app with Material Design
- Automated launch scripts (Windows/Linux)

### ✅ Execution Proof
- Server logs showing aggregation
- Client training outputs
- Arduino image transmission logs
- Android app screenshots
- API test results

### ✅ Reproducibility
- Complete environment setup guide
- Common pitfalls and solutions
- Fallback options for missing hardware
- Docker alternative
- Verification checklist

