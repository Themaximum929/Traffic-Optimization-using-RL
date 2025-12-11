# Federated Edge Intelligence System for Traffic Optimization

## Project Overview
Complete federated learning system for smart city traffic optimization using heterogeneous edge devices (Arduino BLE 33 Sense + Android + Server).

## Quick Start
```bash
cd federated_learning
python test_system.py  # Verify installation
python run_system.py   # Start all services
```

## Key Features
- ✓ Federated Learning (FedAvg algorithm)
- ✓ Edge Computing (Arduino TFLite inference)
- ✓ Privacy Preservation (Differential Privacy + Secure Aggregation)
- ✓ Model Compression (192.63 KB, fits Arduino)
- ✓ Mobile Notifications (Android app)
- ✓ Real-time Monitoring

## Documentation
- [Deployment Guide](federated_learning/DEPLOYMENT_GUIDE.md)
- [Implementation Status](federated_learning/IMPLEMENTATION_STATUS.md)
- [Project Summary](FEDERATED_LEARNING_SUMMARY.md)
- [Project Checklist](federated_learning/PROJECT_CHECKLIST.md)

## System Architecture
```
Server (FL Coordinator) ←→ Edge Devices (Arduino) ←→ Mobile App (Android)
```

## Model Performance
- Original: best_model.h5 (LSTM traffic prediction)
- Compressed: 192.63 KB (TFLite)
- Inference: <50ms per prediction
- Accuracy: MAE ~5-10 km/h

