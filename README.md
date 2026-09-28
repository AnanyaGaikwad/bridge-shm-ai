# 🌉 BridgeGuard AI: Deep Learning Bridge Structural Health Monitoring (SHM)

A production-grade Deep Learning system for **continuous multi-sensor vibration anomaly detection**, **dynamic damage localization**, and **Structural Health Index (SHI) prediction** for highway and railway bridges.

---

## 🌟 Key Highlights

- **Physical Dynamics Engine**: Simulates a 3-span continuous girder bridge ($L = 220\text{ m}$) with Euler-Bernoulli modal dynamics, Poisson-distributed vehicular traffic loads (cars & heavy freight trucks), tire/suspension bounce, diurnal thermal stiffness drift ($\pm 3-5\%$), and ambient wind turbulence.
- **Structural Damage Injection**:
  - Localized stiffness reduction (concrete spalling, rebar corrosion, fatigue degradation).
  - Dynamic breathing crack impact (non-linear $2\omega_0$ and $4\omega_0$ harmonics).
  - Pier bearing seizure (restraint/support degradation).
- **Hybrid Conv1D-BiLSTM Autoencoder**:
  - **Encoder**: Multi-scale 1D Dilated Convolutions extract spatial-temporal modal wavelets across 8 triaxial accelerometer channels, followed by Bidirectional LSTM layers capturing modal decay and dynamic reverberation.
  - **Bottleneck**: Low-dimensional latent space capturing normal bridge operational manifolds.
  - **Decoder**: Symmetrical BiLSTM + ConvTranspose1D reconstructing multi-sensor acceleration waveforms.
- **Traffic-Invariant RMS Energy Normalization**: Eliminates false alarms caused by traffic volume surges (e.g. 40-ton truck vs 1.5-ton car) while magnifying structural mode shape and frequency anomalies.
- **Dynamic Statistical Thresholding**: Calibrates extreme value anomaly thresholds $\tau$ and computes continuous Structural Health Index ($SHI \in [0\%, 100\%]$).
- **Sensor-Level Damage Localization**: Attributes reconstruction error across individual sensor stations to pinpoint the exact damaged span/pier.
- **Interactive Streamlit Web Dashboard**: Real-time 2D bridge digital twin, oscilloscope waveforms, FFT frequency spectrum, damage injection studio, SHI gauge, and drag-and-drop CSV importer.

---

## 📐 Bridge Geometry & Sensor Station Topology

The modeled bridge is a continuous 3-span girder bridge:
- **Span 1**: 60 meters ($0\text{ m} \to 60\text{ m}$)
- **Span 2 (Main Span)**: 100 meters ($60\text{ m} \to 160\text{ m}$)
- **Span 3**: 60 meters ($160\text{ m} \to 220\text{ m}$)

| Sensor ID | Location ($x$) | Span | Structural Function | Nominal $f_1$ |
| :--- | :---: | :---: | :--- | :---: |
| `S1_S1Q` | 25.0 m | Span 1 | Quarter-point deflection | 2.45 Hz |
| `S2_S1M` | 45.0 m | Span 1 | Midspan dynamic bending | 2.45 Hz |
| `S3_P1` | 60.0 m | Pier 1 | Pier support & bearing restraint | 4.75 Hz |
| `S4_S2Q` | 85.0 m | Span 2 | Main span quarter-point | 2.45 Hz |
| `S5_S2M` | 110.0 m | Span 2 | **Main span midspan (Critical max bending)** | 2.45 Hz |
| `S6_S2Q3` | 135.0 m | Span 2 | Main span three-quarter point | 2.45 Hz |
| `S7_P2` | 160.0 m | Pier 2 | Pier support & bearing restraint | 4.75 Hz |
| `S8_S3M` | 190.0 m | Span 3 | Span 3 midspan dynamic bending | 2.45 Hz |

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        A[Multi-Channel Accelerometers\n8 Stations @ 100 Hz] --> B[Butterworth Bandpass Filter\n0.5 - 35 Hz Zero-Phase]
        B --> C[Sliding Window Segmentation\n256 Samples / 64 Stride]
        C --> D[Multi-Channel RMS Energy Normalization]
    end

    subgraph Deep Learning Model
        D --> E[Multi-Scale Conv1D Encoder\nChannels: 8 -> 32 -> 64 -> 96]
        E --> F[Bidirectional LSTM\nHidden Size: 64]
        F --> G[Latent Space Bottleneck\nDimension: 48]
        G --> H[Bidirectional LSTM Decoder\nHidden Size: 48]
        H --> I[ConvTranspose1D Decoder\nChannels: 96 -> 64 -> 32 -> 8]
    end

    subgraph Decision & Health Engine
        I --> J[Reconstruction Error Matrix\ne_i = ||x_i - x_hat_i||^2]
        J --> K[Statistical Dynamic Threshold\ntau = Baseline Extreme Percentile]
        K --> L{Anomaly Detector\ne_i > tau}
        L -->|Yes| M[Flag Structural Anomaly\nAlert Maintenance Team]
        L -->|No| N[Nominal Baseline Operation]
        J --> O[Structural Health Index SHI\nContinuous Score: 0 to 100%]
        J --> P[Spatial Damage Localization\nSensor-wise Error Attribution]
    end
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
cd /Users/ananya/.gemini/antigravity/scratch/bridge-shm-ai
source .venv/bin/activate
```

Dependencies installed: `torch`, `streamlit`, `plotly`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`.

### 2. Run Baseline Training & Automated Evaluation

```bash
python quickstart.py
```

This will:
1. Simulate baseline healthy bridge vibrations under variable traffic and temperatures.
2. Train the Conv1D-BiLSTM Autoencoder with Smooth L1 Loss and Cosine Annealing on Apple Silicon / CUDA / CPU.
3. Calibrate statistical anomaly thresholds $\tau$.
4. Generate benchmark scenarios (`data/sample_healthy.csv` and `data/sample_damaged.csv`).
5. Evaluate detection sensitivity, false alarm rate, and damage localization attribution.

### 3. Launch the Interactive Web Dashboard

```bash
streamlit run app.py
```
Or with custom port:
```bash
streamlit run app.py --server.port 8502
```

Access the dashboard at: **http://localhost:8502**

---

## 📊 Dashboard Modules

1. **⚡ Live Simulation & Damage Studio**:
   - Customize traffic density, vehicle speed, ambient temperature, and wind.
   - Inject real-time structural defects: stiffness reduction (cracks/fatigue), impact breathing cracks, and frozen bearings.
   - Real-time oscilloscope waveform and FFT frequency spectrum viewer.
2. **📈 Anomaly Detection & Health Scoring**:
   - Live Structural Health Index (SHI) gauge:
     - 🟢 **Healthy (>= 88%)**: Nominal structural operation.
     - 🟡 **Advisory (72% - 88%)**: Minor stiffness degradation detected; schedule maintenance inspection.
     - 🟠 **Warning (48% - 72%)**: Significant stiffness loss; implement traffic speed restrictions.
     - 🔴 **Critical Alarm (< 48%)**: Severe structural defect / crack propagation; immediate closure/inspection.
   - Continuous reconstruction error timeline with dynamic calibrated threshold line.
   - Spatial damage attribution heatmap across the 8 sensor stations.
3. **🌉 Bridge Digital Twin & Frequencies**:
   - Interactive 2D schematic of piers, abutments, and accelerometer nodes.
   - Theoretical eigenmode shapes ($\phi_1, \phi_2, \phi_3$) along the 220m deck.
4. **🧪 Model Architecture & Performance**:
   - Neural network parameters, layer breakdowns, and training loss curves.
   - ROC-AUC, PR-AUC, F1-Score, and false alarm rate benchmarks.
   - Interactive model retraining panel with custom hyperparameters.
5. **📂 Custom Sensor Data CSV**:
   - Drag-and-drop CSV uploader for physical bridge sensor streams (e.g. Z24 benchmark).
   - Automated preprocessing, neural inference, and downloadable PDF/CSV diagnostic reports.

---

## 📁 Repository Structure

```
bridge-shm-ai/
├── app.py                     # Streamlit Interactive Web Application
├── quickstart.py              # CLI automated training and benchmark script
├── requirements.txt           # Python dependency requirements
├── README.md                  # Comprehensive technical documentation
├── data/
│   ├── sample_healthy.csv     # 60s benchmark baseline vibration telemetry
│   └── sample_damaged.csv     # 60s benchmark vibration with injected crack @ 110m
├── checkpoints/
│   └── bridge_shm_pipeline.pt # Trained Conv1D-BiLSTM model, scaler, and threshold bundle
└── src/
    ├── pipeline.py            # End-to-end SHM pipeline manager
    ├── simulation/
    │   └── bridge_physics.py  # Continuous multi-span dynamic physics engine
    ├── data/
    │   ├── preprocessing.py   # Bandpass filtering, windowing, and RMS energy scaler
    │   └── dataset.py         # PyTorch Dataset, DataLoader, and CSV parser
    ├── models/
    │   ├── conv1d_bilstm_autoencoder.py # Hybrid Conv1D-BiLSTM neural network
    │   └── health_index.py    # Threshold calibration, SHI formula, and damage localization
    └── training/
        ├── trainer.py         # PyTorch training loop with MPS/CUDA acceleration
        └── evaluator.py       # ROC-AUC, Precision, Recall, and confusion matrix evaluator
```
