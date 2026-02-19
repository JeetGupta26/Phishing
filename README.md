# Phishing Detection Platform

A production-grade, multi-signal phishing detection system featuring automated retraining, drift monitoring, and a calibrated inference microservice.

## 🚀 Overview
This platform uses machine learning (LightGBM) and real-time intelligence signals (DNS, SSL, Reputation) to detect phishing URLs. It is designed with enterprise governance, featuring a Champion-Challenger retraining framework and automated drift detection.

## 🛠️ Tech Stack
- **Language**: Python 3.10+
- **ML Framework**: LightGBM, Scikit-learn
- **API**: FastAPI, Uvicorn
- **Monitoring**: PSI (Population Stability Index), KS-Test
- **Infrastructure**: Docker, DNS/WHOIS lookups

---

## ⚙️ Installation

### 1. Prerequisites
- Python 3.10 or higher
- System libraries for WHOIS (on Windows/Linux):
  ```bash
  # Windows: Ensure 'whois' is in PATH or pip install whois
  # Linux: sudo apt-get install whois
  ```

### 2. Setup Environment
```bash
git clone <repo-url>
cd Phishing2.0
pip install -r requirements.txt
```

### 3. API Keys (Optional)
Create a `.env` file for Safe Browsing (optional for full multi-signal logic):
```env
SAFE_BROWSING_API_KEY=your_key_here
```

---

## 🏃 Running the Pipeline

### Step 1: Baseline Training (Phase 1-3)
Train the regularized and calibrated LightGBM model:
```bash
python core_model_v4.py
```
*Outputs: Optimized model in `v4_artifacts/`.*

### Step 2: Initialize Registry & Monitoring (Phase 4)
Setup the model registry and run a monitoring simulation:
```bash
python simulate_monitoring.py
```
*Outputs: Registered model in `model_registry/production/`.*

### Step 3: Run Retraining Pipeline (Phase 5)
Trigger the Champion-Challenger retraining flow:
```bash
python run_retraining_pipeline.py
```
*Outputs: Retraining report and potential model promotion in `retraining/`.*

### Step 4: Start Inference Microservice (Phase 6)
Deploy the API for production inference:
```bash
# Manual run
python service/main.py

# Or via Docker
docker build -t phishing-service .
docker run -p 8000:8000 phishing-service
```

### Step 5: Execute Certification Suite (Phase 7)
Validate the system under load and adversarial stress:
```bash
python validation/certifier.py
```
*Outputs: `validation/enterprise_certification.json` and executive summary.*

---

## 🧩 Directory Structure
- `feature_layers/`: Specialized feature extraction (Lexical, DNS, SSL, etc.)
- `monitoring/`: Drift detection and model versioning registry.
- `service/`: Production FastAPI microservice.
- `retraining/`: Automated retraining and evaluation logic.
- `validation/`: Benchmarking and adversarial testing suite.

## 📊 API Usage
**Endpoint**: `POST /predict`
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"url": "http://example-phish.com"}'
```

---
**Certified Status**: PASS ✅
*(See `validation/executive_summary.txt` for latest certification details)*
