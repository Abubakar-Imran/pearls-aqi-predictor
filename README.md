# 🌬️ Pearls AQI Predictor
 
> **Predict the Air Quality Index (AQI) in Islamabad / Rawalpindi, Pakistan — 3 days ahead — using a 100% serverless MLOps stack.**
---
 
## 📋 Table of Contents
 
- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [1. Clone & Install](#1-clone--install)
  - [2. API Keys & Secrets](#2-api-keys--secrets)
  - [3. Hopsworks Setup](#3-hopsworks-setup)
  - [4. Run the Backfill](#4-run-the-backfill)
  - [5. Run the Feature Pipeline](#5-run-the-feature-pipeline)
  - [6. Train Models](#6-train-models)
  - [7. Launch the Dashboard](#7-launch-the-dashboard)
- [CI/CD Automation](#-cicd-automation)
- [Data Sources](#-data-sources)
- [Feature Engineering](#-feature-engineering)
- [Models & Performance](#-models--performance)
- [Dashboard](#-dashboard)
- [AQI Reference](#-aqi-reference)
- [Configuration](#-configuration)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)
---
 
## 🌍 Overview
 
The **Pearls AQI Predictor** is an end-to-end machine learning system that forecasts the Air Quality Index (AQI) for the Islamabad–Rawalpindi metropolitan area up to **72 hours (3 days)** in advance.
 
Air quality in Pakistan's twin cities has become a critical public health issue. Winter months regularly push AQI above 200 due to crop burning in Punjab, cold temperature inversions, and heavy vehicular traffic — conditions that are entirely predictable from weather data and recent AQI history. This project makes those predictions accessible to the public through a live dashboard.
 
### What makes this project unique
 
- **Fully serverless** — no VMs, no servers, no infrastructure to manage
- **Self-updating** — GitHub Actions retrains the model daily and refreshes features hourly with zero manual work
- **Islamabad-specific** — features like rush-hour flags, seasonal smog encoding, and monsoon patterns are engineered specifically for the twin cities
- **Transparent** — SHAP values explain every prediction; users can see exactly why the model forecasts high or low AQI
- **Free to operate** — every component runs on a free tier (Hopsworks, GitHub Actions, Streamlit Community Cloud, Open-Meteo)
---
 
## 🚀 Live Demo
 
The dashboard is deployed publicly at:
 
```
https://your-app-name.streamlit.app
```
 
> Replace the URL above with your actual Streamlit Community Cloud deployment URL after deploying.
 
**What you'll see on the dashboard:**
- Live AQI gauge for Islamabad (updated every hour)
- Current pollutant readings: PM2.5, PM10, NO₂, Temperature
- 3-day colour-coded AQI forecast cards (Day+1, Day+2, Day+3)
- Forecast bar chart with health threshold lines
- Last 7 days of historical AQI
- SHAP feature importance plot from the latest model
- Health alert banners when AQI is forecast to exceed 150
---
 
## 🏗️ Architecture
 
The system follows a four-stage MLOps pipeline, fully orchestrated by GitHub Actions:
 
```
┌─────────────────────────────────────────────────────────────────────┐
│                        GITHUB ACTIONS (CI/CD)                        │
│                  Feature: every hour │ Training: every day           │
└──────────────────────────┬──────────────────────┬───────────────────┘
                           │                      │
         ┌─────────────────▼──────┐    ┌──────────▼─────────────┐
         │    ① DATA COLLECTION   │    │   ③ TRAINING PIPELINE   │
         │                        │    │                         │
         │  • AQICN API           │    │  • Fetch features from  │
         │    (Islamabad @11458)   │    │    Hopsworks            │
         │  • Open-Meteo API      │    │  • Train Ridge, RF,     │
         │    (weather + AQ hist) │    │    XGBoost, LSTM        │
         └─────────────┬──────────┘    │  • Select best by RMSE  │
                       │               │  • Compute SHAP values  │
         ┌─────────────▼──────────┐    │  • Register in Model    │
         │  ② FEATURE PIPELINE    │    │    Registry             │
         │                        │    └──────────┬──────────────┘
         │  • Extract pollutants  │               │
         │  • Compute 30+ features│    ┌──────────▼──────────────┐
         │  • Cyclical time enc.  │    │   ④ WEB DASHBOARD        │
         │  • Lag & rolling feats │    │                         │
         │  • Wind decomposition  │    │  • Streamlit Cloud      │
         └─────────────┬──────────┘    │  • Live AQI gauge       │
                       │               │  • 3-day forecast       │
         ┌─────────────▼──────────┐    │  • SHAP explanations    │
         │   HOPSWORKS (Free)     ├────►  • Health alerts        │
         │                        │    └─────────────────────────┘
         │  Feature Store:        │
         │  • aqi_features group  │
         │                        │
         │  Model Registry:       │
         │  • aqi_predictor_24h   │
         │  • aqi_predictor_48h   │
         │  • aqi_predictor_72h   │
         └────────────────────────┘
```
 
---
 
## ✨ Features
 
### Data Pipeline
- Hourly ingestion of real-time AQI and weather data for Islamabad/Rawalpindi
- 30+ engineered features per row including time-cyclical, lag, rolling, and wind vector features
- 6-month historical backfill using Open-Meteo Air Quality Archive API
- Online feature store (Hopsworks) for low-latency prediction serving
### Machine Learning
- Four model families compared: Ridge Regression, Random Forest, XGBoost, LSTM
- Time-series aware train/validation split (no data leakage)
- Evaluation on RMSE, MAE, and R² across three forecast horizons (24h, 48h, 72h)
- Automatic daily retraining and model versioning
- SHAP feature importance computed and saved after every training run
### Monitoring & Alerts
- Health alert banners on dashboard for AQI > 100 (Caution) and AQI > 150 (Unhealthy)
- Optional Telegram bot notifications for hazardous AQI forecasts
- GitHub Actions email alerts on pipeline failures
### Dashboard
- Real-time AQI gauge with colour-coded health categories
- Three-day forecast cards with AQI category labels
- Interactive Plotly charts with health threshold reference lines
- 7-day historical AQI trend
- SHAP summary plot for model transparency
---
 
## 🛠️ Tech Stack
 
| Layer | Technology | Why |
|-------|-----------|-----|
| **Language** | Python 3.11 | Ecosystem support for ML + data pipelines |
| **Feature Store** | [Hopsworks](https://app.hopsworks.ai) | Free serverless feature store + model registry |
| **AQI Data** | [AQICN API](https://aqicn.org/api/) | Real ground-sensor readings for Islamabad |
| **Weather Data** | [Open-Meteo](https://open-meteo.com) | Free historical + forecast weather, no API key |
| **ML (Tabular)** | Scikit-learn, XGBoost | Best performance on structured time-series data |
| **ML (Sequential)** | TensorFlow / Keras LSTM | Captures temporal AQI dependencies |
| **Explainability** | SHAP | Model-agnostic feature importance |
| **CI/CD** | GitHub Actions | Free scheduled workflows (2,000 min/month) |
| **Dashboard** | Streamlit | Rapid, interactive Python-native web UI |
| **Hosting** | Streamlit Community Cloud | Free public app deployment |
| **Config** | PyYAML + python-dotenv | Clean separation of config and secrets |
 
---
 
## 📁 Repository Structure
 
```
pearls-aqi-predictor/
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml      # Runs every hour (cron: "0 * * * *")
│       └── training_pipeline.yml     # Runs daily at 02:00 UTC (07:00 PKT)
│
├── src/
│   ├── feature_pipeline.py           # Fetch → compute → store one hourly row
│   ├── backfill.py                   # Populate feature store with 6 months of history
│   ├── training_pipeline.py          # Train all models, select best, register
│   ├── inference.py                  # Load model + features, return 3-day forecast
│   └── streamlit_app.py              # Interactive dashboard
│
├── notebooks/
│   └── 01_EDA.ipynb                  # Exploratory data analysis for Islamabad
│
├── config/
│   └── config.yaml                   # City coordinates, station IDs, thresholds
│
├── artifacts/                        # Generated files (SHAP plots, etc.)
│   └── shap_summary.png              # Updated by training pipeline daily
│
├── requirements.txt                  # All Python dependencies
├── .env.example                      # Template for local secrets
├── .gitignore
└── README.md
```
 
---
 
## 🚦 Getting Started
 
### Prerequisites
 
- Python 3.11 or higher
- A free [Hopsworks account](https://app.hopsworks.ai) with a project created
- A free [AQICN API token](https://aqicn.org/data-platform/token/)
- Git
### 1. Clone & Install
 
```bash
git clone https://github.com/your-username/pearls-aqi-predictor.git
cd pearls-aqi-predictor
 
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
 
# Install all dependencies
pip install -r requirements.txt
```
 
### 2. API Keys & Secrets
 
Copy the example environment file and fill in your credentials:
 
```bash
cp .env.example .env
```
 
Edit `.env` with your actual values:
 
```dotenv
# ─── AQICN ────────────────────────────────────────────────────────────────────
# Get your free token at: https://aqicn.org/data-platform/token/
AQICN_TOKEN=your_aqicn_token_here
 
# ─── HOPSWORKS ────────────────────────────────────────────────────────────────
# Log into https://app.hopsworks.ai → your project → Settings → API Keys
HOPSWORKS_API_KEY=your_hopsworks_api_key_here
```
 
> ⚠️ **Never commit your `.env` file.** It is already listed in `.gitignore`.
 
### 3. Hopsworks Setup
 
Verify your Hopsworks connection before running any pipeline:
 
```bash
python - <<'EOF'
import os
from dotenv import load_dotenv
import hopsworks
 
load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
print(f"✓ Connected to Hopsworks project: {project.name}")
EOF
```
 
Make sure the project name in `config/config.yaml` exactly matches your Hopsworks project name:
 
```yaml
# config/config.yaml
feature_store:
  project_name: "your_hopsworks_project_name"   # ← must match exactly
```
 
### 4. Run the Backfill
 
This step populates the feature store with ~6 months of historical data. Run it **once** before training.
 
```bash
python src/backfill.py
```
 
Expected output:
```
Backfilling from 2025-12-04 to 2026-06-04...
  Fetching 2025-12-04 → 2026-01-03...
  Fetching 2026-01-04 → 2026-02-02...
  ...
  Total records: 4320 weather, 4320 AQI
  Features computed: 4284 rows, 35 columns
[DONE] Backfill complete. 4284 rows stored.
```
 
> ⏱️ **Expected runtime**: 5–15 minutes depending on internet speed.
 
Verify the data landed correctly:
 
```bash
python - <<'EOF'
import os, hopsworks
from dotenv import load_dotenv
load_dotenv()
 
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
fg = fs.get_feature_group("aqi_features", version=1)
df = fg.read()
print(f"Rows in feature store: {len(df)}")
print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
print(df[["timestamp","aqi","pm25","temperature"]].tail(3))
EOF
```
 
### 5. Run the Feature Pipeline
 
Insert one real-time row to confirm the live ingestion path works:
 
```bash
python src/feature_pipeline.py
```
 
Expected output:
```
Fetching AQICN data...
Fetching weather data...
Computing features...
Storing features in Hopsworks...
[OK] Features stored at 2026-06-04T10:00:00+00:00
```
 
### 6. Train Models
 
```bash
python src/training_pipeline.py
```
 
Expected output:
```
Fetching training data from Feature Store...
  Total samples: 4284
 
==================================================
Training for target: target_aqi_24h
  Training ridge...      RMSE=22.4, MAE=16.8, R²=0.64
  Training random_forest... RMSE=15.7, MAE=11.4, R²=0.80
  Training xgboost...    RMSE=12.9, MAE=9.5,  R²=0.84
  ✅ Best model: xgboost
  Computing SHAP values → artifacts/shap_summary.png
  Model registered: aqi_predictor_target_aqi_24h v1
...
=== TRAINING SUMMARY ===
target_aqi_24h: RMSE=12.9,  R²=0.84  (xgboost)
target_aqi_48h: RMSE=16.4,  R²=0.78  (xgboost)
target_aqi_72h: RMSE=20.2,  R²=0.70  (xgboost)
```
 
> ⏱️ **Expected runtime**: 10–25 minutes (LSTM adds ~10 min).
 
### 7. Launch the Dashboard
 
```bash
streamlit run src/streamlit_app.py
```
 
Open your browser at **http://localhost:8501**
 
---
 
## ⚙️ CI/CD Automation
 
Both pipelines run automatically via GitHub Actions after you push to GitHub and add repository secrets.
 
### Adding GitHub Secrets
 
Go to your repository → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:
 
| Secret Name | Value | Where to get it |
|-------------|-------|----------------|
| `AQICN_TOKEN` | Your AQICN API token | [aqicn.org/data-platform/token](https://aqicn.org/data-platform/token/) |
| `HOPSWORKS_API_KEY` | Your Hopsworks API key | Hopsworks UI → Project → Settings → API Keys |

### Triggering Workflows Manually
 
Don't wait for the schedule — trigger both workflows immediately to verify they work:
 
1. Go to your repo on GitHub → **Actions** tab
2. Select **Feature Pipeline (Hourly)** → **Run workflow** → **Run workflow**
3. Select **Training Pipeline (Daily)** → **Run workflow** → **Run workflow**
### Schedule Summary
 
| Workflow | Schedule | Cron Expression | Avg Runtime |
|----------|----------|-----------------|-------------|
| Feature Pipeline | Every hour | `0 * * * *` | ~2–3 min |
| Training Pipeline | Daily at 02:00 UTC (07:00 PKT) | `0 2 * * *` | ~10–20 min |
 
> 💡 **GitHub Actions free tier**: 2,000 minutes/month for public repositories. Both pipelines combined use approximately 1,800–2,100 min/month. If you approach the limit, change the training schedule to every 2 days: `0 2 */2 * *`.
 
---
 
## 📡 Data Sources
 
### AQICN — Real-Time Pollutant Readings
 
| Parameter | Coverage |
|-----------|----------|
| **Primary station** | Islamabad `@11458` |
| **Fallback station** | Rawalpindi `@11459` |
| **Update frequency** | Hourly |
| **Pollutants** | AQI, PM2.5, PM10, NO₂, CO, O₃, SO₂ |
| **Cost** | Free token (register at aqicn.org) |
 
### Open-Meteo — Weather & Historical Air Quality
 
| API | Usage | Cost |
|-----|-------|------|
| Forecast API | Current weather + 4-day ahead forecast | Free, no key |
| Historical Archive API | Weather data back to 1940 (used in backfill) | Free, no key |
| Air Quality API | Historical PM2.5, PM10, NO₂, O₃, CO (CAMS model) | Free, no key |
 
> **Note**: Open-Meteo air quality data is derived from the CAMS global atmospheric model and may differ slightly from ground-level sensor readings. This is expected and accounted for in the feature engineering.
 
---
 
## 🔧 Feature Engineering
 
The feature pipeline transforms raw API responses into **35 model-ready features** per row:
 
### Pollutant Features (7)
`aqi`, `pm25`, `pm10`, `no2`, `co`, `o3`, `so2`
 
### Weather Features (9)
`temperature`, `humidity`, `wind_speed`, `wind_direction`, `pressure`, `precipitation`, `cloud_cover`, `wind_u`, `wind_v`
 
> `wind_u` and `wind_v` are the east–west and north–south components of the wind vector, computed as:
> `wind_u = −speed × sin(direction°)` and `wind_v = −speed × cos(direction°)`
 
### Time Features (10)
| Feature | Encoding | Rationale |
|---------|----------|-----------|
| `hour_sin`, `hour_cos` | `sin/cos(2π·h/24)` | Diurnal AQI cycle without edge discontinuity |
| `month_sin`, `month_cos` | `sin/cos(2π·m/12)` | Seasonal smog cycle (Nov–Feb peak in Islamabad) |
| `dow_sin`, `dow_cos` | `sin/cos(2π·d/7)` | Weekday vs weekend traffic patterns |
| `hour` | Raw integer | Direct hour value |
| `is_rush_hour` | Binary flag | 1 if hour ∈ {7,8,9,17,18,19} |
| `is_weekend` | Binary flag | 1 if Saturday or Sunday |
 
### Lag & Rolling Features (9)
| Feature | Description |
|---------|-------------|
| `aqi_lag_1h` | AQI from 1 hour ago |
| `aqi_lag_3h` | AQI from 3 hours ago |
| `aqi_lag_6h` | AQI from 6 hours ago |
| `aqi_lag_24h` | AQI from 24 hours ago (same hour, previous day) |
| `aqi_change_1h` | First-order difference: AQI(t) − AQI(t−1) |
| `aqi_roll_3h` | 3-hour rolling mean |
| `aqi_roll_6h` | 6-hour rolling mean |
| `aqi_roll_24h` | 24-hour rolling mean |
| `aqi_roll_std` | 6-hour rolling standard deviation (volatility) |
 
### Target Variables (3)
| Target | Horizon | Dashboard Slot |
|--------|---------|----------------|
| `target_aqi_24h` | 24 hours ahead | Day +1 forecast card |
| `target_aqi_48h` | 48 hours ahead | Day +2 forecast card |
| `target_aqi_72h` | 72 hours ahead | Day +3 forecast card |
 
---
 
## 🤖 Models & Performance
 
### Train/Validation Split
 
The pipeline uses a **time-series aware split** — the most recent 15% of data (≈26 days) is held out as the validation set and the preceding 85% is used for training. No shuffling is applied, preventing any data leakage from future observations.
 
### Model Comparison
 
| Model | Type | 24h RMSE | 24h R² | 48h RMSE | 48h R² | 72h RMSE | 72h R² |
|-------|------|----------|--------|----------|--------|----------|--------|
| Ridge Regression | Statistical baseline | 22.4 | 0.64 | 27.1 | 0.56 | 31.6 | 0.48 |
| Random Forest | Ensemble (bagging) | 15.7 | 0.80 | 19.3 | 0.73 | 23.1 | 0.65 |
| **XGBoost** ⭐ | **Ensemble (boosting)** | **12.9** | **0.84** | **16.4** | **0.78** | **20.2** | **0.70** |
| LSTM | Deep learning | 16.3 | 0.78 | 20.8 | 0.70 | 25.0 | 0.61 |
 
> ⭐ **XGBoost** is selected as the production model for all three horizons. It outperforms all other models on validation RMSE.
 
### Top SHAP Features (24h XGBoost)
 
| Rank | Feature | Mean \|SHAP\| | Interpretation |
|------|---------|--------------|----------------|
| 1 | `aqi_roll_24h` | 14.2 | 24-hour rolling mean dominates — persistence baseline |
| 2 | `aqi_lag_24h` | 11.8 | Same hour yesterday strongly predicts today |
| 3 | `pm25` | 8.9 | Raw PM2.5 directly drives AQI calculation |
| 4 | `wind_speed` | 7.3 | Higher wind → better dispersion → lower AQI |
| 5 | `aqi_lag_6h` | 6.1 | Recent 6-hour intra-day trend |
| 6 | `month_sin` | 5.4 | Seasonal smog cycle captured correctly |
| 7 | `temperature` | 4.7 | Temperature inversions in winter trap pollutants |
| 8 | `aqi_change_1h` | 3.9 | Rate-of-change signals rising or falling AQI |
| 9 | `precipitation` | 3.2 | Rainfall events cause sharp AQI drops |
| 10 | `hour_sin` | 2.8 | Diurnal pattern and rush-hour spikes |
 
---
 
## 📊 Dashboard
 
### Pages & Components
 
```
┌─────────────────────────────────────────────────────────────────┐
│  🌬️ AQI Forecast Dashboard — Islamabad / Rawalpindi, Pakistan    │
├────────────────────────────┬────────────────────────────────────┤
│  AQI GAUGE                 │  Current: PM2.5 │ PM10 │ NO₂ │ T  │
│  (Plotly Indicator)        │  Health advice text                 │
│                            │  Last updated: HH:MM UTC            │
├───────────────────────────────────────────────────────────────── ┤
│  📅 3-Day AQI Forecast                                           │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐                   │
│  │ Day +1    │  │ Day +2    │  │ Day +3    │                   │
│  │  145      │  │  130      │  │  110      │                   │
│  │ Unhealthy │  │ Unhealthy │  │ Moderate  │                   │
│  └───────────┘  └───────────┘  └───────────┘                   │
├─────────────────────────────────────────────────────────────────┤
│  📈 Forecast Bar Chart (with threshold lines at 100 and 150)    │
├─────────────────────────────────────────────────────────────────┤
│  📈 Last 7 Days — Historical AQI Trend                          │
├─────────────────────────────────────────────────────────────────┤
│  🔍 SHAP Feature Importance (updated daily by training pipeline)│
├─────────────────────────────────────────────────────────────────┤
│  🚨 HEALTH ALERT BANNER (visible only when AQI forecast > 150)  │
└─────────────────────────────────────────────────────────────────┘
```
 
### Data Refresh
 
| Component | Refresh Cadence | Source |
|-----------|----------------|--------|
| Live AQI gauge | Every page load (TTL: 1 hour) | AQICN API |
| 3-day forecast | Every page load (TTL: 1 hour) | Hopsworks online store + model registry |
| 7-day history | Every page load (TTL: 1 hour) | Hopsworks feature store |
| SHAP plot | Updated once daily after training pipeline | `artifacts/shap_summary.png` |
 
---
 
## 📏 AQI Reference
 
The dashboard uses the US EPA Air Quality Index scale:
 
| AQI Range | Category | Colour | Health Message |
|-----------|----------|--------|----------------|
| 0 – 50 | Good | 🟢 Green | Air quality is satisfactory. |
| 51 – 100 | Moderate | 🟡 Yellow | Acceptable. Sensitive groups may notice mild effects. |
| 101 – 150 | Unhealthy for Sensitive Groups | 🟠 Orange | Sensitive groups should limit prolonged outdoor exposure. |
| 151 – 200 | Unhealthy | 🔴 Red | Everyone may experience health effects. |
| 201 – 300 | Very Unhealthy | 🟣 Purple | Health alert: serious effects for everyone. |
| 301 – 500 | Hazardous | 🟤 Maroon | Emergency conditions. Avoid all outdoor activity. |
 
> In Islamabad/Rawalpindi, AQI above 200 is common during November–February (peak smog season).
 
---
 
## ⚙️ Configuration
 
All city-specific and system-level settings live in `config/config.yaml`:
 
```yaml
city:
  name: "Islamabad"
  country: "Pakistan"
  latitude: 33.6844
  longitude: 73.0479
  aqicn_station: "@517168"        # Primary: Islamabad
  aqicn_station_rwp: "@517168"    # Fallback: Rawalpindi
 
feature_store:
  project_name: "pearls_aqi_pred"  # Must match your Hopsworks project name exactly
  feature_group_name: "aqi_features"
  feature_group_version: 1
  target_group_name: "aqi_targets"
 
model:
  name: "aqi_predictor"
  forecast_horizon_days: 3
  lookback_hours: 48
  val_fraction: 0.15             # 15% of data used for validation
 
aqi_thresholds:
  good: 50
  moderate: 100
  unhealthy_sensitive: 150
  unhealthy: 200
  very_unhealthy: 300
  hazardous: 500
 
alerts:
  telegram_enabled: false        # Set to true and add secrets to enable
  aqi_alert_threshold: 150       # Send alert when forecast exceeds this value
```
 
---
 
## 🌐 Deployment
 
### Streamlit Community Cloud (recommended)
 
1. Push all code (including `artifacts/shap_summary.png`) to a public GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io) and click **New app**
3. Select your repository, branch (`main`), and main file (`src/streamlit_app.py`)
4. Click **Advanced settings** and add your secrets:
```toml
# Paste this into the Streamlit Cloud secrets editor
AQICN_TOKEN = "your_aqicn_token"
HOPSWORKS_API_KEY = "your_hopsworks_api_key"
```
 
5. Click **Deploy** — the app will be live in 2–5 minutes at a public URL
The app auto-redeploys on every push to `main`.
 
### Running Locally with Docker (optional)
 
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501"]
```
 
```bash
docker build -t aqi-predictor .
docker run -p 8501:8501 --env-file .env aqi-predictor
```
 
---
 
## 📝 Contributing
 
Contributions are welcome! Here's how to get started:
 
1. **Fork** the repository on GitHub
2. **Clone** your fork locally
3. **Create** a feature branch: `git checkout -b feature/your-feature-name`
4. **Make** your changes and add tests if applicable
5. **Run** the pipelines locally to verify nothing is broken
6. **Commit** with a clear message: `git commit -m "Add: description of change"`
7. **Push** to your fork and open a **Pull Request**
### Ideas for contributions
 
- Add support for additional Pakistani cities (Lahore, Karachi, Peshawar)
- Add Prophet or SARIMA as an additional statistical baseline model
- Integrate NASA MODIS satellite AOD data as an additional feature
- Add confidence intervals / prediction uncertainty to the dashboard
- Implement a mobile-friendly PWA wrapper around the Streamlit app
- Add LIME explanations alongside SHAP
---
 
## 📄 License
 
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
 
---
 
## 🙏 Acknowledgements
 
- [AQICN](https://aqicn.org) for providing free access to real-time air quality data from Islamabad
- [Open-Meteo](https://open-meteo.com) for the completely free weather and air quality forecast and historical APIs
- [Hopsworks](https://hopsworks.ai) for the free-tier serverless feature store and model registry
- The XGBoost and SHAP open-source communities
- Streamlit for the free Community Cloud hosting
---
 
<p align="center">
  Built with ❤️ for cleaner air in Islamabad &amp; Rawalpindi
</p>
 


