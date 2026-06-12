<div align="center">

# 🌬️ Pearls AQI Predictor

### End-to-End Air Quality Index Prediction System
**Islamabad / Rawalpindi, Pakistan — 3-Day Forecast — 100% Serverless**

---

*Developed by **Abubakar Imran** as part of the*
**10 Pearls Shine Internship Program — Cohort 8 | Data Sciences**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [1 · Clone & Install](#1--clone--install)
  - [2 · API Keys & Secrets](#2--api-keys--secrets)
  - [3 · Hopsworks Setup](#3--hopsworks-setup)
  - [4 · Run the Backfill](#4--run-the-backfill)
  - [5 · Run the Feature Pipeline](#5--run-the-feature-pipeline)
  - [6 · Train Models](#6--train-models)
  - [7 · Launch the Dashboard](#7--launch-the-dashboard)
- [Data Sources](#-data-sources)
- [Feature Engineering](#-feature-engineering)
- [Models & Performance](#-models--performance)
- [CI/CD Automation](#-cicd-automation)
- [Dashboard](#-dashboard)
- [Configuration](#-configuration)
- [AQI Reference](#-aqi-reference)
- [Deployment](#-deployment)
- [Limitations & Future Work](#-limitations--future-work)
- [Acknowledgements](#-acknowledgements)

---

## 🌍 Overview

The **Pearls AQI Predictor** is an end-to-end machine learning system that forecasts the Air Quality Index (AQI) for the **Islamabad–Rawalpindi metropolitan area** of Pakistan up to **72 hours (3 days)** in advance. It ingests live environmental data every hour, stores engineered features in a cloud feature store, retrains prediction models daily, and serves forecasts through a publicly accessible interactive dashboard — all at **zero infrastructure cost**.

Air quality in the twin cities has become a critical public health issue. During winter months (November–February), AQI readings regularly exceed 200 — the "Very Unhealthy" threshold — driven by crop burning in Punjab, cold temperature inversions, rush-hour traffic, and industrial emissions. These patterns are predictable from environmental data. This project makes those predictions accessible to anyone.

### Why This Project Stands Out

| Property | Value |
|----------|-------|
| **Fully serverless** | No VMs, no containers, no infrastructure to manage |
| **Self-updating** | GitHub Actions retrains models daily and refreshes features hourly — zero manual work |
| **Islamabad-specific** | Rush-hour flags, seasonal smog encoding, monsoon patterns built specifically for the twin cities |
| **Transparent** | SHAP values explain every prediction on the live dashboard |
| **Free to operate** | Hopsworks · GitHub Actions · Streamlit Community Cloud · Open-Meteo — all free tiers |
| **Author** | Abubakar Imran — 10 Pearls Shine Internship Program, Cohort 8, Data Sciences |

---

## 🚀 Live Demo

```
https://pearls-aqi-predictor-abubakar.streamlit.app/
```

> Replace the URL above with your actual Streamlit Community Cloud URL after deploying.

**What the dashboard shows:**
- 🔵 Live AQI gauge for Islamabad (updated every hour from AQICN station `@517168`)
- 📊 Current pollutant readings — PM2.5, PM10, NO₂, Temperature
- 📅 3-day colour-coded AQI forecast cards (Day+1, Day+2, Day+3)
- 📈 Forecast bar chart with health threshold reference lines at AQI = 100 and 150
- 🕐 Last 7 days of historical AQI from the feature store
- 🔍 SHAP feature importance plot from the latest daily training run
- 🚨 Health alert banners when forecast AQI exceeds unhealthy thresholds

---

## 🏗️ System Architecture

The system follows a four-stage MLOps pipeline orchestrated entirely by GitHub Actions:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      GITHUB ACTIONS  (CI/CD Orchestrator)                 │
│         Feature Pipeline: every hour    │   Training Pipeline: daily      │
│         cron: "0 * * * *"               │   cron: "0 2 * * *" (07:00 PKT) │
└───────────────────────┬─────────────────────────────┬────────────────────┘
                        │                             │
      ┌─────────────────▼──────┐          ┌───────────▼────────────────────┐
      │   ① DATA COLLECTION     │          │   ③ TRAINING PIPELINE           │
      │                         │          │                                 │
      │   AQICN API             │          │   • Fetch features from         │
      │   Station: @517168      │          │     Hopsworks feature store     │
      │   (Islamabad)           │          │   • Engineer lag/rolling feats  │
      │                         │          │   • Train Ridge, RF, XGBoost    │
      │   Open-Meteo API        │          │   • Select best model by RMSE   │
      │   (weather, no key)     │          │   • Compute SHAP values         │
      └──────────────┬──────────┘          │   • Register versioned model    │
                     │                     └───────────┬────────────────────-┘
      ┌──────────────▼──────────┐                      │
      │   ② FEATURE PIPELINE    │          ┌───────────▼────────────────────┐
      │                         │          │   ④ STREAMLIT DASHBOARD          │
      │   35 features/row       │          │                                 │
      │   Cyclical time enc.    │          │   • Live AQI gauge (AQICN)      │
      │   Wind U/V components   │          │   • 3-day forecast cards        │
      │   Lag & rolling feats   │          │   • SHAP importance plot        │
      │   Rush-hour flags       │          │   • Health alert banners        │
      └──────────────┬──────────┘          │   • 7-day history chart         │
                     │                     └─────────────────────────────────┘
      ┌──────────────▼──────────────────────────────────────────────────────┐
      │                      HOPSWORKS  (Free Tier)                          │
      │   Feature Store: aqi_features (v1) — online-enabled feature group   │
      │   Model Registry: aqi_predictor_24h / _48h / _72h (versioned)       │
      └─────────────────────────────────────────────────────────────────────┘
```

### Architecture Principles

- **Feature Store as integration layer** — Hopsworks decouples data production (hourly pipeline) from data consumption (training + inference); both components are independently deployable
- **Model Registry for versioning** — every daily run creates a new version; the dashboard always serves the best-performing version by RMSE via `get_best_model()`
- **Graceful degradation** — dashboard falls back to sample forecast data if the model registry is unavailable, ensuring the UI never crashes
- **No data leakage** — train/validation split is strictly chronological; validation set is always the most recent 15% of data

---

## 📁 Repository Structure

```
pearls-aqi-predictor/
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml      # Runs every hour  (cron: "0 * * * *")
│       └── training_pipeline.yml     # Runs daily at 02:00 UTC / 07:00 PKT
│
├── src/
│   ├── feature_pipeline.py           # Fetch → compute → store one hourly row
│   ├── backfill.py                   # Populate feature store with 6-month history
│   ├── training_pipeline.py          # Train models, evaluate, register best
│   ├── inference.py                  # Load model + latest features → 3-day forecast
│   └── streamlit_app.py              # Interactive web dashboard
│
├── notebooks/
│   └── 01_EDA.ipynb                  # Exploratory data analysis for Islamabad
│
├── config/
│   └── config.yaml                   # City coordinates, station ID, thresholds
│
├── artifacts/
│   └── shap_summary.png              # Updated by training pipeline after every run
│
├── model_artifacts/                  # Local model artefacts (git-ignored)
├── data/                             # Local CSV backfill backup (git-ignored)
├── requirements.txt
├── .env.example                      # Template — copy to .env and fill in secrets
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology | Role | Cost |
|-------|-----------|------|------|
| **Language** | Python 3.11 | All pipeline and dashboard code | Open source |
| **Feature Store** | [Hopsworks](https://app.hopsworks.ai) | Centralized feature storage + online serving | Free tier |
| **Model Registry** | Hopsworks (built-in) | Versioned model storage and retrieval | Included |
| **AQI Data** | [AQICN API](https://aqicn.org/api/) | Real-time ground-sensor pollutant readings | Free token |
| **Weather Data** | [Open-Meteo](https://open-meteo.com) | Current + historical weather, no key needed | Completely free |
| **ML — Tabular** | scikit-learn, XGBoost | Ridge, Random Forest, XGBoost regression | Open source |
| **Explainability** | SHAP | Feature importance and model transparency | Open source |
| **CI/CD** | GitHub Actions | Scheduled pipeline orchestration | 2,000 min/month free |
| **Dashboard** | Streamlit | Interactive Python-native web UI | Free community cloud |
| **Visualization** | Plotly | Gauge, bar charts, line charts | Open source |
| **Serialization** | joblib | Model persistence (`.pkl` files) | Open source |
| **Config** | PyYAML + python-dotenv | Configuration and secrets management | Open source |

---

## 🚦 Getting Started

### Prerequisites

- Python 3.11 or higher
- A free [Hopsworks account](https://app.hopsworks.ai) — create a project named `pearls_aqi_pred`
- A free [AQICN API token](https://aqicn.org/data-platform/token/)
- Git

---

### 1 · Clone & Install

```bash
git clone https://github.com/Abubakar-Imran/pearls-aqi-predictor.git
cd pearls-aqi-predictor

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

**Verify key imports after installation:**

```bash
python -c "import hopsworks; print('hopsworks OK')"
python -c "import xgboost;   print('xgboost   OK')"
python -c "import shap;      print('shap      OK')"
python -c "import streamlit; print('streamlit OK')"
```

---

### 2 · API Keys & Secrets

```bash
cp .env.example .env
```

Open `.env` and fill in your credentials:

```dotenv
# ── AQICN ─────────────────────────────────────────────────────────────────────
# Register for a free token at: https://aqicn.org/data-platform/token/
AQICN_TOKEN=your_aqicn_token_here

# ── HOPSWORKS ─────────────────────────────────────────────────────────────────
# Log into https://app.hopsworks.ai → your project → Settings → API Keys
HOPSWORKS_API_KEY=your_hopsworks_api_key_here
```

> ⚠️ **Never commit your `.env` file.** It is listed in `.gitignore` by default.

---

### 3 · Hopsworks Setup

Verify your connection before running any pipeline:

```bash
python - <<'EOF'
import os
from dotenv import load_dotenv
import hopsworks

load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
print(f"✓ Connected to project: {project.name}")
fs = project.get_feature_store()
print(f"✓ Feature store ready: {fs.name}")
EOF
```

Make sure the `project_name` in `config/config.yaml` **exactly matches** your Hopsworks project name (case-sensitive):

```yaml
feature_store:
  project_name: "pearls_aqi_pred"   # ← must match your Hopsworks project exactly
```

> **Dependency note**: The workflow files explicitly pre-install `hopsworks[python]==4.7.*` and `confluent-kafka` before `requirements.txt`. This prevents a known issue where GitHub Actions pip caching restores a stale environment missing `confluent-kafka`, which Hopsworks needs for its feature store write path.

---

### 4 · Run the Backfill

This one-time step populates the feature store with ~6 months of historical data before the first model training run.

```bash
python src/backfill.py
```

**Expected output:**
```
Backfilling from 2025-12-09 to 2026-06-09...
  Fetching 2025-12-09 → 2026-01-08...
  Fetching 2026-01-09 → 2026-02-07...
  ...
  Total records: 4320 weather, 4320 AQI
  Features computed: 4284 rows, 35 columns
[DONE] Backfill complete. 4284 rows stored.
```

> ⏱️ **Expected runtime**: 5–15 minutes depending on internet speed.

**Verify the data was stored correctly:**

```bash
python - <<'EOF'
import os, hopsworks
from dotenv import load_dotenv
load_dotenv()

project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
fg = fs.get_feature_group("aqi_features", version=1)
df = fg.read()
print(f"Rows in feature store : {len(df)}")
print(f"Date range            : {df['timestamp'].min()} → {df['timestamp'].max()}")
print(df[["timestamp", "aqi", "pm25", "temperature"]].tail(3))
EOF
```

**Minimum acceptable dataset for training:**

| Metric | Minimum | Recommended |
|--------|---------|-------------|
| Total rows | 1,000 | 3,000+ |
| Date range | 60 days | 180 days |
| Missing AQI % | < 20% | < 10% |

---

### 5 · Run the Feature Pipeline

Insert one live row to confirm the end-to-end ingestion path works:

```bash
python src/feature_pipeline.py
```

**Expected output:**
```
Fetching AQICN data...
Fetching weather data...
Computing features...
Storing features in Hopsworks...
[OK] Features stored at 2026-06-09T10:00:00+00:00
```

**Common errors and fixes:**

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `AQICN API error: nauth` | Invalid or expired token | Regenerate at https://aqicn.org/data-platform/token/ |
| `AQICN API error: Over Quota` | Rate limit hit | Wait 1 hour; station `@517168` has hourly limits |
| `KeyError: 'pm25'` | Station has no PM2.5 sensor reading right now | The safe extractor returns `NaN` — this is handled automatically |
| `RestAPIError` from Hopsworks | Wrong project name or invalid API key | Check `config.yaml` project name matches Hopsworks exactly |
| Feature group schema mismatch | Column added/removed after first insert | Delete the feature group in Hopsworks UI and re-run backfill |

---

### 6 · Train Models

```bash
python src/training_pipeline.py
```

**Expected output:**
```
Fetching training data from Feature Store...
  Total samples: 4284

==================================================
Training for target: target_aqi_24h
  Training ridge...         RMSE=22.4, MAE=16.8, R²=0.64
  Training random_forest... RMSE=15.7, MAE=11.4, R²=0.80
  Training xgboost...       RMSE=12.9, MAE=9.5,  R²=0.84
  ✅ Best model: xgboost
  Computing SHAP values → artifacts/shap_summary.png
  Model registered: aqi_predictor_target_aqi_24h v1
...
=== TRAINING SUMMARY ===
target_aqi_24h: RMSE=12.9, MAE=9.5,  R²=0.84  (xgboost)
target_aqi_48h: RMSE=16.4, MAE=12.1, R²=0.78  (xgboost)
target_aqi_72h: RMSE=20.2, MAE=13.7, R²=0.70  (xgboost)
```

> ⏱️ **Expected runtime**: 10–25 minutes.

**If model R² is below 0.60, work through this checklist:**

1. Verify backfill data has sufficient rows (`len(df) > 2000`)
2. Check lag feature correlation: `df["aqi_lag_24h"].corr(df["aqi"])` — should be > 0.70
3. Check for excessive NaN values: `df[FEATURE_COLS].isna().mean()` — should be < 0.15
4. Try increasing XGBoost trees: `n_estimators=500, max_depth=8, learning_rate=0.03`

---

### 7 · Launch the Dashboard

```bash
streamlit run src/streamlit_app.py
```

Open your browser at **http://localhost:8501**

**Dashboard validation checklist:**
- [ ] AQI gauge shows a non-zero real value from AQICN
- [ ] "Last updated" timestamp is within the last 2 hours
- [ ] All four metric cards (PM2.5, PM10, NO₂, Temperature) show values
- [ ] Three forecast day cards appear with coloured AQI values
- [ ] Forecast bar chart renders with threshold lines at 100 and 150
- [ ] No "Could not load live model predictions" warning appears
- [ ] SHAP image renders at the bottom of the page
- [ ] Health alert banner appears/disappears correctly based on forecast values

---

## 📡 Data Sources

### AQICN — Real-Time Pollutant Readings

[AQICN](https://aqicn.org) aggregates data from thousands of ground-level monitoring stations globally. This project uses the Islamabad station for all live ingestion.

| Parameter | Value |
|-----------|-------|
| **Station ID** | `@517168` (Islamabad) |
| **API Endpoint** | `https://api.waqi.info/feed/@517168/?token={TOKEN}` |
| **Update Frequency** | Hourly |
| **Authentication** | Free token — [register here](https://aqicn.org/data-platform/token/) |
| **Pollutants** | AQI, PM2.5, PM10, NO₂, CO, O₃, SO₂ |

### Open-Meteo — Weather & Historical Air Quality

[Open-Meteo](https://open-meteo.com) is a completely free weather API requiring no API key or registration.

| API | Usage in this Project |
|-----|-----------------------|
| **Forecast API** | Current weather conditions — used by the live feature pipeline |
| **Historical Archive API** | Hourly weather back to 1940 — used by the backfill script |
| **Air Quality API** | CAMS-model derived PM2.5/PM10/NO₂/O₃/CO — historical pollutant data for backfill |

> **Note**: Open-Meteo air quality figures are derived from the CAMS global atmospheric model and may differ slightly from ground-sensor readings. This known domain gap is documented in the Limitations section.

---

## 🔧 Feature Engineering

The feature pipeline transforms raw API responses into **35 model-ready features** per row. All features are stored in the Hopsworks online feature store and served at low latency to both the training pipeline and the inference module.

### Pollutant Features (7)
Raw readings extracted safely from AQICN — API quirks (dashes, nulls) converted to `NaN`:

`aqi` · `pm25` · `pm10` · `no2` · `co` · `o3` · `so2`

### Weather Features (9)
Seven raw values from Open-Meteo plus two derived wind-vector components:

`temperature` · `humidity` · `wind_speed` · `wind_direction` · `pressure` · `precipitation` · `cloud_cover` · `wind_u` · `wind_v`

**Wind decomposition** eliminates the circular discontinuity of raw wind direction:

```python
wind_u = −wind_speed × sin(radians(wind_direction))  # east–west
wind_v = −wind_speed × cos(radians(wind_direction))  # north–south
```

This allows the model to learn directional effects (e.g., north-westerly winds bringing cleaner air from the Margalla Hills) without treating 359° as numerically distant from 1°.

### Temporal Features (10)

| Feature | Encoding | Captures |
|---------|----------|----------|
| `hour_sin` / `hour_cos` | `sin/cos(2π × h / 24)` | Diurnal AQI cycle — rush-hour peaks |
| `month_sin` / `month_cos` | `sin/cos(2π × m / 12)` | Seasonal smog cycle (Nov–Feb peak in Islamabad) |
| `dow_sin` / `dow_cos` | `sin/cos(2π × d / 7)` | Weekday vs. weekend traffic emission patterns |
| `hour` | Raw integer 0–23 | Direct lookup by tree-based models |
| `is_rush_hour` | Binary | 1 if hour ∈ {7, 8, 9, 17, 18, 19} — Islamabad peak traffic |
| `is_weekend` | Binary | 1 if Saturday or Sunday — ~15% lower AQI than weekdays |

### Lag & Rolling Features (9)

| Feature | Description | Window |
|---------|-------------|--------|
| `aqi_lag_1h` | AQI 1 hour ago | 1 h |
| `aqi_lag_3h` | AQI 3 hours ago | 3 h |
| `aqi_lag_6h` | AQI 6 hours ago — captures intra-day trend | 6 h |
| `aqi_lag_24h` | AQI at the same hour yesterday | 24 h |
| `aqi_change_1h` | `AQI(t) − AQI(t−1)` — rate of change | Δ 1 h |
| `aqi_roll_3h` | 3-hour rolling mean | 3 h mean |
| `aqi_roll_6h` | 6-hour rolling mean | 6 h mean |
| `aqi_roll_24h` | 24-hour rolling mean | 24 h mean |
| `aqi_roll_std` | 6-hour rolling standard deviation (volatility) | 6 h std |

### Target Variables (3)

| Target | Horizon | Dashboard Slot |
|--------|---------|----------------|
| `target_aqi_24h` | AQI 24 hours ahead (`df.shift(−24)`) | Day +1 forecast card |
| `target_aqi_48h` | AQI 48 hours ahead (`df.shift(−48)`) | Day +2 forecast card |
| `target_aqi_72h` | AQI 72 hours ahead (`df.shift(−72)`) | Day +3 forecast card |

Each target gets its own independently trained model, so the 24h model is specifically optimised for that horizon rather than sharing parameters with the 72h model.

---

## 🤖 Models & Performance

### Train/Validation Split

A **strictly chronological split** with no shuffling: the most recent **15%** of data (~26 days) is held out as the validation set and the earlier **85%** is used for training. This mirrors real-world deployment — the model always predicts future timestamps it has never seen during training.

### Model Comparison

| Model | 24h RMSE | 24h MAE | 24h R² | 48h RMSE | 48h R² | 72h RMSE | 72h R² |
|-------|----------|---------|--------|----------|--------|----------|--------|
| Ridge Regression | 22.4 | 16.8 | 0.64 | 27.1 | 0.56 | 31.6 | 0.48 |
| Random Forest | 15.7 | 11.4 | 0.80 | 19.3 | 0.73 | 23.1 | 0.65 |
| **XGBoost** ⭐ | **12.9** | **9.5** | **0.84** | **16.4** | **0.78** | **20.2** | **0.70** |

> ⭐ **XGBoost** is selected as the production model at all three horizons. It achieves the best RMSE and all targets exceed the pre-defined R² thresholds (0.70 / 0.60 / 0.50 for 24h / 48h / 72h respectively).

### Top SHAP Features (XGBoost 24h model)

| Rank | Feature | Mean \|SHAP\| | Physical Interpretation |
|------|---------|:------------:|------------------------|
| 1 | `aqi_roll_24h` | 14.2 | 24-hour rolling mean dominates — AQI is highly persistent |
| 2 | `aqi_lag_24h` | 11.8 | Same hour yesterday is the strongest single lag predictor |
| 3 | `pm25` | 8.9 | Raw PM2.5 directly drives the AQI breakpoint calculation |
| 4 | `wind_speed` | 7.3 | Most important met. feature — high wind disperses pollutants |
| 5 | `aqi_lag_6h` | 6.1 | Recent intra-day trend signals accelerating or slowing change |
| 6 | `month_sin` | 5.4 | Seasonal smog cycle — winter months correctly dominate |
| 7 | `temperature` | 4.7 | Cold temperature inversions trap PM2.5 near ground level |
| 8 | `aqi_change_1h` | 3.9 | Rate-of-change: whether AQI is rising or falling |
| 9 | `precipitation` | 3.2 | Rainfall causes sharp AQI drops via wet deposition |
| 10 | `hour_sin` | 2.8 | Diurnal cycle — rush-hour peaks correctly flagged |

SHAP values are recomputed after every daily training run and the resulting plot is saved to `artifacts/shap_summary.png`, which is displayed live on the dashboard.

---

## ⚙️ CI/CD Automation

Both pipelines run automatically via GitHub Actions once you push to GitHub and configure repository secrets.

### GitHub Repository Secrets

Go to your repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:

| Secret Name | Where to Get It |
|-------------|----------------|
| `AQICN_TOKEN` | [aqicn.org/data-platform/token](https://aqicn.org/data-platform/token/) |
| `HOPSWORKS_API_KEY` | Hopsworks UI → Project → Settings → API Keys |

### Pipeline Schedules

| Workflow | File | Schedule | Cron | Avg Runtime |
|----------|------|----------|------|-------------|
| Feature Pipeline | `feature_pipeline.yml` | Every hour | `0 * * * *` | ~2–3 min |
| Training Pipeline | `training_pipeline.yml` | Daily, 07:00 PKT | `0 2 * * *` | ~10–20 min |

### Triggering Manually (Recommended First Test)

Don't wait for the schedule — trigger both workflows immediately to verify they work:

1. Go to your repo on GitHub → **Actions** tab
2. Click **Feature Pipeline (Hourly)** → **Run workflow** → **Run workflow**
3. Wait for it to complete (green ✓), then check Hopsworks for the new row
4. Click **Training Pipeline (Daily)** → **Run workflow** → **Run workflow**
5. Wait for it to complete and check Hopsworks Model Registry for a new model version

### GitHub Actions Usage

| Workflow | Frequency | Avg Runtime | Monthly Minutes |
|----------|-----------|-------------|-----------------|
| Feature Pipeline | 720 runs/month | ~2 min | ~1,440 min |
| Training Pipeline | 30 runs/month | ~15 min | ~450 min |
| **Total** | | | **~1,890 min** |

> 💡 GitHub Actions free tier provides **2,000 minutes/month** for public repositories. The combined pipelines use approximately 1,890 min/month. If you approach the limit, reduce the training frequency to every 2 days: `cron: "0 2 */2 * *"`.

### Dependency Management Note

The feature pipeline workflow installs Hopsworks with an explicit pre-install step:

```yaml
- name: Install Hopsworks dependencies first
  run: |
    pip install --upgrade pip
    pip install "hopsworks[python]==4.7.*" confluent-kafka
    pip install -r requirements.txt
```

This is intentional — GitHub Actions pip caching can restore a stale environment that lacks `confluent-kafka`, which Hopsworks requires for its feature store write path. The explicit install guarantees availability regardless of cache state.

---

## 📊 Dashboard

### Component Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│  🌬️  AQI Forecast Dashboard — Islamabad / Rawalpindi, Pakistan            │
├───────────────────────────────────┬──────────────────────────────────────┤
│  AQI GAUGE  (Plotly Indicator)    │  PM2.5 │ PM10 │ NO₂ │ Temperature   │
│  0 ─────────── 150 ──────── 300   │  Health advice text                  │
│  Colour-coded by EPA categories   │  Last updated: HH:MM UTC             │
├──────────────────────────────────────────────────────────────────────────┤
│  📅  3-Day AQI Forecast                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │
│  │  Day + 1    │  │  Day + 2    │  │  Day + 3    │                      │
│  │    145      │  │    130      │  │    110      │                      │
│  │  Unhealthy  │  │  Unhealthy  │  │  Moderate   │                      │
│  └─────────────┘  └─────────────┘  └─────────────┘                      │
├──────────────────────────────────────────────────────────────────────────┤
│  📈  Forecast Bar Chart  (with reference lines at AQI 100 and 150)        │
├──────────────────────────────────────────────────────────────────────────┤
│  📈  Last 7 Days — Historical AQI Trend                                   │
├──────────────────────────────────────────────────────────────────────────┤
│  🔍  SHAP Feature Importance  (updated daily by training pipeline)        │
├──────────────────────────────────────────────────────────────────────────┤
│  🚨  HEALTH ALERT BANNER  (shown only when forecast AQI > 150)            │
└──────────────────────────────────────────────────────────────────────────┘
```

### Data Refresh

| Component | Refresh Cadence | Source |
|-----------|----------------|--------|
| Live AQI gauge | Every page load (TTL: 1 h) | AQICN API — station `@517168` |
| Pollutant metric cards | Every page load (TTL: 1 h) | AQICN API |
| 3-day forecast cards | Every page load (TTL: 1 h) | Hopsworks online store + model registry |
| 7-day history chart | Every page load (TTL: 1 h) | Hopsworks feature store |
| SHAP importance plot | Once daily after training | `artifacts/shap_summary.png` |

### Health Alert Thresholds

| Forecast AQI | Alert Type | Dashboard Response |
|:------------:|------------|-------------------|
| 0 – 100 | None | No banner displayed |
| 101 – 150 | Caution | `st.warning()` — yellow banner advising sensitive groups |
| 151 – 200 | Unhealthy | `st.error()` — red banner advising masks and reduced outdoor time |
| 200+ | Emergency | `st.error()` — red banner advising all to avoid outdoor activity |

---

## ⚙️ Configuration

All city-specific and system-level parameters are in `config/config.yaml`:

```yaml
city:
  name: "Islamabad"
  country: "Pakistan"
  latitude: 33.6844
  longitude: 73.0479
  aqicn_station: "@517168"             # Islamabad monitoring station

feature_store:
  project_name: "pearls_aqi_pred"     # Must match Hopsworks project exactly
  feature_group_name: "aqi_features"
  feature_group_version: 1

model:
  name: "aqi_predictor"
  forecast_horizon_days: 3
  lookback_hours: 48
  val_fraction: 0.15                   # 15% most-recent data used for validation

aqi_thresholds:
  good: 50
  moderate: 100
  unhealthy_sensitive: 150
  unhealthy: 200
  very_unhealthy: 300
  hazardous: 500
```

> The modular config design means redirecting the entire system to a different city requires only updating the `city` block — no code changes needed.

---

## 📏 AQI Reference

The dashboard uses the **US EPA Air Quality Index** scale:

| AQI Range | Category | Colour | Health Guidance |
|:---------:|----------|:------:|----------------|
| 0 – 50 | Good | 🟢 Green | Satisfactory air quality; outdoor activities are safe |
| 51 – 100 | Moderate | 🟡 Yellow | Acceptable; very sensitive individuals may notice mild effects |
| 101 – 150 | Unhealthy for Sensitive Groups | 🟠 Orange | Elderly, children, and asthmatics should reduce outdoor exposure |
| 151 – 200 | Unhealthy | 🔴 Red | Everyone may experience health effects; use N95 masks outdoors |
| 201 – 300 | Very Unhealthy | 🟣 Purple | Health alert — serious effects likely for the general population |
| 301 – 500 | Hazardous | 🟤 Maroon | Emergency conditions — avoid all outdoor activity |

> In Islamabad/Rawalpindi, AQI above 200 is a **recurring phenomenon** during November–February (peak smog season), making 72-hour advance forecasting particularly valuable for public health planning.

---

## 🌐 Deployment

### Streamlit Community Cloud (Recommended)

1. Push all code — including `artifacts/shap_summary.png` — to a public GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repository, branch (`main`), and main file (`src/streamlit_app.py`)
4. Click **Advanced settings** → add secrets:

```toml
# Paste into the Streamlit Cloud secrets editor
AQICN_TOKEN        = "your_aqicn_token"
HOPSWORKS_API_KEY  = "your_hopsworks_api_key"
```

5. Click **Deploy** — the app will be live in 2–5 minutes at a public URL
6. The app **auto-redeploys** on every push to `main`

### Local Development

```bash
git clone https://github.com/Abubakar-Imran/pearls-aqi-predictor.git
cd pearls-aqi-predictor
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # Add your API keys to .env
streamlit run src/streamlit_app.py
```

### Docker (Optional)

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
docker build -t pearls-aqi-predictor .
docker run -p 8501:8501 --env-file .env pearls-aqi-predictor
```

---

## ⚠️ Limitations & Future Work

### Known Limitations

| Limitation | Detail |
|-----------|--------|
| **Single station dependency** | Live pipeline depends on one AQICN sensor. If the station goes offline, lag features propagate `NaN` into model inputs. |
| **CAMS vs. ground-sensor discrepancy** | Backfill uses Open-Meteo CAMS model air quality — systematically underestimates PM2.5 peaks compared to ground sensors. |
| **No uncertainty quantification** | Predictions are point estimates. No confidence intervals are currently produced or displayed. |
| **GitHub Actions minute limits** | ~1,890 min/month approaches the 2,000 min free tier; pipeline optimisation is required to stay within quota. |
| **No sequence model** | The current training pipeline uses only tabular models (Ridge, RF, XGBoost). An LSTM would better capture multi-step temporal patterns. |

### Suggested Future Extensions

- Integrate AQICN historical data (via partner API) to replace CAMS-derived training data, eliminating the domain gap
- Add NASA MODIS satellite Aerosol Optical Depth (AOD) as an additional feature — literature shows 5–10% RMSE reduction for PM2.5 forecasting
- Implement probabilistic forecasting using quantile regression or conformal prediction to add forecast confidence intervals
- Add a SARIMA or Prophet statistical baseline for additional model comparison
- Extend to additional Pakistani cities (Lahore, Karachi, Peshawar) — the modular city config makes this straightforward
- Implement multi-station ensemble averaging using additional AQICN stations across Islamabad and Rawalpindi sectors
- Add LSTM/GRU sequence model to the training pipeline comparison

---

## 🙏 Acknowledgements

- **[AQICN](https://aqicn.org)** — Free access to real-time air quality sensor data from Islamabad
- **[Open-Meteo](https://open-meteo.com)** — Completely free historical and forecast weather and air quality APIs; no key required
- **[Hopsworks](https://hopsworks.ai)** — Free-tier serverless feature store and model registry
- The **XGBoost** and **SHAP** open-source communities
- **Streamlit** for the free Community Cloud hosting
- **10 Pearls Technology** for the Shine Internship Program opportunity

---

