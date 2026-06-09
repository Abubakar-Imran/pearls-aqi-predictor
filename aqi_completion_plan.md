# Pearls AQI Predictor — Completion Plan
**Date**: 2026-06-04 | **City**: Islamabad / Rawalpindi, Pakistan

---

## Overview

All source files exist. The remaining work is entirely about **running, verifying, debugging, and deploying** each component — then producing the final report. This plan is ordered by dependency: you cannot train without backfill data, and you cannot deploy without a trained model.

```
Phase A: Environment & Keys        (~30 min)
Phase B: Run Feature Pipeline      (~30 min)
Phase C: Run Backfill              (~1–2 hrs)
Phase D: Run Training Pipeline     (~1–2 hrs)
Phase E: Validate Dashboard        (~1 hr)
Phase F: GitHub Actions & CI/CD    (~45 min)
Phase G: Optional Enhancements     (~2–4 hrs)
Phase H: Final Report              (~3–4 hrs)
```

---

## Phase A — Environment & Credential Verification

> Goal: Confirm every external dependency (APIs, Hopsworks, Python packages) works before running any pipeline.

### A1. Install and verify Python dependencies

```bash
pip install -r requirements.txt

# Spot-check key imports
python -c "import hopsworks; print('hopsworks OK')"
python -c "import xgboost; print('xgboost OK')"
python -c "import shap; print('shap OK')"
python -c "import streamlit; print('streamlit OK')"
```

**Common issues and fixes:**

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: hopsworks` | `pip install hopsworks==3.7.0` |
| `OSError: libgomp` on Linux | `sudo apt-get install libgomp1` |
| `xgboost` build error on Windows | Install via `conda install -c conda-forge xgboost` |

---

### A2. Verify AQICN API token

```bash
# Replace YOUR_TOKEN with what's in your .env
curl "https://api.waqi.info/feed/@11458/?token=YOUR_TOKEN"

# Expected: {"status":"ok","data":{"aqi":...}}
# If you get "nauth" or "Invalid key": regenerate at https://aqicn.org/data-platform/token/
```

---

### A3. Verify Open-Meteo (no key needed)

```bash
curl "https://api.open-meteo.com/v1/forecast?latitude=33.6844&longitude=73.0479&current=temperature_2m&forecast_days=1"
# Expected: {"current":{"temperature_2m": ...}}
```

---

### A4. Verify Hopsworks connection

```python
# Run this as a standalone script: python check_hopsworks.py
import os
from dotenv import load_dotenv
import hopsworks

load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
print(f"Connected to project: {project.name}")
fs = project.get_feature_store()
print(f"Feature store: {fs.name}")
```

**If login fails:**
- Log into https://app.hopsworks.ai
- Go to your project → Settings → API Keys → copy the key
- Paste it in your `.env` as `HOPSWORKS_API_KEY=...`
- Make sure the project name in `config/config.yaml` exactly matches your Hopsworks project name

---

## Phase B — Run Feature Pipeline (First Live Data Point)

> Goal: Insert one real row of features into Hopsworks to confirm the full write path works.

### B1. Run the feature pipeline once manually

```bash
python src/feature_pipeline.py
```

**Expected output:**
```
Fetching AQICN data...
Fetching weather data...
Computing features...
Storing features in Hopsworks...
[OK] Features stored at 2026-06-04T...
```

### B2. Verify the row landed in Hopsworks

```python
# python verify_feature_store.py
import hopsworks, os
from dotenv import load_dotenv

load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
fg = fs.get_feature_group("aqi_features", version=1)
df = fg.read()
print(f"Total rows in feature store: {len(df)}")
print(df[["timestamp","aqi","pm25","temperature"]].tail(3))
```

### B3. Common pipeline errors and fixes

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `AQICN API error: Over Quota` | Rate limit hit | Wait 1 hr or use station `@11459` (Rawalpindi) as fallback |
| `KeyError: 'pm25'` in `extract_pollutants` | Station has no PM2.5 sensor right now | Add `.get("pm25", {}).get("v", None)` with explicit None handling |
| `hopsworks.client.exceptions.RestAPIError` | Wrong project name | Check `config.yaml` project name matches Hopsworks exactly |
| Feature group schema mismatch | Column added/removed after first insert | Delete the feature group in Hopsworks UI and re-create |

---

## Phase C — Run Backfill (Historical Training Data)

> Goal: Populate Hopsworks with at least 90 days of hourly data (2,160+ rows) before training.

### C1. Run backfill for 6 months

```bash
python src/backfill.py
# Expected runtime: 5–15 minutes depending on internet speed
```

**Watch for these specific issues:**

#### Issue 1: Open-Meteo Air Quality API coverage for Pakistan
Open-Meteo uses CAMS (Copernicus) model data. Islamabad is covered but verify:
```python
import requests
r = requests.get(
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    "?latitude=33.6844&longitude=73.0479"
    "&hourly=pm2_5,european_aqi&start_date=2025-12-01&end_date=2025-12-07"
)
data = r.json()
print(data["hourly"]["pm2_5"][:5])   # Should return numeric values, not all None
```
If all `None`, switch to `us_aqi` instead of `european_aqi` in `backfill.py`.

#### Issue 2: Hopsworks insert takes too long for large batches
Insert in chunks of 30 days at a time:
```python
# In backfill.py, after building features_df:
CHUNK_SIZE = 30 * 24   # 30 days of hourly rows
for i in range(0, len(features_df), CHUNK_SIZE):
    chunk = features_df.iloc[i:i+CHUNK_SIZE]
    fg.insert(chunk, write_options={"wait_for_job": False})
    print(f"Inserted rows {i} to {i+len(chunk)}")
```

#### Issue 3: NaN-heavy lag features in the first 24 rows
This is expected — the first 24 rows will have NaN lags because there's no earlier data. The training pipeline already calls `dropna()`, so these will be excluded automatically.

### C2. Verify backfill data volume

```python
# Should show 3,000+ rows after a 6-month backfill
fg = fs.get_feature_group("aqi_features", version=1)
df = fg.read()
print(f"Total rows: {len(df)}")
print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"Missing AQI values: {df['aqi'].isna().sum()}")
print(df.describe()[["aqi","pm25","temperature"]])
```

**Minimum acceptable dataset for training:**

| Metric | Minimum | Recommended |
|--------|---------|-------------|
| Total rows | 1,000 | 3,000+ |
| Date range | 60 days | 180 days |
| Missing AQI % | < 20% | < 10% |

---

## Phase D — Run Training Pipeline

> Goal: Train models for all 3 forecast horizons (24h, 48h, 72h), compare results, save SHAP plot, register in Hopsworks.

### D1. Run training

```bash
python src/training_pipeline.py
```

**Expected output:**
```
Fetching training data from Feature Store...
  Total samples: 3240

==================================================
Training for target: target_aqi_24h
  Training ridge for target_aqi_24h...
    RMSE=18.4, MAE=13.2, R²=0.71
  Training random_forest for target_aqi_24h...
    RMSE=14.1, MAE=10.8, R²=0.82
  Training xgboost for target_aqi_24h...
    RMSE=12.6, MAE=9.4, R²=0.86
  ✅ Best model for target_aqi_24h: xgboost
  Computing SHAP values...
  SHAP plot saved to artifacts/shap_summary.png
  Model registered: aqi_predictor_target_aqi_24h v1
...

=== TRAINING SUMMARY ===
target_aqi_24h: RMSE=12.6, MAE=9.4, R²=0.86 (xgboost)
target_aqi_48h: RMSE=15.3, MAE=11.2, R²=0.79 (xgboost)
target_aqi_72h: RMSE=18.1, MAE=13.7, R²=0.72 (xgboost)
```

### D2. Model performance targets

These are realistic targets for AQI prediction on Pakistani air quality data:

| Horizon | Acceptable R² | Good R² | RMSE target |
|---------|--------------|---------|-------------|
| 24h | > 0.70 | > 0.82 | < 20 AQI units |
| 48h | > 0.60 | > 0.75 | < 25 AQI units |
| 72h | > 0.50 | > 0.65 | < 30 AQI units |

### D3. If model performance is poor (R² < 0.5)

Work through this checklist in order:

**1. Check feature quality first**
```python
df = fg.read()
print(df[["aqi","aqi_lag_24h","aqi_roll_24h"]].corr())
# aqi_lag_24h should correlate 0.7+ with aqi
```

**2. Check AQI range — Islamabad can spike above 200 in winter**
```python
print(df["aqi"].describe())
# If std > 80 and max > 300, normalize targets: use log(aqi+1) as target
```

**3. Add winter dummy feature**
```python
# In training_pipeline.py FEATURE_COLS, add:
df["is_winter"] = df["month"].apply(lambda m: int(m in [11, 12, 1, 2]))
# Winter in Islamabad = heavy smog from crop burning in Punjab
```

**4. Increase tree depth for XGBoost**
```python
xgb.XGBRegressor(max_depth=8, n_estimators=500, learning_rate=0.03)
```

**5. Try a gradient boosting ensemble**
```python
from sklearn.ensemble import VotingRegressor
ensemble = VotingRegressor([("rf", rf_model), ("xgb", xgb_model)], weights=[0.4, 0.6])
```

### D4. Verify model is accessible from Model Registry

```python
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
mr = project.get_model_registry()
models = mr.get_models("aqi_predictor_target_aqi_24h")
print(f"Registered versions: {[m.version for m in models]}")
best = mr.get_best_model("aqi_predictor_target_aqi_24h", metric="rmse", direction="min")
print(f"Best model: v{best.version}, RMSE={best.training_metrics}")
```

---

## Phase E — Validate Streamlit Dashboard End-to-End

> Goal: Confirm the live app loads real AQI, fetches model predictions, and renders all visualizations correctly.

### E1. Launch locally

```bash
streamlit run src/streamlit_app.py
# Opens at http://localhost:8501
```

### E2. Dashboard validation checklist

Go through each of these manually:

- [ ] AQI gauge shows a non-zero real value from AQICN
- [ ] "Last updated" timestamp is recent (within 1–2 hours)
- [ ] All 4 pollutant metric cards (PM2.5, PM10, NO₂, Temp) show values
- [ ] 3 forecast day cards appear with colored AQI values
- [ ] Forecast bar chart renders with threshold lines at 100 and 150
- [ ] No "Could not load live model predictions" warning (means model loaded OK)
- [ ] SHAP image appears at the bottom
- [ ] Health alert banner appears/disappears correctly based on forecast values

### E3. If "Could not load live model predictions" warning appears

This means the inference code in `streamlit_app.py` failed to pull from Hopsworks. Debug it:

```python
# Run this standalone to isolate the issue:
import hopsworks, os, joblib
from dotenv import load_dotenv
load_dotenv()

project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
mr = project.get_model_registry()

# Step 1: List available models
for m in mr.get_models():
    print(m.name, m.version)

# Step 2: Download and load
best = mr.get_best_model("aqi_predictor_target_aqi_24h", metric="rmse", direction="min")
saved_dir = best.download()
print(f"Downloaded to: {saved_dir}")

import os
print(os.listdir(saved_dir))   # Confirm .pkl file is there
```

### E4. Add historical AQI chart to dashboard (enhancement)

The dashboard currently shows current + 3-day forecast. Add a 7-day history chart for more context:

```python
# In streamlit_app.py, add this section after the forecast chart:
@st.cache_data(ttl=3600)
def load_recent_history():
    project = hopsworks.login(api_key_value=st.secrets["HOPSWORKS_API_KEY"])
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    last_7d = df[df["timestamp"] >= pd.Timestamp.now() - pd.Timedelta(days=7)]
    return last_7d[["timestamp","aqi","pm25","pm10"]].sort_values("timestamp")

st.subheader("📈 Last 7 Days — Historical AQI")
hist_df = load_recent_history()
fig_hist = px.line(hist_df, x="timestamp", y="aqi",
                   title="Islamabad AQI — Past 7 Days")
fig_hist.add_hline(y=100, line_dash="dash", line_color="orange")
fig_hist.add_hline(y=150, line_dash="dash", line_color="red")
st.plotly_chart(fig_hist, use_container_width=True)
```

### E5. Deploy to Streamlit Community Cloud

1. Push all code including `artifacts/shap_summary.png` to GitHub
2. Go to https://share.streamlit.io → **New app**
3. Repository: your GitHub repo | Branch: `main` | Main file: `src/streamlit_app.py`
4. Click **Advanced settings** → paste secrets:

```toml
AQICN_TOKEN = "your_token"
HOPSWORKS_API_KEY = "your_key"
```

5. Click **Deploy** — takes 2–5 minutes
6. Copy the public URL (e.g., `https://yourname-aqi-islamabad.streamlit.app`)

---

## Phase F — GitHub Actions CI/CD Verification

> Goal: Confirm both scheduled workflows run successfully in the cloud without manual intervention.

### F1. Add secrets to GitHub repository

1. Go to your repo on GitHub → **Settings** → **Secrets and variables** → **Actions**
2. Add these repository secrets:

| Name | Value |
|------|-------|
| `AQICN_TOKEN` | Your AQICN token |
| `HOPSWORKS_API_KEY` | Your Hopsworks API key |

### F2. Trigger workflows manually to test

Do not wait for the schedule — trigger immediately:

1. Go to GitHub repo → **Actions** tab
2. Click **Feature Pipeline (Hourly)** → **Run workflow** → **Run workflow**
3. Watch the run complete (should take < 3 minutes)
4. Click **Training Pipeline (Daily)** → **Run workflow** → **Run workflow**
5. Watch the run complete (should take < 15 minutes)

### F3. Diagnose common GitHub Actions failures

**Failure: `ModuleNotFoundError`**
```yaml
# Add this to both workflow yml files after the install step:
      - name: Verify imports
        run: python -c "import hopsworks, xgboost, shap, streamlit"
```

**Failure: `HOPSWORKS_API_KEY` not found**
```yaml
# Make sure your script reads the env variable correctly:
      - name: Debug secrets
        run: |
          python -c "import os; print('Key present:', bool(os.getenv('HOPSWORKS_API_KEY')))"
        env:
          HOPSWORKS_API_KEY: ${{ secrets.HOPSWORKS_API_KEY }}
```

**Failure: Feature pipeline timeout (> 15 min)**
Add a timeout guard in `feature_pipeline.py`:
```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Feature pipeline timed out")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(600)   # 10-minute hard limit
```

### F4. Confirm scheduled runs appear in Actions history

After 24 hours, check:
- At least 1 training pipeline run (scheduled at 2 AM UTC)
- At least 2 feature pipeline runs (scheduled every hour)

Green checkmarks = pipeline is live and automated.

---

## Phase G — Optional Enhancements (for Higher Marks)

### G1. Add LSTM model to training pipeline

The current training pipeline uses only Scikit-learn / XGBoost. Adding a TensorFlow LSTM scores higher on the "variety of forecasting models" requirement:

```python
# Add to training_pipeline.py

def prepare_sequences(df, feature_cols, target_col, seq_len=24):
    """Convert tabular data to sequences for LSTM."""
    X, y = [], []
    for i in range(seq_len, len(df)):
        X.append(df[feature_cols].iloc[i-seq_len:i].values)
        y.append(df[target_col].iloc[i])
    return np.array(X), np.array(y)


def train_lstm(X_train_seq, y_train, X_val_seq, y_val):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True,
                             input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(X_train_seq, y_train, epochs=20, batch_size=32,
              validation_data=(X_val_seq, y_val), verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])
    preds = model.predict(X_val_seq).flatten()
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    return model, {"rmse": round(rmse,4), "mae": round(mae,4), "r2": round(r2,4)}
```

### G2. Add alerting integration via Telegram Bot (free)

```python
# src/alert_checker.py

import requests, os
from dotenv import load_dotenv
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_alert(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message})

def check_and_alert():
    token = os.getenv("AQICN_TOKEN")
    resp = requests.get(f"https://api.waqi.info/feed/@11458/?token={token}")
    aqi = resp.json()["data"]["aqi"]
    if aqi > 150:
        send_telegram_alert(
            f"🚨 AQI ALERT — Islamabad/Rawalpindi\n"
            f"Current AQI: {aqi} (UNHEALTHY)\n"
            f"Wear N95 masks outdoors."
        )
        print(f"Alert sent! AQI = {aqi}")
    else:
        print(f"AQI = {aqi}, no alert needed.")

if __name__ == "__main__":
    check_and_alert()
```

**Setup steps:**
1. Message `@BotFather` on Telegram → `/newbot` → copy the token
2. Message `@userinfobot` on Telegram → copy your chat ID
3. Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to `.env` and GitHub secrets
4. Add this to `feature_pipeline.yml` workflow at the end:
```yaml
      - name: Check AQI and send alerts
        env:
          AQICN_TOKEN: ${{ secrets.AQICN_TOKEN }}
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: python src/alert_checker.py
```

### G3. Enhance EDA notebook with Islamabad-specific analysis

Add these cells to `notebooks/01_EDA.ipynb`:

```python
# Cell: Seasonal Analysis — Islamabad has extreme winter smog
monthly_aqi = df.groupby("month")["aqi"].agg(["mean","std","max"])
monthly_aqi.index = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]
monthly_aqi.plot(kind="bar", title="Monthly AQI Statistics — Islamabad")
# Expected pattern: Nov-Feb worst, Jul-Sep best (monsoon)

# Cell: Diurnal (Daily) Cycle
hourly_aqi = df.groupby("hour")["aqi"].mean()
hourly_aqi.plot(title="Average AQI by Hour of Day")
# Expected: peaks at 7-9 AM and 6-8 PM (rush hours)

# Cell: Wind Speed vs AQI (key relationship)
import seaborn as sns
sns.scatterplot(data=df, x="wind_speed", y="aqi", hue="month", alpha=0.4)
# Expected: inverse relationship — higher wind disperses pollutants

# Cell: Weekend vs Weekday AQI
df.groupby("is_weekend")["aqi"].mean().plot(
    kind="bar", title="AQI: Weekday vs Weekend",
    xticks=[0,1], xlabel="", color=["steelblue","tomato"]
)
plt.xticks([0,1], ["Weekday","Weekend"], rotation=0)
```

---

## Phase H — Final Report

> Goal: Produce the 4th deliverable — a detailed report documenting the project.

### H1. Report structure

Write the report as a PDF or Word document with the following sections:

```
1. Introduction
   - Problem statement: AQI in Islamabad/Rawalpindi
   - Why AQI matters in Pakistan (context, health statistics)
   - Objectives and scope

2. System Architecture
   - Architecture diagram (from the implementation guide)
   - Technology stack and justification for each choice
   - Free-tier services used and their limitations

3. Data Sources & Feature Engineering
   - AQICN station details for Islamabad (@11458)
   - Open-Meteo air quality and weather data
   - Full feature list with descriptions (30+ features)
   - Backfill strategy and date range

4. Exploratory Data Analysis
   - AQI distribution histogram
   - Monthly seasonality (winter smog pattern)
   - Diurnal cycle (rush hour peaks)
   - Correlation heatmap
   - Wind speed vs AQI scatter
   - Key EDA findings specific to Islamabad

5. Model Development
   - Model comparison table (RMSE, MAE, R² for Ridge / RF / XGBoost / LSTM)
   - Time-series train/validation split strategy
   - Hyperparameter tuning decisions
   - Why XGBoost performed best (or whichever did)

6. SHAP Feature Importance
   - SHAP summary plot image
   - Interpretation: which features drive AQI most?
   - Expected top features: aqi_lag_24h, pm25, aqi_roll_24h, month

7. MLOps Pipeline
   - GitHub Actions schedule screenshots
   - Hopsworks Feature Store screenshot
   - Model Registry screenshot with versioned models

8. Web Application
   - Screenshots of Streamlit dashboard
   - Live dashboard URL
   - Health alert system description

9. Challenges & Solutions
   - Incomplete PM data from AQICN station (some pollutants missing)
   - Handling NaN lag features at dataset boundary
   - Open-Meteo AQI model vs ground-level sensor discrepancy
   - GitHub Actions memory limits for TensorFlow

10. Conclusion & Future Work
    - Accuracy achieved vs targets
    - Potential improvements (more stations, satellite data)
    - Production scaling considerations
```

### H2. Key screenshots to collect

Before writing the report, capture these screenshots:

- [ ] Hopsworks feature store → Feature Groups → `aqi_features` → Data Explorer showing rows
- [ ] Hopsworks Model Registry → `aqi_predictor_target_aqi_24h` → metrics tab
- [ ] GitHub Actions → Actions tab showing green workflow runs
- [ ] Streamlit dashboard showing live AQI gauge
- [ ] Streamlit dashboard showing 3-day forecast cards
- [ ] `artifacts/shap_summary.png` (generated by training pipeline)

---

## Final Completion Checklist

### Code & Pipelines
- [ ] `python src/feature_pipeline.py` runs without errors
- [ ] `python src/backfill.py` completes and stores 2,000+ rows
- [ ] `python src/training_pipeline.py` registers models with R² > 0.70 (24h)
- [ ] `streamlit run src/streamlit_app.py` loads with live data and model predictions
- [ ] Both GitHub Actions workflows show green on manual trigger

### Deliverable 1: End-to-End AQI System
- [ ] Features stored in Hopsworks Feature Store (real Islamabad data)
- [ ] Models registered in Hopsworks Model Registry (versioned)
- [ ] Inference pipeline pulls live features and returns 3-day forecast

### Deliverable 2: Automated Pipeline
- [ ] Feature pipeline scheduled every hour (GitHub Actions)
- [ ] Training pipeline scheduled daily (GitHub Actions)
- [ ] At least 1 successful scheduled run visible in GitHub Actions history

### Deliverable 3: Interactive Dashboard
- [ ] Deployed public URL on Streamlit Community Cloud
- [ ] Live AQI gauge (real data from AQICN)
- [ ] 3-day AQI forecast (from trained model)
- [ ] SHAP feature importance plot visible
- [ ] Health alert banners working
- [ ] (Optional) 7-day historical chart

### Deliverable 4: Detailed Report
- [ ] EDA section with Islamabad-specific seasonal findings
- [ ] Model comparison table with RMSE / MAE / R² for all models
- [ ] SHAP plot included with interpretation
- [ ] Architecture diagram included
- [ ] Screenshots of live pipeline runs and dashboard
- [ ] Challenges section documenting what went wrong and how it was fixed

---

*Completion Plan v1.0 — Pearls AQI Predictor | Target completion: all phases within 2–3 days*
