"""
AQI Forecast Dashboard for Islamabad / Rawalpindi.
Deploy on Streamlit Community Cloud.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import hopsworks
from dotenv import load_dotenv
import yaml

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
CITY = CONFIG.get("city", {})
LAT = CITY.get("latitude")
LON = CITY.get("longitude")

st.set_page_config(
    page_title="AQI Forecast - Islamabad/Rawalpindi",
    page_icon="AQI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

AQI_LEVELS = [
    (0, 50, "Good", "#00e400", "Air quality is satisfactory."),
    (51, 100, "Moderate", "#ffff00", "Sensitive individuals may feel mild discomfort."),
    (101, 150, "Unhealthy (Sensitive)", "#ff7e00", "Sensitive groups should limit exposure."),
    (151, 200, "Unhealthy", "#ff0000", "Everyone may experience health effects."),
    (201, 300, "Very Unhealthy", "#8f3f97", "Health alert: serious effects for everyone."),
    (301, 500, "Hazardous", "#7e0023", "Emergency conditions. Avoid outdoor activity."),
]


def get_secret(name: str) -> str:
    try:
        return str(st.secrets[name])
    except Exception:
        return os.getenv(name, "")


def classify_aqi(value: float) -> tuple:
    try:
        numeric = round(float(value))
    except (TypeError, ValueError):
        numeric = 0
    for lo, hi, label, color, advice in AQI_LEVELS:
        if lo <= numeric <= hi:
            return label, color, advice
    return "Hazardous", "#7e0023", "Emergency conditions."


def get_aqi_gauge(value: float) -> go.Figure:
    label, color, _ = classify_aqi(value)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"Current AQI<br><span style='font-size:14px;color:{color}'>{label}</span>"},
            gauge={
                "axis": {"range": [0, 300], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": "#00e400"},
                    {"range": [51, 100], "color": "#ffff00"},
                    {"range": [101, 150], "color": "#ff7e00"},
                    {"range": [151, 200], "color": "#ff0000"},
                    {"range": [201, 300], "color": "#8f3f97"},
                ],
            },
        )
    )
    fig.update_layout(height=300, margin=dict(t=60, b=20))
    return fig


@st.cache_data(ttl=3600)
def load_current_aqi():
    token = get_secret("AQICN_TOKEN")
    if not token:
        return {
            "aqi": 0,
            "pm25": 0,
            "pm10": 0,
            "no2": 0,
            "temperature": 0,
            "humidity": 0,
            "wind": 0,
            "updated": "Missing AQICN token",
        }
    url = f"https://api.waqi.info/feed/geo:{LAT};{LON}/?token={token}"
    resp = requests.get(url, timeout=30)
    data = resp.json().get("data", {})
    aqi_value = data.get("aqi", 0)
    try:
        aqi_value = float(aqi_value)
    except (TypeError, ValueError):
        aqi_value = 0.0
    iaqi = data.get("iaqi", {})
    return {
        "aqi": aqi_value,
        "pm25": iaqi.get("pm25", {}).get("v", 0),
        "pm10": iaqi.get("pm10", {}).get("v", 0),
        "no2": iaqi.get("no2", {}).get("v", 0),
        "temperature": iaqi.get("t", {}).get("v", 0),
        "humidity": iaqi.get("h", {}).get("v", 0),
        "wind": iaqi.get("w", {}).get("v", 0),
        "updated": data.get("time", {}).get("s", ""),
    }


@st.cache_data(ttl=3600)
def load_forecast():
    try:
        project = hopsworks.login(
            api_key_value=get_secret("HOPSWORKS_API_KEY")
        )
        mr = project.get_model_registry()
        
        # Load 24h model
        model_24 = mr.get_best_model("aqi_predictor_target_aqi_24h", metric="rmse", direction="min")
        saved_model_dir_24 = model_24.download()
        clf_24 = joblib.load(os.path.join(saved_model_dir_24, "aqi_target_aqi_24h_model.pkl"))

        # Load 48h model
        model_48 = mr.get_best_model("aqi_predictor_target_aqi_48h", metric="rmse", direction="min")
        saved_model_dir_48 = model_48.download()
        clf_48 = joblib.load(os.path.join(saved_model_dir_48, "aqi_target_aqi_48h_model.pkl"))

        # Load 72h model
        model_72 = mr.get_best_model("aqi_predictor_target_aqi_72h", metric="rmse", direction="min")
        saved_model_dir_72 = model_72.download()
        clf_72 = joblib.load(os.path.join(saved_model_dir_72, "aqi_target_aqi_72h_model.pkl"))

        fs = project.get_feature_store()
        fg = fs.get_feature_group("aqi_features", version=1)
        df = fg.read(online=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp")
        if len(df) < 25:
            raise RuntimeError("Not enough history to compute lag features.")
        df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
        df["aqi_lag_1h"] = df["aqi"].shift(1)
        df["aqi_lag_3h"] = df["aqi"].shift(3)
        df["aqi_lag_6h"] = df["aqi"].shift(6)
        df["aqi_lag_24h"] = df["aqi"].shift(24)
        df["aqi_change_1h"] = df["aqi"] - df["aqi_lag_1h"]
        df["aqi_roll_3h"] = df["aqi"].rolling(3).mean()
        df["aqi_roll_6h"] = df["aqi"].rolling(6).mean()
        df["aqi_roll_24h"] = df["aqi"].rolling(24).mean()
        df["aqi_roll_std"] = df["aqi"].rolling(6).std()
        latest = df.tail(1)

        feature_cols = [
            "pm25",
            "pm10",
            "no2",
            "co",
            "o3",
            "temperature",
            "humidity",
            "wind_speed",
            "pressure",
            "precipitation",
            "cloud_cover",
            "wind_u",
            "wind_v",
            "hour_sin",
            "hour_cos",
            "month_sin",
            "month_cos",
            "dow_sin",
            "dow_cos",
            "is_rush_hour",
            "is_weekend",
            "aqi_lag_1h",
            "aqi_lag_3h",
            "aqi_lag_6h",
            "aqi_lag_24h",
            "aqi_change_1h",
            "aqi_roll_3h",
            "aqi_roll_6h",
            "aqi_roll_24h",
            "aqi_roll_std",
        ]

        X = latest[feature_cols]
        if X.isna().any(axis=None):
            valid = df[feature_cols].dropna()
            if valid.empty:
                raise RuntimeError("Latest features contain NaNs; check history.")
            X = valid.tail(1)
            
        pred_24 = float(clf_24.predict(X)[0])
        pred_48 = float(clf_48.predict(X)[0])
        pred_72 = float(clf_72.predict(X)[0])

        today = datetime.now()
        return [
            {"date": (today + timedelta(days=1)).strftime("%A, %b %d"), "aqi": max(0, pred_24)},
            {"date": (today + timedelta(days=2)).strftime("%A, %b %d"), "aqi": max(0, pred_48)},
            {"date": (today + timedelta(days=3)).strftime("%A, %b %d"), "aqi": max(0, pred_72)},
        ]
    except Exception as exc:
        st.warning(f"Could not load live model predictions. Showing sample data. ({exc})")
        today = datetime.now()
        return [
            {"date": (today + timedelta(days=1)).strftime("%A, %b %d"), "aqi": 145},
            {"date": (today + timedelta(days=2)).strftime("%A, %b %d"), "aqi": 130},
            {"date": (today + timedelta(days=3)).strftime("%A, %b %d"), "aqi": 110},
        ]


def main():
    st.title("AQI Forecast Dashboard")
    st.markdown("### Islamabad / Rawalpindi, Pakistan")
    st.divider()

    with st.spinner("Loading real-time AQI data..."):
        current = load_current_aqi()
        forecast = load_forecast()

    label, color, advice = classify_aqi(current["aqi"])

    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(get_aqi_gauge(current["aqi"]), width="stretch")
    with col2:
        st.markdown(f"#### Air Quality: {label}")
        st.info(advice)
        st.markdown(f"Last updated: {current.get('updated', 'N/A')}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PM2.5", f"{current['pm25']:.0f} ug/m3")
        m2.metric("PM10", f"{current['pm10']:.0f} ug/m3")
        m3.metric("NO2", f"{current['no2']:.0f} ppb")
        m4.metric("Temp", f"{current['temperature']:.1f} C")

    st.divider()

    st.subheader("3-Day AQI Forecast")
    cols = st.columns(3)
    for i, day in enumerate(forecast):
        lbl, clr, _ = classify_aqi(day["aqi"])
        with cols[i]:
            st.markdown(f"**{day['date']}**")
            st.markdown(
                f"<div style='background:{clr};padding:16px;border-radius:12px;"
                "text-align:center;color:#000;font-size:28px;font-weight:bold'>"
                f"{day['aqi']:.0f}</div>",
                unsafe_allow_html=True,
            )
            st.caption(lbl)

    st.divider()

    forecast_df = pd.DataFrame(forecast)
    fig = px.bar(
        forecast_df,
        x="date",
        y="aqi",
        color="aqi",
        color_continuous_scale=["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97"],
        range_color=[0, 300],
        labels={"aqi": "Predicted AQI", "date": "Date"},
        title="Predicted AQI - Next 3 Days",
    )
    fig.add_hline(
        y=100,
        line_dash="dash",
        line_color="orange",
        annotation_text="Moderate threshold (100)",
    )
    fig.add_hline(
        y=150,
        line_dash="dash",
        line_color="red",
        annotation_text="Unhealthy threshold (150)",
    )
    st.plotly_chart(fig, width="stretch")

    if os.path.exists("artifacts/shap_summary.png"):
        st.divider()
        st.subheader("Feature Importance (SHAP)")
        st.image("artifacts/shap_summary.png", caption="SHAP Summary Plot")

    if any(d["aqi"] > 150 for d in forecast):
        st.error(
            "Health alert: AQI is forecasted to reach unhealthy levels in the next 3 days."
        )
    elif any(d["aqi"] > 100 for d in forecast):
        st.warning(
            "Caution: AQI is forecasted to be moderate to unhealthy. Sensitive groups should reduce exposure."
        )

    st.caption("Data: AQICN | Open-Meteo | Model: Pearls AQI Predictor")


if __name__ == "__main__":
    main()
