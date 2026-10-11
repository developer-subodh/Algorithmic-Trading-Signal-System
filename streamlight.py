import streamlit as st
import joblib
import numpy as np
import pandas as pd


st.set_page_config(page_title="AlphaWave ML Dashboard", layout="centered")

st.title("📊 AlphaWave ML: Algorithmic Trading Signal System")
st.write("Move the sliders to input technical indicators and generate live trading signals.")


@st.cache_resource
def load_trading_model():
    return joblib.load('trading_model.pkl')

try:
    model = load_trading_model()
    st.success("🤖 XGBoost Engine Loaded Successfully (Validation Accuracy: 99.83%)")
except Exception as e:
    st.error(f"❌ Model file not found: {e}. Make sure 'trading_model.pkl' is in this folder.")

st.markdown("---")
st.subheader("🛠️ Adjust Market & Technical Vectors")


col1, col2 = st.columns(2)

with col1:
    rsi_value = st.slider("RSI (14 Days)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    macd_value = st.slider("MACD Value", min_value=-5.0, max_value=5.0, value=0.0, step=0.01)
    volatility_stretch = st.slider("Volatility Stretch Ratio", min_value=0.1, max_value=5.0, value=1.0, step=0.01)
    volatility_range = st.slider("Volatility Range (High-Low)", min_value=0.0, max_value=20.0, value=2.0, step=0.1)

with col2:
    dist_to_upper = st.slider("Distance to Upper Bollinger Band", min_value=-10.0, max_value=50.0, value=5.0, step=0.1)
    rsi_diff = st.slider("RSI Momentum Diff (Velocity)", min_value=-20.0, max_value=20.0, value=0.0, step=0.1)
    macd_direction = st.selectbox("MACD Trend Direction", options=[0, 1], format_func=lambda x: "1 (Bullish Regime)" if x == 1 else "0 (Bearish Regime)")
    volume_scaled = st.slider("Scaled Volume Factor", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)


era_encoded = 4 
open_log, close_log, high_log, low_log, sma_log, bbu_log = 3.5, 3.5, 3.6, 3.4, 3.5, 3.7

st.markdown("---")


if st.button("🚀 Generate Trading Signal", type="primary"):

    # 15 फीचर्स को बिल्कुल सही क्रम में एरे में सजाना
    input_data = np.array([[
        era_encoded, volume_scaled, volatility_range, volatility_stretch,
        rsi_value, rsi_diff, macd_value, macd_direction,
        open_log, close_log, high_log, low_log, sma_log, bbu_log,
        dist_to_upper
    ]])


    prediction = model.predict(input_data)[0]

    st.subheader("🎯 Live Model Output:")


    if prediction == 1:
        st.markdown("<h2 style='color: #00cc96;'>🟢 SIGNAL: BUY TIME (Market Up Day)</h2>", unsafe_allow_html=True)
        st.balloons()  # स्क्रीन पर गुब्बारे उड़ेंगे!
    else:
        st.markdown("<h2 style='color: #ef553b;'>🔴 SIGNAL: RISK ZONE (Market Down Day)</h2>", unsafe_allow_html=True)
