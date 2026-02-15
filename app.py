import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("ridge_model.pkl")

st.set_page_config(page_title="Energy Demand Forecast", layout="centered")

# -----------------------------
# HEADER
# -----------------------------
st.title("⚡ Energy Demand Prediction Dashboard")

st.markdown("""
Use the sliders below to simulate real-world conditions.
The model predicts hourly electricity demand based on temperature,
time, and historical trends.
""")

st.markdown("---")

# -----------------------------
# INPUT SECTION
# -----------------------------
st.subheader("🔧 Adjust Conditions")

col1, col2 = st.columns(2)

with col1:
    temp = st.slider("🌡 Outdoor Temperature (°C)", -10, 50, 30)
    hour = st.slider("⏰ Hour of Day", 0, 23, 12)
    holiday = st.selectbox("🎉 Is Holiday?", [0, 1])

with col2:
    day = st.slider("📅 Day of Month", 1, 31, 15)
    month = st.slider("🗓 Month", 1, 12, 6)
    day_of_week = st.slider("📆 Day of Week (0=Mon)", 0, 6, 2)

st.markdown("#### 🔁 Recent Demand Trends")

col3, col4 = st.columns(2)

with col3:
    lag1 = st.number_input("Previous Hour Demand", value=500)

with col4:
    lag24 = st.number_input("Previous Day Same Hour", value=480)

rolling_mean = st.number_input("3-Hour Rolling Average", value=490)
rolling_std = st.number_input("3-Hour Rolling Std Dev", value=20)

# -----------------------------
# ENGINEERED FEATURES
# -----------------------------
temp_hour = temp * hour
temp_squared = temp ** 2

input_data = pd.DataFrame({
    'Outdoor_Temp': [temp],
    'Is_Holiday': [holiday],
    'Hour': [hour],
    'Day': [day],
    'Month': [month],
    'Day_of_Week': [day_of_week],
    'Lag_1': [lag1],
    'Lag_24': [lag24],
    'DayOfWeek': [day_of_week],
    'Rolling_Mean_3': [rolling_mean],
    'Rolling_Std_3': [rolling_std],
    'Temp_Hour_Interaction': [temp_hour],
    'Temp_Squared': [temp_squared]
})

st.markdown("---")

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔮 Predict Demand"):

    prediction = model.predict(input_data)[0]

    st.success(f"⚡ Estimated Energy Demand: {prediction:.2f} MW")

    # Simple Explanation
    st.markdown("### 📌 What This Means")

    if prediction > 700:
        st.warning("⚠ High demand expected. Extra power generation may be required.")
    elif prediction > 500:
        st.info("ℹ Moderate demand. Grid operating normally.")
    else:
        st.success("✅ Low demand period. Good opportunity to reduce production costs.")

    # Temperature Spike Simulation
    spike_data = input_data.copy()
    spike_data['Outdoor_Temp'] += 10
    spike_data['Temp_Hour_Interaction'] = spike_data['Outdoor_Temp'] * spike_data['Hour']
    spike_data['Temp_Squared'] = spike_data['Outdoor_Temp'] ** 2

    spike_prediction = model.predict(spike_data)[0]

    st.markdown("### 🌡 Impact of +10°C Temperature Increase")

    colA, colB = st.columns(2)
    colA.metric("Normal", f"{prediction:.2f} MW")
    colB.metric("After +10°C", f"{spike_prediction:.2f} MW")

    # Smaller Graph
    fig, ax = plt.subplots(figsize=(4,3))   # Smaller size here
    ax.bar(["Normal", "+10°C"], [prediction, spike_prediction])
    ax.set_ylabel("MW")
    ax.set_title("Temperature Impact")
    st.pyplot(fig)

st.markdown("---")

st.caption("Model: Ridge Regression | R² ≈ 0.96 | Includes lag & non-linear features")
