import streamlit as st
import numpy as np
import joblib

st.write("App Started Successfully")

try:
    model = joblib.load("energy_model.pkl")
    scaler = joblib.load("scaler.pkl")
    st.write("Model and Scaler Loaded Successfully")

    st.title("⚡ Energy Demand Prediction")

    Outdoor_Temp = st.number_input("Outdoor Temperature (°C)", value=25.0)
    Is_Holiday = st.selectbox("Is it a Holiday?", [0, 1])
    Hour = st.slider("Hour of Day", 0, 23, 12)
    Day = st.slider("Day of Month", 1, 31, 15)
    Month = st.slider("Month", 1, 12, 6)
    DayOfWeek = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 3)

    Temp_Hour_Interaction = Outdoor_Temp * Hour
    Temp_Squared = Outdoor_Temp ** 2

    input_data = np.array([[
        Outdoor_Temp,
        Is_Holiday,
        Hour,
        Day,
        Month,
        DayOfWeek,
        Temp_Hour_Interaction,
        Temp_Squared
    ]])

    input_scaled = scaler.transform(input_data)

    if st.button("Predict"):
        prediction = model.predict(input_scaled)
        st.success(f"Predicted Energy Demand: {prediction[0]:.2f} MW")

except Exception as e:
    st.error(f"Error occurred: {e}")
