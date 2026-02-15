import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Energy Demand Dashboard",
    layout="wide"
)

# -----------------------------
# Load Model, Scaler & Dataset
# -----------------------------
model = joblib.load("energy_model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("voltedge_energy_usage_P6.csv")

df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')

# -----------------------------
# Title Section
# -----------------------------
st.title("⚡ Energy Demand Prediction Dashboard")
st.markdown("Predict electricity demand and explore data insights.")

st.markdown("---")

# -----------------------------
# Input Section
# -----------------------------
st.subheader("📥 Enter Input Details")

col1, col2 = st.columns(2)

with col1:
    Outdoor_Temp = st.number_input("Outdoor Temperature (°C)", value=25.0)
    Hour = st.slider("Hour of Day", 0, 23, 12)
    holiday_option = st.radio("Is it a Holiday?", ["No", "Yes"])
    Is_Holiday = 1 if holiday_option == "Yes" else 0

with col2:
    Day = st.slider("Day of Month", 1, 31, 15)
    Month = st.slider("Month", 1, 12, 6)

    weekday_dict = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6
    }

    weekday_name = st.selectbox("Day of Week", list(weekday_dict.keys()))
    DayOfWeek = weekday_dict[weekday_name]

# -----------------------------
# Feature Engineering
# -----------------------------
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

st.markdown("---")

# -----------------------------
# Prediction Section
# -----------------------------
if st.button("🔮 Predict Energy Demand"):

    prediction = model.predict(input_scaled)[0]

    st.success(f"Predicted Energy Demand: {prediction:.2f} MW")

    st.markdown("### 📌 Prediction Insight")

    explanation = ""

    # Temperature influence
    if Outdoor_Temp > 30:
        explanation += "- High temperature may increase cooling demand (AC usage).\n"
    elif Outdoor_Temp < 10:
        explanation += "- Low temperature may increase heating demand.\n"
    else:
        explanation += "- Moderate temperature suggests balanced energy usage.\n"

    # Hour influence
    if 6 <= Hour <= 10:
        explanation += "- Morning hours typically show increased residential consumption.\n"
    elif 18 <= Hour <= 22:
        explanation += "- Evening peak hours usually result in higher electricity usage.\n"
    else:
        explanation += "- Selected hour falls outside major peak periods.\n"

    # Holiday influence
    if Is_Holiday == 1:
        explanation += "- Holiday factor may alter commercial and residential consumption patterns.\n"
    else:
        explanation += "- Regular working day consumption pattern applied.\n"

    st.info(explanation)

    st.markdown("""
    **Model Analysis Based On:**
    - Time of day (Hour)
    - Outdoor temperature
    - Day of week
    - Holiday indicator
    - Interaction and polynomial features
    """)
st.markdown("---")
st.subheader("📊 How Energy Demand Depends on Key Factors")

st.markdown("""
**🔹 Temperature Effect:**  
Energy demand tends to increase during very high temperatures due to cooling requirements (AC usage) and during very low temperatures due to heating demand. Moderate temperatures generally result in balanced consumption.

**🔹 Time of Day (Hour):**  
Energy consumption typically peaks during morning and evening hours when residential and commercial activities are high. Midday and late-night hours usually show lower demand.

**🔹 Holiday Influence:**  
On holidays, commercial electricity usage may decrease while residential usage may increase, slightly altering overall demand patterns.

**🔹 Interaction Features:**  
The model also considers the combined effect of temperature and hour (e.g., hot evenings may significantly increase demand).

The prediction is generated using a trained Random Forest regression model that learns these patterns from historical data.
""")

st.markdown("---")

# -----------------------------
# Dashboard Graph Section
# -----------------------------
if st.button("📊 Show Analysis Dashboard"):

    st.subheader("📊 Energy Data Insights")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # 1️⃣ Scatter Plot
    with col1:
        fig1, ax1 = plt.subplots(figsize=(4,3))
        sns.scatterplot(
            x=df['Outdoor_Temp'],
            y=df['Energy_Demand_MW'],
            ax=ax1
        )
        ax1.set_title("Temp vs Demand")
        ax1.set_xlabel("Temperature")
        ax1.set_ylabel("Demand")
        st.pyplot(fig1)

    # 2️⃣ Hourly Trend
    with col2:
        hourly_avg = df.groupby("Hour")["Energy_Demand_MW"].mean()
        fig2, ax2 = plt.subplots(figsize=(4,3))
        hourly_avg.plot(ax=ax2)
        ax2.set_title("Avg Demand by Hour")
        ax2.set_xlabel("Hour")
        ax2.set_ylabel("Avg Demand")
        st.pyplot(fig2)

    # 3️⃣ Holiday Comparison
    with col3:
        fig3, ax3 = plt.subplots(figsize=(4,3))
        sns.boxplot(
            x=df['Is_Holiday'],
            y=df['Energy_Demand_MW'],
            ax=ax3
        )
        ax3.set_title("Holiday vs Demand")
        ax3.set_xlabel("Is Holiday")
        ax3.set_ylabel("Demand")
        st.pyplot(fig3)

    # 4️⃣ Correlation Heatmap
    with col4:
        fig4, ax4 = plt.subplots(figsize=(4,3))
        sns.heatmap(
            df.corr(),
            cmap="coolwarm",
            ax=ax4,
            cbar=False
        )
        ax4.set_title("Correlation Heatmap")
        st.pyplot(fig4)
