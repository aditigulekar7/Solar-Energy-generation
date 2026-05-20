import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(page_title="Solar Energy Prediction", layout="wide")

st.title("☀️ Solar Energy Generation Prediction System")
st.write("Predict solar power generation using Machine Learning and weather parameters.")

# ---------------------------------
# Load Dataset
# ---------------------------------
data = pd.read_csv("Solar Energy generation.csv")

# ---------------------------------
# Display Dataset
# ---------------------------------
st.subheader("Dataset Preview")
st.dataframe(data.head())

# ---------------------------------
# Data Preprocessing
# ---------------------------------

# Remove missing values

data = data.dropna()

# Features and Target

X = data.drop("generated_power_kw", axis=1)
y = data["generated_power_kw"]

# ---------------------------------
# Train Test Split
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# ---------------------------------
# Model Training
# ---------------------------------
model = LinearRegression()
model.fit(X_train, y_train)
# ---------------------------------
# Model Prediction
# ---------------------------------
predictions = model.predict(X_test)

# ---------------------------------
# Model Evaluation
# ---------------------------------
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

st.subheader("Model Performance")
st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"R² Score: {r2:.2f}")

# ---------------------------------
# User Input Section
# ---------------------------------
st.subheader("Enter Weather Parameters")
# Creating input fields

temperature = st.number_input("Temperature (°C)", value=25.0)
humidity = st.number_input("Relative Humidity (%)", value=50.0)
pressure = st.number_input("Sea Level Pressure", value=1013.0)
precipitation = st.number_input("Total Precipitation", value=0.0)
snowfall = st.number_input("Snowfall Amount", value=0.0)
cloud_cover = st.number_input("Total Cloud Cover", value=20.0)
high_cloud = st.number_input("High Cloud Cover", value=10.0)
medium_cloud = st.number_input("Medium Cloud Cover", value=10.0)
low_cloud = st.number_input("Low Cloud Cover", value=10.0)
radiation = st.number_input("Shortwave Radiation", value=300.0)
wind_speed_10 = st.number_input("Wind Speed 10m", value=5.0)
wind_direction_10 = st.number_input("Wind Direction 10m", value=180.0)
wind_speed_80 = st.number_input("Wind Speed 80m", value=5.0)
wind_direction_80 = st.number_input("Wind Direction 80m", value=180.0)
wind_speed_900 = st.number_input("Wind Speed 900mb", value=5.0)
wind_direction_900 = st.number_input("Wind Direction 900mb", value=180.0)
wind_gust = st.number_input("Wind Gust", value=10.0)
angle_incidence = st.number_input("Angle of Incidence", value=45.0)
zenith = st.number_input("Zenith", value=60.0)
azimuth = st.number_input("Azimuth", value=180.0)

# ---------------------------------
# Prediction Button
# ---------------------------------

if st.button("Predict Solar Power"):

    input_data = pd.DataFrame([[
        temperature,
        humidity,
        pressure,
        precipitation,
        snowfall,
        cloud_cover,
        high_cloud,
        medium_cloud,
        low_cloud,
        radiation,
        wind_speed_10,
        wind_direction_10,
        wind_speed_80,
        wind_direction_80,
        wind_speed_900,
        wind_direction_900,
        wind_gust,
        angle_incidence,
        zenith,
        azimuth
    ]], columns=X.columns)

    prediction = model.predict(input_data)

    st.success(f"Predicted Solar Power Generation: {prediction[0]:.2f} kW")

# ---------------------------------
# Visualization
# ---------------------------------

st.subheader("Actual vs Predicted Power")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(y_test.values[:100], label="Actual Power")
ax.plot(predictions[:100], label="Predicted Power")
ax.set_xlabel("Samples")
ax.set_ylabel("Generated Power (kW)")
ax.legend()

st.pyplot(fig)


