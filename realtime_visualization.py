import streamlit as st
import pandas as pd
import boto3
from datetime import datetime
import time

# DynamoDB Configuration
dynamodb = boto3.resource('dynamodb', region_name='us-east-2')
table = dynamodb.Table('AgricultureMonitoring')

st.set_page_config(page_title="🌾 Agriculture Monitoring", layout="wide")
st.title("🌾 Agriculture Monitoring System with AWS Integration")

# Fetch data from DynamoDB
@st.cache_data(ttl=60)  # Refresh data every 60 seconds
def fetch_data():
    response = table.scan()
    items = response['Items']
    
    # Convert data to DataFrame
    if not items:
        return pd.DataFrame(columns=["timestamp", "temperature", "humidity", "soil_moisture", "soil_nitrogen", "soil_phosphorus", "soil_potassium"])
    
    df = pd.DataFrame(items)
    
    # Convert data types
    for col in ["temperature", "humidity", "soil_moisture", "soil_nitrogen", "soil_phosphorus", "soil_potassium"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(inplace=True)
    
    # Sort by timestamp to ensure proper time series plotting
    df.sort_values('timestamp', inplace=True)

    return df

# Real-time data display
df = fetch_data()

if df.empty:
    st.warning("No data available. Please check if the Lambda ingestion script is running.")
else:
    st.subheader("📊 Live Sensor Data Visualization")

    # Use timestamp as the x-axis for better time-based visualization
    st.line_chart(df.set_index("timestamp")[["temperature", "humidity", "soil_moisture", "soil_nitrogen", "soil_phosphorus", "soil_potassium"]])

    # Display current sensor readings
    st.subheader("📈 Current Sensor Readings")
    col1, col2, col3 = st.columns(3)
    latest_data = df.iloc[-1]

    col1.metric("🌡 Temperature (°C)", f"{latest_data['temperature']:.2f}")
    col1.metric("💧 Humidity (%)", f"{latest_data['humidity']:.2f}")
    col1.metric("🌿 Soil Moisture", f"{latest_data['soil_moisture']}")
    col2.metric("🧪 Nitrogen (mg/kg)", f"{latest_data['soil_nitrogen']:.2f}")
    col2.metric("🧪 Phosphorus (mg/kg)", f"{latest_data['soil_phosphorus']:.2f}")
    col3.metric("🧪 Potassium (mg/kg)", f"{latest_data['soil_potassium']:.2f}")

    # Download data as CSV
    st.markdown("### 📥 Download Data")
    st.download_button("Download Data as CSV", df.to_csv(index=False).encode('utf-8'), "agriculture_data.csv", "text/csv")
