import streamlit as st
import pandas as pd
import boto3
from datetime import datetime
import time

# Streamlit configuration
st.set_page_config(page_title="🌾 Agriculture Monitoring", layout="wide")
st.title("🌾 Agriculture Monitoring System with AWS Integration")

# Fetch data from DynamoDB using credentials from Streamlit Secrets
@st.cache_data(ttl=60)
def fetch_data():
    try:
        aws_access_key_id = st.secrets["aws"]["aws_access_key_id"]
        aws_secret_access_key = st.secrets["aws"]["aws_secret_access_key"]
        region_name = st.secrets["aws"]["region_name"]

        # Log AWS credentials to verify
        st.write(f"Region: {region_name}")

        # Initialize DynamoDB resource
        dynamodb = boto3.resource(
            'dynamodb',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )

        # Check if table exists
        table = dynamodb.Table('AgricultureMonitoring')
        st.write("Attempting to scan the table...")

        response = table.scan()
        items = response.get('Items', [])
        st.write(f"Items fetched: {len(items)}")

        if not items:
            return pd.DataFrame(columns=["timestamp", "temperature", "humidity", "soil_moisture", "soil_nitrogen", "soil_phosphorus", "soil_potassium"])

        df = pd.DataFrame(items)

        for col in ["temperature", "humidity", "soil_moisture", "soil_nitrogen", "soil_phosphorus", "soil_potassium"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(inplace=True)
        df.sort_values('timestamp', inplace=True)

        return df

    except Exception as e:
        st.error(f"Error fetching data from DynamoDB: {e}")
        return pd.DataFrame(columns=["timestamp", "temperature", "humidity", "soil_moisture", "soil_nitrogen", "soil_phosphorus", "soil_potassium"])


    except Exception as e:
        st.error(f"Error fetching data from DynamoDB: {e}")
        return pd.DataFrame(columns=["timestamp", "temperature", "humidity", "soil_moisture", "soil_nitrogen", "soil_phosphorus", "soil_potassium"])

# Fetch and display data
df = fetch_data()

if df.empty:
    st.warning("No data available. Please check if the Lambda ingestion script is running.")
else:
    st.subheader("📊 Live Sensor Data Visualization")
    
    # Plot live sensor data
    st.line_chart(df.set_index("timestamp")[["temperature", "humidity", "soil_moisture", "soil_nitrogen", "soil_phosphorus", "soil_potassium"]])

    st.subheader("📈 Current Sensor Readings")
    col1, col2, col3 = st.columns(3)

    # Display the latest data values
    latest_data = df.iloc[-1]
    col1.metric("🌡 Temperature (°C)", f"{latest_data['temperature']:.2f}")
    col1.metric("💧 Humidity (%)", f"{latest_data['humidity']:.2f}")
    col1.metric("🌿 Soil Moisture", f"{latest_data['soil_moisture']:.2f}")
    col2.metric("🧪 Nitrogen (mg/kg)", f"{latest_data['soil_nitrogen']:.2f}")
    col2.metric("🧪 Phosphorus (mg/kg)", f"{latest_data['soil_phosphorus']:.2f}")
    col3.metric("🧪 Potassium (mg/kg)", f"{latest_data['soil_potassium']:.2f}")

    # Option to download data as CSV
    st.markdown("### 📥 Download Data")
    st.download_button("Download Data as CSV", df.to_csv(index=False).encode('utf-8'), "agriculture_data.csv", "text/csv")
