import streamlit as st
import pandas as pd
import boto3
from datetime import datetime
import requests

# ================================
# Configuration & AWS Setup
# ================================
st.set_page_config(page_title="🌾 Agriculture Monitoring", layout="wide")
st.title("🌾 Agriculture Monitoring System with AWS Integration")

# -------------------------------
# Cached AWS Session (Improves Performance)
# -------------------------------
@st.cache_resource
def get_dynamodb_session():
    return boto3.Session(
        aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
        aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
        region_name=st.secrets["aws"]["region_name"]
    )

# -------------------------------
# Fetch Sensor Data from DynamoDB
# -------------------------------
@st.cache_data(ttl=60)  # Refresh data every 60 seconds
def fetch_data():
    try:
        session = get_dynamodb_session()
        dynamodb = session.resource('dynamodb')
        table = dynamodb.Table('AgricultureMonitoring')
        response = table.scan()
        items = response.get('Items', [])

        if not items:
            return pd.DataFrame(columns=[
                "timestamp", "temperature", "humidity", "soil_moisture",
                "soil_nitrogen", "soil_phosphorus", "soil_potassium"
            ])

        df = pd.DataFrame(items)
        numeric_cols = ["temperature", "humidity", "soil_moisture", 
                        "soil_nitrogen", "soil_phosphorus", "soil_potassium"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(inplace=True)
        df.sort_values('timestamp', inplace=True)
        return df

    except Exception as e:
        st.error(f"⚠️ Error fetching data from DynamoDB: {e}")
        return pd.DataFrame(columns=[
            "timestamp", "temperature", "humidity", "soil_moisture",
            "soil_nitrogen", "soil_phosphorus", "soil_potassium"
        ])

# -------------------------------
# Hugging Face LLM Integration
# -------------------------------
def interpret_data(prompt):
    hf_api_token = st.secrets["huggingface"]["api_token"]
    headers = {"Authorization": f"Bearer {hf_api_token}"}
    model = "davin-ai/agriculture-bert"  # Alternative: "facebook/bart-large-cnn"
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    payload = {"inputs": prompt}

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            output = response.json()
            return output[0].get("generated_text", "⚠️ No response received.")
        else:
            return f"⚠️ API Error: {response.text}"
    except requests.exceptions.Timeout:
        return "⏳ Request timed out. Try again."
    except Exception as e:
        return f"⚠️ API Error: {e}"

# ================================
# Sidebar: Choose View Mode
# ================================
view_mode = st.sidebar.radio("Select View", ["Dashboard", "Simple"])

# ================================
# Dashboard View (Existing Visualization)
# ================================
if view_mode == "Dashboard":
    df = fetch_data()

    if df.empty:
        st.warning("🚨 No data available. Ensure the Lambda ingestion script is running.")
    else:
        st.subheader("📊 Live Sensor Data Visualization")
        st.line_chart(df.set_index("timestamp")[[
            "temperature", "humidity", "soil_moisture",
            "soil_nitrogen", "soil_phosphorus", "soil_potassium"
        ]])

        st.subheader("📈 Current Sensor Readings")
        latest_data = df.iloc[-1]

        col1, col2, col3 = st.columns(3)
        col1.metric("🌡 Temperature (°C)", f"{latest_data['temperature']:.2f}")
        col1.metric("💧 Humidity (%)", f"{latest_data['humidity']:.2f}")
        col1.metric("🌿 Soil Moisture", f"{latest_data['soil_moisture']:.2f}")

        col2.metric("🧪 Nitrogen (mg/kg)", f"{latest_data['soil_nitrogen']:.2f}")
        col2.metric("🧪 Phosphorus (mg/kg)", f"{latest_data['soil_phosphorus']:.2f}")
        col3.metric("🧪 Potassium (mg/kg)", f"{latest_data['soil_potassium']:.2f}")

        st.markdown("### 📥 Download Data")
        st.download_button(
            "Download Data as CSV", df.to_csv(index=False).encode('utf-8'),
            "agriculture_data.csv", "text/csv"
        )

# ================================
# Simple Chat View (LLM Interpretation)
# ================================
elif view_mode == "Simple":
    st.header("🌾 Agricultural Data Interpretation Chat")
    df = fetch_data()

    if df.empty:
        st.warning("🚨 No sensor data available for interpretation.")
    else:
        latest_data = df.iloc[-1]
        initial_prompt = f"""
        You are an expert agronomist analyzing real-time sensor data from a farm.
        Interpret the following sensor readings in simple terms:
        
        🌡 Temperature: {latest_data['temperature']:.2f} °C
        💧 Humidity: {latest_data['humidity']:.2f} %
        🌿 Soil Moisture: {latest_data['soil_moisture']}
        🧪 Nitrogen: {latest_data['soil_nitrogen']:.2f} mg/kg
        🧪 Phosphorus: {latest_data['soil_phosphorus']:.2f} mg/kg
        🧪 Potassium: {latest_data['soil_potassium']:.2f} mg/kg
        
        Respond with:
        1. **Current Condition Analysis**
        2. **Potential Impact**
        3. **Recommendations**
        """

        if st.button("Get Initial Interpretation"):
            st.session_state["initial_interpretation"] = interpret_data(initial_prompt)

        if st.session_state.get("initial_interpretation"):
            st.markdown("**Initial Interpretation:**")
            st.info(st.session_state["initial_interpretation"])

    # Chat Interface
    st.subheader("💬 Ask Follow-Up Questions")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.text_input("Your question:")
    if st.button("Send"):
        if user_input:
            st.session_state["chat_history"].append(("User", user_input))
            prompt = f"""
            You are an agronomy expert. Here are the latest sensor readings:
            
            🌡 Temperature: {latest_data['temperature']:.2f} °C
            💧 Humidity: {latest_data['humidity']:.2f} %
            🌿 Soil Moisture: {latest_data['soil_moisture']}
            🧪 Nitrogen: {latest_data['soil_nitrogen']:.2f} mg/kg
            🧪 Phosphorus: {latest_data['soil_phosphorus']:.2f} mg/kg
            🧪 Potassium: {latest_data['soil_potassium']:.2f} mg/kg
            
            The farmer asks: "{user_input}"
            """

            response = interpret_data(prompt)
            st.session_state["chat_history"].append(("LLM", response))
            st.rerun()

    # Display chat history
    if st.session_state["chat_history"]:
        st.markdown("### 🗨️ Chat History")
        for role, msg in st.session_state["chat_history"]:
            st.write(f"**{role}:** {msg}")
