import streamlit as st
import pandas as pd
import boto3
from datetime import datetime
import time
import requests

# ================================
# Configuration & AWS Setup
# ================================
st.set_page_config(page_title="🌾 Agriculture Monitoring", layout="wide")
st.title("🌾 Agriculture Monitoring System with AWS Integration")

# -------------------------------
# Fetch sensor data from DynamoDB
# -------------------------------
@st.cache_data(ttl=60)  # Refresh data every 60 seconds
def fetch_data():
    try:
        session = boto3.Session(
            aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
            aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
            region_name=st.secrets["aws"]["region_name"]
        )
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
        for col in ["temperature", "humidity", "soil_moisture", 
                    "soil_nitrogen", "soil_phosphorus", "soil_potassium"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(inplace=True)
        df.sort_values('timestamp', inplace=True)
        return df

    except Exception as e:
        st.error(f"Error fetching data from DynamoDB: {e}")
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
    # You can change this model to any Hugging Face model capable of text generation (e.g., "google/flan-t5-large")
    model = "google/flan-t5-large"
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    payload = {"inputs": prompt}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            output = response.json()
            # Assume output is a list and the generated text is in the first element under "generated_text"
            return output[0]["generated_text"]
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error calling Hugging Face API: {e}"

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
        st.warning("No data available. Please check if the Lambda ingestion script is running.")
    else:
        st.subheader("📊 Live Sensor Data Visualization")
        st.line_chart(df.set_index("timestamp")[[
            "temperature", "humidity", "soil_moisture",
            "soil_nitrogen", "soil_phosphorus", "soil_potassium"
        ]])

        st.subheader("📈 Current Sensor Readings")
        col1, col2, col3 = st.columns(3)
        latest_data = df.iloc[-1]
        col1.metric("🌡 Temperature (°C)", f"{latest_data['temperature']:.2f}")
        col1.metric("💧 Humidity (%)", f"{latest_data['humidity']:.2f}")
        col1.metric("🌿 Soil Moisture", f"{latest_data['soil_moisture']:.2f}")
        col2.metric("🧪 Nitrogen (mg/kg)", f"{latest_data['soil_nitrogen']:.2f}")
        col2.metric("🧪 Phosphorus (mg/kg)", f"{latest_data['soil_phosphorus']:.2f}")
        col3.metric("🧪 Potassium (mg/kg)", f"{latest_data['soil_potassium']:.2f}")

        st.markdown("### 📥 Download Data")
        st.download_button("Download Data as CSV", df.to_csv(index=False).encode('utf-8'),
                           "agriculture_data.csv", "text/csv")

# ================================
# Simple Chat View (LLM Interpretation)
# ================================
elif view_mode == "Simple":
    st.header("Agricultural Data Interpretation Chat")
    df = fetch_data()
    if df.empty:
        st.warning("No sensor data available for interpretation.")
    else:
        # Use the latest sensor data for initial interpretation
        latest_data = df.iloc[-1]
        initial_prompt = (
            f"Interpret the following agricultural sensor readings in simple terms: \n"
            f"Temperature: {latest_data['temperature']:.2f} °C, "
            f"Humidity: {latest_data['humidity']:.2f} %, "
            f"Soil Moisture: {latest_data['soil_moisture']}, "
            f"Nitrogen: {latest_data['soil_nitrogen']:.2f} mg/kg, "
            f"Phosphorus: {latest_data['soil_phosphorus']:.2f} mg/kg, "
            f"Potassium: {latest_data['soil_potassium']:.2f} mg/kg.\n"
            "Explain their agricultural significance."
        )

        if st.button("Get Initial Interpretation"):
            initial_interpretation = interpret_data(initial_prompt)
            st.session_state.initial_interpretation = initial_interpretation
        else:
            st.session_state.initial_interpretation = st.session_state.get("initial_interpretation", "")

        if st.session_state.initial_interpretation:
            st.markdown("**Initial Interpretation:**")
            st.info(st.session_state.initial_interpretation)

    # Chat Interface
    st.subheader("Ask Follow-Up Questions")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_input = st.text_input("Your question:", key="user_question")
    if st.button("Send"):
        if user_input:
            st.session_state.chat_history.append(("User", user_input))
            # Use the latest sensor data context if desired. You can append it to the prompt.
            prompt = user_input
            response = interpret_data(prompt)
            st.session_state.chat_history.append(("LLM", response))
            st.experimental_rerun()

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Chat History")
        for role, msg in st.session_state.chat_history:
            st.write(f"**{role}:** {msg}")
