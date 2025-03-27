import streamlit as st
import pandas as pd
import boto3
from datetime import datetime
import time
from huggingface_hub import InferenceApi

# -------------------------------
# Streamlit configuration
st.set_page_config(page_title="🌾 Agriculture Monitoring", layout="wide")
st.title("🌾 Agriculture Monitoring System with AWS & Hugging Face")

# -------------------------------
# Sidebar Mode Selection
mode = st.sidebar.radio("Choose Mode", options=["Dashboard", "Simple"])

# -------------------------------
# Fetch data from DynamoDB using credentials from Streamlit Secrets
@st.cache_data(ttl=60)
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
        for col in ["temperature", "humidity", "soil_moisture", "soil_nitrogen", "soil_phosphorus", "soil_potassium"]:
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
# Function to interpret results using Hugging Face Inference API
def interpret_results(latest_data):
    prompt = (
        "You are an agricultural data analyst. Interpret the following sensor data:\n"
        f"Timestamp: {latest_data['timestamp']}\n"
        f"Temperature: {latest_data['temperature']} °C\n"
        f"Humidity: {latest_data['humidity']} %\n"
        f"Soil Moisture: {latest_data['soil_moisture']}\n"
        f"Nitrogen: {latest_data['soil_nitrogen']} mg/kg\n"
        f"Phosphorus: {latest_data['soil_phosphorus']} mg/kg\n"
        f"Potassium: {latest_data['soil_potassium']} mg/kg\n\n"
        "Provide a detailed interpretation of these readings for crop health and suggest improvements."
    )

    hf_inference = InferenceApi(repo_id="google/flan-t5-xl", token=st.secrets["huggingface"]["api_key"])
    try:
        # Request a raw response so we can manually parse the output.
        raw_response = hf_inference(prompt, raw_response=True)
        try:
            # Try to parse the response as JSON.
            response_json = raw_response.json()
            interpretation = response_json.get("generated_text", "")
        except Exception:
            # If JSON parsing fails, fall back to plain text.
            interpretation = raw_response.text
        return interpretation.strip()
    except Exception as e:
        return f"Error generating interpretation: {e}"

# -------------------------------
# Simple Chat Interface using Hugging Face for conversation
def chat_with_expert(messages):
    prompt = "You are an agricultural expert providing insights based on sensor data.\n"
    for msg in messages:
        prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
    
    hf_inference = InferenceApi(repo_id="google/flan-t5-xl", token=st.secrets["huggingface"]["api_key"])
    try:
        raw_response = hf_inference(prompt, raw_response=True)
        try:
            response_json = raw_response.json()
            return response_json.get("generated_text", "").strip()
        except Exception:
            return raw_response.text.strip()
    except Exception as e:
        return f"Error generating response: {e}"

# -------------------------------
# Main App Logic
df = fetch_data()

if mode == "Dashboard":
    # Dashboard Mode: Visualizations and Metrics
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

elif mode == "Simple":
    # Simple Mode: Chat Interface
    st.subheader("💬 Chat with the Agriculture Expert")
    
    if df.empty:
        st.info("No data available for interpretation.")
    else:
        latest_data = df.iloc[-1]
        interpretation = interpret_results(latest_data)
        st.markdown("#### Interpretation of Latest Results:")
        st.write(interpretation)
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.text_input("Enter your question or comment:")

    if st.button("Send"):
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": "You are an agricultural expert."}] + st.session_state["chat_history"]
        response = chat_with_expert(messages)
        st.session_state["chat_history"].append({"role": "assistant", "content": response})

    st.markdown("#### Chat History:")
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Expert:** {msg['content']}")
