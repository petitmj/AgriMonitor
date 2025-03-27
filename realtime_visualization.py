import streamlit as st
import pandas as pd
import boto3
import asyncio
import aiohttp
from datetime import datetime
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# ================================
# Configuration & AWS Setup
# ================================
st.set_page_config(page_title="🌾 Agriculture Monitoring", layout="wide")
st.title("🌾 Agriculture Monitoring System with AWS Integration")

# -------------------------------
# AWS Session Setup
# -------------------------------
@st.cache_resource
def get_dynamodb_session():
    """Creates a cached AWS session for improved performance."""
    try:
        return boto3.Session(
            aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
            aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
            region_name=st.secrets["aws"]["region_name"]
        )
    except (NoCredentialsError, PartialCredentialsError) as e:
        st.error(f"⚠️ AWS Credentials Error: {e}")
        return None

# -------------------------------
# Fetch Sensor Data from DynamoDB
# -------------------------------
@st.cache_data(ttl=60)
def fetch_data():
    """Fetches real-time sensor data from DynamoDB."""
    session = get_dynamodb_session()
    if not session:
        return pd.DataFrame()

    try:
        dynamodb = session.resource('dynamodb')
        table = dynamodb.Table('AgricultureMonitoring')
        items = []

        # Implement pagination for large datasets
        response = table.scan()
        items.extend(response.get('Items', []))
        while 'LastEvaluatedKey' in response:
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            items.extend(response.get('Items', []))

        if not items:
            return pd.DataFrame(columns=[
                "timestamp", "temperature", "humidity", "soil_moisture",
                "soil_nitrogen", "soil_phosphorus", "soil_potassium"
            ])

        df = pd.DataFrame(items)
        numeric_cols = ["temperature", "humidity", "soil_moisture", 
                        "soil_nitrogen", "soil_phosphorus", "soil_potassium"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        df.dropna(inplace=True)
        df.sort_values('timestamp', inplace=True)
        return df

    except Exception as e:
        st.error(f"⚠️ Error fetching data from DynamoDB: {e}")
        return pd.DataFrame()

# -------------------------------
# Hugging Face LLM Integration (Async)
# -------------------------------
async def interpret_data(prompt):
    """Interprets agricultural sensor data using an LLM (Hugging Face API)."""
    hf_api_token = st.secrets["huggingface"]["api_token"]
    headers = {"Authorization": f"Bearer {hf_api_token}"}
    model = "facebook/bart-large-cnn"
    API_URL = f"https://api-inference.huggingface.co/models/{model}"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=30) as response:
                if response.status == 200:
                    output = await response.json()
                    return output[0].get("generated_text", "⚠️ No response received.")
                else:
                    return f"⚠️ API Error: {response.status} - {await response.text()}"
        except asyncio.TimeoutError:
            return "⏳ Request timed out. Try again."
        except Exception as e:
            return f"⚠️ API Error: {e}"

# ================================
# Sidebar: Choose View Mode
# ================================
view_mode = st.sidebar.radio("Select View", ["Dashboard", "Chat"])

# ================================
# Dashboard View (Sensor Visualization)
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
# Chat View (AI-powered Interpretation)
# ================================
elif view_mode == "Chat":
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
            with st.spinner("Analyzing data..."):
                st.session_state["initial_interpretation"] = asyncio.run(interpret_data(initial_prompt))

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
            with st.spinner("Thinking..."):
                response = asyncio.run(interpret_data(user_input))
            st.session_state["chat_history"].append(("LLM", response))
            st.rerun()

    # Display chat history
    if st.session_state["chat_history"]:
        st.markdown("### 🗨️ Chat History")
        for role, msg in st.session_state["chat_history"]:
            st.write(f"**{role}:** {msg}")
