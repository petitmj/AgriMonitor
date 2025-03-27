import streamlit as st
import pandas as pd
import boto3
import requests
from datetime import datetime

# Streamlit configuration
st.set_page_config(page_title="🌾 Agriculture Monitoring", layout="wide")
st.title("🌾 Agriculture Monitoring System with AWS Integration")

# Sidebar mode selection
mode = st.sidebar.radio("Choose Mode", ["Dashboard", "Simple"])

# Helper function: Create a DynamoDB table resource using credentials from secrets
def get_dynamodb_table():
    session = boto3.Session(
        aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
        aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
        region_name=st.secrets["aws"]["region_name"]
    )
    dynamodb = session.resource('dynamodb')
    table = dynamodb.Table('AgricultureMonitoring')
    return table

# Fetch data from DynamoDB and return as a DataFrame
@st.cache_data(ttl=60)
def fetch_data():
    try:
        table = get_dynamodb_table()
        response = table.scan()
        items = response.get('Items', [])
        
        # Return an empty DataFrame if no items are found
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

# Fetch data once (cached)
df = fetch_data()

if mode == "Dashboard":
    if df.empty:
        st.warning("No data available. Please check if the Lambda ingestion script is running.")
    else:
        st.subheader("📊 Live Sensor Data Visualization")
        # Plot live sensor data with timestamp on x-axis
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
        st.download_button("Download Data as CSV", df.to_csv(index=False).encode('utf-8'), "agriculture_data.csv", "text/csv")
        
elif mode == "Simple":
    st.subheader("💬 Chat with the Agricultural Interpreter")
    if df.empty:
        st.warning("No data available to interpret.")
    else:
        # Get the latest sensor reading
        latest_data = df.iloc[-1]
        # Build an initial prompt with the sensor data
        initial_prompt = (
            f"Interpret the following agricultural sensor data:\n"
            f"Temperature: {latest_data['temperature']:.2f} °C\n"
            f"Humidity: {latest_data['humidity']:.2f} %\n"
            f"Soil Moisture: {latest_data['soil_moisture']:.2f}\n"
            f"Nitrogen: {latest_data['soil_nitrogen']:.2f} mg/kg\n"
            f"Phosphorus: {latest_data['soil_phosphorus']:.2f} mg/kg\n"
            f"Potassium: {latest_data['soil_potassium']:.2f} mg/kg\n"
            "Explain the agricultural significance of these readings in simple terms."
        )
        
        # Function to call the HuggingFace Inference API for interpretation
        def get_interpretation(prompt):
            api_url = "https://api-inference.huggingface.co/models/google/flan-t5-small"  # Change to your preferred model
            headers = {"Authorization": f"Bearer {st.secrets['huggingface']['api_token']}"}
            payload = {"inputs": prompt}
            response = requests.post(api_url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                # Expected response format: list with dict containing "generated_text"
                if isinstance(result, list) and "generated_text" in result[0]:
                    return result[0]["generated_text"]
                else:
                    return "Unexpected response format from the model."
            else:
                return f"Error {response.status_code}: {response.text}"
        
        # Initialize chat history in session state if not present
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = [("System", initial_prompt)]
        
        # Display chat history
        for speaker, message in st.session_state["chat_history"]:
            st.markdown(f"**{speaker}:** {message}")
        
        # User input for chat
        user_input = st.text_input("Your question (or press Enter to see the interpretation):")
        if st.button("Send") or user_input:
            if user_input:
                st.session_state["chat_history"].append(("User", user_input))
                combined_prompt = initial_prompt + "\nUser: " + user_input + "\nSystem:"
            else:
                combined_prompt = initial_prompt + "\nSystem:"
            # Get the interpretation from HuggingFace
            interpretation = get_interpretation(combined_prompt)
            st.session_state["chat_history"].append(("System", interpretation))
            st.experimental_rerun()
