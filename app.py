import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
@st.cache_data
def load_data():
    file_path = "agriculture_dataset.csv"  # Ensure the dataset is in the same directory
    return pd.read_csv(file_path)

# Prepare the dataset for ML model
def preprocess_data(data):
    label_encoders = {}
    for column in ["Plant Name", "Soil Type", "Growth Stage", "Available Waste Products"]:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    return data, label_encoders

# Train the ML model
@st.cache_resource
def train_model(data):
    X = data[["Plant Name", "Temperature (°C)", "Soil Type", "Soil pH", "Growth Stage"]]
    y = data["Available Waste Products"]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

# Safely encode user inputs with fallback
def safe_transform(label_encoder, value):
    if value in label_encoder.classes_:
        return label_encoder.transform([value])[0]
    else:
        return label_encoder.transform([label_encoder.classes_[0]])[0]  # Use the first class as a fallback

# Main Streamlit App
def main():
    st.title("🌱 Agricultural Waste Recommendation System")
    st.markdown("This app predicts the recommended organic waste products for your plants based on the provided inputs.")
    st.write("---")

    # Load and preprocess data
    data = load_data()
    processed_data, label_encoders = preprocess_data(data)
    model = train_model(processed_data)

    # Center the input fields
    st.write("### Enter the following details:")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        plant_name = st.selectbox(
            "Select Plant Name",
            sorted(data["Plant Name"].unique()),
            format_func=lambda x: label_encoders["Plant Name"].inverse_transform([x])[0],
        )
        temperature = st.slider("Average Temperature (°C)", 15.0, 35.0, step=0.1)
        soil_type = st.selectbox(
            "Select Soil Type",
            sorted(data["Soil Type"].unique()),
            format_func=lambda x: label_encoders["Soil Type"].inverse_transform([x])[0],
        )
        soil_ph = st.slider("Soil pH", 5.5, 8.5, step=0.1)
        growth_stage = st.selectbox(
            "Select Growth Stage",
            sorted(data["Growth Stage"].unique()),
            format_func=lambda x: label_encoders["Growth Stage"].inverse_transform([x])[0],
        )

    st.write("---")

    # Create columns for both buttons side by side
    col1, col2 = st.columns([1, 1])

    with col1:
        predict_button = st.button("Predict")
    
    with col2:
        # Chatbot Button (Always Available)
        chatbot_button = st.button("Chat with Bot")
    
    # Predict button functionality
    if predict_button:
        # Preprocess user inputs safely
        input_data = pd.DataFrame({
            "Plant Name": [safe_transform(label_encoders["Plant Name"], plant_name)],
            "Temperature (°C)": [temperature],
            "Soil Type": [safe_transform(label_encoders["Soil Type"], soil_type)],
            "Soil pH": [soil_ph],
            "Growth Stage": [safe_transform(label_encoders["Growth Stage"], growth_stage)],
        })

        # Prediction
        prediction_encoded = model.predict(input_data)[0]
        prediction = label_encoders["Available Waste Products"].inverse_transform([prediction_encoded])[0]

        # Display result
        st.success(f"### Recommended Organic Waste Product: {prediction}")

    # Chatbot modal (Initially hidden)
    if chatbot_button:
        st.markdown("""
            <style>
                #chatbot-box {
                    display: block;
                    position: fixed;
                    bottom: 10px;
                    right: 10px;
                    width: 350px;
                    height: 500px;
                    z-index: 100;
                    border: 1px solid #ccc;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    background-color: white;
                }
                iframe {
                    width: 100%;
                    height: 100%;
                    border: none;
                }
            </style>
            <div id="chatbot-box">
                <iframe src="https://cdn.botpress.cloud/webchat/v2.2/shareable.html?configUrl=https://files.bpcontent.cloud/2025/01/04/15/20250104152613-5ONG25PS.json"></iframe>
            </div>
        """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
