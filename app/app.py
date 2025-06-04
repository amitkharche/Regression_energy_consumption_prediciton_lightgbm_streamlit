
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load the model
model_path = Path(__file__).parent.parent / 'models' / 'best_lightgbm_model.pkl'
data_path = Path(__file__).parent.parent / 'data' / 'energy_data.csv'

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found. Please ensure the model is trained and saved.")
    st.stop()

try:
    data = pd.read_csv(data_path)
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data = data.drop(columns=['DateTime'])
    data = data.fillna(data.median())
except FileNotFoundError:
    st.error("Data file not found.")
    st.stop()

# Split the data into features and target
X = data.drop(columns=['EnergyConsumption'])

# Streamlit app
st.title('Energy Consumption Prediction')
st.write('This app predicts energy consumption based on various features.')

# User input
temperature = st.slider('Temperature (°C)', float(X['Temperature'].min()), float(X['Temperature'].max()), float(X['Temperature'].mean()))
humidity = st.slider('Humidity (%)', float(X['Humidity'].min()), float(X['Humidity'].max()), float(X['Humidity'].mean()))
appliance_usage = st.slider('Appliance Usage (kWh)', float(X['ApplianceUsage'].min()), float(X['ApplianceUsage'].max()), float(X['ApplianceUsage'].mean()))
light_usage = st.slider('Light Usage (kWh)', float(X['LightUsage'].min()), float(X['LightUsage'].max()), float(X['LightUsage'].mean()))
occupancy = st.slider('Occupancy', int(X['Occupancy'].min()), int(X['Occupancy'].max()), int(X['Occupancy'].mean()))

# Prediction
input_data = pd.DataFrame({
    'Temperature': [temperature],
    'Humidity': [humidity],
    'ApplianceUsage': [appliance_usage],
    'LightUsage': [light_usage],
    'Occupancy': [occupancy]
})

st.subheader("Input Summary")
st.write(input_data)

prediction = model.predict(input_data)
st.success(f'Predicted Energy Consumption: {prediction[0]:.2f} kWh')

import matplotlib.pyplot as plt

# Feature importance section
st.subheader("🔍 Feature Importance")

try:
    importances = model.feature_importances_
    feature_names = ['Temperature', 'Humidity', 'ApplianceUsage', 'LightUsage', 'Occupancy']

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(feature_names, importances, color='skyblue')
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances")
    ax.invert_yaxis()
    st.pyplot(fig)

except AttributeError:
    st.warning("Feature importances are not available for this model.")

