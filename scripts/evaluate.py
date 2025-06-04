import streamlit as st
import pandas as pd
import joblib
import os

# Load the model using joblib
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_lightgbm_model.pkl')
model = joblib.load(model_path)

# Load and preprocess the data
data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'energy_data.csv'))
data['DateTime'] = pd.to_datetime(data['DateTime'])
data = data.drop(columns=['DateTime'])
data = data.fillna(data.median())

# Split the data into features and target
X = data.drop(columns=['EnergyConsumption'])
y = data['EnergyConsumption']

# Streamlit app
st.title('Energy Consumption Prediction')
st.write('This app predicts energy consumption based on various features.')

# User input
temperature = st.slider('Temperature', float(X['Temperature'].min()), float(X['Temperature'].max()), float(X['Temperature'].mean()))
humidity = st.slider('Humidity', float(X['Humidity'].min()), float(X['Humidity'].max()), float(X['Humidity'].mean()))
appliance_usage = st.slider('Appliance Usage', float(X['ApplianceUsage'].min()), float(X['ApplianceUsage'].max()), float(X['ApplianceUsage'].mean()))
light_usage = st.slider('Light Usage', float(X['LightUsage'].min()), float(X['LightUsage'].max()), float(X['LightUsage'].mean()))
occupancy = st.slider('Occupancy', int(X['Occupancy'].min()), int(X['Occupancy'].max()), int(X['Occupancy'].mean()))

# Prediction
input_data = pd.DataFrame({
    'Temperature': [temperature],
    'Humidity': [humidity],
    'ApplianceUsage': [appliance_usage],
    'LightUsage': [light_usage],
    'Occupancy': [occupancy]
})
prediction = model.predict(input_data)
st.write(f'🔋 **Predicted Energy Consumption:** {prediction[0]:.2f}')
