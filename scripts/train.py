import os
import joblib
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from preprocess import preprocess_data  # Ensure this import works relative to your script location

def train_model(df):
    # Split the data into features and target
    X = df.drop(columns=['EnergyConsumption'])
    y = df['EnergyConsumption']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define and train the model using scikit-learn API
    model = LGBMRegressor(
        objective='regression',
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.9
    )
    
    try:
        model.fit(X_train, y_train)
        print("✅ Model trained successfully.")
        
        # Ensure the models directory exists
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(model_dir, exist_ok=True)

        # Save the model
        model_path = os.path.join(model_dir, 'best_lightgbm_model.pkl')
        joblib.dump(model, model_path)
        print(f"✅ Model saved at: {model_path}")
    except Exception as e:
        print(f"❌ Error during training or saving: {e}")
    
    return model

# Run this script directly
if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'energy_data.csv')
    df = preprocess_data(data_path)
    train_model(df)
