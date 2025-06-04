
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib

def train_model(df):
    # Split the data into features and target
    X = df.drop(columns=['EnergyConsumption'])
    y = df['EnergyConsumption']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)
    
    # Define parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    
    # Train the model
    model = lgb.train(params, train_data, valid_sets=[test_data], early_stopping_rounds=10)
    
    # Save the model
    joblib.dump(model, 'models/lightgbm_model.pkl')
    
    return model

# Example usage:
# df = preprocess_data('data/energy_data.csv')
# model = train_model(df)
