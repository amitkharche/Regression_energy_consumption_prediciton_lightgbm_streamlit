
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import joblib

def cross_validate_model(df):
    # Split the data into features and target
    X = df.drop(columns(['EnergyConsumption']))
    y = df['EnergyConsumption']
    
    # Load the model
    model = joblib.load('models/lightgbm_model.pkl')
    
    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    
    # Save the cross-validation results
    with open('output/cross_validation_results.txt', 'w') as f:
        f.write(f'Cross-validation RMSE scores: {scores}\n')
        f.write(f'Mean RMSE: {scores.mean()}\n')
    
    return scores

# Example usage:
# df = preprocess_data('data/energy_data.csv')
# scores = cross_validate_model(df)
