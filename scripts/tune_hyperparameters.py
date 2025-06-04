
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import joblib

def tune_hyperparameters(df):
    # Split the data into features and target
    X = df.drop(columns(['EnergyConsumption']))
    y = df['EnergyConsumption']
    
    # Define the parameter grid
    param_grid = {
        'num_leaves': [31, 50, 100],
        'learning_rate': [0.01, 0.05, 0.1],
        'feature_fraction': [0.7, 0.9, 1.0]
    }
    
    # Create a LightGBM model
    model = lgb.LGBMRegressor(objective='regression', metric='rmse', boosting_type='gbdt')
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_root_mean_squared_error')
    grid_search.fit(X, y)
    
    # Save the best model
    joblib.dump(grid_search.best_estimator_, 'models/best_lightgbm_model.pkl')
    
    # Save the GridSearchCV results
    with open('output/grid_search_results.txt', 'w') as f:
        f.write(f'Best parameters: {grid_search.best_params_}\n')
        f.write(f'Best RMSE: {grid_search.best_score_}\n')
    
    return grid_search.best_estimator_

# Example usage:
# df = preprocess_data('data/energy_data.csv')
# best_model = tune_hyperparameters(df)
