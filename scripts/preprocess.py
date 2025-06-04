
import pandas as pd

def preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Convert DateTime column to datetime format
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Drop the DateTime column
    df = df.drop(columns=['DateTime'])
    
    # Fill missing values with the median of each column
    df = df.fillna(df.median())
    
    return df

# Example usage:
# df = preprocess_data('data/energy_data.csv')
# print(df.head())
