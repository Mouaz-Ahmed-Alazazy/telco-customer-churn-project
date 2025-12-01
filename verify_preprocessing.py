import pandas as pd
from Preprocessing import PreProcessing
import os

def verify():
    print("Starting verification...")
    preprocessor = PreProcessing()
    
    # 1. Load Data
    filepath = 'train.csv'
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    df = preprocessor.load_data(filepath)
    if df is not None:
        print(f"1. Data loaded successfully. Shape: {df.shape}")
    else:
        print("1. Failed to load data.")
        return

    # 2. Handle Missing Values
    df = preprocessor.handle_missing_values(df)
    if df.isnull().sum().sum() == 0:
        print("2. Missing values handled successfully.")
    else:
        print("2. Failed to handle missing values.")

    # 3. Encode Categorical
    df = preprocessor.encode_categorical(df)
    if df.select_dtypes(include=['object']).shape[1] == 0:
        print("3. Categorical variables encoded successfully.")
    else:
        print("3. Failed to encode categorical variables.")

    # 4. Scale Data & 5. Handle Imbalance & 6. Feature Selection & 7. Split Data
    if 'Churn' in df.columns:
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        # 4. Scale Data
        X_scaled = preprocessor.scale_data(X, scaler_type='standard')
        print(f"4. Data scaled successfully. Mean of first col: {X_scaled.iloc[:, 0].mean():.2f}")
        
        # 5. Handle Imbalance
        try:
            X_res, y_res = preprocessor.handle_imbalance(X_scaled, y)
            print(f"5. Imbalance handled successfully. Resampled shape: {X_res.shape}")
        except Exception as e:
            print(f"5. Failed to handle imbalance: {e}")

        # 6. Feature Selection
        try:
            X_selected, selected_features = preprocessor.select_features(X_res, y_res, k=10)
            print(f"6. Feature selection successful. Selected features: {list(selected_features)}")
        except Exception as e:
            print(f"6. Failed to select features: {e}")

        # 7. Split Data
        try:
            X_train, X_test, y_train, y_test = preprocessor.split_data(X_selected, y_res)
            print(f"7. Data split successfully. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        except Exception as e:
            print(f"7. Failed to split data: {e}")

    else:
        print("Target column 'Churn' not found. Skipping steps 4-7.")

if __name__ == "__main__":
    verify()
