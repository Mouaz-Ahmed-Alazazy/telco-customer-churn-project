import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


class PreProcessing:
    def __init__(self):
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.scaler = None

    def load_data(self, filepath):
        """
        Loads the dataset from a CSV file.
        """
        try:
            df = pd.read_csv(filepath)
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None

    def handle_missing_values(self, df):
        """
        Fills missing values with 0.
        """
        df_clean = df.copy()
        df_clean.fillna(0, inplace=True)
        return df_clean

    def encode_categorical(self, df):
        """
        Encodes categorical columns using OrdinalEncoder.
        """
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_encoded[col] = df_encoded[col].astype(str)
        df_encoded[categorical_cols] = self.encoder.fit_transform(df_encoded[categorical_cols])
        return df_encoded

    def scale_data(self, df, scaler_type='standard'):
        """
        Scales numerical features using the specified scaler.
        scaler_type: 'standard', 'minmax', 'maxabs'
        """
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'maxabs':
            self.scaler = MaxAbsScaler()
        else:
            raise ValueError("Invalid scaler_type. Choose from 'standard', 'minmax', 'maxabs'.")
        
        cols_to_scale = df.columns
        if 'Churn' in cols_to_scale:
            cols_to_scale = df.drop('Churn', axis=1).columns
            
        df_scaled = df.copy()
        df_scaled[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        
        return df_scaled

    def handle_imbalance(self, X, y):
        """
        Applies SMOTE to handle class imbalance.
        """
        sm = SMOTE(random_state=42)
        X_resampled, y_resampled = sm.fit_resample(X, y)
        return X_resampled, y_resampled

    def select_features(self, X, y, k=10):
        """
        Selects top k features using SelectKBest with f_classif.
        """
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        return X_new, selected_features

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Splits data into training and testing sets.
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    preprocessor = PreProcessing()
    
    df = preprocessor.load_data('train.csv')
    
    if df is not None:
        df = preprocessor.handle_missing_values(df)
        df = preprocessor.encode_categorical(df)
        
    if 'Churn' in df.columns:
        X = df.drop('Churn', axis=1)
        y = df['Churn']
            
        # X_scaled = preprocessor.scale_data(X, scaler_type='standard')
            
        X_res, y_res = preprocessor.handle_imbalance(X, y)
            
        print(f"Original shape: {X.shape}, Resampled shape: {X_res.shape}")

    model = LogisticRegression()
    model.fit(X_res, y_res)
    y_pred = model.predict(X_res)
    print(classification_report(y_res, y_pred))

    
