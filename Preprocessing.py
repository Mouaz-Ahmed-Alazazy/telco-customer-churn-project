import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC as support_vector_machine
from sklearn.metrics import classification_report
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.feature_selection import SelectKBest, f_classif


class PreProcessing:
    """
    A preprocessing pipeline for telco customer churn data.
    """
    
    def __init__(self, scaler_type: str = 'standard', n_features: Optional[int] = None):
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.scaler = self._initialize_scaler(scaler_type)
        self.feature_selector = SelectKBest(score_func=f_classif, k=n_features) if n_features else None
        self.categorical_cols: List[str] = []
        self.is_fitted = False
        self.scaler_type = scaler_type
        
    def _initialize_scaler(self, scaler_type: str):
        """Initialize the appropriate scaler based on the type."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'maxabs': MaxAbsScaler()
        }
        if scaler_type not in scalers:
            raise ValueError(f"Invalid scaler_type '{scaler_type}'. Choose from {list(scalers.keys())}")
        return scalers[scaler_type]
    
    def load_data(self, filepath: str):
        """
        Load dataset from a CSV file.
        """
        try:
            df = pd.read_csv(filepath)
            print(f"[OK] Loaded data from {filepath} - Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"[ERROR] File not found at {filepath}")
            return None
        except Exception as e:
            print(f"[ERROR] Error loading file: {e}")
            return None
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values with 0.
        """        
        df_clean = df.copy()
        missing_count = df_clean.isnull().sum().sum()
        if missing_count > 0:
            print(f"[OK] Filled {missing_count} missing values with 0")
        df_clean.fillna(0, inplace=True)
        return df_clean
    
    def fit(self, df: pd.DataFrame, target_col: Optional[str] = 'Churn'):
        """
        Fit the preprocessor on training data.
        """
        df_fit = df.copy()
        
        # Identify categorical columns
        self.categorical_cols = df_fit.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target if present
        if target_col and target_col in self.categorical_cols:
            self.categorical_cols.remove(target_col)
        
        # Fit encoder on categorical columns
        if self.categorical_cols:
            df_fit[self.categorical_cols] = df_fit[self.categorical_cols].astype(str)
            self.encoder.fit(df_fit[self.categorical_cols])
            # Apply encoding
            df_fit[self.categorical_cols] = self.encoder.transform(df_fit[self.categorical_cols])
            print(f"[OK] Fitted encoder on {len(self.categorical_cols)} categorical columns")
        
        # Get ALL numerical columns for scaling (after encoding)
        cols_to_scale = df_fit.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in cols_to_scale:
            cols_to_scale.remove(target_col)
        
        # Fit scaler on numerical columns (including encoded categoricals)
        if cols_to_scale:
            self.scaler.fit(df_fit[cols_to_scale])
            print(f"[OK] Fitted {self.scaler_type} scaler on {len(cols_to_scale)} numerical columns")
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame, target_col: Optional[str] = 'Churn') -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")
        
        df_transformed = df.copy()
        
        # Transform categorical columns
        if self.categorical_cols:
            df_transformed[self.categorical_cols] = df_transformed[self.categorical_cols].astype(str)
            df_transformed[self.categorical_cols] = self.encoder.transform(df_transformed[self.categorical_cols])
        
        # Transform ALL numerical columns (after encoding)
        numerical_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numerical_cols:
            numerical_cols.remove(target_col)
        
        if numerical_cols:
            df_transformed[numerical_cols] = self.scaler.transform(df_transformed[numerical_cols])
        
        return df_transformed

    
    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = 'Churn') -> pd.DataFrame:
        """
        Fit on data and then transform it.
        """        
        return self.fit(df, target_col).transform(df, target_col)
    
    def handle_imbalance(self, X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to handle class imbalance.
        """
        original_counts = y.value_counts().to_dict()
        
        sm = SMOTE(random_state=random_state)
        X_resampled, y_resampled = sm.fit_resample(X, y)
        
        new_counts = pd.Series(y_resampled).value_counts().to_dict()
        print(f"[OK] Applied SMOTE - Before: {original_counts}, After: {new_counts}")
        
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=y.name)
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 10) -> Tuple[pd.DataFrame, List[str]]:
        
        """
        Select top k features using SelectKBest with f_classif.
        """        
        
        if self.feature_selector is None or self.feature_selector.k != k:
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        
        X_selected = self.feature_selector.fit_transform(X, y)
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        print(f"[OK] Selected {k} features: {selected_features}")
        
        return pd.DataFrame(X_selected, columns=selected_features), selected_features


def main():
    """
    Main function demonstrating the preprocessing pipeline.
    """
    print("="*60)
    print("Telco Customer Churn - Data Preprocessing Pipeline")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = PreProcessing(scaler_type='standard')
    
    # Load training and test data
    print("\n[1/7] Loading Data...")
    df_train = preprocessor.load_data('train.csv')
    df_test = preprocessor.load_data('test.csv')
    
    # drop leaking features
    drop_columns = ['customerID','Churn Score','Churn Reason','Churn Category','Satisfaction Score','Customer Status']
    df_train = df_train.drop(columns=drop_columns, errors='ignore')
    df_test = df_test.drop(columns=drop_columns, errors='ignore')

    if df_train is None or df_test is None:
        print("[ERROR] Failed to load data. Exiting.")
        return
    
    

    # Handle missing values
    print("\n[2/7] Handling Missing Values...")
    df_train = preprocessor.handle_missing_values(df_train)
    df_test = preprocessor.handle_missing_values(df_test)
    
    # Split features and target
    print("\n[3/7] Splitting Features and Target...")
    if 'Churn' not in df_train.columns:
        print("[ERROR] Target column 'Churn' not found in training data")
        return
    
    X_train = df_train.drop('Churn', axis=1)
    y_train = df_train['Churn']
    
    if 'Churn' not in df_test.columns:
        print("[ERROR] Target column 'Churn' not found in test data")
        return
    
    X_test = df_test.drop('Churn', axis=1)
    y_test = df_test['Churn']
    
    print(f"[OK] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Fit preprocessor on training data and transform both sets
    print("\n[4/7] Encoding and Scaling...")
    preprocessor.fit(X_train, target_col=None)
    X_train_transformed = preprocessor.transform(X_train, target_col=None)
    X_test_transformed = preprocessor.transform(X_test, target_col=None)
    
    # Handle class imbalance (ONLY on training data)
    print("\n[5/7] Handling Class Imbalance (Training Data Only)...")
    X_train_resampled, y_train_resampled = preprocessor.handle_imbalance(X_train_transformed, y_train)
    print(f"[OK] Training shape after SMOTE: {X_train_resampled.shape}")
    
    # Select top features
    print("\n[5.5/7] Selecting Top Features...")
    X_train_selected, selected_features = preprocessor.select_features(X_train_resampled, y_train_resampled, k=10)
    X_test_selected = X_test_transformed[selected_features]

    # Train and evaluate Logistic Regression model
    print("\n[6/7] Training Logistic Regression Model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_selected, y_train_resampled)
    print("[OK] Model trained successfully")

    # Train SVM model
    svm_model = support_vector_machine(max_iter=1000, kernel='rbf', random_state=42)
    svm_model.fit(X_train_selected, y_train_resampled)
    print("[OK] SVM Model trained successfully")
    svm_y_pred_test = svm_model.predict(X_test_selected)
    print("\nSVM Test Set Classification Report:")
    print("-"*60)
    print(classification_report(y_test, svm_y_pred_test))

    # Evaluate Logistic Regression on test data
    print("\n[7/7] Evaluating on Test Data...")
    y_pred_test = model.predict(X_test_selected)

    print("\nLogistic Regression Test Set Classification Report:")
    print("-"*60)
    print(classification_report(y_test, y_pred_test))
    
    print("="*60)
    print("Preprocessing Pipeline Completed Successfully!")
    print("="*60)
    
    print("columns of train data:")
    print(X_train_selected.columns)

if __name__ == "__main__":
    main()
