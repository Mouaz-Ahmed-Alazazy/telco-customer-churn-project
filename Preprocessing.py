#top highest corelation features with target variable 'Churn'
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC as support_vector_machine
from sklearn.metrics import classification_report
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier


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
    
    def fit_feature_selector(self, X: pd.DataFrame, y: pd.Series, k: int = 10) -> List[str]:
        """
        Fit feature selector on training data and return selected feature names.
        This should be called on training (or train+validation) data only.
        """
        if self.feature_selector is None or self.feature_selector.k != k:
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        
        self.feature_selector.fit(X, y)
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        print(f"[OK] Selected {k} features: {selected_features}")
        
        return selected_features
    
    def transform_features(self, X: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
        """
        Transform data to use only selected features.
        Can be applied to train, validation, or test data.
        """
        return X[selected_features]

    def select_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
                              X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        """
        Select hyperparameters for the three models using explicit train/validation split.
        Train on X_train/y_train, validate on X_val/y_val.
        No cross-validation since we have an explicit validation set.
        """
        results = {}

        # Logistic Regression
        lr_param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        }
        print("\n[Hyperparameters] Tuning Logistic Regression...")
        best_lr_score = 0
        best_lr_params = {}
        
        for C in lr_param_grid['C']:
            for penalty in lr_param_grid['penalty']:
                for solver in lr_param_grid['solver']:
                    model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=2000, random_state=42)
                    model.fit(X_train, y_train)
                    val_score = model.score(X_val, y_val)
                    if val_score > best_lr_score:
                        best_lr_score = val_score
                        best_lr_params = {'C': C, 'penalty': penalty, 'solver': solver}
        
        results['LogisticRegression'] = {'best_params': best_lr_params, 'best_score': best_lr_score}
        print(f"Best Params for LR: {best_lr_params} | Validation Score: {best_lr_score:.4f}")

        # Decision Tree
        dt_param_grid = {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        print("\n[Hyperparameters] Tuning Decision Tree...")
        best_dt_score = 0
        best_dt_params = {}
        
        for max_depth in dt_param_grid['max_depth']:
            for min_split in dt_param_grid['min_samples_split']:
                for min_leaf in dt_param_grid['min_samples_leaf']:
                    for criterion in dt_param_grid['criterion']:
                        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_split,
                                                       min_samples_leaf=min_leaf, criterion=criterion, random_state=42)
                        model.fit(X_train, y_train)
                        val_score = model.score(X_val, y_val)
                        if val_score > best_dt_score:
                            best_dt_score = val_score
                            best_dt_params = {'max_depth': max_depth, 'min_samples_split': min_split,
                                            'min_samples_leaf': min_leaf, 'criterion': criterion}
        
        results['DecisionTree'] = {'best_params': best_dt_params, 'best_score': best_dt_score}
        print(f"Best Params for DT: {best_dt_params} | Validation Score: {best_dt_score:.4f}")

        # SVM - Reduced grid for computational efficiency
        svm_param_grid = {
            'C': [0.1, 1, 10],  # Reduced from [0.1, 1, 10, 100]
            'gamma': ['scale'],  # Reduced from ['scale', 'auto']
            'kernel': ['rbf']    # Reduced from ['rbf', 'linear'] - RBF usually performs better
        }
        print("\n[Hyperparameters] Tuning SVM...")
        print(f"[INFO] Testing {len(svm_param_grid['C']) * len(svm_param_grid['gamma']) * len(svm_param_grid['kernel'])} parameter combinations...")
        best_svm_score = 0
        best_svm_params = {}
        
        combination_num = 0
        total_combinations = len(svm_param_grid['C']) * len(svm_param_grid['gamma']) * len(svm_param_grid['kernel'])
        
        for C in svm_param_grid['C']:
            for gamma in svm_param_grid['gamma']:
                for kernel in svm_param_grid['kernel']:
                    combination_num += 1
                    print(f"  Testing combination {combination_num}/{total_combinations}: C={C}, gamma={gamma}, kernel={kernel}...", end=" ")
                    model = support_vector_machine(C=C, gamma=gamma, kernel=kernel, 
                                                   probability=True, max_iter=1000, random_state=42)
                    model.fit(X_train, y_train)
                    val_score = model.score(X_val, y_val)
                    print(f"Score: {val_score:.4f}")
                    if val_score > best_svm_score:
                        best_svm_score = val_score
                        best_svm_params = {'C': C, 'gamma': gamma, 'kernel': kernel}
        
        results['SVM'] = {'best_params': best_svm_params, 'best_score': best_svm_score}
        print(f"Best Params for SVM: {best_svm_params} | Validation Score: {best_svm_score:.4f}")

        return results
        


def main():

    print("="*60)
    print("Telco Customer Churn")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = PreProcessing(scaler_type='standard')
    
    # ========================================
    # STEP 1: Load ALL three datasets
    # ========================================
    print("\n[STEP 1/9] Loading Data...")
    df_train = preprocessor.load_data('train.csv')
    df_validation = preprocessor.load_data('validation.csv')
    df_test = preprocessor.load_data('test.csv')
    
    if df_train is None or df_validation is None or df_test is None:
        print("[ERROR] Failed to load one or more datasets. Exiting.")
        return
    
    print(f"[OK] Loaded 3 datasets - Train: {df_train.shape}, Validation: {df_validation.shape}, Test: {df_test.shape}")
    
    # ========================================
    # Exploratory Analysis (on training data only)
    # ========================================
    print("\n[INFO] Exploratory Data Analysis (Training Data Only)...")
    churned_customers = df_train[df_train['Churn'] == 1]
    non_churned_customers = df_train[df_train['Churn'] == 0]
    
    print(f"  Churned Customers Mean Tenure: {churned_customers['Tenure in Months'].mean():.2f} months")
    print(f"  Non-Churned Customers Mean Tenure: {non_churned_customers['Tenure in Months'].mean():.2f} months")
    print(f"  Class Balance - Churned: {len(churned_customers)}, Non-Churned: {len(non_churned_customers)}")
    
    # ========================================
    # Drop leaking features from ALL datasets
    # ========================================
    print("\n[STEP 2/9] Dropping Data Leakage Features...")
    drop_columns = ['customerID', 'Churn Score', 'Churn Reason', 'Churn Category', 
                   'Satisfaction Score', 'Customer Status']
    df_train = df_train.drop(columns=drop_columns, errors='ignore')
    df_validation = df_validation.drop(columns=drop_columns, errors='ignore')
    df_test = df_test.drop(columns=drop_columns, errors='ignore')
    print(f"[OK] Dropped {len(drop_columns)} potential leakage columns from all datasets")
    
    # ========================================
    # Handle missing values in ALL datasets
    # ========================================
    print("\n[STEP 3/9] Handling Missing Values...")
    df_train = preprocessor.handle_missing_values(df_train)
    df_validation = preprocessor.handle_missing_values(df_validation)
    df_test = preprocessor.handle_missing_values(df_test)
    
    # ========================================
    # Split features and target for ALL datasets
    # ========================================
    print("\n[STEP 4/9] Splitting Features and Target...")
    if 'Churn' not in df_train.columns or 'Churn' not in df_validation.columns or 'Churn' not in df_test.columns:
        print("[ERROR] Target column 'Churn' not found in one or more datasets")
        return
    
    X_train = df_train.drop('Churn', axis=1)
    y_train = df_train['Churn']
    
    X_validation = df_validation.drop('Churn', axis=1)
    y_validation = df_validation['Churn']
    
    X_test = df_test.drop('Churn', axis=1)
    y_test = df_test['Churn']
    
    print(f"[OK] Split completed - Train: {X_train.shape}, Validation: {X_validation.shape}, Test: {X_test.shape}")
    
    # ========================================
    # Fit preprocessor ONLY on training data
    # ========================================
    print("\n[STEP 5/9] Fitting Preprocessing (TRAINING DATA ONLY)...")
    preprocessor.fit(X_train, target_col=None)
    print("[OK] Preprocessor fitted on training data only (no validation/test contamination)")
    
    # Transform all three datasets using the fitted preprocessor
    X_train_transformed = preprocessor.transform(X_train, target_col=None)
    X_validation_transformed = preprocessor.transform(X_validation, target_col=None)
    X_test_transformed = preprocessor.transform(X_test, target_col=None)
    print("[OK] All datasets transformed using training-fitted preprocessor")
    
    # ========================================
    # Handle class imbalance (ONLY on training data)
    # ========================================
    print("\n[STEP 6/9] Handling Class Imbalance (TRAINING DATA ONLY)...")
    X_train_resampled, y_train_resampled = preprocessor.handle_imbalance(X_train_transformed, y_train)
    print(f"[OK] SMOTE applied to training data only - New shape: {X_train_resampled.shape}")
    print("[INFO] Validation and test data remain unchanged (class imbalance NOT applied)")
    
    # ========================================
    # Feature selection: Fit on TRAIN, apply to all
    # ========================================
    print("\n[STEP 7/9] Feature Selection...")
    print("[INFO] Fitting feature selector on training data only...")
    selected_features = preprocessor.fit_feature_selector(X_train_resampled, y_train_resampled, k=7)
    
    # Apply selected features to all datasets
    X_train_selected = preprocessor.transform_features(X_train_resampled, selected_features)
    X_validation_selected = preprocessor.transform_features(X_validation_transformed, selected_features)
    X_test_selected = preprocessor.transform_features(X_test_transformed, selected_features)
    print(f"[OK] Feature selection applied to all datasets - Selected {len(selected_features)} features")
    
    # ========================================
    # Hyperparameter tuning using train and validation
    # ========================================
    print("\n[STEP 8/9] Hyperparameter Tuning (Train on TRAIN, Validate on VALIDATION)...")
    print("[INFO] This may take a few minutes...")
    best_params = preprocessor.select_hyperparameters(X_train_selected, y_train_resampled,
                                                      X_validation_selected, y_validation)
    
    print("\n" + "="*80)
    print("Hyperparameter Tuning Completed - Best Parameters:")
    print("="*80)
    for model_name, result in best_params.items():
        print(f"\n{model_name}:")
        print(f"  Parameters: {result['best_params']}")
        print(f"  Validation Score: {result['best_score']:.4f}")
    
    # ========================================
    # Final Training: Combine train+validation
    # ========================================
    print("\n" + "="*80)
    print("[STEP 9/9] Final Model Training (TRAIN + VALIDATION combined)")
    print("="*80)
    
    # Combine training and validation for final training
    X_final_train = pd.concat([X_train_selected, X_validation_selected], axis=0)
    y_final_train = pd.concat([y_train_resampled, y_validation], axis=0)
    print(f"[OK] Combined train+validation for final training - Shape: {X_final_train.shape}")
    
    # Train final models with best hyperparameters
    print("\n[INFO] Training final models with best hyperparameters...")
    
    # Logistic Regression
    lr_params = best_params['LogisticRegression']['best_params']
    final_lr_model = LogisticRegression(**lr_params, max_iter=2000, random_state=42)
    final_lr_model.fit(X_final_train, y_final_train)
    print("[OK] Logistic Regression trained with best params")
    
    # Decision Tree
    dt_params = best_params['DecisionTree']['best_params']
    final_dt_model = DecisionTreeClassifier(**dt_params, random_state=42)
    final_dt_model.fit(X_final_train, y_final_train)
    print("[OK] Decision Tree trained with best params")
    
    # SVM
    svm_params = best_params['SVM']['best_params']
    final_svm_model = support_vector_machine(**svm_params, probability=True, random_state=42)
    final_svm_model.fit(X_final_train, y_final_train)
    print("[OK] SVM trained with best params")
    
    # ========================================
    # FINAL EVALUATION: Test set (ONLY ONCE!)
    # ========================================
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET (Used Only Once)")
    print("="*80)
    
    print("\n" + "-"*80)
    print("Logistic Regression - Test Set Performance:")
    print("-"*80)
    y_pred_lr = final_lr_model.predict(X_test_selected)
    print(classification_report(y_test, y_pred_lr))
    
    print("\n" + "-"*80)
    print("Decision Tree - Test Set Performance:")
    print("-"*80)
    y_pred_dt = final_dt_model.predict(X_test_selected)
    print(classification_report(y_test, y_pred_dt))
    
    print("\n" + "-"*80)
    print("SVM - Test Set Performance:")
    print("-"*80)
    y_pred_svm = final_svm_model.predict(X_test_selected)
    print(classification_report(y_test, y_pred_svm))
    
    # Ensemble - Soft Voting
    print("\n" + "-"*80)
    print("Soft Voting Ensemble - Test Set Performance:")
    print("-"*80)
    voting_soft = VotingClassifier(
        estimators=[('lr', final_lr_model), ('dt', final_dt_model), ('svm', final_svm_model)],
        voting='soft'
    )
    voting_soft.fit(X_final_train, y_final_train)
    y_pred_soft = voting_soft.predict(X_test_selected)
    print(classification_report(y_test, y_pred_soft))
    
    # Ensemble - Hard Voting
    print("\n" + "-"*80)
    print("Hard Voting Ensemble - Test Set Performance:")
    print("-"*80)
    voting_hard = VotingClassifier(
        estimators=[('lr', final_lr_model), ('dt', final_dt_model), ('svm', final_svm_model)],
        voting='hard'
    )
    voting_hard.fit(X_final_train, y_final_train)
    y_pred_hard = voting_hard.predict(X_test_selected)
    print(classification_report(y_test, y_pred_hard))
    
    print("\n" + "="*80)
    print("Pipeline Completed Successfully")
    print("="*80)


if __name__ == "__main__":
    main()