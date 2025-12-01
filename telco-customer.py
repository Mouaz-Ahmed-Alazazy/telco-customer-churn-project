import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif

class PreProcessing:
    def __init__(self, df):
        self.df = df

    def handle_missing_values(self, threshold=0.5):
        # Drop columns with more than threshold% missing values
        self.df = self.df.loc[:, self.df.isnull().mean() < threshold]
        # Drop rows with any remaining missing values
        self.df = self.df.dropna()
        return self.df

    def encode_categorical(self):
        # Convert categorical variables to dummy/indicator variables
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)
        return self.df

    def scale_numerical(self):
        # Scale numerical features using Min-Max scaling
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            self.df[col] = (self.df[col] - self.df[col].min()) / (self.df[col].max() - self.df[col].min())
        return self.df

    def split_data(self, target, test_size=0.2, random_state=42):
        X = self.df.drop(columns=[target])
        y = self.df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def feature_selection(self, X, y, k=10):
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        return X_new, selected_features