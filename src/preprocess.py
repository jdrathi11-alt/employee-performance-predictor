"""
preprocess.py
Data cleaning and feature engineering for Employee Performance Predictor.
Uses only pandas and numpy - no sklearn dependencies.
"""

import pandas as pd
import numpy as np


def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, drop duplicates, fill nulls, return clean DataFrame."""
    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)   # synthetic data has no nulls; real data might
    return df


class LabelEncoder:
    """Simple LabelEncoder implementation."""
    def __init__(self):
        self.mapping_ = {}
        self.classes_ = []

    def fit_transform(self, series):
        self.classes_ = sorted(series.unique().tolist())
        self.mapping_ = {val: idx for idx, val in enumerate(self.classes_)}
        return series.map(self.mapping_)

    def transform(self, series):
        return series.map(self.mapping_)


class StandardScaler:
    """Simple StandardScaler implementation."""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit_transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        return (X - self.mean_) / self.std_

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return (X - self.mean_) / self.std_


def encode_and_scale(df: pd.DataFrame):
    """
    Encode categoricals, scale numerics.
    Returns (X, y, feature_names, scaler).
    """
    df = df.copy()

    # Drop non-predictive columns
    df.drop(columns=["employee_id", "performance_score"], inplace=True)

    # Encode all object columns with custom LabelEncoder
    encoders = {}
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=["performance_label"])
    y = df["performance_label"].values

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns.tolist(), scaler


if __name__ == "__main__":
    df = load_and_clean("data/employee_data.csv")
    print("Shape after cleaning:", df.shape)
    print("Null counts:\n", df.isnull().sum())
    X, y, features, _ = encode_and_scale(df)
    print("Feature count:", len(features))
    print("Features:", features)
