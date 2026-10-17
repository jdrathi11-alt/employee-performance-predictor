"""
predict.py
Load saved model and predict performance for new employees.
Also generates an HR insights report in outputs/.
Uses only pandas, numpy, and pickle - no sklearn/joblib dependencies.
"""

import os
import pickle
import numpy as np
import pandas as pd

from preprocess import load_and_clean, encode_and_scale, LabelEncoder

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model():
    with open(f"{MODEL_DIR}/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{MODEL_DIR}/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names


def predict_employees(df_new: pd.DataFrame) -> pd.DataFrame:
    """
    Predict performance for a DataFrame of employees.
    Uses the same encode_and_scale pipeline as training.
    Returns df_new with added prediction columns.
    """
    model, scaler, feature_names = load_model()

    df = df_new.copy()

    # Encode categoricals using custom LabelEncoder
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for drop_col in ["employee_id", "performance_score", "performance_label"]:
        if drop_col in cat_cols:
            cat_cols.remove(drop_col)

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Select only the features used during training
    available = [f for f in feature_names if f in df.columns]
    X = df[available].values

    # Scale using saved scaler
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]

    result = df_new.copy()
    result["predicted_label"] = preds
    result["high_performer_probability"] = np.round(probs, 3)
    result["prediction"] = result["predicted_label"].map(
        {1: "High Performer", 0: "Low Performer"}
    )
    return result


def generate_hr_insights(df_predicted: pd.DataFrame):
    """Print and save a simple HR insights summary."""
    total = len(df_predicted)
    high = (df_predicted["predicted_label"] == 1).sum()
    low = total - high

    lines = [
        "=" * 50,
        "       HR INSIGHTS REPORT",
        "=" * 50,
        f"Total Employees Analysed : {total}",
        f"Predicted High Performers: {high} ({high/total*100:.1f}%)",
        f"Predicted Low Performers : {low}  ({low/total*100:.1f}%)",
        "",
        "--- Top 5 High Performers (by probability) ---",
    ]

    top5 = df_predicted.nlargest(5, "high_performer_probability")[
        ["employee_id", "department", "high_performer_probability", "prediction"]
    ].to_string(index=False)
    lines.append(top5)

    lines += [
        "",
        "--- Top 5 At-Risk Employees (lowest probability) ---",
    ]
    bot5 = df_predicted.nsmallest(5, "high_performer_probability")[
        ["employee_id", "department", "high_performer_probability", "prediction"]
    ].to_string(index=False)
    lines.append(bot5)

    lines.append("=" * 50)
    report = "\n".join(lines)
    print(report)

    path = f"{OUTPUT_DIR}/hr_insights_report.txt"
    with open(path, "w") as f:
        f.write(report)
    print(f"\nReport saved → {path}")

    csv_path = f"{OUTPUT_DIR}/predictions.csv"
    df_predicted.to_csv(csv_path, index=False)
    print(f"Predictions CSV saved → {csv_path}")


if __name__ == "__main__":
    df = load_and_clean("data/employee_data.csv")
    result = predict_employees(df)
    generate_hr_insights(result)
