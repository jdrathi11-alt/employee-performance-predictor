"""
main.py
Full pipeline: generate → clean → EDA → train → predict → insights.
Run this single file to reproduce the entire project end-to-end.
"""

import sys
import os

# Allow imports from src/ when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from generate_data import generate_dataset
from preprocess import load_and_clean
from eda import run_eda
from train_model import train_and_evaluate
from predict import predict_employees, generate_hr_insights


def main():
    print("=" * 55)
    print("   EMPLOYEE PERFORMANCE PREDICTOR")
    print("=" * 55)

    # --- Step 1: Generate synthetic dataset ---
    print("\n[1/5] Generating synthetic HR dataset...")
    os.makedirs("data", exist_ok=True)
    df_raw = generate_dataset()
    df_raw.to_csv("data/employee_data.csv", index=False)
    print(f"      {len(df_raw)} employees generated → data/employee_data.csv")

    # --- Step 2: Load & clean ---
    print("\n[2/5] Loading and cleaning data...")
    df = load_and_clean("data/employee_data.csv")
    print(f"      Shape after cleaning: {df.shape}")

    # --- Step 3: EDA ---
    print("\n[3/5] Running Exploratory Data Analysis...")
    run_eda(df)

    # --- Step 4: Train model ---
    print("\n[4/5] Training and evaluating model...")
    _, _, _, best_acc = train_and_evaluate("data/employee_data.csv")
    print(f"      Best model accuracy: {best_acc:.4f}")

    # --- Step 5: Predict + HR insights ---
    print("\n[5/5] Generating predictions and HR insights...")
    result = predict_employees(df)
    generate_hr_insights(result)

    print("\n" + "=" * 55)
    print("   PIPELINE COMPLETE")
    print("   Outputs → outputs/   Plots → outputs/plots/")
    print("=" * 55)


if __name__ == "__main__":
    main()
