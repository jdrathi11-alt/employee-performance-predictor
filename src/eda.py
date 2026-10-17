"""
eda.py
Exploratory Data Analysis — saves plots to outputs/plots/.
Run standalone or imported by main.py.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PLOT_DIR = "outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2")


def _save(name: str):
    path = f"{PLOT_DIR}/{name}.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def run_eda(df: pd.DataFrame):
    print("\n=== EDA ===")
    print(df.describe())
    print("\nClass balance:\n", df["performance_label"].value_counts())

    # 1. Class distribution
    plt.figure(figsize=(5, 4))
    counts = df["performance_label"].value_counts()
    sns.barplot(x=["Low Performer (0)", "High Performer (1)"], y=counts.values)
    plt.title("Performance Label Distribution")
    plt.ylabel("Count")
    _save("01_class_distribution")

    # 2. Performance by department
    plt.figure(figsize=(8, 4))
    dept_perf = df.groupby("department")["performance_label"].mean().sort_values()
    sns.barplot(x=dept_perf.index, y=dept_perf.values)
    plt.title("Avg High-Performer Rate by Department")
    plt.ylabel("High Performer Rate")
    plt.xticks(rotation=30)
    _save("02_dept_performance")

    # 3. Correlation heatmap (numeric cols only)
    plt.figure(figsize=(10, 7))
    num_df = df.select_dtypes(include="number").drop(columns=["performance_label"], errors="ignore")
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    _save("03_correlation_heatmap")

    # 4. Training hours vs performance score
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="performance_label", y="training_hours", data=df)
    plt.xticks([0, 1], ["Low Performer", "High Performer"])
    plt.title("Training Hours vs Performance")
    _save("04_training_vs_performance")

    # 5. Satisfaction score distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="satisfaction_score", hue="performance_label",
                 bins=20, kde=True, palette={0: "#e74c3c", 1: "#2ecc71"})
    plt.title("Satisfaction Score by Performance")
    _save("05_satisfaction_distribution")

    # 6. Salary vs experience scatter
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df.sample(300, random_state=42),
                    x="experience_years", y="salary",
                    hue="performance_label", alpha=0.7,
                    palette={0: "#e74c3c", 1: "#2ecc71"})
    plt.title("Salary vs Experience (sample 300)")
    _save("06_salary_vs_experience")

    print("EDA complete. 6 plots saved.")


if __name__ == "__main__":
    df = pd.read_csv("data/employee_data.csv")
    run_eda(df)
