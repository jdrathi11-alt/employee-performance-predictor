"""
generate_data.py
Synthetic HR dataset generator for Employee Performance Predictor.
Produces a realistic CSV with ~1000 employees.
"""

import numpy as np
import pandas as pd

# ponytail: fixed seed for reproducibility, no config file needed
SEED = 42
N = 1000


def generate_dataset(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    departments = ["Engineering", "Sales", "HR", "Marketing", "Finance", "Operations"]
    education = ["Bachelor's", "Master's", "PhD", "High School"]
    genders = ["Male", "Female", "Non-Binary"]

    dept = rng.choice(departments, n)
    edu = rng.choice(education, n, p=[0.5, 0.3, 0.1, 0.1])
    gender = rng.choice(genders, n, p=[0.5, 0.45, 0.05])
    age = rng.integers(22, 60, n)
    experience = np.clip(rng.integers(0, age - 21, n), 0, 35)
    salary = (30000 + experience * 2500 + rng.integers(-5000, 5000, n)).astype(int)
    training_hours = rng.integers(0, 100, n)
    projects_completed = rng.integers(1, 20, n)
    avg_monthly_hours = rng.integers(140, 300, n)
    satisfaction_score = np.round(rng.uniform(1.0, 5.0, n), 1)
    manager_rating = rng.integers(1, 6, n)          # 1–5 scale
    promotions_last_5yr = rng.integers(0, 4, n)
    absences = rng.integers(0, 30, n)

    # Derive performance score (1–5) from key signals — realistic causal rules
    perf_raw = (
        0.25 * manager_rating
        + 0.20 * (satisfaction_score / 5 * 5)
        + 0.20 * (projects_completed / 20 * 5)
        + 0.15 * (training_hours / 100 * 5)
        + 0.10 * (promotions_last_5yr / 3 * 5)
        - 0.10 * (absences / 30 * 5)
        + rng.normal(0, 0.3, n)          # noise
    )
    perf_score = np.clip(np.round(perf_raw, 1), 1.0, 5.0)

    # Binary target: 1 = High Performer (score >= 3.5)
    performance_label = (perf_score >= 3.5).astype(int)

    df = pd.DataFrame({
        "employee_id": [f"EMP{str(i).zfill(4)}" for i in range(1, n + 1)],
        "age": age,
        "gender": gender,
        "education": edu,
        "department": dept,
        "experience_years": experience,
        "salary": salary,
        "training_hours": training_hours,
        "projects_completed": projects_completed,
        "avg_monthly_hours": avg_monthly_hours,
        "satisfaction_score": satisfaction_score,
        "manager_rating": manager_rating,
        "promotions_last_5yr": promotions_last_5yr,
        "absences": absences,
        "performance_score": perf_score,
        "performance_label": performance_label,   # target column
    })

    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    df.to_csv("data/employee_data.csv", index=False)
    print(f"Dataset saved → data/employee_data.csv  ({len(df)} rows)")
    print(df.head())
    print(f"\nClass balance:\n{df['performance_label'].value_counts()}")
