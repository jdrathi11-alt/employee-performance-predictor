"""
Virtual simulation: tweak one feature, see prediction change.
"""
import sys; sys.path.insert(0, '.')
from predict import load_model, predict_employees
from preprocess import load_and_clean
import numpy as np

df = load_and_clean("data/employee_data.csv")
model, scaler, features = load_model()

# Pick a low-probability employee and simulate better satisfaction
sim = df.iloc[[0]].copy()
sim["manager_rating"] = 5.0  # max out
sim["satisfaction_score"] = 5.0
sim_result = predict_employees(sim)
print(f"Before: {sim_result['prediction'].values[0]} (prob: {sim_result['high_performer_probability'].values[0]})")
