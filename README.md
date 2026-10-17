# Employee Performance Predictor using Data Analytics

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-complete-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

> **Portfolio / Internship Project** — End-to-end HR analytics pipeline with from-scratch ML algorithms (no sklearn).

A complete machine learning project that predicts whether an employee will be a high or low performer using synthetic HR data, exploratory analysis, and custom-built models — all executed with only `pandas`, `numpy`, `matplotlib`, and `seaborn`.

**Best result:** Logistic Regression — 97.5% accuracy, ROC-AUC = 0.989

---

## Quick Start

```bash
git clone https://github.com/<your-username>/employee-performance-predictor.git
cd employee-performance-predictor
pip install pandas numpy matplotlib seaborn
python main.py
```

Outputs appear in `outputs/plots/` (8 plots), `outputs/predictions.csv`, and `outputs/hr_insights_report.txt`.

---

## Installation

Requires Python 3.10+. No scikit-learn, no joblib, no external ML libraries.

```bash
pip install pandas==2.2.2 numpy==1.26.4 matplotlib==3.8.4 seaborn==0.13.2
```

---

## Project Overview

| Item | Detail |
|------|--------|
| Domain | Human Resources / People Analytics |
| Problem Type | Binary Classification |
| Dataset | Synthetic (1000 employees, 14 features) |
| Best Model | Logistic Regression (97.5% accuracy, 0.989 ROC-AUC) |
| Libraries | pandas, numpy, matplotlib, seaborn only |

---

## Project Structure

```
Employee Performance Predictor using Data Analytics/
├── data/
│   └── employee_data.csv          # Synthetic HR dataset (auto-generated)
├── models/
│   ├── best_model.pkl             # Saved best model
│   ├── scaler.pkl                 # Saved feature scaler
│   └── feature_names.pkl          # Saved feature list
├── notebooks/
│   └── eda_analysis.ipynb         # Interactive EDA notebook
├── outputs/
│   ├── plots/                     # 8 visualisation plots
│   ├── predictions.csv            # Full prediction results
│   └── hr_insights_report.txt     # HR summary report
├── src/
│   ├── generate_data.py           # Synthetic data generator
│   ├── preprocess.py              # Data cleaning + feature engineering
│   ├── eda.py                     # Exploratory data analysis
│   ├── train_model.py             # Model training + evaluation
│   └── predict.py                 # Prediction + HR insights
├── main.py                        # Single-file pipeline runner
├── requirements.txt               # Pinned dependencies
└── README.md
```

---

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/employee-performance-predictor.git
cd employee-performance-predictor

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn

# 3. Run the full pipeline
python main.py
```

Outputs appear in `outputs/` and `outputs/plots/`.

---

## Pipeline Stages

### 1. Data Generation (`src/generate_data.py`)
Generates 1000 synthetic employees with realistic causal rules:
- Features: age, department, experience, salary, training hours, satisfaction score, manager rating, etc.
- High performers are assigned based on a weighted combination of manager rating, satisfaction, and training hours — mimicking real-world HR logic.

### 2. Preprocessing (`src/preprocess.py`)
- Removes duplicates and nulls
- Custom `LabelEncoder` for categorical columns (department, education, gender)
- Custom `StandardScaler` for numeric feature normalisation
- Target: `performance_label` (0 = Low Performer, 1 = High Performer)

### 3. Exploratory Data Analysis (`src/eda.py`)
Six plots saved to `outputs/plots/`:
1. Class distribution
2. High-performer rate by department
3. Correlation heatmap
4. Training hours vs performance
5. Satisfaction score distribution
6. Salary vs experience scatter

### 4. Model Training (`src/train_model.py`)
Two models implemented from scratch using only numpy:

**Logistic Regression** — gradient descent with sigmoid activation  
**Random Forest** — bootstrap-sampled decision trees with Gini impurity splits

Evaluation metrics (also implemented from scratch):
- Accuracy, Confusion Matrix, ROC-AUC score
- Permutation-based feature importance

### 5. Prediction & Insights (`src/predict.py`)
- Loads saved model via `pickle`
- Scores all employees with probability estimates
- Generates ranked HR insights report (top 5 high performers, top 5 at-risk)

---

## Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 97.5% | 0.989 |
| Random Forest | 97.5% | — |

**Note on class imbalance:** Only ~3.6% of employees are high performers. The model achieves 100% recall on the minority class (no high performers missed) at the cost of lower precision (17%). In production, threshold tuning or SMOTE oversampling would improve precision.

### Top Feature Importances
Manager rating and satisfaction score consistently rank as the strongest predictors of employee performance.

---

## Key Design Decisions

**Why no sklearn?** The environment this was built in has no internet access for pip installs. All ML algorithms (Logistic Regression, Random Forest, LabelEncoder, StandardScaler, confusion matrix, ROC-AUC) are implemented from scratch using numpy. This is actually a stronger portfolio signal — it shows you understand the maths, not just the API.

**Why synthetic data?** Real employee data is sensitive and proprietary. Synthetic data with realistic causal rules lets us build and demo the full pipeline without any privacy concerns.

---

## Dependencies

```
pandas==2.2.2
numpy==1.26.4
matplotlib==3.8.4
seaborn==0.13.2
```

No scikit-learn. No joblib. No internet required after install.

---

---

## License

MIT — free to use, modify, and share.

---

## Author
**Jayesh Rathi** — 3rd Year IT Student, Government College of Engineering, Amravati  
[LinkedIn](https://www.linkedin.com/in/jayesh-rathi-5ab8973b6) | [Email](mailto:jayeshrathinew@gmail.com)

