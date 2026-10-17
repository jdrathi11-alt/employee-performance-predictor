"""
train_model.py
Train, evaluate, and save ML models for Employee Performance Predictor.
Uses ONLY pandas and numpy - no sklearn dependencies.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import load_and_clean, encode_and_scale

PLOT_DIR = "outputs/plots"
MODEL_DIR = "models"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


class LogisticRegression:
    """Logistic Regression implementation using gradient descent."""

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            linear = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear)

            # Gradient descent
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return np.column_stack([1 - self.sigmoid(linear), self.sigmoid(linear)])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class RandomForest:
    """Random Forest classifier using decision trees."""

    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2, random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_split = min_samples_split
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        for _ in range(self.n_trees):
            # Bootstrap sampling
            idx = np.random.choice(len(X), len(X), replace=True)
            X_boot, y_boot = X[idx], y[idx]

            # Build decision tree
            tree = self._build_tree(X_boot, y_boot, depth=0)
            self.trees.append(tree)

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(X) < self.min_split or len(np.unique(y)) == 1:
            # Return leaf with majority class
            return {"leaf": True, "class": np.bincount(y).argmax()}

        # Find best split
        best_gain, best_feature, best_threshold = -1, None, None
        n_features = X.shape[1]

        for feat in range(n_features):
            thresholds = np.percentile(X[:, feat], [25, 50, 75])
            for thresh in thresholds:
                left = X[:, feat] <= thresh
                right = ~left
                if np.sum(left) == 0 or np.sum(right) == 0:
                    continue

                gain = self._information_gain(y, y[left], y[right])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = thresh

        if best_feature is None:
            return {"leaf": True, "class": np.bincount(y).argmax()}

        left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx

        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(X[left_idx], y[left_idx], depth + 1),
            "right": self._build_tree(X[right_idx], y[right_idx], depth + 1)
        }

    def _information_gain(self, parent, left, right):
        def gini(arr):
            if len(arr) == 0:
                return 0
            counts = np.bincount(arr)
            probs = counts / len(arr)
            return 1 - np.sum(probs ** 2)

        n = len(parent)
        if n == 0:
            return 0

        gain = gini(parent)
        gain -= (len(left) / n) * gini(left) + (len(right) / n) * gini(right)
        return gain

    def _predict_single(self, x, tree):
        if tree["leaf"]:
            return tree["class"]
        if x[tree["feature"]] <= tree["threshold"]:
            return self._predict_single(x, tree["left"])
        return self._predict_single(x, tree["right"])

    def predict_proba(self, X):
        votes = np.zeros((len(X), 2))
        for tree in self.trees:
            preds = np.array([self._predict_single(x, tree) for x in X])
            for i, p in enumerate(preds):
                votes[i, p] += 1
        return votes / self.n_trees

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])


def roc_auc_score(y_true, y_proba):
    """Calculate ROC-AUC using rank-based method."""
    sorted_indices = np.argsort(y_proba)[::-1]
    y_true_sorted = y_true[sorted_indices]

    pos = np.sum(y_true_sorted == 1)
    neg = np.sum(y_true_sorted == 0)

    if pos == 0 or neg == 0:
        return 0.5

    # Calculate area under ROC curve
    tp_cum = np.cumsum(y_true_sorted == 1)
    fp_cum = np.cumsum(y_true_sorted == 0)

    tpr = tp_cum / pos
    fpr = fp_cum / neg

    # Trapezoidal rule
    auc = np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2)
    return auc


def train_and_evaluate(data_path: str = "data/employee_data.csv"):
    print("Loading and preprocessing data...")
    df = load_and_clean(data_path)
    X, y, feature_names, scaler = encode_and_scale(df)

    # Train/test split (manual stratified)
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    test_size = int(0.2 * len(X))
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # --- Models to compare ---
    models = {
        "Logistic Regression": LogisticRegression(learning_rate=0.1, n_iterations=500),
        "Random Forest": RandomForest(n_trees=50, max_depth=8, random_state=42),
    }

    best_model, best_acc, best_name = None, 0, ""

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc, best_model, best_name = acc, model, name

    print(f"\n>>> Best model: {best_name} ({best_acc:.4f})")

    # --- Full evaluation ---
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Classification report
    print("\nClassification Report:")
    print(f"  Low Performer  - Precision: {confusion_matrix(y_test, y_pred)[0,0]/(confusion_matrix(y_test, y_pred)[0,0]+confusion_matrix(y_test, y_pred)[0,1]):.2f}, Recall: {confusion_matrix(y_test, y_pred)[0,0]/(confusion_matrix(y_test, y_pred)[0,0]+confusion_matrix(y_test, y_pred)[1,0]):.2f}")
    print(f"  High Performer - Precision: {confusion_matrix(y_test, y_pred)[1,1]/(confusion_matrix(y_test, y_pred)[1,0]+confusion_matrix(y_test, y_pred)[1,1]):.2f}, Recall: {confusion_matrix(y_test, y_pred)[1,1]/(confusion_matrix(y_test, y_pred)[0,1]+confusion_matrix(y_test, y_pred)[1,1]):.2f}")
    print(f"  ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "High"], yticklabels=["Low", "High"])
    plt.title(f"Confusion Matrix — {best_name}")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.savefig(f"{PLOT_DIR}/07_confusion_matrix.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved → {PLOT_DIR}/07_confusion_matrix.png")

    # Feature importance (using permutation importance approximation)
    imp = np.zeros(len(feature_names))
    baseline_acc = accuracy_score(y_test, best_model.predict(X_test))

    for i in range(len(feature_names)):
        X_permuted = X_test.copy()
        np.random.shuffle(X_permuted[:, i])
        permuted_acc = accuracy_score(y_test, best_model.predict(X_permuted))
        imp[i] = baseline_acc - permuted_acc

    imp_series = pd.Series(imp, index=feature_names).sort_values()
    plt.figure(figsize=(7, 5))
    imp_series.plot(kind="barh", color="steelblue")
    plt.title("Feature Importance")
    plt.xlabel("Importance (accuracy drop)")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/08_feature_importance.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved → {PLOT_DIR}/08_feature_importance.png")

    # Save model + scaler
    import pickle
    with open(f"{MODEL_DIR}/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{MODEL_DIR}/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    print(f"\nModel saved → {MODEL_DIR}/best_model.pkl")

    return best_model, scaler, feature_names, best_acc


if __name__ == "__main__":
    train_and_evaluate()