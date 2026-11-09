"""
Lab 5: Decision Tree Classifier
Module: decision_tree_model.py

Provides reusable functions for dataset preparation, pipeline construction,
decision tree training, evaluation metrics, and diagnostic plot generation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

NUMERICAL_COLS: List[str] = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
CATEGORICAL_COLS: List[str] = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation"]
TARGET_COL: str = "status"
DROP_COLS: List[str] = ["sl_no", "salary"]


def load_dataset(data_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the placement dataset using dynamic relative resolution."""
    if data_path is None:
        data_path = Path("datasets/Placement_Data_Full_Class.csv") if Path("datasets/Placement_Data_Full_Class.csv").exists() else Path("../datasets/Placement_Data_Full_Class.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    return pd.read_csv(data_path)


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate target and feature matrix, excluding identifiers and leakage attributes."""
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target '{TARGET_COL}' missing from dataset.")
    y = df[TARGET_COL]
    X = df.drop(columns=[c for c in DROP_COLS + [TARGET_COL] if c in df.columns])
    return X, y


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Build a leak-free ColumnTransformer for numerical and categorical features."""
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])
    return ColumnTransformer([("num", num_pipe, NUMERICAL_COLS), ("cat", cat_pipe, CATEGORICAL_COLS)])


def build_model_pipeline(max_depth: Optional[int] = None, criterion: str = "gini", random_state: int = 42) -> Pipeline:
    """Assemble end-to-end Pipeline with preprocessor and DecisionTreeClassifier."""
    clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=random_state)
    return Pipeline([("preprocessor", build_preprocessing_pipeline()), ("classifier", clf)])


def train_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit the model pipeline on training data."""
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Compute standard classification evaluation metrics."""
    y_pred = pipeline.predict(X_test)
    classes = pipeline.named_steps["classifier"].classes_
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="Placed"),
        "recall": recall_score(y_test, y_pred, pos_label="Placed"),
        "f1_score": f1_score(y_test, y_pred, pos_label="Placed"),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=classes),
        "classification_report": classification_report(y_test, y_pred, target_names=classes),
        "classes": classes,
        "y_pred": y_pred
    }


def save_diagnostic_plots(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, output_dir: Optional[Path] = None) -> None:
    """Generate and export confusion matrix, decision tree, and feature importance plots."""
    out = output_dir or (Path("Lab5/outputs") if Path("Lab5").exists() else Path("outputs"))
    out.mkdir(parents=True, exist_ok=True)
    res = evaluate_model(pipeline, X_test, y_test)

    # 1. Confusion Matrix
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues", xticklabels=res["classes"], yticklabels=res["classes"])
    plt.title("Confusion Matrix — Student Placement Prediction")
    plt.xlabel("Predicted Status"); plt.ylabel("True Status"); plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png"); plt.close()

    # 2. Decision Tree Visualization
    clf = pipeline.named_steps["classifier"]
    feat_names = [n.replace("num__", "").replace("cat__", "") for n in pipeline.named_steps["preprocessor"].get_feature_names_out()]
    plt.figure(figsize=(18, 9), dpi=300)
    plot_tree(clf, feature_names=feat_names, class_names=list(res["classes"]), filled=True, rounded=True, fontsize=8, max_depth=3)
    plt.title("Decision Tree Visualization (Truncated Depth = 3)")
    plt.tight_layout(); plt.savefig(out / "decision_tree.png"); plt.close()

    # 3. Feature Importance
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    valid_idx = [i for i in sorted_idx if importances[i] > 0]
    plt.figure(figsize=(9, 5), dpi=300)
    sns.barplot(x=[importances[i] for i in valid_idx], y=[feat_names[i] for i in valid_idx], hue=[feat_names[i] for i in valid_idx], palette="viridis", legend=False)
    plt.title("Feature Importance — Gini Score"); plt.xlabel("Importance"); plt.ylabel("Feature")
    plt.tight_layout(); plt.savefig(out / "feature_importance.png"); plt.close()


def main() -> None:
    """Execute pipeline training, evaluation, and plot generation."""
    print("=== LAB 5: DECISION TREE CLASSIFIER ===")
    df = load_dataset()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training instances: {X_train.shape[0]} | Testing instances: {X_test.shape[0]}")

    pipeline = build_model_pipeline(random_state=42)
    pipeline = train_model(pipeline, X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test)
    print(f"\n--- Model Evaluation (Baseline Decision Tree) ---")
    print(f"Accuracy : {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f} | F1-Score : {metrics['f1_score']:.4f}")
    print("\nClassification Report:\n", metrics["classification_report"])
    print("Confusion Matrix:\n", metrics["confusion_matrix"])

    save_diagnostic_plots(pipeline, X_test, y_test)
    print("\nSaved plots: confusion_matrix.png, decision_tree.png, feature_importance.png")


if __name__ == "__main__":
    main()
