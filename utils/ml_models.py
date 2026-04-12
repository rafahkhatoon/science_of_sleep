"""
utils/ml_models.py
Random Forest classification for sleep quality prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


FEATURE_COLS = [
    "Stress Level",
    "Sleep Duration",
    "Heart Rate",
    "Daily Steps",
    "Physical Activity Level",
    "BMI_enc",
    "Age",
]

FEATURE_DISPLAY = [
    "Stress Level",
    "Sleep Duration",
    "Heart Rate",
    "Daily Steps",
    "Physical Activity",
    "BMI Category",
    "Age",
]


def run_random_forest(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42):
    """
    Train a Random Forest to predict High vs Low sleep quality.

    Returns
    -------
    model, X_test, y_test, y_pred, feature_names, accuracy, report_df
    """
    df_model = df.copy()

    # Target
    if "Sleep Quality Label" not in df_model.columns:
        df_model["Sleep Quality Label"] = df_model["Quality of Sleep"].apply(
            lambda x: "High Sleep Quality" if x >= 7 else "Low Sleep Quality"
        )

    # Ensure BMI_enc exists
    if "BMI_enc" not in df_model.columns:
        le = LabelEncoder()
        df_model["BMI_enc"] = le.fit_transform(df_model["BMI Category"])

    available = [c for c in FEATURE_COLS if c in df_model.columns]
    display   = [FEATURE_DISPLAY[FEATURE_COLS.index(c)] for c in available]

    X = df_model[available].fillna(df_model[available].median())
    y = df_model["Sleep Quality Label"]

    # Guard: need at least 2 classes
    if y.nunique() < 2:
        # Fallback — flip one label
        y.iloc[0] = "Low Sleep Quality" if y.iloc[0] == "High Sleep Quality" else "High Sleep Quality"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=4,
        random_state=random_state,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).T.round(3)

    return model, X_test, y_test, y_pred, display, acc, report_df


def run_classification_report(y_test, y_pred) -> pd.DataFrame:
    report = classification_report(y_test, y_pred, output_dict=True)
    return pd.DataFrame(report).T.round(3)
