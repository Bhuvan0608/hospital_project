"""
No-show prediction model for the Smart Hospital Management and
Appointment Prediction System.

Reads the CSV files created by generate_data.py (folder: data/), trains
three models, compares them, and saves the best one plus charts for your
report and slides.

Usage
  pip install pandas scikit-learn matplotlib joblib
  python train_no_show_model.py

Outputs (folder: model/)
  no_show_model.joblib      best trained model (used later by the Flask app)
  model_comparison.csv      metrics for all three models
  roc_curves.png            ROC curves
  confusion_matrix.png      confusion matrix of the best model
  feature_importance.png    which features matter most
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")  # draw charts to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = "data"
OUT_DIR = "model"
SEED = 42

NUMERIC = ["lead_days", "age", "day_of_week", "hour", "sms_reminder_sent",
           "prior_appointments", "prior_no_shows", "prior_no_show_rate"]
CATEGORICAL = ["department"]


# ----------------------------------------------------------------------
# 1. Build the dataset
# ----------------------------------------------------------------------
def build_dataset():
    appts = pd.read_csv(os.path.join(DATA_DIR, "appointments.csv"),
                        parse_dates=["booked_at", "appointment_date"])
    patients = pd.read_csv(os.path.join(DATA_DIR, "patients.csv"), parse_dates=["date_of_birth"])
    doctors = pd.read_csv(os.path.join(DATA_DIR, "doctors.csv"))
    depts = pd.read_csv(os.path.join(DATA_DIR, "departments.csv"))

    df = (appts
          .merge(patients[["patient_id", "date_of_birth"]], on="patient_id")
          .merge(doctors[["doctor_id", "department_id"]], on="doctor_id")
          .merge(depts[["department_id", "name"]].rename(columns={"name": "department"}),
                 on="department_id"))

    # Only appointments with a known outcome can teach the model.
    # Cancelled ones are removed: the patient told the hospital in advance,
    # so the slot was not a surprise no-show.
    df = df[df["status"].isin(["completed", "no_show"])].copy()
    df["no_show"] = (df["status"] == "no_show").astype(int)

    # Features known at booking time
    slot = pd.to_timedelta(df["slot_time"])
    df["appointment_dt"] = df["appointment_date"] + slot
    df["lead_days"] = (df["appointment_dt"] - df["booked_at"]).dt.total_seconds() / 86400
    df["age"] = (df["appointment_date"] - df["date_of_birth"]).dt.days // 365
    df["day_of_week"] = df["appointment_date"].dt.weekday
    df["hour"] = slot.dt.components.hours

    # Patient history: only appointments BEFORE this one (no peeking ahead)
    df = df.sort_values(["patient_id", "appointment_dt"]).reset_index(drop=True)
    g = df.groupby("patient_id")["no_show"]
    df["prior_appointments"] = g.cumcount()
    df["prior_no_shows"] = g.cumsum() - df["no_show"]
    df["prior_no_show_rate"] = (df["prior_no_shows"] / df["prior_appointments"]).fillna(0)

    return df.sort_values("appointment_dt").reset_index(drop=True)


# ----------------------------------------------------------------------
# 2. Models
# ----------------------------------------------------------------------
def make_models():
    scaled = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    unscaled = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return {
        "Logistic Regression": Pipeline([
            ("prep", scaled),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "Random Forest": Pipeline([
            ("prep", unscaled),
            ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                             class_weight="balanced", random_state=SEED,
                                             n_jobs=-1))]),
        "Gradient Boosting": Pipeline([
            ("prep", unscaled),
            ("model", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05,
                                                     max_iter=200, class_weight="balanced",
                                                     random_state=SEED))]),
    }


# ----------------------------------------------------------------------
# 3. Train, compare, save
# ----------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = build_dataset()
    print(f"Rows used: {len(df)}   no-show rate: {df['no_show'].mean() * 100:.1f}%")

    # Time-based split: train on the earlier 75% of appointments and test on
    # the most recent 25%, like predicting the future from the past.
    cut = int(len(df) * 0.75)
    train, test = df.iloc[:cut], df.iloc[cut:]
    X_train, y_train = train[NUMERIC + CATEGORICAL], train["no_show"]
    X_test, y_test = test[NUMERIC + CATEGORICAL], test["no_show"]
    print(f"Train: {len(train)} rows   Test: {len(test)} rows\n")

    results, fitted = [], {}
    for name, pipe in make_models().items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results.append({
            "model": name,
            "roc_auc": roc_auc_score(y_test, proba),
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
        })
        fitted[name] = pipe

    res = pd.DataFrame(results).round(3)
    res.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)
    print(res.to_string(index=False))

    best_name = res.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    best = fitted[best_name]
    print(f"\nBest model (highest ROC-AUC): {best_name}")

    # ROC curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random guess")
    ax.set_title("ROC curves: no-show prediction")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "roc_curves.png"), dpi=150)
    plt.close(fig)

    # Confusion matrix of the best model
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_estimator(
        best, X_test, y_test, display_labels=["Attended", "No-show"],
        cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix: {best_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # Feature importance (permutation: how much does ROC-AUC drop if a
    # feature is shuffled?)
    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc",
                                 n_repeats=5, random_state=SEED)
    imp_df = (pd.Series(imp.importances_mean, index=NUMERIC + CATEGORICAL)
              .sort_values())
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    imp_df.plot.barh(ax=ax, color="#2a6fb0")
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.set_title("Feature importance")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)
    print("\nMost important features:")
    print(imp_df.sort_values(ascending=False).round(4).to_string())

    joblib.dump({"model": best, "model_name": best_name,
                 "features": NUMERIC + CATEGORICAL},
                os.path.join(OUT_DIR, "no_show_model.joblib"))
    print(f"\nSaved model and charts in the '{OUT_DIR}' folder.")


if __name__ == "__main__":
    main()
