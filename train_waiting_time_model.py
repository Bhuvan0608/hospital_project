"""
Waiting-time prediction model for the Smart Hospital Management and
Appointment Prediction System.

Predicts how many minutes a patient will wait past their booked slot.
Reads the CSV files created by generate_data.py (folder: data/), trains
regression models, compares them with simple baselines, and saves the best
one plus charts for your report and slides.

Two kinds of estimate are compared:
  * BOOKING-TIME  uses only what is known in advance (doctor's workload,
                  slot position, time of day ...).
  * LIVE          also uses how late the doctor is running right now (the
                  previous patient's wait). This is available on the day,
                  for example on the waiting-room display.

Usage
  python train_waiting_time_model.py        (same libraries as the no-show model)

Outputs (folder: model/)
  waiting_time_model.joblib       best LIVE model (used later by the Flask app)
  waiting_model_comparison.csv    metrics for all models
  waiting_actual_vs_predicted.png
  waiting_feature_importance.png
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = "data"
OUT_DIR = "model"
SEED = 42

BOOKING_FEATURES = ["position_in_day", "day_load", "utilization", "hour",
                    "day_of_week", "consultation_min"]
LIVE_FEATURES = BOOKING_FEATURES + ["prev_wait", "gap_to_prev"]
CATEGORICAL = ["department"]


# ----------------------------------------------------------------------
# 1. Build the dataset
# ----------------------------------------------------------------------
def build_dataset():
    appts = pd.read_csv(os.path.join(DATA_DIR, "appointments.csv"),
                        parse_dates=["appointment_date"])
    doctors = pd.read_csv(os.path.join(DATA_DIR, "doctors.csv"))
    depts = pd.read_csv(os.path.join(DATA_DIR, "departments.csv"))
    sched = pd.read_csv(os.path.join(DATA_DIR, "doctor_schedules.csv"))

    df = (appts
          .merge(doctors[["doctor_id", "department_id", "consultation_min"]], on="doctor_id")
          .merge(depts[["department_id", "name"]].rename(columns={"name": "department"}),
                 on="department_id"))

    df["slot"] = pd.to_timedelta(df["slot_time"])
    df["hour"] = df["slot"].dt.components.hours
    df["day_of_week"] = df["appointment_date"].dt.weekday

    # Doctor's workload that day. Counted over every booking (including ones
    # that later cancel), because that is what the hospital knows on the day.
    key = ["doctor_id", "appointment_date"]
    df = df.sort_values(key + ["slot"]).reset_index(drop=True)
    df["day_load"] = df.groupby(key)["appointment_id"].transform("count")
    df["position_in_day"] = df.groupby(key).cumcount() + 1

    # How full the doctor's schedule is that day
    capacity = (sched.groupby(["doctor_id", "day_of_week"])["max_patients"]
                .sum().rename("capacity").reset_index())
    df = df.merge(capacity, on=["doctor_id", "day_of_week"], how="left")
    df["utilization"] = df["day_load"] / df["capacity"]

    # Only completed visits have a real waiting time to learn from
    df = df[(df["status"] == "completed") & df["waiting_minutes"].notna()].copy()
    df = df.sort_values(key + ["slot"])

    # LIVE features: the previous seen patient's wait that day, and how many
    # minutes earlier that patient's slot was. 0 for the day's first patient.
    g = df.groupby(key)
    df["prev_wait"] = g["waiting_minutes"].shift(1).fillna(0)
    prev_slot = g["slot"].shift(1)
    df["gap_to_prev"] = ((df["slot"] - prev_slot).dt.total_seconds() / 60).fillna(0)

    return df.sort_values(["appointment_date", "slot"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# 2. Models
# ----------------------------------------------------------------------
def make_pipeline(model, numeric, scale):
    num = StandardScaler() if scale else "passthrough"
    prep = ColumnTransformer([
        ("num", num, numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def make_live_models():
    return {
        "Linear Regression": make_pipeline(LinearRegression(), LIVE_FEATURES, True),
        "Random Forest": make_pipeline(
            RandomForestRegressor(n_estimators=300, min_samples_leaf=10,
                                  random_state=SEED, n_jobs=-1), LIVE_FEATURES, False),
        "Gradient Boosting": make_pipeline(
            HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05,
                                          max_iter=200, random_state=SEED),
            LIVE_FEATURES, False),
    }


# ----------------------------------------------------------------------
# 3. Train, compare, save
# ----------------------------------------------------------------------
def score(name, y_true, pred):
    pred = np.clip(pred, 0, None)  # a wait cannot be negative
    return {"model": name,
            "MAE_min": mean_absolute_error(y_true, pred),
            "RMSE_min": np.sqrt(mean_squared_error(y_true, pred)),
            "R2": r2_score(y_true, pred)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = build_dataset()
    print(f"Completed visits used: {len(df)}")
    print(f"Waiting time: average {df['waiting_minutes'].mean():.1f} min, "
          f"median {df['waiting_minutes'].median():.0f} min, "
          f"{(df['waiting_minutes'] == 0).mean() * 100:.0f}% had no wait\n")

    # Time-based split: earlier 75% for training, latest 25% for testing
    cut = int(len(df) * 0.75)
    train, test = df.iloc[:cut], df.iloc[cut:]
    y_train, y_test = train["waiting_minutes"], test["waiting_minutes"]
    print(f"Train: {len(train)} rows   Test: {len(test)} rows\n")

    results = []

    # Baselines: always guess the same number
    for label, strategy in [("Baseline: always the average", "mean"),
                            ("Baseline: always the median", "median")]:
        dummy = DummyRegressor(strategy=strategy).fit(train[LIVE_FEATURES], y_train)
        results.append(score(label, y_test, dummy.predict(test[LIVE_FEATURES])))

    # Booking-time model (no live information)
    booking = make_pipeline(
        HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05,
                                      max_iter=200, random_state=SEED),
        BOOKING_FEATURES, False)
    booking.fit(train[BOOKING_FEATURES + CATEGORICAL], y_train)
    results.append(score("Booking-time only (Gradient Boosting)", y_test,
                         booking.predict(test[BOOKING_FEATURES + CATEGORICAL])))

    # Live models
    X_train = train[LIVE_FEATURES + CATEGORICAL]
    X_test = test[LIVE_FEATURES + CATEGORICAL]
    live = {}
    for name, pipe in make_live_models().items():
        pipe.fit(X_train, y_train)
        live[name] = pipe
        results.append(score(f"Live: {name}", y_test, pipe.predict(X_test)))

    res = pd.DataFrame(results).round(3)
    res.to_csv(os.path.join(OUT_DIR, "waiting_model_comparison.csv"), index=False)
    print(res.to_string(index=False))

    live_rows = res[res["model"].str.startswith("Live:")]
    best_row = live_rows.sort_values("MAE_min").iloc[0]
    best_name = best_row["model"].replace("Live: ", "")
    best = live[best_name]
    median_mae = res.loc[res["model"] == "Baseline: always the median", "MAE_min"].iloc[0]
    print(f"\nBest live model (lowest MAE): {best_name}")
    print(f"Its average error is {best_row['MAE_min']:.1f} min, versus "
          f"{median_mae:.1f} min for always guessing the median wait.")

    # Actual vs predicted
    pred = np.clip(best.predict(X_test), 0, None)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_test, pred, s=8, alpha=0.35, color="#2a6fb0")
    top = max(y_test.max(), pred.max())
    ax.plot([0, top], [0, top], "k--", label="Perfect prediction")
    ax.set_xlabel("Actual waiting time (min)")
    ax.set_ylabel("Predicted waiting time (min)")
    ax.set_title(f"Actual vs predicted: {best_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "waiting_actual_vs_predicted.png"), dpi=150)
    plt.close(fig)

    # Feature importance
    imp = permutation_importance(best, X_test, y_test,
                                 scoring="neg_mean_absolute_error",
                                 n_repeats=5, random_state=SEED)
    imp_df = (pd.Series(imp.importances_mean, index=LIVE_FEATURES + CATEGORICAL)
              .sort_values())
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    imp_df.plot.barh(ax=ax, color="#2a6fb0")
    ax.set_xlabel("Increase in average error (min) when the feature is shuffled")
    ax.set_title("Feature importance: waiting time")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "waiting_feature_importance.png"), dpi=150)
    plt.close(fig)
    print("\nMost important features:")
    print(imp_df.sort_values(ascending=False).round(3).to_string())

    joblib.dump({"model": best, "model_name": best_name,
                 "features": LIVE_FEATURES + CATEGORICAL},
                os.path.join(OUT_DIR, "waiting_time_model.joblib"))
    print(f"\nSaved model and charts in the '{OUT_DIR}' folder.")


if __name__ == "__main__":
    main()
