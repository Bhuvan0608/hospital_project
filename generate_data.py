"""
Synthetic data generator for the Smart Hospital Management and
Appointment Prediction System.

What it does
  1. Generates realistic fake data for every table in schema.sql
     (except `predictions`, which your ML model will fill later).
  2. Saves each table as a CSV file in a folder called `data/`.
  3. Loads the same data into your MySQL database `smart_hospital`.

Usage
  pip install mysql-connector-python
  python generate_data.py                 # generate CSVs + load into MySQL
  python generate_data.py --no-db         # generate CSVs only

WARNING: loading into MySQL first EMPTIES all tables, so the script can be
re-run safely, but any data you added by hand will be deleted.
"""

import argparse
import csv
import getpass
import hashlib
import math
import os
import random
from datetime import date, datetime, time, timedelta

# ----------------------------------------------------------------------
# Settings you can change
# ----------------------------------------------------------------------
SEED = 42               # same seed = same data every run
N_PATIENTS = 1500
DOCTORS_PER_DEPT = 2    # 8 departments x 2 = 16 doctors
HISTORY_DAYS = 90       # past days of completed/no-show/cancelled history
FUTURE_DAYS = 7         # upcoming days of 'scheduled' appointments

DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_USER = "root"
DB_NAME = "smart_hospital"

OUT_DIR = "data"

# ----------------------------------------------------------------------
# Reference data
# ----------------------------------------------------------------------
DEPARTMENTS = {
    "General Medicine": ["Fever and cold", "General check-up", "Fatigue", "Blood pressure check"],
    "Cardiology": ["Chest discomfort", "Palpitations", "Heart check-up", "Blood pressure follow-up"],
    "Orthopedics": ["Knee pain", "Back pain", "Fracture follow-up", "Joint stiffness"],
    "Pediatrics": ["Child fever", "Vaccination", "Growth check-up", "Cough and cold"],
    "Dermatology": ["Skin rash", "Acne treatment", "Hair fall", "Allergy"],
    "ENT": ["Ear pain", "Sore throat", "Sinus problem", "Hearing check"],
    "Gynecology": ["Routine check-up", "Prenatal visit", "Menstrual problem", "Follow-up"],
    "Neurology": ["Headache", "Dizziness", "Numbness", "Follow-up visit"],
}

# Small per-department tendency to have no-shows (used by the generator only)
DEPT_NOSHOW_EFFECT = {
    "General Medicine": 0.10, "Cardiology": -0.25, "Orthopedics": 0.05,
    "Pediatrics": -0.15, "Dermatology": 0.30, "ENT": 0.10,
    "Gynecology": -0.10, "Neurology": -0.05,
}

NOTES = [
    "Patient examined, vitals normal.", "Advised rest and fluids.",
    "Symptoms improving, continue treatment.", "Tests recommended.",
    "Follow-up advised if symptoms persist.", "Condition stable.",
]
PRESCRIPTIONS = [
    "Paracetamol 500mg twice daily for 3 days", "Vitamin D3 weekly for 8 weeks",
    "Cetirizine 10mg at night for 5 days", "Antacid after meals for 7 days",
    "Pain relief gel, apply twice daily", "No medication, lifestyle advice only",
]

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Arjun", "Sai", "Karthik", "Rahul", "Rohan", "Suresh",
    "Ramesh", "Venkat", "Praveen", "Anil", "Kiran", "Naveen", "Ravi", "Manoj", "Harish",
    "Ananya", "Diya", "Priya", "Lakshmi", "Sravani", "Divya", "Meena", "Kavya", "Pooja",
    "Swathi", "Anjali", "Deepa", "Harini", "Madhavi", "Sunita", "Rekha", "Bhavana",
]
LAST_NAMES = [
    "Reddy", "Naidu", "Sharma", "Rao", "Kumar", "Varma", "Chowdary", "Prasad",
    "Raju", "Babu", "Murthy", "Shetty", "Gupta", "Singh", "Patel", "Das",
]
CITIES = ["Vijayawada", "Guntur", "Visakhapatnam", "Tirupati", "Nellore", "Kakinada", "Rajahmundry"]

# Doctors work Monday-Saturday. 0 = Monday ... 5 = Saturday, 6 (Sunday) = off.
WORK_DAYS = [0, 1, 2, 3, 4, 5]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def placeholder_hash(text):
    # Placeholder only. The real Flask app should use werkzeug's
    # generate_password_hash instead.
    return hashlib.sha256(text.encode()).hexdigest()


def minutes_to_time(m):
    return time(m // 60, m % 60)


def make_slots(start_h, end_h, step_min):
    """Slot start times (in minutes after midnight) for one session."""
    slots, m = [], start_h * 60
    while m + step_min <= end_h * 60:
        slots.append(m)
        m += step_min
    return slots


def no_show_probability(lead_days, age, dow, hour, sms, dept_effect, reliability):
    z = -1.75 + 0.045 * lead_days + dept_effect + reliability - 0.7 * sms
    if 18 <= age <= 30:
        z += 0.4
    elif age >= 60:
        z -= 0.35
    elif age < 12:
        z -= 0.2
    if dow == 0:
        z += 0.2
    elif dow == 5:
        z += 0.35
    if hour < 10:
        z -= 0.15
    elif hour >= 15:
        z += 0.25
    return 1.0 / (1.0 + math.exp(-z))


# ----------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------
def generate():
    random.seed(SEED)
    today = date.today()

    data = {k: [] for k in [
        "users", "departments", "patients", "doctors",
        "doctor_schedules", "appointments", "consultations",
    ]}

    # --- users: 1 admin, then doctors, then patients -----------------
    user_id = 1
    data["users"].append({
        "user_id": user_id, "email": "admin@smarthospital.com",
        "password_hash": placeholder_hash("admin123"), "role": "admin",
        "created_at": datetime.now().replace(microsecond=0),
    })

    # --- departments -------------------------------------------------
    dept_list = []
    for i, (name, reasons) in enumerate(DEPARTMENTS.items(), start=1):
        data["departments"].append({
            "department_id": i, "name": name,
            "description": f"{name} department",
        })
        dept_list.append({"id": i, "name": name, "reasons": reasons,
                          "effect": DEPT_NOSHOW_EFFECT[name]})

    # --- doctors + schedules ----------------------------------------
    doctors = []
    schedule_id = 1
    doctor_id = 1
    used_names = set()
    for dept in dept_list:
        for _ in range(DOCTORS_PER_DEPT):
            user_id += 1
            while True:
                name = f"Dr. {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
                if name not in used_names:
                    used_names.add(name)
                    break
            cm = random.choice([10, 15, 15, 20, 25])
            data["users"].append({
                "user_id": user_id, "email": f"doctor{doctor_id}@smarthospital.com",
                "password_hash": placeholder_hash("doctor123"), "role": "doctor",
                "created_at": datetime.now().replace(microsecond=0),
            })
            data["doctors"].append({
                "doctor_id": doctor_id, "user_id": user_id,
                "department_id": dept["id"], "full_name": name,
                "specialization": dept["name"],
                "qualification": random.choice(["MBBS, MD", "MBBS, MS", "MBBS, DNB"]),
                "consultation_min": cm,
            })

            # Morning session every working day; afternoon on some days.
            afternoon_days = set(random.sample(WORK_DAYS, 3)) if random.random() < 0.6 else set()
            day_slots = {}
            for dow in WORK_DAYS:
                sessions = [(9, 13)]
                if dow in afternoon_days:
                    sessions.append((14, 17))
                slots = []
                for s, e in sessions:
                    sl = make_slots(s, e, cm)
                    slots += sl
                    data["doctor_schedules"].append({
                        "schedule_id": schedule_id, "doctor_id": doctor_id,
                        "day_of_week": dow, "start_time": time(s, 0),
                        "end_time": time(e, 0), "max_patients": len(sl),
                    })
                    schedule_id += 1
                day_slots[dow] = slots

            doctors.append({"id": doctor_id, "dept": dept, "cm": cm, "slots": day_slots})
            doctor_id += 1

    # --- patients ----------------------------------------------------
    patients = []
    for pid in range(1, N_PATIENTS + 1):
        user_id += 1
        age = int(random.triangular(1, 90, 35))
        dob = today - timedelta(days=age * 365 + random.randint(0, 364))
        data["users"].append({
            "user_id": user_id, "email": f"patient{pid}@example.com",
            "password_hash": placeholder_hash("patient123"), "role": "patient",
            "created_at": datetime.now().replace(microsecond=0),
        })
        data["patients"].append({
            "patient_id": pid, "user_id": user_id,
            "full_name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            "date_of_birth": dob, "gender": random.choices(["M", "F", "Other"], weights=[49, 49, 2])[0],
            "phone": str(random.randint(6, 9)) + "".join(random.choice("0123456789") for _ in range(9)),
            "address": random.choice(CITIES),
        })
        # Hidden trait: how reliable this patient is. Never saved to the
        # database; it only shapes the generated no-show behaviour.
        patients.append({"id": pid, "age": age, "reliability": random.gauss(0, 0.8)})

    # --- appointments ------------------------------------------------
    appt_id = 1
    consult_id = 1
    start_day = today - timedelta(days=HISTORY_DAYS)
    end_day = today + timedelta(days=FUTURE_DAYS)

    d = start_day
    while d <= end_day:
        dow = d.weekday()
        is_future = d >= today
        if dow in WORK_DAYS:
            for doc in doctors:
                slots = doc["slots"][dow]
                if not slots:
                    continue
                n = max(1, int(len(slots) * random.uniform(0.45, 0.95)))
                chosen = sorted(random.sample(slots, n))

                t_free = 0  # minute at which the doctor is next free
                for slot_min in chosen:
                    pat = random.choice(patients)
                    lead_days = min(int(random.expovariate(1 / 7.0)), 30)
                    appt_dt = datetime.combine(d, minutes_to_time(slot_min))
                    booked_at = appt_dt - timedelta(days=lead_days,
                                                    hours=random.randint(0, 6),
                                                    minutes=random.randint(0, 59))
                    sms = 1 if random.random() < 0.6 else 0

                    status, wait, actual_start = "scheduled", None, None
                    if not is_future:
                        p_ns = no_show_probability(
                            lead_days, pat["age"], dow, slot_min // 60, sms,
                            doc["dept"]["effect"], pat["reliability"])
                        r = random.random()
                        if r < p_ns:
                            status = "no_show"
                        elif r < p_ns + min(0.05 + 0.004 * lead_days, 0.15):
                            status = "cancelled"
                        else:
                            status = "completed"
                            # Occasional walk-in patients delay the doctor.
                            if random.random() < 0.15:
                                t_free = max(t_free, slot_min) + random.randint(8, 20)
                            start = max(slot_min, t_free)
                            wait = int(round(start - slot_min))
                            duration = doc["cm"] * 0.9 * random.lognormvariate(0, 0.35)
                            if random.random() < 0.05:
                                duration *= 2
                            t_free = start + duration
                            actual_start = minutes_to_time(min(int(start), 23 * 60 + 59))

                    data["appointments"].append({
                        "appointment_id": appt_id, "patient_id": pat["id"],
                        "doctor_id": doc["id"],
                        "booked_at": booked_at.replace(microsecond=0),
                        "appointment_date": d, "slot_time": minutes_to_time(slot_min),
                        "status": status, "reason": random.choice(doc["dept"]["reasons"]),
                        "sms_reminder_sent": sms,
                        "actual_start_time": actual_start,
                        "waiting_minutes": wait,
                    })

                    if status == "completed" and random.random() < 0.8:
                        follow = d + timedelta(days=random.choice([7, 14, 30])) if random.random() < 0.3 else None
                        data["consultations"].append({
                            "consultation_id": consult_id, "appointment_id": appt_id,
                            "notes": random.choice(NOTES),
                            "prescription": random.choice(PRESCRIPTIONS),
                            "follow_up_date": follow,
                        })
                        consult_id += 1
                    appt_id += 1
        d += timedelta(days=1)

    return data


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def save_csv(data):
    os.makedirs(OUT_DIR, exist_ok=True)
    for table, rows in data.items():
        path = os.path.join(OUT_DIR, f"{table}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  saved {path}  ({len(rows)} rows)")


# Insert order matters because of foreign keys.
LOAD_ORDER = ["users", "departments", "patients", "doctors",
              "doctor_schedules", "appointments", "consultations"]


def load_into_mysql(data):
    try:
        import mysql.connector
    except ImportError:
        print("\nmysql-connector-python is not installed, so MySQL was skipped.")
        print("Install it with:  pip install mysql-connector-python")
        print("then run this script again.")
        return

    password = getpass.getpass(f"MySQL password for {DB_USER}: ")
    conn = mysql.connector.connect(host=DB_HOST, port=DB_PORT, user=DB_USER,
                                   password=password, database=DB_NAME)
    cur = conn.cursor()

    cur.execute("SET FOREIGN_KEY_CHECKS = 0")
    cur.execute("SHOW TABLES LIKE 'doctor_leaves'")
    tables = ["predictions", "consultations", "appointments", "doctor_schedules"]
    if cur.fetchall():  # exists once the web app has upgraded the database
        tables.append("doctor_leaves")
    for table in tables + ["doctors", "patients", "departments", "users"]:
        cur.execute(f"TRUNCATE TABLE {table}")
    cur.execute("SET FOREIGN_KEY_CHECKS = 1")

    for table in LOAD_ORDER:
        rows = data[table]
        cols = list(rows[0].keys())
        sql = (f"INSERT INTO {table} ({', '.join(cols)}) "
               f"VALUES ({', '.join(['%s'] * len(cols))})")
        values = [tuple(r[c] for c in cols) for r in rows]
        for i in range(0, len(values), 2000):
            cur.executemany(sql, values[i:i + 2000])
        print(f"  loaded {table}  ({len(rows)} rows)")

    conn.commit()
    cur.close()
    conn.close()
    print("\nDone. Data is now in the smart_hospital database.")


def summary(data):
    appts = data["appointments"]
    past = [a for a in appts if a["status"] != "scheduled"]
    counts = {}
    for a in past:
        counts[a["status"]] = counts.get(a["status"], 0) + 1
    waits = [a["waiting_minutes"] for a in past if a["waiting_minutes"] is not None]
    print(f"\nAppointments: {len(appts)} total, {len(appts) - len(past)} upcoming")
    for s, c in sorted(counts.items()):
        print(f"  {s:10s} {c:6d}  ({100 * c / len(past):.1f}% of past)")
    if waits:
        print(f"  average waiting time: {sum(waits) / len(waits):.1f} min, "
              f"max {max(waits)} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-db", action="store_true",
                        help="only create CSV files, do not touch MySQL")
    args = parser.parse_args()

    print("Generating data...")
    data = generate()
    summary(data)
    print("\nSaving CSV files...")
    save_csv(data)
    if not args.no_db:
        print("\nLoading into MySQL...")
        load_into_mysql(data)
