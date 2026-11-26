"""
Smart Hospital Management and Appointment Prediction System
Flask web app, stage 1: login, patient registration, doctor search,
appointment booking and cancellation.

Run:   python app.py      then open  http://127.0.0.1:5000
"""

import csv
import functools
import hashlib
import os
import secrets
from datetime import date, datetime, time, timedelta

import mysql.connector
from flask import (Flask, abort, flash, g, redirect, render_template,
                   request, send_from_directory, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

import config
from predictor import Predictor, risk_level

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

BOOKING_WINDOW_DAYS = 30
OVERDUE_MINUTES = 60   # a pending patient this far past their slot needs a status update

# Trained models live in ../model (created by the two training scripts)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         getattr(config, "MODEL_DIR", "../model"))
predictor = Predictor(MODEL_DIR)
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ----------------------------------------------------------------------
# Database helpers
# ----------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = mysql.connector.connect(
            host=config.DB_HOST, port=config.DB_PORT, user=config.DB_USER,
            password=config.DB_PASSWORD, database=config.DB_NAME)
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def query(sql, params=(), one=False):
    cur = get_db().cursor(dictionary=True)
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    if one:
        return rows[0] if rows else None
    return rows


def to_time(value):
    """MySQL returns TIME columns as timedelta; turn them into time."""
    if isinstance(value, timedelta):
        total = int(value.total_seconds())
        return time(total // 3600 % 24, total % 3600 // 60)
    if isinstance(value, time):
        return value
    return time.fromisoformat(str(value))


def minutes_of(t):
    return t.hour * 60 + t.minute


@app.template_filter("time12")
def time12(value):
    return to_time(value).strftime("%I:%M %p").lstrip("0")


STATUS_TEXT = {"pending": "Awaiting confirmation", "scheduled": "Confirmed",
               "completed": "Completed", "cancelled": "Cancelled", "no_show": "No show"}


@app.template_filter("status_text")
def status_text(value):
    return STATUS_TEXT.get(value, value)


@app.template_filter("time24")
def time24(value):
    return to_time(value).strftime("%H:%M")


@app.template_filter("date_nice")
def date_nice(value):
    return value.strftime("%a, %d %b %Y")


# ----------------------------------------------------------------------
# Security: passwords, login check, form tokens
# ----------------------------------------------------------------------
def verify_password(stored, supplied):
    # New accounts use werkzeug hashes. The generated demo accounts use a
    # plain SHA-256 placeholder, so accept that too.
    if stored.startswith(("scrypt:", "pbkdf2:")):
        return check_password_hash(stored, supplied)
    return secrets.compare_digest(stored, hashlib.sha256(supplied.encode()).hexdigest())


def login_required(*roles):
    def decorator(view):
        @functools.wraps(view)
        def wrapped(*args, **kwargs):
            if "user_id" not in session:
                return redirect(url_for("login"))
            if roles and session.get("role") not in roles:
                flash("You do not have access to that page.", "error")
                return redirect(url_for("home"))
            return view(*args, **kwargs)
        return wrapped
    return decorator


def csrf_token():
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_hex(16)
    return session["_csrf"]


app.jinja_env.globals["csrf_token"] = csrf_token


@app.before_request
def check_csrf():
    if request.method == "POST":
        expected = session.get("_csrf")
        sent = request.form.get("_csrf", "")
        if not expected or not secrets.compare_digest(expected, sent):
            abort(400, "Invalid or expired form. Go back, refresh the page and try again.")


# ----------------------------------------------------------------------
# Login, logout, registration
# ----------------------------------------------------------------------
@app.route("/")
def home():
    role = session.get("role")
    if role == "patient":
        return redirect(url_for("patient_dashboard"))
    if role == "doctor":
        return redirect(url_for("doctor_home"))
    if role == "admin":
        return redirect(url_for("admin_home"))
    return redirect(url_for("login"))


LOGIN_ROLES = [("patient", "Patient"), ("doctor", "Doctor"), ("admin", "Admin")]


@app.route("/login", methods=["GET", "POST"])
def login():
    selected = "patient"
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        role = request.form.get("role", "")
        if role not in dict(LOGIN_ROLES):
            flash("Please choose whether you are logging in as a patient, doctor or admin.", "error")
            return render_template("login.html", roles=LOGIN_ROLES, selected=selected)
        selected = role
        user = query("SELECT user_id, password_hash, role FROM users WHERE email = %s",
                     (email,), one=True)
        if user and verify_password(user["password_hash"], password):
            if user["role"] != role:
                article = "an" if role == "admin" else "a"
                flash(f"This account is not {article} {role} account. "
                      "Choose the correct login type and try again.", "error")
                return render_template("login.html", roles=LOGIN_ROLES, selected=selected)
            profile_id, name = None, "Administrator"
            if user["role"] == "patient":
                p = query("SELECT patient_id, full_name FROM patients WHERE user_id = %s",
                          (user["user_id"],), one=True)
                profile_id, name = p["patient_id"], p["full_name"]
            elif user["role"] == "doctor":
                d = query("SELECT doctor_id, full_name FROM doctors WHERE user_id = %s",
                          (user["user_id"],), one=True)
                profile_id, name = d["doctor_id"], d["full_name"]
            session.clear()
            session["user_id"] = user["user_id"]
            session["role"] = user["role"]
            session["profile_id"] = profile_id
            session["name"] = name
            return redirect(url_for("home"))
        flash("Wrong email or password.", "error")
    return render_template("login.html", roles=LOGIN_ROLES, selected=selected)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html", form={})

    form = request.form
    name = form.get("full_name", "").strip()
    email = form.get("email", "").strip().lower()
    password = form.get("password", "")
    gender = form.get("gender", "")
    phone = form.get("phone", "").strip()
    address = form.get("address", "").strip()

    errors = []
    if len(name) < 2:
        errors.append("Please enter your full name.")
    if "@" not in email or len(email) > 120:
        errors.append("Please enter a valid email address.")
    if len(password) < 6:
        errors.append("Password must be at least 6 characters.")
    try:
        dob = date.fromisoformat(form.get("date_of_birth", ""))
        if dob > date.today() or dob.year < date.today().year - 120:
            errors.append("Please enter a valid date of birth.")
    except ValueError:
        dob = None
        errors.append("Please enter your date of birth.")
    if gender not in ("M", "F", "Other"):
        errors.append("Please choose a gender.")
    if not (phone.isdigit() and len(phone) == 10):
        errors.append("Phone number must be 10 digits.")

    if errors:
        for e in errors:
            flash(e, "error")
        return render_template("register.html", form=form)

    db = get_db()
    cur = db.cursor()
    try:
        cur.execute("INSERT INTO users (email, password_hash, role) VALUES (%s, %s, 'patient')",
                    (email, generate_password_hash(password)))
        user_id = cur.lastrowid
        cur.execute(
            "INSERT INTO patients (user_id, full_name, date_of_birth, gender, phone, address) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (user_id, name, dob, gender, phone, address or None))
        db.commit()
    except mysql.connector.IntegrityError:
        db.rollback()
        flash("That email is already registered. Try logging in.", "error")
        return render_template("register.html", form=form)
    finally:
        cur.close()

    flash("Account created. Please log in.", "success")
    return redirect(url_for("login"))


# ----------------------------------------------------------------------
# Patient: my appointments, doctor search, booking
# ----------------------------------------------------------------------
@app.route("/patient")
@login_required("patient")
def patient_dashboard():
    rows = query("""
        SELECT a.appointment_id, a.doctor_id, a.appointment_date, a.slot_time, a.status, a.reason,
               a.cancelled_by, a.cancel_reason,
               d.full_name AS doctor_name, dep.name AS department
        FROM appointments a
        JOIN doctors d ON d.doctor_id = a.doctor_id
        JOIN departments dep ON dep.department_id = d.department_id
        WHERE a.patient_id = %s
        ORDER BY a.appointment_date DESC, a.slot_time DESC""", (session["profile_id"],))

    now = datetime.now()
    upcoming, history, attention = [], [], []
    for r in rows:
        when = datetime.combine(r["appointment_date"], to_time(r["slot_time"]))
        if r["status"] in ("pending", "scheduled") and when >= now:
            upcoming.append(r)
        else:
            history.append(r)
            if r["status"] == "cancelled" and r["cancelled_by"] == "doctor" and when >= now:
                attention.append(r)
    upcoming.reverse()  # soonest first
    attention.reverse()
    return render_template("patient_dashboard.html", upcoming=upcoming, history=history[:30],
                           total_history=len(history), attention=attention)


@app.route("/doctors")
@login_required("patient")
def doctors():
    dept_id = request.args.get("department", type=int)
    q = request.args.get("q", "").strip()

    sql = """SELECT d.doctor_id, d.full_name, d.specialization, d.qualification,
                    d.consultation_min, dep.name AS department
             FROM doctors d JOIN departments dep ON dep.department_id = d.department_id
             WHERE 1 = 1"""
    params = []
    if dept_id:
        sql += " AND d.department_id = %s"
        params.append(dept_id)
    if q:
        sql += " AND d.full_name LIKE %s"
        params.append(f"%{q}%")
    sql += " ORDER BY dep.name, d.full_name"
    docs = query(sql, tuple(params))

    days = {}
    for r in query("SELECT DISTINCT doctor_id, day_of_week FROM doctor_schedules"):
        days.setdefault(r["doctor_id"], set()).add(r["day_of_week"])
    for d in docs:
        d["days"] = ", ".join(DAY_NAMES[i] for i in sorted(days.get(d["doctor_id"], [])))

    departments = query("SELECT department_id, name FROM departments ORDER BY name")
    return render_template("doctors.html", doctors=docs, departments=departments,
                           dept_id=dept_id, q=q)


def slot_grid(doctor, day):
    """Every bookable slot of a doctor on a date, as ({value, minutes, free}, why_none).

    why_none explains an empty list: 'range' (date outside the booking window),
    'leave' (doctor on leave) or 'off' (doctor does not work that weekday).
    """
    today = date.today()
    if day < today or day > today + timedelta(days=BOOKING_WINDOW_DAYS):
        return [], "range"
    if query("SELECT leave_id FROM doctor_leaves WHERE doctor_id = %s AND leave_date = %s",
             (doctor["doctor_id"], day), one=True):
        return [], "leave"

    step = doctor["consultation_min"]
    sessions = query(
        "SELECT start_time, end_time FROM doctor_schedules "
        "WHERE doctor_id = %s AND day_of_week = %s ORDER BY start_time",
        (doctor["doctor_id"], day.weekday()))
    if not sessions:
        return [], "off"

    taken = {minutes_of(to_time(r["slot_time"])) for r in query(
        "SELECT slot_time FROM appointments "
        "WHERE doctor_id = %s AND appointment_date = %s AND status <> 'cancelled'",
        (doctor["doctor_id"], day))}

    now_min = minutes_of(datetime.now().time())
    slots = []
    for s in sessions:
        m = minutes_of(to_time(s["start_time"]))
        end = minutes_of(to_time(s["end_time"]))
        while m + step <= end:
            if not (day == today and m <= now_min):
                slots.append({"value": time(m // 60, m % 60).strftime("%H:%M"),
                              "minutes": m, "free": m not in taken})
            m += step
    return slots, None


def available_slots(doctor, day):
    """Only the free slots of one doctor on one date."""
    return [s for s in slot_grid(doctor, day)[0] if s["free"]]


def label_slot(s):
    s["label"] = time12(time(s["minutes"] // 60, s["minutes"] % 60))
    return s


WHY_TEXT = {
    "leave": "The doctor is on leave on this date.",
    "off": "The doctor does not see patients on this weekday.",
    "range": "That date is outside the booking window.",
}


def find_alternatives(doctor_id, day, want_min):
    """Other doctors (and other times) when the chosen doctor or slot is unavailable."""
    base = """SELECT d.doctor_id, d.full_name, d.consultation_min, d.department_id,
                     dep.name AS department
              FROM doctors d JOIN departments dep ON dep.department_id = d.department_id"""
    this = query(base + " WHERE d.doctor_id = %s", (doctor_id,), one=True)
    if this is None:
        return None
    others = query(base + " WHERE d.doctor_id <> %s ORDER BY dep.name, d.full_name", (doctor_id,))

    grid, why = slot_grid(this, day)
    free_here = [s for s in grid if s["free"]]
    if want_min is not None and why is None:
        why = "booked" if any(s["minutes"] == want_min and not s["free"] for s in grid) else None
    same_doc = []
    if want_min is not None:
        near = sorted(free_here, key=lambda s: abs(s["minutes"] - want_min))[:4]
        same_doc = [label_slot(s) for s in sorted(near, key=lambda s: s["minutes"])]

    exact, nearby = [], []
    for o in others:
        free = available_slots(o, day)
        if not free:
            continue
        same_dept = o["department_id"] == this["department_id"]
        if want_min is None:
            exact.append({"doctor": o, "same_dept": same_dept,
                          "slots": [label_slot(s) for s in free[:4]]})
            continue
        match = [s for s in free if s["minutes"] == want_min]
        if match:
            exact.append({"doctor": o, "same_dept": same_dept, "slots": [label_slot(match[0])]})
        else:
            best = min(free, key=lambda s: abs(s["minutes"] - want_min))
            if abs(best["minutes"] - want_min) <= 60:
                nearby.append({"doctor": o, "same_dept": same_dept, "slots": [label_slot(best)]})
    exact.sort(key=lambda g: not g["same_dept"])
    if want_min is not None:
        nearby.sort(key=lambda g: (abs(g["slots"][0]["minutes"] - want_min), not g["same_dept"]))
    return {"this": this, "why": why, "exact": exact[:6], "nearby": nearby[:4], "same_doc": same_doc}


@app.route("/alternatives/<int:doctor_id>")
@login_required("patient")
def alternatives(doctor_id):
    try:
        day = date.fromisoformat(request.args.get("date", ""))
    except ValueError:
        abort(400)
    want = parse_hhmm(request.args.get("time", ""))
    data = find_alternatives(doctor_id, day, minutes_of(want) if want else None)
    if data is None:
        abort(404)
    want_label = time12(want) if want else None
    return render_template("_alternatives.html", day=day, want=want, want_label=want_label,
                           why_text=("That time slot is already booked." if data["why"] == "booked"
                                     else WHY_TEXT.get(data["why"],
                                                       "No free slots are left on this date.")),
                           **data)


@app.route("/book/<int:doctor_id>", methods=["GET", "POST"])
@login_required("patient")
def book(doctor_id):
    doctor = query("""SELECT d.doctor_id, d.full_name, d.specialization, d.qualification,
                             d.consultation_min, dep.name AS department
                      FROM doctors d JOIN departments dep ON dep.department_id = d.department_id
                      WHERE d.doctor_id = %s""", (doctor_id,), one=True)
    if doctor is None:
        abort(404)

    today = date.today()
    raw = request.values.get("date", "")
    try:
        chosen = date.fromisoformat(raw)
    except ValueError:
        chosen = today

    slots, why = slot_grid(doctor, chosen)
    free_count = sum(1 for sl in slots if sl["free"])

    if request.method == "POST":
        slot_value = request.form.get("slot", "")
        reason = request.form.get("reason", "").strip()[:255] or None
        if slot_value not in {sl["value"] for sl in slots if sl["free"]}:
            flash(f"{doctor['full_name']} is not available at that time.", "error")
            return redirect(url_for("book", doctor_id=doctor_id, date=chosen.isoformat(),
                                    unavailable=slot_value))
        slot_time = time.fromisoformat(slot_value)
        result = create_appointment(session["profile_id"], doctor_id, chosen, slot_time, reason)
        if isinstance(result, int):
            score_new_booking(doctor_id, chosen, result)
            flash(f"Appointment requested with {doctor['full_name']} on {date_nice(chosen)} at "
                  f"{time12(slot_time)}. It is confirmed once the doctor accepts.", "success")
            return redirect(url_for("patient_dashboard"))
        flash(result, "error")
        return redirect(url_for("book", doctor_id=doctor_id, date=chosen.isoformat(),
                                unavailable=slot_value))

    unavailable = request.args.get("unavailable", "")
    for sl in slots:
        label_slot(sl)
    # Pop up the suggestions when a booking just failed or the whole day has no free slot.
    auto_open = bool(unavailable) or (not free_count and why != "range")
    return render_template("book.html", doctor=doctor, chosen=chosen, slots=slots,
                           free_count=free_count, why_text=WHY_TEXT.get(why, ""),
                           auto_open=auto_open, auto_time=unavailable,
                           min_date=today,
                           max_date=today + timedelta(days=BOOKING_WINDOW_DAYS))


def create_appointment(patient_id, doctor_id, day, slot_time, reason):
    """Insert the booking. Returns the new appointment id, or an error message."""
    db = get_db()
    cur = db.cursor(dictionary=True)
    try:
        cur.execute(
            "SELECT appointment_id FROM appointments WHERE patient_id = %s "
            "AND appointment_date = %s AND slot_time = %s AND status IN ('pending', 'scheduled')",
            (patient_id, day, slot_time))
        if cur.fetchall():
            return "You already have an appointment at that time."

        cur.execute(
            "SELECT appointment_id, status FROM appointments "
            "WHERE doctor_id = %s AND appointment_date = %s AND slot_time = %s FOR UPDATE",
            (doctor_id, day, slot_time))
        existing = cur.fetchall()
        now = datetime.now().replace(microsecond=0)

        if existing:
            row = existing[0]
            if row["status"] != "cancelled":
                db.rollback()
                return "Sorry, that slot was just taken. Please choose another."
            # The slot belongs to a cancelled booking (the table allows only one
            # row per doctor, date and slot), so reuse that row.
            appointment_id = row["appointment_id"]
            cur.execute("DELETE FROM predictions WHERE appointment_id = %s", (appointment_id,))
            cur.execute(
                "UPDATE appointments SET patient_id = %s, status = 'pending', "
                "booked_at = %s, reason = %s, sms_reminder_sent = 0, "
                "actual_start_time = NULL, waiting_minutes = NULL, "
                "cancelled_by = NULL, cancel_reason = NULL "
                "WHERE appointment_id = %s",
                (patient_id, now, reason, appointment_id))
        else:
            cur.execute(
                "INSERT INTO appointments (patient_id, doctor_id, booked_at, "
                "appointment_date, slot_time, status, reason) "
                "VALUES (%s, %s, %s, %s, %s, 'pending', %s)",
                (patient_id, doctor_id, now, day, slot_time, reason))
            appointment_id = cur.lastrowid
        db.commit()
        return appointment_id
    except mysql.connector.IntegrityError:
        db.rollback()
        return "Sorry, that slot was just taken. Please choose another."
    finally:
        cur.close()


@app.route("/appointments/<int:appt_id>/cancel", methods=["POST"])
@login_required("patient")
def cancel_appointment(appt_id):
    appt = query(
        "SELECT appointment_date, slot_time, status FROM appointments "
        "WHERE appointment_id = %s AND patient_id = %s",
        (appt_id, session["profile_id"]), one=True)
    if appt is None:
        flash("Appointment not found.", "error")
    elif appt["status"] not in ("pending", "scheduled") or \
            datetime.combine(appt["appointment_date"], to_time(appt["slot_time"])) < datetime.now():
        flash("This appointment can no longer be cancelled.", "error")
    else:
        db = get_db()
        cur = db.cursor()
        cur.execute("UPDATE appointments SET status = 'cancelled', cancelled_by = 'patient', "
                    "cancel_reason = NULL WHERE appointment_id = %s", (appt_id,))
        db.commit()
        cur.close()
        flash("Appointment cancelled.", "success")
    return redirect(url_for("patient_dashboard"))


# ----------------------------------------------------------------------
# Predictions: no-show risk and waiting-time estimates
# ----------------------------------------------------------------------
_base_rate = {"value": None}


def no_show_base_rate():
    """Overall share of no-shows among finished appointments."""
    if _base_rate["value"] is None:
        row = query("SELECT SUM(CASE WHEN status = 'no_show' THEN 1 ELSE 0 END) AS ns, "
                    "COUNT(*) AS n FROM appointments "
                    "WHERE status IN ('completed', 'no_show')", one=True)
        n = int(row["n"] or 0)
        _base_rate["value"] = float(row["ns"] or 0) / n if n else 0.2
    return _base_rate["value"]


def load_day(doctor_id, day):
    """One doctor's appointments for one date, with patient details."""
    doctor = query("""SELECT d.doctor_id, d.full_name, d.consultation_min, dep.name AS department
                      FROM doctors d JOIN departments dep ON dep.department_id = d.department_id
                      WHERE d.doctor_id = %s""", (doctor_id,), one=True)
    cap = query("SELECT COALESCE(SUM(max_patients), 0) AS cap FROM doctor_schedules "
                "WHERE doctor_id = %s AND day_of_week = %s",
                (doctor_id, day.weekday()), one=True)
    rows = query("""SELECT a.appointment_id, a.patient_id, a.booked_at, a.appointment_date,
                           a.slot_time, a.status, a.reason, a.sms_reminder_sent,
                           a.actual_start_time, a.waiting_minutes,
                           p.full_name AS patient_name, p.date_of_birth,
                           (c.consultation_id IS NOT NULL) AS has_notes
                    FROM appointments a
                    JOIN patients p ON p.patient_id = a.patient_id
                    LEFT JOIN consultations c ON c.appointment_id = a.appointment_id
                    WHERE a.doctor_id = %s AND a.appointment_date = %s
                    ORDER BY a.slot_time""", (doctor_id, day))
    for r in rows:
        r["slot_t"] = to_time(r["slot_time"])
        r["slot_min"] = minutes_of(r["slot_t"])
        r["age"] = (day - r["date_of_birth"]).days // 365
    return doctor, int(cap["cap"]), rows


def patient_history(patient_ids):
    """Finished appointments of the given patients: {id: [(datetime, was_no_show)]}."""
    if not patient_ids:
        return {}
    marks = ", ".join(["%s"] * len(patient_ids))
    rows = query("SELECT patient_id, appointment_date, slot_time, status FROM appointments "
                 f"WHERE status IN ('completed', 'no_show') AND patient_id IN ({marks})",
                 tuple(patient_ids))
    history = {}
    for r in rows:
        when = datetime.combine(r["appointment_date"], to_time(r["slot_time"]))
        history.setdefault(r["patient_id"], []).append((when, r["status"] == "no_show"))
    return history


def score_day(doctor, capacity, day, rows):
    """Adds no_show_prob, risk and est_wait to the open (pending or confirmed) rows."""
    for r in rows:
        r["no_show_prob"] = r["risk"] = r["est_wait"] = None
        r["overdue"] = False
    pending = [r for r in rows if r["status"] in ("pending", "scheduled")]
    if not pending:
        return

    # --- no-show probability ------------------------------------------
    if predictor.has_no_show:
        history = patient_history(sorted({r["patient_id"] for r in pending}))
        records = []
        for r in pending:
            when = datetime.combine(day, r["slot_t"])
            earlier = [no_show for (w, no_show) in history.get(r["patient_id"], []) if w < when]
            n_prior, n_ns = len(earlier), sum(earlier)
            records.append({
                "lead_days": max((when - r["booked_at"]).total_seconds() / 86400, 0.0),
                "age": r["age"],
                "day_of_week": day.weekday(),
                "hour": r["slot_t"].hour,
                "sms_reminder_sent": int(r["sms_reminder_sent"]),
                "prior_appointments": n_prior,
                "prior_no_shows": n_ns,
                "prior_no_show_rate": n_ns / n_prior if n_prior else 0.0,
                "department": doctor["department"],
            })
        for r, p in zip(pending, predictor.no_show(records, no_show_base_rate())):
            r["no_show_prob"] = p
            r["risk"] = risk_level(p)

    # --- live waiting time (today only) --------------------------------
    if predictor.has_wait and day == date.today():
        now_min = minutes_of(datetime.now().time())
        day_load = len(rows)  # every booking that day, as in the training data
        done = [r for r in rows if r["status"] == "completed" and r["waiting_minutes"] is not None]
        records, targets = [], []
        for position, r in enumerate(rows, start=1):
            if r["status"] != "scheduled":
                continue
            earlier = [d for d in done if d["slot_min"] < r["slot_min"]]
            prev = earlier[-1] if earlier else None
            records.append({
                "position_in_day": position,
                "day_load": day_load,
                "utilization": day_load / (capacity or day_load),
                "hour": r["slot_t"].hour,
                "day_of_week": day.weekday(),
                "consultation_min": doctor["consultation_min"],
                "prev_wait": int(prev["waiting_minutes"]) if prev else 0,
                "gap_to_prev": (r["slot_min"] - prev["slot_min"]) if prev else 0,
                "department": doctor["department"],
            })
            targets.append(r)
        for r, w in zip(targets, predictor.wait(records)):
            already = max(0, now_min - r["slot_min"])  # minutes already past the slot
            if already > OVERDUE_MINUTES:
                # Still "scheduled" long after the slot: nobody marked the patient
                # as seen or no-show, so a wait estimate would be meaningless.
                r["overdue"] = True
            elif w is not None:
                r["est_wait"] = int(round(max(w, already)))


def save_predictions(rows):
    """Store each scheduled row's prediction in the `predictions` table."""
    rows = [r for r in rows if r.get("no_show_prob") is not None]
    if not rows:
        return
    db = get_db()
    cur = db.cursor()
    version = predictor.version()
    now = datetime.now().replace(microsecond=0)
    for r in rows:
        prob = round(r["no_show_prob"], 4)
        cur.execute("SELECT prediction_id FROM predictions WHERE appointment_id = %s",
                    (r["appointment_id"],))
        if cur.fetchall():
            cur.execute("UPDATE predictions SET no_show_probability = %s, predicted_wait_min = %s, "
                        "model_version = %s, predicted_at = %s WHERE appointment_id = %s",
                        (prob, r["est_wait"], version, now, r["appointment_id"]))
        else:
            cur.execute("INSERT INTO predictions (appointment_id, no_show_probability, "
                        "predicted_wait_min, model_version, predicted_at) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (r["appointment_id"], prob, r["est_wait"], version, now))
    db.commit()
    cur.close()


def score_new_booking(doctor_id, day, appointment_id):
    """Predict and store the no-show risk of a booking just made. Never fails the booking."""
    try:
        doctor, capacity, rows = load_day(doctor_id, day)
        score_day(doctor, capacity, day, rows)
        save_predictions([r for r in rows if r["appointment_id"] == appointment_id])
    except Exception:
        app.logger.exception("Could not store the prediction for the new booking")


# ----------------------------------------------------------------------
# Doctor: daily schedule, mark seen / no-show, consultation notes
# ----------------------------------------------------------------------
@app.route("/doctor")
@login_required("doctor")
def doctor_home():
    try:
        day = date.fromisoformat(request.args.get("date", ""))
    except ValueError:
        day = date.today()

    doctor, capacity, rows = load_day(session["profile_id"], day)
    score_day(doctor, capacity, day, rows)
    try:
        save_predictions([r for r in rows if r["status"] in ("pending", "scheduled")])
    except Exception:
        get_db().rollback()
        app.logger.exception("Could not store predictions")

    visible = [r for r in rows if r["status"] != "cancelled"]
    seen = [r for r in visible if r["status"] == "completed"]
    confirmed = [r for r in visible if r["status"] == "scheduled"]
    to_accept = [r for r in visible if r["status"] == "pending"]
    waits = [int(r["waiting_minutes"]) for r in seen if r["waiting_minutes"] is not None]
    expected = [r["no_show_prob"] for r in confirmed + to_accept if r["no_show_prob"] is not None]
    today = date.today()
    leave = query("SELECT reason FROM doctor_leaves WHERE doctor_id = %s AND leave_date = %s",
                  (session["profile_id"], day), one=True)
    leave_days = query("SELECT leave_date, reason FROM doctor_leaves WHERE doctor_id = %s "
                       "AND leave_date >= %s ORDER BY leave_date LIMIT 10",
                       (session["profile_id"], today))
    summary = {
        "booked": len(visible),
        "seen": len(seen),
        "waiting": len(confirmed) - sum(1 for r in confirmed if r["overdue"]),
        "overdue": sum(1 for r in confirmed if r["overdue"]),
        "to_accept": len(to_accept),
        "no_shows": sum(1 for r in visible if r["status"] == "no_show"),
        "avg_wait": round(sum(waits) / len(waits)) if waits else None,
        "expected_no_shows": round(sum(expected), 1) if expected else None,
    }
    return render_template("doctor_home.html", doctor=doctor, day=day, rows=visible,
                           summary=summary, is_today=(day == date.today()),
                           hidden_cancelled=len(rows) - len(visible),
                           leave=leave, leave_days=leave_days, can_edit_leave=(day >= today),
                           open_count=len(confirmed) + len(to_accept),
                           models_ok=predictor.has_no_show and predictor.has_wait)


def own_appointment(appt_id):
    return query("SELECT appointment_id, appointment_date, slot_time, status FROM appointments "
                 "WHERE appointment_id = %s AND doctor_id = %s",
                 (appt_id, session["profile_id"]), one=True)


@app.route("/doctor/appointments/<int:appt_id>/seen", methods=["POST"])
@login_required("doctor")
def mark_seen(appt_id):
    a = own_appointment(appt_id)
    if a is None:
        flash("Appointment not found.", "error")
    elif a["status"] != "scheduled" or a["appointment_date"] != date.today():
        flash("Only today's confirmed appointments can be updated.", "error")
    else:
        now = datetime.now()
        slot_dt = datetime.combine(a["appointment_date"], to_time(a["slot_time"]))
        wait = max(0, int((now - slot_dt).total_seconds() // 60))
        db = get_db()
        cur = db.cursor()
        if wait > OVERDUE_MINUTES:
            # The slot was long ago, so the real wait is unknown. Recording a huge
            # number would corrupt the data the waiting-time model learns from.
            cur.execute("UPDATE appointments SET status = 'completed' WHERE appointment_id = %s",
                        (appt_id,))
            message = ("Marked as seen. The slot was more than an hour ago, "
                       "so no waiting time was recorded.")
        else:
            cur.execute("UPDATE appointments SET status = 'completed', actual_start_time = %s, "
                        "waiting_minutes = %s WHERE appointment_id = %s",
                        (now.time().replace(microsecond=0), wait, appt_id))
            message = f"Marked as seen. Waiting time recorded: {wait} min."
        db.commit()
        cur.close()
        flash(message, "success")
    return redirect(url_for("doctor_home"))


@app.route("/doctor/appointments/<int:appt_id>/noshow", methods=["POST"])
@login_required("doctor")
def mark_no_show(appt_id):
    a = own_appointment(appt_id)
    if a is None:
        flash("Appointment not found.", "error")
    elif a["status"] != "scheduled" or a["appointment_date"] != date.today():
        flash("Only today's confirmed appointments can be updated.", "error")
    else:
        db = get_db()
        cur = db.cursor()
        cur.execute("UPDATE appointments SET status = 'no_show' WHERE appointment_id = %s",
                    (appt_id,))
        db.commit()
        cur.close()
        flash("Marked as no-show.", "success")
    return redirect(url_for("doctor_home"))


@app.route("/doctor/appointments/<int:appt_id>/accept", methods=["POST"])
@login_required("doctor")
def accept_appointment(appt_id):
    a = own_appointment(appt_id)
    if a is None:
        flash("Appointment not found.", "error")
        return redirect(url_for("doctor_home"))
    when = datetime.combine(a["appointment_date"], to_time(a["slot_time"]))
    if a["status"] != "pending":
        flash("Only bookings waiting for acceptance can be accepted.", "error")
    elif when < datetime.now():
        flash("That time has already passed.", "error")
    else:
        db = get_db()
        cur = db.cursor()
        cur.execute("UPDATE appointments SET status = 'scheduled' WHERE appointment_id = %s",
                    (appt_id,))
        db.commit()
        cur.close()
        flash("Appointment accepted. The patient now sees it as confirmed.", "success")
    return redirect(url_for("doctor_home", date=a["appointment_date"].isoformat()))


@app.route("/doctor/appointments/<int:appt_id>/cancel", methods=["POST"])
@login_required("doctor")
def doctor_cancel(appt_id):
    reason = request.form.get("reason", "").strip()[:255] or "Cancelled by the doctor"
    a = own_appointment(appt_id)
    if a is None:
        flash("Appointment not found.", "error")
        return redirect(url_for("doctor_home"))
    when = datetime.combine(a["appointment_date"], to_time(a["slot_time"]))
    if a["status"] not in ("pending", "scheduled"):
        flash("Only waiting or confirmed appointments can be cancelled.", "error")
    elif when < datetime.now():
        flash("Past appointments cannot be cancelled.", "error")
    else:
        db = get_db()
        cur = db.cursor()
        cur.execute("UPDATE appointments SET status = 'cancelled', cancelled_by = 'doctor', "
                    "cancel_reason = %s WHERE appointment_id = %s", (reason, appt_id))
        db.commit()
        cur.close()
        flash("Appointment cancelled. The patient will see your reason and can pick another doctor.",
              "success")
    return redirect(url_for("doctor_home", date=a["appointment_date"].isoformat()))


@app.route("/doctor/leave", methods=["POST"])
@login_required("doctor")
def mark_leave():
    try:
        day = date.fromisoformat(request.form.get("date", ""))
    except ValueError:
        flash("Please choose a valid date.", "error")
        return redirect(url_for("doctor_home"))
    today = date.today()
    if day < today:
        flash("You can only mark today or a future date as leave.", "error")
        return redirect(url_for("doctor_home"))
    reason = request.form.get("reason", "").strip()[:200] or "Doctor on leave"
    doctor_id = session["profile_id"]

    db = get_db()
    cur = db.cursor()
    try:
        cur.execute("INSERT INTO doctor_leaves (doctor_id, leave_date, reason) VALUES (%s, %s, %s)",
                    (doctor_id, day, reason))
        sql = ("UPDATE appointments SET status = 'cancelled', cancelled_by = 'doctor', "
               "cancel_reason = %s WHERE doctor_id = %s AND appointment_date = %s "
               "AND status IN ('pending', 'scheduled')")
        params = [reason, doctor_id, day]
        if day == today:  # slots that already passed are left alone
            sql += " AND slot_time > %s"
            params.append(datetime.now().time().replace(microsecond=0))
        cur.execute(sql, tuple(params))
        cancelled = cur.rowcount
        db.commit()
    except mysql.connector.IntegrityError:
        db.rollback()
        flash("That day is already marked as leave.", "error")
        return redirect(url_for("doctor_home", date=day.isoformat()))
    finally:
        cur.close()
    flash(f"Leave marked for {date_nice(day)}. {cancelled} appointment"
          f"{'' if cancelled == 1 else 's'} cancelled and no new bookings are possible that day.",
          "success")
    return redirect(url_for("doctor_home", date=day.isoformat()))


@app.route("/doctor/leave/remove", methods=["POST"])
@login_required("doctor")
def remove_leave():
    try:
        day = date.fromisoformat(request.form.get("date", ""))
    except ValueError:
        return redirect(url_for("doctor_home"))
    db = get_db()
    cur = db.cursor()
    cur.execute("DELETE FROM doctor_leaves WHERE doctor_id = %s AND leave_date = %s "
                "AND leave_date >= %s", (session["profile_id"], day, date.today()))
    db.commit()
    cur.close()
    flash("Leave removed. Patients can book that day again; cancelled appointments stay cancelled.",
          "success")
    return redirect(url_for("doctor_home", date=day.isoformat()))


@app.route("/doctor/appointments/<int:appt_id>/notes", methods=["GET", "POST"])
@login_required("doctor")
def consultation_notes(appt_id):
    a = query("""SELECT a.appointment_id, a.appointment_date, a.slot_time, a.status, a.reason,
                        p.full_name AS patient_name
                 FROM appointments a JOIN patients p ON p.patient_id = a.patient_id
                 WHERE a.appointment_id = %s AND a.doctor_id = %s""",
              (appt_id, session["profile_id"]), one=True)
    if a is None:
        abort(404)
    if a["status"] != "completed":
        flash("Notes can be added once the patient has been seen.", "error")
        return redirect(url_for("doctor_home"))

    existing = query("SELECT notes, prescription, follow_up_date FROM consultations "
                     "WHERE appointment_id = %s", (appt_id,), one=True)
    form = existing or {}

    if request.method == "POST":
        notes = request.form.get("notes", "").strip()[:5000] or None
        prescription = request.form.get("prescription", "").strip()[:5000] or None
        follow_raw = request.form.get("follow_up_date", "").strip()
        follow_up = None
        if follow_raw:
            try:
                follow_up = date.fromisoformat(follow_raw)
            except ValueError:
                follow_up = None
            if follow_up is None or follow_up < a["appointment_date"]:
                flash("Follow-up date must be on or after the appointment date.", "error")
                return render_template("doctor_notes.html", a=a, form=request.form)

        db = get_db()
        cur = db.cursor()
        if existing:
            cur.execute("UPDATE consultations SET notes = %s, prescription = %s, "
                        "follow_up_date = %s WHERE appointment_id = %s",
                        (notes, prescription, follow_up, appt_id))
        else:
            cur.execute("INSERT INTO consultations (appointment_id, notes, prescription, "
                        "follow_up_date) VALUES (%s, %s, %s, %s)",
                        (appt_id, notes, prescription, follow_up))
        db.commit()
        cur.close()
        flash("Consultation notes saved.", "success")
        return redirect(url_for("doctor_home", date=a["appointment_date"].isoformat()))

    return render_template("doctor_notes.html", a=a, form=form)


# ----------------------------------------------------------------------
# Admin: dashboard, model performance, prediction monitoring
# ----------------------------------------------------------------------
MODEL_IMAGES = [
    ("roc_curves.png", "No-show model: ROC curves"),
    ("confusion_matrix.png", "No-show model: confusion matrix"),
    ("feature_importance.png", "No-show model: feature importance"),
    ("waiting_actual_vs_predicted.png", "Waiting-time model: actual vs predicted"),
    ("waiting_feature_importance.png", "Waiting-time model: feature importance"),
]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def pct(part, whole):
    return round(100.0 * part / whole, 1) if whole else None


def rate_items(groups, order=None):
    """groups: {label: [finished, no_shows]} -> list for a horizontal bar chart."""
    labels = order or sorted(groups)
    items = []
    for label in labels:
        n, ns = groups.get(label, [0, 0])
        items.append({"label": label, "n": n, "rate": pct(ns, n)})
    top = max([i["rate"] or 0 for i in items] + [1])
    for i in items:
        i["width"] = round(100 * (i["rate"] or 0) / top)
    return items


def read_csv_table(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return {"head": rows[0], "body": rows[1:]} if rows else None


def admin_stats(today):
    db = get_db()
    cur = db.cursor()
    cur.execute("""SELECT a.appointment_date, a.booked_at, a.status, a.sms_reminder_sent,
                          a.waiting_minutes, dep.name
                   FROM appointments a
                   JOIN doctors d ON d.doctor_id = a.doctor_id
                   JOIN departments dep ON dep.department_id = d.department_id
                   WHERE a.appointment_date < %s""", (today,))
    past = cur.fetchall()
    cur.close()

    finished = [r for r in past if r[2] in ("completed", "no_show")]
    no_shows = sum(1 for r in finished if r[2] == "no_show")
    cancelled = sum(1 for r in past if r[2] == "cancelled")
    waits = [r[4] for r in finished if r[4] is not None]

    by_dept, by_day, by_sms, by_lead = {}, {}, {}, {}
    for date_, booked, status, sms, _wait, dept in finished:
        lead = (date_ - booked.date()).days
        bucket = ("0-2 days" if lead <= 2 else "3-7 days" if lead <= 7
                  else "8-14 days" if lead <= 14 else "15+ days")
        for table, key in ((by_dept, dept), (by_day, WEEKDAYS[date_.weekday()]),
                           (by_sms, "Reminder sent" if sms else "No reminder"),
                           (by_lead, bucket)):
            g = table.setdefault(key, [0, 0])
            g[0] += 1
            g[1] += 1 if status == "no_show" else 0

    # Weekly stacked columns (last 13 weeks with data)
    weeks = {}
    this_week = today - timedelta(days=today.weekday())
    for date_, _b, status, _s, _w, _d in past:
        if status not in ("completed", "no_show", "cancelled"):
            continue
        start = date_ - timedelta(days=date_.weekday())
        if start >= this_week:
            continue  # the current week is not finished yet
        w = weeks.setdefault(start, {"completed": 0, "no_show": 0, "cancelled": 0})
        w[status] += 1
    weekly = []
    chosen = sorted(weeks)[-13:]
    peak = max([sum(weeks[k].values()) for k in chosen] + [1])
    for k in chosen:
        w = weeks[k]
        total = sum(w.values())
        weekly.append({
            "label": k.strftime("%d %b").lstrip("0"),
            "title": f"Week of {k.strftime('%d %b')}: {total} appointments, "
                     f"{w['completed']} completed, {w['no_show']} no-shows, "
                     f"{w['cancelled']} cancelled",
            **{s: round(150.0 * w[s] / peak, 1) for s in w},
        })

    counts = {
        "patients": query("SELECT COUNT(*) AS n FROM patients", one=True)["n"],
        "doctors": query("SELECT COUNT(*) AS n FROM doctors", one=True)["n"],
        "upcoming": query("SELECT COUNT(*) AS n FROM appointments "
                          "WHERE status IN ('pending', 'scheduled') AND appointment_date >= %s",
                          (today,), one=True)["n"],
        "finished": len(finished),
        "no_show_rate": pct(no_shows, len(finished)),
        "cancel_rate": pct(cancelled, len(finished) + cancelled),
        "avg_wait": round(sum(waits) / len(waits), 1) if waits else None,
    }

    doctors = query("""
        SELECT d.full_name, dep.name AS department,
               SUM(CASE WHEN a.status IN ('completed', 'no_show') THEN 1 ELSE 0 END) AS finished,
               SUM(CASE WHEN a.status = 'no_show' THEN 1 ELSE 0 END) AS no_shows,
               AVG(a.waiting_minutes) AS avg_wait,
               SUM(CASE WHEN a.status IN ('pending', 'scheduled') AND a.appointment_date >= %s
                        THEN 1 ELSE 0 END) AS upcoming
        FROM doctors d
        JOIN departments dep ON dep.department_id = d.department_id
        LEFT JOIN appointments a ON a.doctor_id = d.doctor_id
        GROUP BY d.doctor_id, d.full_name, dep.name
        ORDER BY dep.name, d.full_name""", (today,))
    for d in doctors:
        d["finished"] = int(d["finished"] or 0)
        d["no_show_rate"] = pct(int(d["no_shows"] or 0), d["finished"])
        d["avg_wait"] = round(float(d["avg_wait"]), 1) if d["avg_wait"] is not None else None
        d["upcoming"] = int(d["upcoming"] or 0)

    return {
        "counts": counts,
        "weekly": weekly,
        "dept_chart": rate_items(by_dept),
        "day_chart": rate_items(by_day, [d for d in WEEKDAYS if d in by_day]),
        "sms_chart": rate_items(by_sms, ["No reminder", "Reminder sent"]),
        "lead_chart": rate_items(by_lead, ["0-2 days", "3-7 days", "8-14 days", "15+ days"]),
        "doctors": doctors,
    }


def prediction_check():
    """Compare stored predictions with what happened, for resolved appointments."""
    rows = query("""SELECT pr.no_show_probability AS p, a.status
                    FROM predictions pr JOIN appointments a ON a.appointment_id = pr.appointment_id
                    WHERE a.status IN ('completed', 'no_show')""")
    bands = [("Low (under 15%)", 0.0, 0.15), ("Medium (15-30%)", 0.15, 0.30),
             ("High (30% or more)", 0.30, 1.01)]
    result = []
    for label, lo, hi in bands:
        group = [r for r in rows if lo <= float(r["p"]) < hi]
        n = len(group)
        result.append({
            "label": label, "n": n,
            "predicted": pct(sum(float(r["p"]) for r in group), n),
            "actual": pct(sum(1 for r in group if r["status"] == "no_show"), n),
        })
    return {"total": len(rows), "bands": result}


def risky_upcoming(today, limit=10):
    now = datetime.now()
    rows = query("""
        SELECT a.appointment_id, a.appointment_date, a.slot_time, a.sms_reminder_sent,
               p.full_name AS patient_name, d.full_name AS doctor_name,
               pr.no_show_probability AS prob
        FROM predictions pr
        JOIN appointments a ON a.appointment_id = pr.appointment_id
        JOIN patients p ON p.patient_id = a.patient_id
        JOIN doctors d ON d.doctor_id = a.doctor_id
        WHERE a.status IN ('pending', 'scheduled') AND a.appointment_date >= %s AND a.appointment_date <= %s
        ORDER BY pr.no_show_probability DESC LIMIT 60""",
        (today, today + timedelta(days=7)))
    out = []
    for r in rows:
        if datetime.combine(r["appointment_date"], to_time(r["slot_time"])) < now:
            continue  # slot already passed
        r["prob"] = float(r["prob"])
        r["risk"] = risk_level(r["prob"])
        out.append(r)
    return out[:limit]


@app.route("/admin")
@login_required("admin")
def admin_home():
    today = date.today()
    stats = admin_stats(today)
    return render_template(
        "admin_home.html", **stats, today=today,
        check=prediction_check(), risky=risky_upcoming(today),
        has_predictions_model=predictor.has_no_show,
        model_tables=[t for t in (read_csv_table("model_comparison.csv"),
                                  read_csv_table("waiting_model_comparison.csv")) if t],
        model_titles=["No-show model comparison", "Waiting-time model comparison"],
        model_images=[(n, t) for n, t in MODEL_IMAGES
                      if os.path.exists(os.path.join(MODEL_DIR, n))])


@app.route("/admin/model/<name>")
@login_required("admin")
def admin_model_file(name):
    if name not in {n for n, _ in MODEL_IMAGES}:
        abort(404)
    return send_from_directory(MODEL_DIR, name)


@app.route("/admin/refresh-predictions", methods=["POST"])
@login_required("admin")
def admin_refresh_predictions():
    if not predictor.has_no_show:
        flash("The no-show model was not found in the model folder.", "error")
        return redirect(url_for("admin_home"))
    today = date.today()
    scored = 0
    for d in query("SELECT doctor_id FROM doctors"):
        for offset in range(0, 8):
            day = today + timedelta(days=offset)
            doctor, capacity, rows = load_day(d["doctor_id"], day)
            if not rows:
                continue
            score_day(doctor, capacity, day, rows)
            pending = [r for r in rows if r["status"] in ("pending", "scheduled")]
            save_predictions(pending)
            scored += sum(1 for r in pending if r["no_show_prob"] is not None)
    flash(f"Predictions refreshed for {scored} upcoming appointments.", "success")
    return redirect(url_for("admin_home"))


@app.route("/admin/appointments/<int:appt_id>/reminder", methods=["POST"])
@login_required("admin")
def send_reminder(appt_id):
    a = query("SELECT doctor_id, appointment_date, status, sms_reminder_sent "
              "FROM appointments WHERE appointment_id = %s", (appt_id,), one=True)
    if a is None or a["status"] not in ("pending", "scheduled") or a["appointment_date"] < date.today():
        flash("Reminders can only be sent for upcoming appointments.", "error")
    elif a["sms_reminder_sent"]:
        flash("A reminder was already sent for this appointment.", "error")
    else:
        db = get_db()
        cur = db.cursor()
        cur.execute("UPDATE appointments SET sms_reminder_sent = 1 WHERE appointment_id = %s",
                    (appt_id,))
        db.commit()
        cur.close()
        doctor, capacity, rows = load_day(a["doctor_id"], a["appointment_date"])
        score_day(doctor, capacity, a["appointment_date"], rows)
        mine = [r for r in rows if r["appointment_id"] == appt_id]
        save_predictions(mine)
        new = mine[0]["no_show_prob"] if mine else None
        extra = f" New estimated no-show risk: {round(new * 100)}%." if new is not None else ""
        flash("Reminder marked as sent (simulated: no real SMS is sent)." + extra, "success")
    return redirect(url_for("admin_home"))


# ----------------------------------------------------------------------
# Admin: manage departments and doctors
# ----------------------------------------------------------------------
SCHEDULE_DAYS = list(enumerate(DAY_NAMES[:6]))  # Monday to Saturday


def parse_hhmm(value):
    try:
        return time.fromisoformat(value.strip())
    except (ValueError, AttributeError):
        return None


def render_manage(form=None, selected_days=None):
    departments = query("""
        SELECT dep.department_id, dep.name, dep.description, COUNT(d.doctor_id) AS doctors
        FROM departments dep LEFT JOIN doctors d ON d.department_id = dep.department_id
        GROUP BY dep.department_id, dep.name, dep.description ORDER BY dep.name""")
    doctors = query("""
        SELECT d.doctor_id, d.full_name, dep.name AS department, d.qualification,
               d.consultation_min, u.email
        FROM doctors d
        JOIN departments dep ON dep.department_id = d.department_id
        JOIN users u ON u.user_id = d.user_id
        ORDER BY dep.name, d.full_name""")
    return render_template("admin_manage.html", departments=departments, doctors=doctors,
                           days=SCHEDULE_DAYS, form=form or {},
                           selected_days=[0, 1, 2, 3, 4, 5] if selected_days is None else selected_days)


@app.route("/admin/manage")
@login_required("admin")
def admin_manage():
    return render_manage()


@app.route("/admin/departments/add", methods=["POST"])
@login_required("admin")
def admin_add_department():
    name = request.form.get("name", "").strip()
    description = request.form.get("description", "").strip()[:255] or None
    if not 2 <= len(name) <= 80:
        flash("Department name must be 2 to 80 characters.", "error")
        return redirect(url_for("admin_manage"))
    if query("SELECT department_id FROM departments WHERE LOWER(name) = LOWER(%s)",
             (name,), one=True):
        flash(f"A department called {name} already exists.", "error")
        return redirect(url_for("admin_manage"))
    db = get_db()
    cur = db.cursor()
    try:
        cur.execute("INSERT INTO departments (name, description) VALUES (%s, %s)",
                    (name, description))
        db.commit()
        flash(f"Department {name} added.", "success")
    except mysql.connector.IntegrityError:
        db.rollback()
        flash(f"A department called {name} already exists.", "error")
    finally:
        cur.close()
    return redirect(url_for("admin_manage"))


@app.route("/admin/doctors/add", methods=["POST"])
@login_required("admin")
def admin_add_doctor():
    f = request.form
    name = f.get("full_name", "").strip()
    email = f.get("email", "").strip().lower()
    password = f.get("password", "")
    qualification = f.get("qualification", "").strip()[:100] or None
    days = sorted({int(x) for x in f.getlist("days") if x in ("0", "1", "2", "3", "4", "5")})
    afternoon = f.get("afternoon") == "1"
    dept_id = f.get("department_id", type=int)
    minutes = f.get("consultation_min", type=int)

    errors = []
    if len(name) < 2:
        errors.append("Please enter the doctor's name.")
    if "@" not in email or len(email) > 120:
        errors.append("Please enter a valid email address.")
    if len(password) < 6:
        errors.append("Password must be at least 6 characters.")
    department = query("SELECT department_id, name FROM departments WHERE department_id = %s",
                       (dept_id,), one=True) if dept_id else None
    if department is None:
        errors.append("Please choose a department.")
    if minutes is None or not 5 <= minutes <= 60:
        errors.append("Minutes per patient must be between 5 and 60.")
    if not days:
        errors.append("Choose at least one working day.")

    sessions = []
    m_start, m_end = parse_hhmm(f.get("m_start")), parse_hhmm(f.get("m_end"))
    if m_start is None or m_end is None or m_end <= m_start:
        errors.append("The morning session needs a start time before its end time.")
    else:
        sessions.append((m_start, m_end))
    if afternoon:
        a_start, a_end = parse_hhmm(f.get("a_start")), parse_hhmm(f.get("a_end"))
        if a_start is None or a_end is None or a_end <= a_start:
            errors.append("The afternoon session needs a start time before its end time.")
        elif sessions and a_start < sessions[0][1]:
            errors.append("The afternoon session must start after the morning session ends.")
        else:
            sessions.append((a_start, a_end))
    if minutes and 5 <= minutes <= 60:
        for s, e in sessions:
            if (minutes_of(e) - minutes_of(s)) // minutes < 1:
                errors.append("Each session must be long enough for at least one patient.")
                break

    if not errors and query("SELECT user_id FROM users WHERE email = %s", (email,), one=True):
        errors.append("That email is already registered.")

    if errors:
        for e in errors:
            flash(e, "error")
        return render_manage(form=f, selected_days=days)

    full_name = name if name.lower().startswith("dr") else f"Dr. {name}"
    db = get_db()
    cur = db.cursor()
    try:
        cur.execute("INSERT INTO users (email, password_hash, role) VALUES (%s, %s, 'doctor')",
                    (email, generate_password_hash(password)))
        user_id = cur.lastrowid
        cur.execute(
            "INSERT INTO doctors (user_id, department_id, full_name, specialization, "
            "qualification, consultation_min) VALUES (%s, %s, %s, %s, %s, %s)",
            (user_id, dept_id, full_name, department["name"], qualification, minutes))
        doctor_id = cur.lastrowid
        for dow in days:
            for s, e in sessions:
                cur.execute(
                    "INSERT INTO doctor_schedules (doctor_id, day_of_week, start_time, "
                    "end_time, max_patients) VALUES (%s, %s, %s, %s, %s)",
                    (doctor_id, dow, s, e, (minutes_of(e) - minutes_of(s)) // minutes))
        db.commit()
    except mysql.connector.IntegrityError:
        db.rollback()
        flash("That email is already registered.", "error")
        return render_manage(form=f, selected_days=days)
    finally:
        cur.close()
    flash(f"{full_name} added to {department['name']}. They can log in with {email}.", "success")
    return redirect(url_for("admin_manage"))


def ensure_schema():
    """Upgrade an existing database for booking approval and doctor leave.

    Safe to run on every start: each step is skipped if it is already done.
    """
    db = mysql.connector.connect(
        host=config.DB_HOST, port=config.DB_PORT, user=config.DB_USER,
        password=config.DB_PASSWORD, database=config.DB_NAME)
    cur = db.cursor()
    cur.execute("SELECT COLUMN_TYPE FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = %s "
                "AND TABLE_NAME = 'appointments' AND COLUMN_NAME = 'status'", (config.DB_NAME,))
    ctype = cur.fetchone()[0]
    if isinstance(ctype, (bytes, bytearray)):
        ctype = ctype.decode()
    if "'pending'" not in str(ctype):
        cur.execute("ALTER TABLE appointments MODIFY COLUMN status "
                    "ENUM('scheduled','completed','cancelled','no_show','pending') "
                    "NOT NULL DEFAULT 'scheduled'")
    cur.execute("SELECT COUNT(*) FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = %s "
                "AND TABLE_NAME = 'appointments' AND COLUMN_NAME = 'cancel_reason'",
                (config.DB_NAME,))
    if cur.fetchone()[0] == 0:
        cur.execute("ALTER TABLE appointments ADD COLUMN cancelled_by ENUM('patient','doctor') NULL, "
                    "ADD COLUMN cancel_reason VARCHAR(255) NULL")
    cur.execute("""CREATE TABLE IF NOT EXISTS doctor_leaves (
                       leave_id INT AUTO_INCREMENT PRIMARY KEY,
                       doctor_id INT NOT NULL,
                       leave_date DATE NOT NULL,
                       reason VARCHAR(255) NULL,
                       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                       FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id),
                       UNIQUE (doctor_id, leave_date))""")
    db.commit()
    cur.close()
    db.close()


if __name__ == "__main__":
    try:
        mysql.connector.connect(
            host=config.DB_HOST, port=config.DB_PORT, user=config.DB_USER,
            password=config.DB_PASSWORD, database=config.DB_NAME).close()
    except mysql.connector.Error as err:
        print("\nCould not connect to MySQL:", err)
        print("Check DB_PASSWORD in config.py and that the MySQL80 service is running.\n")
        raise SystemExit(1)
    try:
        ensure_schema()
    except mysql.connector.Error as err:
        print("\nCould not upgrade the database:", err)
        print("Run upgrade_schema.sql in MySQL Workbench once, then start the app again.\n")
        raise SystemExit(1)
    app.run(debug=True)
