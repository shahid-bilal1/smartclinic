
import logging
import math
import os
import re
import secrets
import threading
import time
from datetime import date, datetime, timedelta
from functools import wraps
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw charts to image files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from flask import (Flask, abort, flash, redirect, render_template, request,  # noqa: E402
                   session, url_for)
from markupsafe import Markup  # noqa: E402
from werkzeug.security import check_password_hash, generate_password_hash  # noqa: E402

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CHART_DIR = BASE_DIR / "static" / "charts"

FILES = {
    "patients": BASE_DIR / "patients.csv",
    "appointments": BASE_DIR / "appointments.csv",
    "billing": BASE_DIR / "billing.csv",
}
COLUMNS = {
    "patients": ["patient_id", "full_name", "age", "contact", "gender", "visit_date"],
    "appointments": ["appointment_id", "patient_id", "patient_name", "appointment_date",
                     "appointment_time", "doctor_name", "reason", "status"],
    "billing": ["bill_id", "patient_id", "consultation_fee", "additional_charges",
                "total_amount", "payment_method", "payment_status", "bill_date"],
}

GENDERS = ["Male", "Female", "Other"]
DOCTORS = ["Dr. Anita Rao", "Dr. Faisal Ahmed", "Dr. Priya Nambiar", "Dr. Rakesh Kulkarni"]
TIME_SLOTS = [f"{h:02d}:{m:02d}" for h in range(9, 17) for m in (0, 30)]  # 09:00 - 16:30
STATUSES = ["Scheduled", "Completed", "Cancelled"]
PAY_METHODS = ["Cash", "Card", "UPI", "Insurance"]
PAY_STATUSES = ["Paid", "Unpaid"]
PER_PAGE = 12
DEFAULT_FEE = 500

# Demo users. Passwords can be changed with environment variables (see README).
USERS = {
    "admin": {
        "hash": generate_password_hash(os.environ.get("CLINIC_ADMIN_PASSWORD", "admin123")),
        "role": "admin", "name": "Clinic Admin",
    },
    "reception": {
        "hash": generate_password_hash(os.environ.get("CLINIC_STAFF_PASSWORD", "reception123")),
        "role": "staff", "name": "Front Desk",
    },
}


def _load_secret_key() -> str:
    """Use an env variable, or create a random key once and keep it in .secret_key."""
    env_key = os.environ.get("CLINIC_SECRET_KEY")
    if env_key:
        return env_key
    key_file = BASE_DIR / ".secret_key"
    try:
        if key_file.exists():
            return key_file.read_text().strip()
        key = secrets.token_hex(32)
        key_file.write_text(key)
        return key
    except OSError:
        return secrets.token_hex(32)


app = Flask(__name__)
app.config.update(
    SECRET_KEY=_load_secret_key(),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=2),
)
logging.basicConfig(level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)  # hide chatty chart messages
log = logging.getLogger("clinic")

_data_lock = threading.RLock()   # one writer at a time for CSV files
_plot_lock = threading.Lock()    # Matplotlib is not thread-safe


# ---------------------------------------------------------------------------
# 2. CSV HELPERS
# ---------------------------------------------------------------------------
def write_table(key: str, df: pd.DataFrame) -> None:
    """Save a DataFrame to its CSV file safely (write temp file, then swap)."""
    path = FILES[key]
    tmp = path.with_suffix(".tmp")
    with _data_lock:
        df[COLUMNS[key]].to_csv(tmp, index=False)
        os.replace(tmp, path)


def read_table(key: str) -> pd.DataFrame:
    """
    Read a CSV as text. Missing, empty or damaged files give an empty table
    instead of crashing the website. Rows without an ID and duplicate IDs are dropped.
    """
    cols = COLUMNS[key]
    path = FILES[key]
    with _data_lock:
        if not path.exists():
            write_table(key, pd.DataFrame(columns=cols))
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, OSError) as err:
            log.warning("Could not read %s (%s). Using an empty table.", path.name, err)
            df = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols].astype(str).apply(lambda s: s.str.strip())
    df = df[df[cols[0]] != ""].drop_duplicates(subset=cols[0], keep="first")
    return df.reset_index(drop=True)


def next_id(prefix: str, df: pd.DataFrame, col: str) -> str:
    """P001, P002 ... always one higher than the biggest existing number (so IDs stay unique)."""
    nums = pd.to_numeric(df[col].str.extract(r"(\d+)$")[0], errors="coerce").dropna()
    return f"{prefix}{(int(nums.max()) + 1 if len(nums) else 1):03d}"


def typed_patients() -> pd.DataFrame:
    df = read_table("patients")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["visit_dt"] = pd.to_datetime(df["visit_date"], format="%Y-%m-%d", errors="coerce")
    return df


def typed_appointments() -> pd.DataFrame:
    df = read_table("appointments")
    df["status"] = df["status"].str.title()
    df["appt_dt"] = pd.to_datetime(df["appointment_date"], format="%Y-%m-%d", errors="coerce")
    return df


def typed_billing() -> pd.DataFrame:
    df = read_table("billing")
    for c in ("consultation_fee", "additional_charges", "total_amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # if the total is damaged/missing, rebuild it from the two parts
    df["total_amount"] = df["total_amount"].fillna(df["consultation_fee"].fillna(0) + df["additional_charges"].fillna(0))
    df["payment_status"] = df["payment_status"].str.title()
    df["bill_dt"] = pd.to_datetime(df["bill_date"], format="%Y-%m-%d", errors="coerce")
    return df


def paginate(df: pd.DataFrame, page: int):
    pages = max(1, math.ceil(len(df) / PER_PAGE))
    page = min(max(page, 1), pages)
    return df.iloc[(page - 1) * PER_PAGE: page * PER_PAGE], page, pages


# ---------------------------------------------------------------------------
# 3. VALIDATION
# ---------------------------------------------------------------------------
def parse_date(text: str):
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def parse_money(text: str):
    try:
        value = float(text)
    except (ValueError, TypeError):
        return None
    return round(value, 2) if math.isfinite(value) else None


NAME_RE = re.compile(r"^[A-Za-z][A-Za-z .'\-]{1,59}$")
PHONE_RE = re.compile(r"^\+?\d{10,13}$")


def validate_patient(form, patients: pd.DataFrame, editing_id: str = None):
    errors = []
    name = " ".join(form.get("full_name", "").split())
    age_text = form.get("age", "").strip()
    contact = re.sub(r"[\s\-]", "", form.get("contact", ""))
    gender = form.get("gender", "")
    visit_text = form.get("visit_date", "").strip()

    if not NAME_RE.match(name):
        errors.append("Full name must be 2-60 characters (letters, spaces, . ' - only).")
    if not age_text.isdigit() or not (0 <= int(age_text) <= 120):
        errors.append("Age must be a whole number between 0 and 120.")
    if not PHONE_RE.match(contact):
        errors.append("Contact number must have 10-13 digits (a leading + is allowed).")
    if gender not in GENDERS:
        errors.append("Please choose a gender.")
    visit = parse_date(visit_text)
    if visit is None:
        errors.append("Visit date is not a valid date.")
    elif visit > date.today():
        errors.append("Visit date cannot be in the future.")
    if not errors:
        same = patients[(patients["full_name"].str.lower() == name.lower()) & (patients["contact"] == contact)]
        if editing_id:
            same = same[same["patient_id"] != editing_id]
        if len(same):
            errors.append(f"This patient is already registered as {same.iloc[0]['patient_id']}.")
    data = {"full_name": name, "age": age_text, "contact": contact, "gender": gender, "visit_date": visit_text}
    return data, errors


def free_slots(apps: pd.DataFrame, doctor: str, day: str, ignore_id: str = None):
    busy = apps[(apps["doctor_name"] == doctor) & (apps["appointment_date"] == day) & (apps["status"] != "Cancelled")]
    if ignore_id:
        busy = busy[busy["appointment_id"] != ignore_id]
    return [s for s in TIME_SLOTS if s not in set(busy["appointment_time"])]


def validate_appointment(form, apps: pd.DataFrame, patients: pd.DataFrame, existing=None):
    errors = []
    editing_id = existing["appointment_id"] if existing is not None else None
    pid = form.get("patient_id", "")
    day = form.get("appointment_date", "").strip()
    slot = form.get("appointment_time", "")
    doctor = form.get("doctor_name", "")
    reason = " ".join(form.get("reason", "").split())
    status = form.get("status", "Scheduled") if existing is not None else "Scheduled"

    match = patients[patients["patient_id"] == pid]
    if match.empty:
        errors.append("Please select a registered patient.")
    appt_date = parse_date(day)
    if appt_date is None:
        errors.append("Appointment date is not a valid date.")
    if slot not in TIME_SLOTS:
        errors.append("Please choose a time slot from the list.")
    if doctor not in DOCTORS:
        errors.append("Please choose a doctor from the list.")
    if not (3 <= len(reason) <= 120):
        errors.append("Reason for visit must be 3-120 characters.")
    if status not in STATUSES:
        errors.append("Invalid appointment status.")

    if not errors and status == "Scheduled":
        changed = existing is None or (day, slot) != (existing["appointment_date"], existing["appointment_time"])
        if changed:
            now = datetime.now()
            if appt_date < now.date() or (appt_date == now.date() and slot < now.strftime("%H:%M")):
                errors.append("A scheduled appointment cannot be in the past.")
    if not errors and status != "Cancelled":
        # duplicate protection: same doctor + date + time (cancelled bookings free the slot)
        active = apps[(apps["status"] != "Cancelled") & (apps["appointment_date"] == day)
                      & (apps["appointment_time"] == slot)]
        if editing_id:
            active = active[active["appointment_id"] != editing_id]
        if len(active[active["doctor_name"] == doctor]):
            options = free_slots(apps, doctor, day, editing_id)
            hint = f" Free slots that day: {', '.join(options[:6])}." if options else " That doctor is fully booked that day."
            errors.append(f"{doctor} is already booked on {day} at {slot}.{hint}")
        elif len(active[active["patient_id"] == pid]):
            errors.append("This patient already has another appointment at the same date and time.")
    data = {"patient_id": pid, "patient_name": match.iloc[0]["full_name"] if len(match) else "",
            "appointment_date": day, "appointment_time": slot, "doctor_name": doctor,
            "reason": reason, "status": status}
    return data, errors


def validate_bill(form, patients: pd.DataFrame):
    errors = []
    pid = form.get("patient_id", "")
    fee = parse_money(form.get("consultation_fee", ""))
    extra = parse_money(form.get("additional_charges", "0") or "0")
    method = form.get("payment_method", "")
    status = form.get("payment_status", "")
    bill_text = form.get("bill_date", "").strip()

    if patients[patients["patient_id"] == pid].empty:
        errors.append("Please select a registered patient.")
    if fee is None or not (0 <= fee <= 100000):
        errors.append("Consultation fee must be a number between 0 and 100000.")
    if extra is None or not (0 <= extra <= 100000):
        errors.append("Additional charges must be a number between 0 and 100000.")
    if method not in PAY_METHODS:
        errors.append("Please choose a payment method.")
    if status not in PAY_STATUSES:
        errors.append("Please choose a payment status.")
    bill_date = parse_date(bill_text)
    if bill_date is None:
        errors.append("Bill date is not a valid date.")
    elif bill_date > date.today():
        errors.append("Bill date cannot be in the future.")
    total = round((fee or 0) + (extra or 0), 2)  # total is ALWAYS calculated by the server
    data = {"patient_id": pid, "consultation_fee": f"{fee or 0:.2f}", "additional_charges": f"{extra or 0:.2f}",
            "total_amount": f"{total:.2f}", "payment_method": method, "payment_status": status,
            "bill_date": bill_text}
    return data, errors


# ---------------------------------------------------------------------------
# 4. SECURITY HELPERS: login, roles, CSRF
# ---------------------------------------------------------------------------
_failed_logins = {}  # ip -> (count, first_time)


def current_user():
    name = session.get("user")
    return {"username": name, **USERS[name]} if name in USERS else None


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not current_user():
            flash("Please sign in to open the clinic dashboard.", "error")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapper


def admin_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        user = current_user()
        if not user:
            return redirect(url_for("login", next=request.path))
        if user["role"] != "admin":
            flash("Only the admin account can delete records.", "error")
            return redirect(request.referrer or url_for("dashboard"))
        return view(*args, **kwargs)
    return wrapper


@app.before_request
def csrf_protect():
    """Every POST form must carry the secret token we gave this browser session."""
    if request.method == "POST":
        token = session.get("_csrf")
        if not token or not secrets.compare_digest(token, request.form.get("_csrf", "")):
            abort(400)


@app.after_request
def security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "same-origin"
    if request.endpoint not in ("static", "intro"):
        resp.headers["Cache-Control"] = "no-store"  # do not keep patient pages in browser cache
    return resp


ICONS = {
    "dashboard": '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/>',
    "patients": '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "user": '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>',
    "billing": '<rect x="1" y="4" width="22" height="16" rx="2"/><line x1="1" y1="10" x2="23" y2="10"/>',
    "analytics": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "cross": '<path d="M9 3h6v6h6v6h-6v6H9v-6H3V9h6z"/>',
    "pulse": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "logout": '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/>',
    "search": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "edit": '<path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.12 2.12 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>',
    "trash": '<polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>',
    "eye": '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "x": '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "lock": '<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "arrow": '<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>',
    "back": '<line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/>',
    "plus": '<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>',
    "wallet": '<path d="M20 12V8H6a2 2 0 0 1 0-4h12v4"/><path d="M4 6v12a2 2 0 0 0 2 2h14v-4"/><path d="M18 12a2 2 0 0 0 0 4h4v-4z"/>',
    "stethoscope": '<path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6 6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/><path d="M8 15v1a6 6 0 0 0 6 6 6 6 0 0 0 6-6v-4"/><circle cx="20" cy="10" r="2"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
}


@app.template_global()
def icon(name: str, size: int = 18) -> Markup:
    return Markup(f'<svg class="icon" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
                  f'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
                  f'aria-hidden="true">{ICONS.get(name, "")}</svg>')


@app.template_global()
def page_url(page: int) -> str:
    args = request.args.to_dict()
    args["page"] = page
    return url_for(request.endpoint, **args)


@app.template_filter("money")
def money(value):
    try:
        return f"₹{float(value):,.2f}"
    except (ValueError, TypeError):
        return "—"


@app.template_filter("nicedate")
def nicedate(value):
    parsed = parse_date(str(value))
    return parsed.strftime("%d %b %Y") if parsed else (value or "—")


@app.template_filter("time12")
def time12(value):
    try:
        return datetime.strptime(str(value), "%H:%M").strftime("%I:%M %p").lstrip("0")
    except ValueError:
        return value or "—"


@app.context_processor
def inject_globals():
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_hex(16)
    return {"csrf_token": session["_csrf"], "user": current_user(), "today": date.today().isoformat(),
            "DOCTORS": DOCTORS, "TIME_SLOTS": TIME_SLOTS, "GENDERS": GENDERS, "STATUSES": STATUSES,
            "PAY_METHODS": PAY_METHODS, "PAY_STATUSES": PAY_STATUSES, "year": date.today().year}


@app.errorhandler(400)
def bad_request(_):
    return render_template("error.html", code=400, message="The form expired or was invalid. Please go back, refresh the page and try again."), 400


@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", code=404, message="We could not find that page or record."), 404


@app.errorhandler(500)
def server_error(_):
    return render_template("error.html", code=500, message="Something went wrong on our side. Check the terminal window for details."), 500


# ---------------------------------------------------------------------------
# 5. ROUTES
# ---------------------------------------------------------------------------
@app.route("/")
def intro():
    return render_template("intro.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    target = request.args.get("next", "")
    if not (target.startswith("/") and not target.startswith("//")):
        target = url_for("dashboard")
    if current_user():
        return redirect(target)
    if request.method == "POST":
        ip = request.remote_addr or "?"
        count, first = _failed_logins.get(ip, (0, time.time()))
        if time.time() - first > 300:
            count, first = 0, time.time()
        if count >= 5:
            flash("Too many failed attempts. Please wait 5 minutes and try again.", "error")
            return render_template("login.html"), 429
        username = request.form.get("username", "").strip().lower()
        account = USERS.get(username)
        if account and check_password_hash(account["hash"], request.form.get("password", "")):
            _failed_logins.pop(ip, None)
            session.clear()
            session.permanent = True
            session["user"] = username
            flash(f"Welcome back, {account['name']}!", "success")
            return redirect(target)
        _failed_logins[ip] = (count + 1, first)
        flash("Wrong username or password.", "error")
    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for("intro"))


# ---- Dashboard -------------------------------------------------------------
def bar_rows(labels, values):
    """Turn numbers into 0-100 percentages so CSS can draw bars (no JavaScript needed)."""
    top = max(values) if values and max(values) > 0 else 1
    return [{"label": l, "value": v, "pct": round(100 * v / top)} for l, v in zip(labels, values)]


def build_dashboard():
    pat, apps, bills = typed_patients(), typed_appointments(), typed_billing()
    today = date.today().isoformat()
    stats = {
        "patients": len(pat),
        "today": int(((apps["appointment_date"] == today) & (apps["status"] != "Cancelled")).sum()),
        "completed": int((apps["status"] == "Completed").sum()),
        "revenue": float(bills.loc[bills["payment_status"] == "Paid", "total_amount"].sum()),
        "pending": float(bills.loc[bills["payment_status"] == "Unpaid", "total_amount"].sum()),
    }
    recent = pat.sort_values(["visit_dt", "patient_id"], ascending=False).head(5)
    upcoming = apps[(apps["status"] == "Scheduled") & (apps["appointment_date"] >= today)] \
        .sort_values(["appointment_date", "appointment_time"]).head(6)

    months = pd.period_range(end=date.today(), periods=6, freq="M")
    paid = bills[bills["payment_status"] == "Paid"].dropna(subset=["bill_dt"])
    by_month = paid.groupby(paid["bill_dt"].dt.to_period("M"))["total_amount"].sum()
    revenue_bars = bar_rows([m.strftime("%b") for m in months], [float(by_month.get(m, 0)) for m in months])

    gender_counts = pat["gender"].value_counts()
    gender_bars = bar_rows(GENDERS, [int(gender_counts.get(g, 0)) for g in GENDERS])
    groups = pd.cut(pat["age"].dropna(), bins=[-1, 12, 19, 39, 59, 200],
                    labels=["0-12", "13-19", "20-39", "40-59", "60+"])
    age_counts = groups.value_counts()
    age_bars = bar_rows(["0-12", "13-19", "20-39", "40-59", "60+"],
                        [int(age_counts.get(g, 0)) for g in ["0-12", "13-19", "20-39", "40-59", "60+"]])
    return stats, recent.to_dict("records"), upcoming.to_dict("records"), revenue_bars, gender_bars, age_bars


@app.route("/dashboard")
@login_required
def dashboard():
    stats, recent, upcoming, revenue_bars, gender_bars, age_bars = build_dashboard()
    return render_template("dashboard.html", stats=stats, recent=recent, upcoming=upcoming,
                           revenue_bars=revenue_bars, gender_bars=gender_bars, age_bars=age_bars)


# ---- Patients --------------------------------------------------------------
@app.route("/patients", methods=["GET", "POST"])
@login_required
def patients():
    pat = read_table("patients")
    form = {"visit_date": date.today().isoformat()}
    if request.method == "POST":
        with _data_lock:
            pat = read_table("patients")
            data, errors = validate_patient(request.form, pat)
            if errors:
                for e in errors:
                    flash(e, "error")
                form = request.form
            else:
                new_id = next_id("P", pat, "patient_id")
                write_table("patients", pd.concat([pat, pd.DataFrame([{"patient_id": new_id, **data}])], ignore_index=True))
                flash(f"Patient {data['full_name']} was added with ID {new_id}.", "success")
                return redirect(url_for("patients"))
    q = request.args.get("q", "").strip()
    view = pat
    if q:
        mask = (pat["full_name"].str.contains(q, case=False, regex=False) | pat["patient_id"].str.contains(q, case=False, regex=False)
                | pat["contact"].str.contains(q, case=False, regex=False))
        view = pat[mask]
    view = view.sort_values("patient_id", ascending=False)
    rows, page, pages = paginate(view, request.args.get("page", 1, type=int))
    return render_template("patients.html", rows=rows.to_dict("records"), q=q, page=page, pages=pages,
                           total=len(view), form=form, new_id=next_id("P", pat, "patient_id"))


@app.route("/patients/<pid>")
@login_required
def patient_view(pid):
    pat = read_table("patients")
    match = pat[pat["patient_id"] == pid]
    if match.empty:
        abort(404)
    apps = read_table("appointments")
    bills = typed_billing()
    p_apps = apps[apps["patient_id"] == pid].sort_values(["appointment_date", "appointment_time"], ascending=False)
    p_bills = bills[bills["patient_id"] == pid].sort_values("bill_date", ascending=False)
    summary = {"billed": float(p_bills["total_amount"].sum()),
               "unpaid": float(p_bills.loc[p_bills["payment_status"] == "Unpaid", "total_amount"].sum())}
    return render_template("patient_view.html", p=match.iloc[0].to_dict(), appts=p_apps.to_dict("records"),
                           bills=p_bills.to_dict("records"), summary=summary)


@app.route("/patients/<pid>/edit", methods=["GET", "POST"])
@login_required
def patient_edit(pid):
    with _data_lock:
        pat = read_table("patients")
        match = pat[pat["patient_id"] == pid]
        if match.empty:
            abort(404)
        form = match.iloc[0].to_dict()
        if request.method == "POST":
            data, errors = validate_patient(request.form, pat, editing_id=pid)
            if errors:
                for e in errors:
                    flash(e, "error")
                form = {"patient_id": pid, **request.form.to_dict()}
            else:
                for k, v in data.items():
                    pat.loc[pat["patient_id"] == pid, k] = v
                write_table("patients", pat)
                apps = read_table("appointments")  # keep the copied patient name in sync
                apps.loc[apps["patient_id"] == pid, "patient_name"] = data["full_name"]
                write_table("appointments", apps)
                flash(f"Patient {pid} was updated.", "success")
                return redirect(url_for("patient_view", pid=pid))
    return render_template("patient_form.html", form=form, pid=pid)


@app.route("/patients/<pid>/delete", methods=["GET", "POST"])
@admin_required
def patient_delete(pid):
    pat = read_table("patients")
    match = pat[pat["patient_id"] == pid]
    if match.empty:
        abort(404)
    n_apps = int((read_table("appointments")["patient_id"] == pid).sum())
    n_bills = int((read_table("billing")["patient_id"] == pid).sum())
    if request.method == "POST":
        with _data_lock:
            write_table("patients", pat[pat["patient_id"] != pid])
            apps, bills = read_table("appointments"), read_table("billing")
            write_table("appointments", apps[apps["patient_id"] != pid])
            write_table("billing", bills[bills["patient_id"] != pid])
        flash(f"Patient {pid} and {n_apps} appointment(s) and {n_bills} bill(s) were deleted.", "success")
        return redirect(url_for("patients"))
    note = (f"This will also delete {n_apps} appointment(s) and {n_bills} bill(s) linked to this patient."
            if (n_apps or n_bills) else "This patient has no linked appointments or bills.")
    return render_template("confirm.html", title="Delete patient record",
                           message=f"Delete {match.iloc[0]['full_name']} ({pid})?", note=note,
                           cancel_url=url_for("patient_view", pid=pid))


# ---- Appointments ----------------------------------------------------------
@app.route("/appointments", methods=["GET", "POST"])
@login_required
def appointments():
    pat = read_table("patients")
    form = {"patient_id": request.args.get("patient_id", ""), "appointment_date": date.today().isoformat(),
            "appointment_time": "09:00"}
    if request.method == "POST":
        with _data_lock:
            apps = read_table("appointments")
            data, errors = validate_appointment(request.form, apps, pat)
            if errors:
                for e in errors:
                    flash(e, "error")
                form = request.form
            else:
                new_id = next_id("A", apps, "appointment_id")
                write_table("appointments", pd.concat([apps, pd.DataFrame([{"appointment_id": new_id, **data}])], ignore_index=True))
                flash(f"Appointment {new_id} booked for {data['patient_name']} with {data['doctor_name']}.", "success")
                return redirect(url_for("appointments"))
    apps = read_table("appointments")
    q = request.args.get("q", "").strip()
    status = request.args.get("status", "")
    doctor = request.args.get("doctor", "")
    day = request.args.get("day", "").strip()
    view = apps
    if q:
        view = view[view["patient_name"].str.contains(q, case=False, regex=False)
                    | view["patient_id"].str.contains(q, case=False, regex=False)
                    | view["appointment_id"].str.contains(q, case=False, regex=False)]
    if status in STATUSES:
        view = view[view["status"] == status]
    if doctor in DOCTORS:
        view = view[view["doctor_name"] == doctor]
    if parse_date(day):
        view = view[view["appointment_date"] == day]
    view = view.sort_values(["appointment_date", "appointment_time"], ascending=False)
    rows, page, pages = paginate(view, request.args.get("page", 1, type=int))
    return render_template("appointments.html", rows=rows.to_dict("records"), patients=pat.sort_values("full_name").to_dict("records"),
                           q=q, status=status, doctor=doctor, day=day, page=page, pages=pages, total=len(view), form=form)


@app.route("/appointments/<aid>/edit", methods=["GET", "POST"])
@login_required
def appointment_edit(aid):
    with _data_lock:
        apps = read_table("appointments")
        match = apps[apps["appointment_id"] == aid]
        if match.empty:
            abort(404)
        existing = match.iloc[0]
        pat = read_table("patients")
        form = existing.to_dict()
        if request.method == "POST":
            data, errors = validate_appointment(request.form, apps, pat, existing=existing)
            if errors:
                for e in errors:
                    flash(e, "error")
                form = {"appointment_id": aid, **request.form.to_dict()}
            else:
                for k, v in data.items():
                    apps.loc[apps["appointment_id"] == aid, k] = v
                write_table("appointments", apps)
                flash(f"Appointment {aid} was updated.", "success")
                return redirect(url_for("appointments"))
    return render_template("appointment_form.html", form=form, aid=aid, patients=pat.sort_values("full_name").to_dict("records"))


@app.route("/appointments/<aid>/status", methods=["POST"])
@login_required
def appointment_status(aid):
    new_status = request.form.get("status", "")
    if new_status not in ("Completed", "Cancelled"):
        abort(400)
    with _data_lock:
        apps = read_table("appointments")
        if apps[apps["appointment_id"] == aid].empty:
            abort(404)
        apps.loc[apps["appointment_id"] == aid, "status"] = new_status
        write_table("appointments", apps)
    flash(f"Appointment {aid} marked as {new_status.lower()}.", "success")
    return redirect(request.form.get("back") if (request.form.get("back", "").startswith("/")
                                                 and not request.form.get("back", "").startswith("//"))
                    else url_for("appointments"))


@app.route("/appointments/<aid>/delete", methods=["GET", "POST"])
@admin_required
def appointment_delete(aid):
    apps = read_table("appointments")
    match = apps[apps["appointment_id"] == aid]
    if match.empty:
        abort(404)
    if request.method == "POST":
        write_table("appointments", apps[apps["appointment_id"] != aid])
        flash(f"Appointment {aid} was deleted.", "success")
        return redirect(url_for("appointments"))
    r = match.iloc[0]
    return render_template("confirm.html", title="Delete appointment",
                           message=f"Delete appointment {aid} for {r['patient_name']} on {nicedate(r['appointment_date'])}?",
                           note="Tip: use Cancel instead if you only want to free the time slot.",
                           cancel_url=url_for("appointments"))


# ---- Billing ---------------------------------------------------------------
@app.route("/billing", methods=["GET", "POST"])
@login_required
def billing():
    pat = read_table("patients")
    form = {"bill_date": date.today().isoformat(), "consultation_fee": str(DEFAULT_FEE), "additional_charges": "0",
            "payment_status": "Unpaid", "patient_id": request.args.get("patient_id", "")}
    if request.method == "POST":
        with _data_lock:
            bills = read_table("billing")
            data, errors = validate_bill(request.form, pat)
            if errors:
                for e in errors:
                    flash(e, "error")
                form = request.form
            else:
                new_id = next_id("B", bills, "bill_id")
                write_table("billing", pd.concat([bills, pd.DataFrame([{"bill_id": new_id, **data}])], ignore_index=True))
                flash(f"Bill {new_id} created. Total amount: {money(data['total_amount'])}.", "success")
                return redirect(url_for("billing"))
    bills = typed_billing()
    names = dict(zip(pat["patient_id"], pat["full_name"]))
    bills["patient_name"] = bills["patient_id"].map(names).fillna("Unknown patient")
    totals = {"billed": float(bills["total_amount"].sum()),
              "paid": float(bills.loc[bills["payment_status"] == "Paid", "total_amount"].sum()),
              "unpaid": float(bills.loc[bills["payment_status"] == "Unpaid", "total_amount"].sum())}
    status = request.args.get("status", "")
    q = request.args.get("q", "").strip()
    view = bills
    if status in PAY_STATUSES:
        view = view[view["payment_status"] == status]
    if q:
        view = view[view["patient_name"].str.contains(q, case=False, regex=False)
                    | view["patient_id"].str.contains(q, case=False, regex=False)
                    | view["bill_id"].str.contains(q, case=False, regex=False)]
    view = view.sort_values(["bill_date", "bill_id"], ascending=False)
    rows, page, pages = paginate(view, request.args.get("page", 1, type=int))
    return render_template("billing.html", rows=rows.to_dict("records"), patients=pat.sort_values("full_name").to_dict("records"),
                           totals=totals, status=status, q=q, page=page, pages=pages, total=len(view), form=form)


@app.route("/billing/<bid>/edit", methods=["GET", "POST"])
@login_required
def bill_edit(bid):
    with _data_lock:
        bills = read_table("billing")
        match = bills[bills["bill_id"] == bid]
        if match.empty:
            abort(404)
        pat = read_table("patients")
        form = match.iloc[0].to_dict()
        if request.method == "POST":
            data, errors = validate_bill(request.form, pat)
            if errors:
                for e in errors:
                    flash(e, "error")
                form = {"bill_id": bid, **request.form.to_dict()}
            else:
                for k, v in data.items():
                    bills.loc[bills["bill_id"] == bid, k] = v
                write_table("billing", bills)
                flash(f"Bill {bid} was updated. New total: {money(data['total_amount'])}.", "success")
                return redirect(url_for("billing"))
    return render_template("bill_form.html", form=form, bid=bid, patients=pat.sort_values("full_name").to_dict("records"))


@app.route("/billing/<bid>/toggle", methods=["POST"])
@login_required
def bill_toggle(bid):
    with _data_lock:
        bills = read_table("billing")
        match = bills["bill_id"] == bid
        if not match.any():
            abort(404)
        new = "Unpaid" if bills.loc[match, "payment_status"].iloc[0].title() == "Paid" else "Paid"
        bills.loc[match, "payment_status"] = new
        write_table("billing", bills)
    flash(f"Bill {bid} marked as {new.lower()}.", "success")
    back = request.form.get("back", "")
    return redirect(back if back.startswith("/") and not back.startswith("//") else url_for("billing"))


@app.route("/billing/<bid>/delete", methods=["GET", "POST"])
@admin_required
def bill_delete(bid):
    bills = read_table("billing")
    match = bills[bills["bill_id"] == bid]
    if match.empty:
        abort(404)
    if request.method == "POST":
        write_table("billing", bills[bills["bill_id"] != bid])
        flash(f"Bill {bid} was deleted.", "success")
        return redirect(url_for("billing"))
    return render_template("confirm.html", title="Delete bill", message=f"Delete bill {bid} ({money(match.iloc[0]['total_amount'])})?",
                           note="This cannot be undone.", cancel_url=url_for("billing"))


# ---- Analytics -------------------------------------------------------------
@app.route("/analytics")
@login_required
def analytics():
    with _plot_lock:
        summary, monthly_table = generate_charts()
    return render_template("analytics.html", s=summary, table=monthly_table, v=int(time.time()))


# ---------------------------------------------------------------------------
# 6. ANALYTICS: Pandas + NumPy numbers, Matplotlib + Seaborn charts
#    Only aggregated, non-identifying numbers are used (no names, no phone numbers).
# ---------------------------------------------------------------------------
GREEN, DARK, MINT, BLUE, CORAL, AMBER = "#15966A", "#0E6B4B", "#8FD3B6", "#3A8FB7", "#E07A5F", "#E9A93B"
STATUS_COLORS = {"Completed": GREEN, "Scheduled": BLUE, "Cancelled": CORAL, "Paid": GREEN, "Unpaid": AMBER}
FIG = (7.2, 3.8)


def _save(fig, name):
    for ax in fig.axes:  # remove automatic axis names such as "label" or "None"
        if ax.get_xlabel() in {"label", "status", "month", "None"}:
            ax.set_xlabel("")
    fig.savefig(CHART_DIR / name, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _new_chart(title, xlabel="", ylabel=""):
    fig, ax = plt.subplots(figsize=FIG)
    # (the page already shows the chart title above the picture)
    ax.set_xlabel(xlabel, color="#5b6f67")
    ax.set_ylabel(ylabel, color="#5b6f67")
    sns.despine(ax=ax)
    return fig, ax


def _no_data(name, title):
    fig, ax = _new_chart(title)
    ax.text(0.5, 0.5, "No data available yet", ha="center", va="center", color="#6b7f77", fontsize=12, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    _save(fig, name)


def _label_bars(ax, fmt="{:g}"):
    for c in ax.containers:
        ax.bar_label(c, labels=[fmt.format(v) if v else "" for v in c.datavalues], fontsize=8, padding=2, color="#24352F")


def _safe(name, title, func, *args):
    """Draw one chart; if anything fails the page still loads with a placeholder picture."""
    try:
        func(*args)
    except Exception:  # noqa: BLE001 - we want the page to survive any chart problem
        log.exception("Chart %s failed", name)
        plt.close("all")
        _no_data(name, title)


def generate_charts():
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", rc={"grid.color": "#e3efe9", "axes.edgecolor": "#c8dcd2", "font.size": 10})
    pat, apps, bills = typed_patients(), typed_appointments(), typed_billing()
    apps_ok = apps.dropna(subset=["appt_dt"]).copy()
    apps_ok["month"] = apps_ok["appt_dt"].dt.to_period("M")
    paid = bills[(bills["payment_status"] == "Paid")].dropna(subset=["bill_dt", "total_amount"]).copy()
    paid["month"] = paid["bill_dt"].dt.to_period("M")
    today = pd.Timestamp(date.today())

    # ---- chart 1: appointments per month, split by status (Seaborn bar chart)
    def monthly_appointments():
        if apps_ok.empty:
            return _no_data("appointments_monthly.png", "Appointments per month")
        counts = apps_ok.groupby(["month", "status"]).size().reset_index(name="count")
        months = sorted(counts["month"].unique())[-12:]
        counts = counts[counts["month"].isin(months)]
        counts["label"] = counts["month"].dt.strftime("%b %Y")
        fig, ax = _new_chart("Appointments per month", "", "Appointments")
        sns.barplot(data=counts, x="label", y="count", hue="status", palette=STATUS_COLORS,
                    order=[m.strftime("%b %Y") for m in months], ax=ax)
        _label_bars(ax)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.2)
        ax.legend(title="", frameon=False, loc="upper left", ncol=3)
        ax.tick_params(axis="x", rotation=30)
        _save(fig, "appointments_monthly.png")

    # ---- chart 2: daily appointments, last 30 days + 7-day moving average (NumPy)
    def daily_appointments():
        days = pd.date_range(today - pd.Timedelta(days=29), today)
        series = apps_ok[apps_ok["status"] != "Cancelled"].groupby("appt_dt").size().reindex(days, fill_value=0)
        if apps_ok.empty:
            return _no_data("appointments_daily.png", "Daily appointments (last 30 days)")
        values = series.to_numpy(dtype=float)
        moving = np.convolve(values, np.ones(7) / 7, mode="valid")
        fig, ax = _new_chart("Daily appointments (last 30 days)", "", "Appointments")
        sns.lineplot(x=days, y=values, color=GREEN, marker="o", markersize=4, linewidth=1.8, label="Daily", ax=ax)
        ax.plot(days[6:], moving, color=DARK, linestyle="--", linewidth=2, label="7-day average")
        ax.fill_between(days, values, color=MINT, alpha=0.25)
        ax.set_ylim(0, max(values.max(), 1) * 1.3)
        ax.legend(frameon=False, loc="upper left")
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
        ax.tick_params(axis="x", rotation=30)
        _save(fig, "appointments_daily.png")

    # ---- chart 3: completed / scheduled / cancelled
    def status_chart():
        if apps_ok.empty:
            return _no_data("appointment_status.png", "Appointment outcomes")
        counts = apps_ok["status"].value_counts().reindex(STATUSES, fill_value=0)
        fig, ax = _new_chart("Appointment outcomes", "", "Appointments")
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=STATUS_COLORS, legend=False, ax=ax)
        _label_bars(ax)
        _save(fig, "appointment_status.png")

    # ---- chart 4: monthly revenue (bar + line)
    def revenue_chart():
        if paid.empty:
            return _no_data("revenue_monthly.png", "Monthly clinic revenue (paid bills)")
        months = pd.period_range(end=date.today(), periods=6, freq="M")
        rev = paid.groupby("month")["total_amount"].sum().reindex(months, fill_value=0)
        labels = [m.strftime("%b %Y") for m in months]
        fig, ax = _new_chart("Monthly clinic revenue (paid bills)", "", "Revenue (₹)")
        sns.barplot(x=labels, y=rev.to_numpy(), color=MINT, ax=ax)
        ax.plot(range(len(labels)), rev.to_numpy(), color=DARK, marker="o", linewidth=2)
        ax.bar_label(ax.containers[0], labels=[f"₹{v:,.0f}" for v in rev.to_numpy()], fontsize=8, padding=3)
        ax.tick_params(axis="x", rotation=20)
        _save(fig, "revenue_monthly.png")

    # ---- chart 5: paid vs unpaid
    def payment_chart():
        valid = bills.dropna(subset=["total_amount"])
        if valid.empty:
            return _no_data("payments.png", "Paid vs unpaid bills")
        grouped = valid.groupby("payment_status")["total_amount"].agg(["sum", "count"]).reindex(PAY_STATUSES, fill_value=0)
        fig, ax = _new_chart("Paid vs unpaid bills", "", "Amount (₹)")
        bars = ax.bar(list(grouped.index), grouped["sum"].to_numpy(), color=[STATUS_COLORS[s] for s in grouped.index], width=0.55)
        ax.bar_label(bars, labels=[f"₹{s:,.0f}\n({int(c)} bills)" for s, c in zip(grouped["sum"], grouped["count"])],
                     fontsize=9, padding=3)
        ax.margins(y=0.2)
        _save(fig, "payments.png")

    # ---- chart 6: patient visit trends
    def visit_trend():
        visits = pat.dropna(subset=["visit_dt"]).assign(month=lambda d: d["visit_dt"].dt.to_period("M"))
        done = apps_ok[apps_ok["status"] == "Completed"]
        if visits.empty and done.empty:
            return _no_data("visit_trend.png", "Patient visit trends")
        all_months = pd.period_range(end=date.today(), periods=8, freq="M")
        frame = pd.DataFrame({
            "month": [m.strftime("%b %Y") for m in all_months] * 2,
            "count": list(visits.groupby("month").size().reindex(all_months, fill_value=0)) +
                     list(done.groupby("month").size().reindex(all_months, fill_value=0)),
            "type": ["Latest patient visit date"] * len(all_months) + ["Completed appointments"] * len(all_months)})
        fig, ax = _new_chart("Patient visit trends", "", "Count")
        sns.lineplot(data=frame, x="month", y="count", hue="type", marker="o", palette=[BLUE, GREEN], linewidth=2.2, ax=ax)
        ax.set_ylim(0, max(frame["count"].max(), 1) * 1.25)
        ax.legend(title="", frameon=False, loc="upper left")
        ax.tick_params(axis="x", rotation=30)
        _save(fig, "visit_trend.png")

    # ---- chart 7: age group x gender heatmap (aggregated counts only)
    def heatmap():
        ages = pat.dropna(subset=["age"]).copy()
        if ages.empty:
            return _no_data("age_gender_heatmap.png", "Patients by age group and gender")
        ages["Age group"] = pd.cut(ages["age"], bins=[-1, 12, 19, 39, 59, 200], labels=["0-12", "13-19", "20-39", "40-59", "60+"])
        table = pd.crosstab(ages["Age group"], ages["gender"], dropna=False).reindex(columns=GENDERS, fill_value=0)
        fig, ax = _new_chart("Patients by age group and gender")
        sns.heatmap(table, annot=True, fmt="d", cmap=sns.light_palette(GREEN, as_cmap=True), linewidths=2,
                    linecolor="white", cbar=False, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        _save(fig, "age_gender_heatmap.png")

    for fname, title, fn in [
        ("appointments_monthly.png", "Appointments per month", monthly_appointments),
        ("appointments_daily.png", "Daily appointments", daily_appointments),
        ("appointment_status.png", "Appointment outcomes", status_chart),
        ("revenue_monthly.png", "Monthly revenue", revenue_chart),
        ("payments.png", "Paid vs unpaid", payment_chart),
        ("visit_trend.png", "Patient visit trends", visit_trend),
        ("age_gender_heatmap.png", "Age and gender", heatmap),
    ]:
        _safe(fname, title, fn)

    # ---- summary numbers (NumPy) -------------------------------------------
    amounts = bills["total_amount"].dropna().to_numpy(dtype=float)
    status_counts = apps_ok["status"].value_counts()
    total_appts = int(len(apps_ok))
    paid_total = float(bills.loc[bills["payment_status"] == "Paid", "total_amount"].sum())
    unpaid_total = float(bills.loc[bills["payment_status"] == "Unpaid", "total_amount"].sum())
    weekday_counts = apps_ok.groupby(apps_ok["appt_dt"].dt.dayofweek).size().reindex(range(7), fill_value=0).to_numpy()
    last30 = apps_ok[(apps_ok["appt_dt"] > today - pd.Timedelta(days=30)) & (apps_ok["appt_dt"] <= today)
                     & (apps_ok["status"] != "Cancelled")]
    summary = {
        "patients": int(len(pat)),
        "appointments": total_appts,
        "completed": int(status_counts.get("Completed", 0)),
        "scheduled": int(status_counts.get("Scheduled", 0)),
        "cancelled": int(status_counts.get("Cancelled", 0)),
        "cancel_rate": round(100 * status_counts.get("Cancelled", 0) / total_appts, 1) if total_appts else 0,
        "avg_per_day": round(float(np.mean(last30.groupby("appt_dt").size().reindex(
            pd.date_range(today - pd.Timedelta(days=29), today), fill_value=0))), 1) if total_appts else 0,
        "busiest_day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][int(np.argmax(weekday_counts))] if weekday_counts.sum() else "—",
        "avg_bill": float(np.mean(amounts)) if len(amounts) else 0,
        "median_bill": float(np.median(amounts)) if len(amounts) else 0,
        "p90_bill": float(np.percentile(amounts, 90)) if len(amounts) else 0,
        "paid_total": paid_total, "unpaid_total": unpaid_total,
        "paid_count": int((bills["payment_status"] == "Paid").sum()),
        "unpaid_count": int((bills["payment_status"] == "Unpaid").sum()),
        "collection_rate": round(100 * paid_total / (paid_total + unpaid_total), 1) if (paid_total + unpaid_total) else 0,
        "stamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
    }
    # table: appointments per month by status (last 12 months)
    table = []
    if not apps_ok.empty:
        pivot = pd.crosstab(apps_ok["month"], apps_ok["status"]).reindex(columns=STATUSES, fill_value=0).sort_index().tail(12)
        for month, row in pivot.iterrows():
            table.append({"month": month.strftime("%B %Y"), "total": int(row.sum()), **{k.lower(): int(row[k]) for k in STATUSES}})
        table.reverse()
    return summary, table


if __name__ == "__main__":
    for key in FILES:  # make sure every CSV exists before the first request
        read_table(key)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    print("\n  Smart Clinic Management & Analytics System")
    print("  Open this address in your browser:  http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=os.environ.get("CLINIC_DEBUG") == "1")
