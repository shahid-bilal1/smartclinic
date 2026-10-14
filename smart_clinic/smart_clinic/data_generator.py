"""
data_generator.py
Smart Clinic Management & Analytics System
Yeh file numpy + pandas se apna khud ka dataset generate karti hai
(patients, doctors, appointments, billing) aur data/ folder me CSV save karti hai.
"""
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
rng = np.random.default_rng(42)  # fixed seed = har bar same data

DEPARTMENTS = ["General Medicine", "Pediatrics", "Cardiology", "Dermatology",
               "Orthopedics", "Gynecology", "ENT", "Dentistry"]
FEES = {"General Medicine": 500, "Pediatrics": 600, "Cardiology": 1500,
        "Dermatology": 900, "Orthopedics": 1200, "Gynecology": 1000,
        "ENT": 800, "Dentistry": 700}
DISEASES = ["Fever", "Diabetes", "Hypertension", "Asthma", "Skin Allergy",
            "Back Pain", "Migraine", "Infection", "Checkup", "Dental Pain"]
MALE = ["Ali", "Ahmed", "Hamza", "Usman", "Bilal", "Zain", "Hassan", "Saad", "Faisal", "Imran"]
FEMALE = ["Ayesha", "Fatima", "Sana", "Hina", "Maryam", "Zoya", "Noor", "Iqra", "Areeba", "Laiba"]
SURNAMES = ["Khan", "Sheikh", "Malik", "Raza", "Qureshi", "Butt", "Siddiqui", "Ansari", "Mir", "Chaudhry"]
CITIES = ["Kathua", "Jammu", "Samba", "Udhampur", "Srinagar", "Pathankot"]


def generate_all(n_patients=600, n_appointments=3500):
    os.makedirs(DATA_DIR, exist_ok=True)

    # ---------------- Doctors ----------------
    doctors = pd.DataFrame({
        "doctor_id": [f"D{str(i + 1).zfill(2)}" for i in range(len(DEPARTMENTS))],
        "doctor_name": [f"Dr. {rng.choice(MALE + FEMALE)} {rng.choice(SURNAMES)}" for _ in DEPARTMENTS],
        "department": DEPARTMENTS,
        "consultation_fee": [FEES[d] for d in DEPARTMENTS],
    })

    # ---------------- Patients ----------------
    gender = rng.choice(["Male", "Female"], n_patients, p=[0.48, 0.52])
    names = [f"{rng.choice(MALE if g == 'Male' else FEMALE)} {rng.choice(SURNAMES)}" for g in gender]
    age = np.clip(rng.normal(36, 18, n_patients), 1, 90).astype(int)
    patients = pd.DataFrame({
        "patient_id": [f"P{str(i + 1).zfill(4)}" for i in range(n_patients)],
        "name": names,
        "gender": gender,
        "age": age,
        "city": rng.choice(CITIES, n_patients, p=[0.35, 0.25, 0.12, 0.1, 0.1, 0.08]),
        "phone": ["9" + "".join(map(str, rng.integers(0, 10, 9))) for _ in range(n_patients)],
        "registered_on": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 600, n_patients), unit="D"),
    })

    # ---------------- Appointments ----------------
    dates = pd.to_datetime("2025-10-01") + pd.to_timedelta(rng.integers(0, 365, n_appointments), unit="D")
    # kuch mahino me zyada rush (seasonal flu etc.)
    month_boost = {1: 1.3, 2: 1.1, 7: 1.2, 8: 1.25, 9: 1.2, 12: 1.15}
    keep = np.array([rng.random() < month_boost.get(d.month, 1.0) / 1.3 for d in dates])
    dates = dates[keep]
    n = len(dates)
    hours = np.clip(rng.normal(13, 3.2, n), 9, 20).astype(int)
    dept = rng.choice(DEPARTMENTS, n, p=[0.28, 0.14, 0.1, 0.1, 0.1, 0.1, 0.09, 0.09])
    status = rng.choice(["Completed", "Cancelled", "No-Show"], n, p=[0.82, 0.11, 0.07])
    appointments = pd.DataFrame({
        "appointment_id": [f"A{str(i + 1).zfill(5)}" for i in range(n)],
        "patient_id": rng.choice(patients["patient_id"], n),
        "department": dept,
        "date": dates.normalize(),
        "hour": hours,
        "diagnosis": rng.choice(DISEASES, n),
        "status": status,
    }).sort_values("date").reset_index(drop=True)
    appointments = appointments.merge(doctors[["department", "doctor_id"]], on="department")
    appointments = appointments.sort_values("date").reset_index(drop=True)

    # ---------------- Billing ----------------
    done = appointments[appointments["status"] == "Completed"].copy()
    fee = done["department"].map(FEES)
    lab = rng.choice([0, 0, 300, 600, 1200], len(done))
    medicine = rng.integers(100, 1500, len(done))
    billing = pd.DataFrame({
        "bill_id": [f"B{str(i + 1).zfill(5)}" for i in range(len(done))],
        "appointment_id": done["appointment_id"].values,
        "date": done["date"].values,
        "consultation_fee": fee.values,
        "lab_charges": lab,
        "medicine_charges": medicine,
    })
    billing["total_amount"] = billing[["consultation_fee", "lab_charges", "medicine_charges"]].sum(axis=1)
    billing["payment_method"] = rng.choice(["Cash", "UPI", "Card", "Insurance"], len(billing), p=[0.35, 0.4, 0.15, 0.1])
    billing["payment_status"] = rng.choice(["Paid", "Pending"], len(billing), p=[0.92, 0.08])

    # ---------------- Save ----------------
    patients.to_csv(os.path.join(DATA_DIR, "patients.csv"), index=False)
    doctors.to_csv(os.path.join(DATA_DIR, "doctors.csv"), index=False)
    appointments.to_csv(os.path.join(DATA_DIR, "appointments.csv"), index=False)
    billing.to_csv(os.path.join(DATA_DIR, "billing.csv"), index=False)
    return patients, doctors, appointments, billing


def load_data():
    """CSV maujood na ho to naya generate karo, warna load karo."""
    if not os.path.exists(os.path.join(DATA_DIR, "billing.csv")):
        return generate_all()
    p = pd.read_csv(os.path.join(DATA_DIR, "patients.csv"), parse_dates=["registered_on"])
    d = pd.read_csv(os.path.join(DATA_DIR, "doctors.csv"))
    a = pd.read_csv(os.path.join(DATA_DIR, "appointments.csv"), parse_dates=["date"])
    b = pd.read_csv(os.path.join(DATA_DIR, "billing.csv"), parse_dates=["date"])
    return p, d, a, b


if __name__ == "__main__":
    p, d, a, b = generate_all()
    print(f"Patients: {len(p)} | Doctors: {len(d)} | Appointments: {len(a)} | Bills: {len(b)}")
