"""
Creates FICTIONAL demonstration data for the Smart Clinic system.

Run:  python generate_sample_data.py
This (re)writes patients.csv, appointments.csv and billing.csv.
All names and phone numbers are made up. Do NOT put real patient data in this project.
"""
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
random.seed(42)  # same data every time

NAMES = [
    ("Aarav Sharma", "Male"), ("Diya Patel", "Female"), ("Rohan Mehta", "Male"),
    ("Ananya Iyer", "Female"), ("Vihaan Gupta", "Male"), ("Ishita Nair", "Female"),
    ("Kabir Khan", "Male"), ("Meera Joshi", "Female"), ("Arjun Reddy", "Male"),
    ("Saanvi Kapoor", "Female"), ("Aditya Verma", "Male"), ("Zoya Ansari", "Female"),
    ("Dev Malhotra", "Male"), ("Kavya Singh", "Female"), ("Reyansh Das", "Male"),
    ("Nisha Rao", "Female"), ("Yash Choudhary", "Male"), ("Tara Bose", "Female"),
    ("Imran Shaikh", "Male"), ("Pooja Desai", "Female"), ("Harsh Vora", "Male"),
    ("Riya Menon", "Female"), ("Sameer Qureshi", "Male"), ("Alia Fernandes", "Female"),
    ("Neel Kulkarni", "Male"), ("Sana Merchant", "Female"), ("Omkar Patil", "Male"),
    ("Leena Thomas", "Female"), ("Manav Arora", "Male"), ("Farah Siddiqui", "Female"),
]
DOCTORS = ["Dr. Anita Rao", "Dr. Faisal Ahmed", "Dr. Priya Nambiar", "Dr. Rakesh Kulkarni"]
REASONS = [
    "General check-up", "Follow-up visit", "Fever and cold", "Routine consultation",
    "Blood pressure check", "Vaccination", "Skin consultation", "Health screening",
    "Back pain consultation", "Diet counselling",
]
SLOTS = [f"{h:02d}:{m:02d}" for h in range(9, 17) for m in (0, 30)]
METHODS = ["Cash", "Card", "UPI", "Insurance"]

today = date.today()


def make_patients():
    rows = []
    for i, (name, gender) in enumerate(NAMES, start=1):
        age = random.choice([4, 9, 15, 17, 22, 27, 31, 36, 42, 48, 55, 61, 67, 73])
        phone = "9" + "".join(str(random.randint(0, 9)) for _ in range(9))
        visit = today - timedelta(days=random.randint(0, 150))
        rows.append([f"P{i:03d}", name, age, phone, gender, visit.isoformat()])
    return pd.DataFrame(rows, columns=["patient_id", "full_name", "age", "contact", "gender", "visit_date"])


def make_appointments(patients):
    rows, used = [], set()
    n = 1

    def add(day, status):
        nonlocal n
        for _ in range(50):  # find a free slot
            doctor, slot = random.choice(DOCTORS), random.choice(SLOTS)
            if (doctor, day.isoformat(), slot) not in used:
                used.add((doctor, day.isoformat(), slot))
                p = patients.sample(1, random_state=random.randint(0, 10_000)).iloc[0]
                rows.append([f"A{n:03d}", p.patient_id, p.full_name, day.isoformat(), slot,
                             doctor, random.choice(REASONS), status])
                n += 1
                return

    for back in range(1, 150):  # past appointments
        if random.random() < 0.55:
            day = today - timedelta(days=back)
            if day.weekday() == 6:
                continue
            add(day, random.choices(["Completed", "Cancelled"], weights=[85, 15])[0])
    for _ in range(4):  # today
        add(today, random.choice(["Scheduled", "Completed"]))
    for ahead in range(1, 12):  # upcoming
        if random.random() < 0.6:
            add(today + timedelta(days=ahead), "Scheduled")
    df = pd.DataFrame(rows, columns=["appointment_id", "patient_id", "patient_name", "appointment_date",
                                     "appointment_time", "doctor_name", "reason", "status"])
    return df.sort_values(["appointment_date", "appointment_time"]).reset_index(drop=True).assign(
        appointment_id=lambda d: [f"A{i:03d}" for i in range(1, len(d) + 1)])


def make_billing(appointments):
    rows = []
    done = appointments[appointments["status"] == "Completed"].reset_index(drop=True)
    for i, a in enumerate(done.itertuples(), start=1):
        fee = random.choice([300, 400, 500, 600, 800])
        extra = random.choice([0, 0, 0, 100, 150, 250, 400])
        paid = random.random() < 0.8 or a.appointment_date < (today - timedelta(days=60)).isoformat()
        rows.append([f"B{i:03d}", a.patient_id, f"{fee:.2f}", f"{extra:.2f}", f"{fee + extra:.2f}",
                     random.choice(METHODS), "Paid" if paid else "Unpaid", a.appointment_date])
    return pd.DataFrame(rows, columns=["bill_id", "patient_id", "consultation_fee", "additional_charges",
                                       "total_amount", "payment_method", "payment_status", "bill_date"])


if __name__ == "__main__":
    pat = make_patients()
    app_ = make_appointments(pat)
    bil = make_billing(app_)
    pat.to_csv(BASE / "patients.csv", index=False)
    app_.to_csv(BASE / "appointments.csv", index=False)
    bil.to_csv(BASE / "billing.csv", index=False)
    print(f"Created {len(pat)} patients, {len(app_)} appointments, {len(bil)} bills.")
