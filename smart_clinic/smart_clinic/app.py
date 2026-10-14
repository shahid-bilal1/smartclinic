"""
app.py  -  Smart Clinic Management & Analytics System
Run:  python app.py   ->   http://127.0.0.1:5000
"""
from flask import Flask, render_template
import numpy as np
from data_generator import load_data
from charts import build_all, charts_exist

app = Flask(__name__)

# ============================================================
#  TEAM CONFIG  -  yahan sirf apne links aur photo ke naam badlo
#  Photos ko static/img/ folder me rakho (murtaza.jpg, momin.jpg)
# ============================================================
TEAM = [
    {
        "name": "Murtaza",
        "role": "Co-developer",
        "photo": "murtaza.jpg",
        "portfolio": "https://YOUR-MURTAZA-PORTFOLIO-LINK",   # <-- apna portfolio link yahan
        "about": "Backend logic, data generation aur analytics charts par kaam kiya.",
    },
    {
        "name": "Momin",
        "role": "Co-developer",
        "photo": "momin.jpg",
        "portfolio": "https://YOUR-MOMIN-PORTFOLIO-LINK",     # <-- apna portfolio link yahan
        "about": "Landing page design, Flask routes aur project documentation par kaam kiya.",
    },
]

# ---------- Data + charts (pehli bar run par khud ban jayenge) ----------
patients, doctors, appointments, billing = load_data()
if not charts_exist():
    build_all(patients, doctors, appointments, billing)


def compute_kpis():
    done = appointments[appointments["status"] == "Completed"]
    pending = billing.loc[billing["payment_status"] == "Pending", "total_amount"].sum()
    return {
        "patients": f"{len(patients):,}",
        "appointments": f"{len(appointments):,}",
        "revenue": f"Rs. {billing['total_amount'].sum() / 100000:.1f} L",
        "completion": f"{len(done) / len(appointments) * 100:.0f}%",
        "avg_bill": f"Rs. {int(np.mean(billing['total_amount'])):,}",
        "pending": f"Rs. {pending / 1000:.0f}k",
        "top_dept": appointments["department"].value_counts().idxmax(),
        "doctors": len(doctors),
    }


@app.route("/")
def home():
    recent = (appointments.merge(patients[["patient_id", "name"]], on="patient_id")
              .tail(4)[["name", "department", "status"]].to_dict("records"))
    return render_template("index.html", kpi=compute_kpis(), team=TEAM, recent=recent)


if __name__ == "__main__":
    app.run(debug=True)
