"""
charts.py
matplotlib + seaborn se analytics charts banata hai aur static/charts/ me PNG save karta hai.
"""
import os
import matplotlib
matplotlib.use("Agg")  # server ke liye (window nahi kholta)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "static", "charts")

TEAL, DEEP, MINT, RED = "#0E7C6B", "#0B3D36", "#CFEBE5", "#E5484D"

sns.set_theme(style="whitegrid", rc={
    "axes.edgecolor": "#CFE3DE", "grid.color": "#E3EFEB", "axes.labelcolor": DEEP,
    "xtick.color": DEEP, "ytick.color": DEEP, "text.color": DEEP,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_all(patients, doctors, appointments, billing):
    # 1) Monthly revenue
    m = billing.copy()
    m["month"] = m["date"].dt.to_period("M").dt.to_timestamp()
    monthly = m.groupby("month")["total_amount"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    sns.lineplot(data=monthly, x="month", y="total_amount", color=TEAL, linewidth=3, marker="o", ax=ax)
    ax.fill_between(monthly["month"], monthly["total_amount"], color=TEAL, alpha=0.12)
    ax.set_title("Monthly revenue", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("Revenue (Rs.)")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x/1000:.0f}k")
    fig.autofmt_xdate()
    _save(fig, "revenue.png")

    # 2) Appointments by department
    dep = appointments["department"].value_counts().reset_index()
    dep.columns = ["department", "count"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    sns.barplot(data=dep, y="department", x="count", color=TEAL, ax=ax)
    ax.set_title("Appointments by department", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Appointments"); ax.set_ylabel("")
    _save(fig, "departments.png")

    # 3) Patient age distribution
    fig, ax = plt.subplots(figsize=(7.5, 4))
    sns.histplot(data=patients, x="age", hue="gender", bins=18, multiple="stack",
                 palette={"Male": TEAL, "Female": "#7FD1BF"}, edgecolor="white", ax=ax)
    ax.set_title("Patient age and gender", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Age (years)"); ax.set_ylabel("Patients")
    _save(fig, "age.png")

    # 4) Busy hours heatmap (weekday x hour)
    a = appointments.copy()
    a["weekday"] = a["date"].dt.day_name().str[:3]
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot = a.pivot_table(index="weekday", columns="hour", values="appointment_id",
                          aggfunc="count", fill_value=0).reindex(order)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    sns.heatmap(pivot, cmap=sns.light_palette(TEAL, as_cmap=True), linewidths=1, linecolor="white",
                cbar_kws={"label": "Appointments"}, ax=ax)
    ax.set_title("Busiest hours of the week", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("")
    _save(fig, "heatmap.png")


def charts_exist():
    return all(os.path.exists(os.path.join(OUT, f))
               for f in ["revenue.png", "departments.png", "age.png", "heatmap.png"])
