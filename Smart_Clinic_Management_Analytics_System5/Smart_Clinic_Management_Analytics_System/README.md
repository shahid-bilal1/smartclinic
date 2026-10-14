# Smart Clinic Management & Analytics System

A student project by **Momin & Murtiza**. A green-and-white clinic management website built with
**Flask, HTML/CSS, CSV files, Pandas, NumPy, Matplotlib and Seaborn**. No JavaScript is used.

> **Important:** This is a learning/demo project with **fictional data**. It is **not production-ready** and must
> **not** be used for real patient records. It gives no medical diagnosis or treatment advice.

## Windows installation and run

1. Install Python 3.10+ from https://www.python.org/downloads/ (tick **"Add Python to PATH"** during setup).
2. Unzip the project folder, e.g. to `C:\Smart_Clinic_Management_Analytics_System`.
3. **Easy way:** double-click `run_windows.bat`.
   **Manual way:** open Command Prompt in the folder and run:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   python app.py
   ```
4. Open **http://127.0.0.1:5000** in your browser. Stop the server with `Ctrl + C`.

**Demo logins** (shown on the sign-in page)

| User | Password | Can do |
|------|----------|--------|
| `admin` | `admin123` | everything, including delete |
| `reception` | `reception123` | add / edit / book / bill, no delete |

Change passwords with environment variables `CLINIC_ADMIN_PASSWORD` and `CLINIC_STAFF_PASSWORD`.

To reset the demo data (dates are created relative to today): `python generate_sample_data.py`

## Folder structure
```
app.py                  Flask backend: routes, validation, CSV helpers, analytics
generate_sample_data.py Creates fictional demo CSV data
run_windows.bat         One-click start on Windows
requirements.txt        Python packages
patients.csv  appointments.csv  billing.csv     Data storage
templates/              HTML pages (intro, base, dashboard, patients, appointments, billing, analytics + forms)
static/css/style.css    All styling (green medical theme)
static/images/          SVG illustrations: logo, clinic hero, doctor, team avatars, empty state
static/charts/          Chart PNGs created automatically by the Analytics page
```

## Explanation for a beginner student

**How a request travels.** You click a link, e.g. *Patients*. The browser asks Flask for `/patients`.
The function with `@app.route("/patients")` runs, reads `patients.csv` into a **Pandas DataFrame** (a table in
memory), and passes it to `patients.html`. The template (Jinja2) loops over the rows and builds the page.

**Saving data.** When you submit the Add Patient form, Flask receives it with `request.form`, **validates** it
(name, age 0-120, 10-13 digit phone, date not in the future, no duplicate patient), creates a new unique ID
(`P031`), appends a row and writes the CSV file again. Because the data lives in files, it stays after restarting.

**Key ideas in the code**
- *Validation functions* (`validate_patient`, `validate_appointment`, `validate_bill`) return cleaned data plus a list of errors, shown as red messages. Success shows a green message.
- *Unique IDs*: `next_id()` takes the highest existing number and adds one.
- *No double booking*: the same doctor + date + time cannot be booked twice (cancelled bookings free the slot). The same patient also cannot be in two places at once.
- *Billing*: total = consultation fee + additional charges, always calculated by the server. One button toggles Paid/Unpaid.
- *Dashboard*: every number is computed from the CSVs on each page load, so it updates when records change. Bars are drawn with plain CSS widths and heights.
- *Analytics*: Pandas groups the data (by month, status, ...), NumPy gives the average, median, 90th percentile, a 7-day moving average and the busiest weekday, Matplotlib/Seaborn draw 7 charts saved in `static/charts/`. Only aggregated counts are used, no names or phone numbers.
- *Bad data*: missing, empty or damaged CSV files give empty tables instead of crashing; duplicate IDs and rows without IDs are ignored.
- *Security (basic)*: sign-in with hashed passwords, roles (only admin deletes), CSRF token in every form, login attempt limit, no-cache headers on patient pages, and a random secret key saved in `.secret_key`.

## Pages
Project Intro (`/`) → Sign in → Dashboard, Patients (search, view, edit, delete), Appointments (book, filter, update, mark done, cancel), Billing (create, edit, mark paid/unpaid), Analytics.

## Limitations (be honest in your viva)
CSV files are fine for a demo but not for real clinics: no encryption, no audit log, no concurrent multi-server safety, demo passwords in the source. A real system needs a proper database, HTTPS, strong authentication and legal compliance.
