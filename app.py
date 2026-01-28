from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3
import smtplib
import random
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta

import pickle
import numpy as np
import json
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer



lemmatizer = WordNetLemmatizer()
model = load_model("chatbot_model.h5")
intents = json.loads(open("intents.json", encoding="utf-8").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm sorry, I don't understand that."


app = Flask(__name__)
app.secret_key = "123"
DB = "hr_database.db"

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    # HR table
    c.execute('''
        CREATE TABLE IF NOT EXISTS hr (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            mobile_number TEXT,
            password TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            otp TEXT,
            login_attempts INTEGER DEFAULT 0,
            is_blocked INTEGER DEFAULT 0
        )
    ''')

    # Employees table
    c.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT UNIQUE,
            name TEXT,
            email TEXT UNIQUE,
            mobile TEXT,
            password TEXT,
            department TEXT,
            role TEXT,
            date TEXT,
            added_by_hr INTEGER
        )
    ''')

    # Projects table
    c.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            project_description TEXT,
            progress TEXT,
            start_date TEXT,
            end_date TEXT,
            assigned_employees TEXT
        )
    ''')

    # Meetings table
    c.execute('''
        CREATE TABLE IF NOT EXISTS meetings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            participants TEXT,
            status TEXT
        )
    ''')

    # Tasks table
    c.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            task_name TEXT NOT NULL,
            assigned_to INTEGER,
            deadline TEXT,
            status TEXT,
            FOREIGN KEY (project_id) REFERENCES projects(id),
            FOREIGN KEY (assigned_to) REFERENCES employees(id)
        )
    ''')

    # Chatlogs table
    c.execute('''
        CREATE TABLE IF NOT EXISTS chatlogs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            message TEXT,
            bot_response TEXT,
            timescamp TEXT,
            FOREIGN KEY (user_id) REFERENCES employees(id)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS leave_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT,
            type TEXT,
            start_date TEXT,
            end_date TEXT,
            reason TEXT,
            status TEXT DEFAULT 'Pending',
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
       )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS payroll (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT,
            month TEXT,
            salary REAL,
            bonus REAL,
            status TEXT DEFAULT 'Unpaid',
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
       )
    ''')
    conn.commit()
    conn.close()

init_db()


@app.route("/")
def index():
    if session.get("admin_logged_in"):
        return redirect(url_for("admin_dashboard"))
    return render_template("index.html")

@app.route("/hr_login")
def hr_login_page():
    return redirect(url_for("index"))


@app.route("/hr_register", methods=["POST"])
def hr_register():
    name = request.form.get("name")
    email = request.form.get("email")
    mobile = request.form.get("mobile_number")
    password = request.form.get("password")

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO hr (name, email, mobile_number, password, status) VALUES (?, ?, ?, ?, 'pending')",
            (name, email, mobile, password)
        )
        conn.commit()
        return "REGISTRATION_SUCCESSFUL_WAITING_FOR_ADMIN_APPROVAL"
    except sqlite3.IntegrityError:
        return "EMAIL_ALREADY_EXISTS"
    finally:
        conn.close()

@app.route("/hr_login", methods=["POST"])
def hr_login_post():
    email = request.form.get("email")
    password = request.form.get("password")

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    
    c.execute("SELECT id, name, status, is_blocked, login_attempts FROM hr WHERE email=?", (email,))
    hr_check = c.fetchone()

    if hr_check and hr_check[3] == 1:
        conn.close()
        flash("Your account is BLOCKED due to multiple failed attempts. Contact Admin.", "danger")
        return redirect(url_for("hr_login_page"))

    # Validate login
    if password == "RESEND_OTP":
        c.execute("SELECT id, name, status FROM hr WHERE email=?", (email,))
    else:
        c.execute("SELECT id, name, status FROM hr WHERE email=? AND password=?", (email, password))
    
    hr = c.fetchone()

    if hr:
        if hr[2] != 'approved':
            conn.close()
            return "ACCOUNT_PENDING_APPROVAL_CONTACT_ADMIN"
        
        c.execute("UPDATE hr SET login_attempts=0 WHERE id=?", (hr[0],))
        
        # OTP resend logic
        can_send = True
        if "last_otp_sent" in session:
            time_passed = datetime.now() - datetime.fromisoformat(session["last_otp_sent"])
            if time_passed.total_seconds() < 60:
                can_send = False

        if can_send:
            otp = str(random.randint(100000, 999999))
            c.execute("UPDATE hr SET otp=? WHERE id=?", (otp, hr[0]))
            conn.commit()
            send_otp_email(email, hr[1], otp)
            session["last_otp_sent"] = datetime.now().isoformat()
        
        conn.close()
        session["temp_hr_id"] = hr[0]
        session["temp_hr_email"] = email
        session["otp_tries"] = 0
        return render_template("hr_otp_verify.html", remaining_time=60)
    else:
        if hr_check:
            attempts = hr_check[4] + 1
            if attempts >= 3:
                c.execute("UPDATE hr SET is_blocked=1 WHERE id=?", (hr_check[0],))
                flash("Account Blocked! 3 failed attempts reached.", "danger")
            else:
                c.execute("UPDATE hr SET login_attempts=? WHERE id=?", (attempts, hr_check[0]))
                flash(f"Invalid password. {3 - attempts} attempts remaining.", "danger")
            conn.commit()
        else:
            flash("Invalid HR login credentials", "danger")
        
        conn.close()
        return redirect(url_for("hr_login_page"))

@app.route("/verify_otp", methods=["POST"])
def verify_otp():
    input_otp = request.form.get("otp")
    hr_id = session.get("temp_hr_id")
    sent_time_str = session.get("last_otp_sent")

    if not hr_id or not sent_time_str:
        return redirect(url_for("index"))

    sent_time = datetime.fromisoformat(sent_time_str)
    if datetime.now() > sent_time + timedelta(minutes=1):
        flash("OTP expired. Login again.", "danger")
        return redirect(url_for("hr_login_page"))

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT otp, name FROM hr WHERE id=?", (hr_id,))
    data = c.fetchone()

    if data and data[0] == input_otp:
        c.execute("UPDATE hr SET otp=NULL, login_attempts=0 WHERE id=?", (hr_id,))
        conn.commit()
        conn.close()
        session["hr_id"] = hr_id
        session["hr_name"] = data[1]
        session.pop("temp_hr_id", None)
        session.pop("last_otp_sent", None)
        session.pop("otp_tries", None)
        return redirect(url_for("hr_dashboard"))
    else:
        session["otp_tries"] = session.get("otp_tries", 0) + 1
        if session["otp_tries"] >= 3:
            c.execute("UPDATE hr SET is_blocked=1, otp=NULL WHERE id=?", (hr_id,))
            conn.commit()
            conn.close()
            session.clear()
            flash("Account BLOCKED! 3 incorrect OTP attempts. Contact Admin.", "danger")
            return redirect(url_for("hr_login_page"))
        conn.close()
        flash(f"Invalid OTP. {3 - session['otp_tries']} attempts remaining.", "danger")
        return render_template("hr_otp_verify.html")

@app.route("/apply_leave", methods=["POST"])
def apply_leave():
    if "emp_id" not in session:
        return redirect(url_for("index"))
    
    leave_type = request.form.get("leave_type")
    start = request.form.get("start_date")
    end = request.form.get("end_date")
    reason = request.form.get("reason")

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''INSERT INTO leave_requests (employee_id, type, start_date, end_date, reason) 
                 VALUES (?, ?, ?, ?, ?)''', (session["emp_id"], leave_type, start, end, reason))
    conn.commit()
    conn.close()
    flash("Leave request submitted successfully!", "success")
    return redirect(url_for("employee_dashboard"))

@app.route("/update_leave/<int:leave_id>/<string:status>")
def update_leave(leave_id, status):
    if "hr_id" not in session:
        return redirect(url_for("index"))
    
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("UPDATE leave_requests SET status=? WHERE id=?", (status, leave_id))
    conn.commit()
    conn.close()
    flash(f"Request marked as {status}", "success")
    return redirect(url_for("hr_dashboard"))

@app.route("/admin_stats_json")
def admin_stats_json():
    if not session.get("admin_logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
        
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    
    # 1. HR Status Distribution (Doughnut Chart)
    c.execute("SELECT status, COUNT(*) FROM hr GROUP BY status")
    hr_status = dict(c.fetchall())
    
    # 2. Global Workforce by Department (Bar Chart)
    c.execute("SELECT department, COUNT(*) FROM employees GROUP BY department")
    dept_dist = dict(c.fetchall())
    
    # 3. System Load Simulation (for Telemetry)
    # In a real app, this would query server metrics
    telemetry = {
        "uptime": "99.98%",
        "active_sessions": random.randint(5, 50),
        "db_size": "1.2 MB"
    }
    
    conn.close()
    return jsonify({
        "hr_status": hr_status,
        "dept_dist": dept_dist,
        "telemetry": telemetry,
        "growth": {"labels": ["W1", "W2", "W3", "W4"], "data": [12, 19, 15, 28]}
    })

@app.route("/admin_view_hr/<int:hr_id>")
def admin_view_hr(hr_id):
    if not session.get("admin_logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
        
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    # Fetch employees for a specific HR with their project/meeting status
    c.execute('''
        SELECT e.employee_id, e.name, e.email, e.role, e.department,
        (SELECT COUNT(*) FROM projects WHERE assigned_employees LIKE '%' || e.name || '%') as proj_count
        FROM employees e 
        WHERE e.added_by_hr = ?
    ''', (hr_id,))
    
    employees = []
    for row in c.fetchall():
        employees.append({
            "employee_id": row[0],
            "name": row[1],
            "email": row[2],
            "role": row[3],
            "department": row[4],
            "project_status": "Active" if row[5] > 0 else "Idle",
            "meeting_status": "Synchronized" # Placeholder for meeting logic
        })
    
    conn.close()
    return jsonify({"employees": employees})

@app.route("/process_payroll", methods=["POST"])
def process_payroll():
    # Authorization and dynamic data entry
    emp_id = request.form.get("employee_id")
    salary = request.form.get("salary")
    bonus = request.form.get("bonus", 0)
    month = datetime.now().strftime("%B %Y")
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT name, email FROM employees WHERE employee_id=?", (emp_id,))
    employee = c.fetchone()
    if employee:
        c.execute("INSERT INTO payroll (employee_id, month, salary, bonus, status) VALUES (?, ?, ?, ?, 'Paid')", 
                  (emp_id, month, salary, bonus))
        conn.commit()
        send_payslip_email(employee[1], employee[0], month, salary, bonus)
    conn.close()
    return redirect(url_for("hr_dashboard"))

@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        if email == "admin" and password == "123":
            session["admin_logged_in"] = True
            flash("Admin Authenticated", "success")
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid Admin Access", "danger")
            return redirect(url_for("index"))
    return render_template("admin_login.html")

@app.route("/admin_dashboard")
def admin_dashboard():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))
    
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''
        SELECT 
            h.id, 
            h.name, 
            h.email, 
            h.mobile_number, 
            h.status,
            (SELECT COUNT(*) FROM employees WHERE added_by_hr = h.id) as emp_count,
            h.is_blocked
        FROM hr h
    ''')
    hrs = c.fetchall()
    conn.close()
    return render_template("admin_dashboard.html", hrs=hrs)

@app.route("/approve_hr/<int:hr_id>")
def approve_hr(hr_id):
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("UPDATE hr SET status='approved' WHERE id=?", (hr_id,))
    conn.commit()
    conn.close()
    flash("HR Account Approved", "success")
    return redirect(url_for("admin_dashboard"))

@app.route("/unblock_hr/<int:hr_id>")
def unblock_hr(hr_id):
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("UPDATE hr SET is_blocked=0, login_attempts=0 WHERE id=?", (hr_id,))
    conn.commit()
    conn.close()
    flash("HR Account Unblocked Successfully", "success")
    return redirect(url_for("admin_dashboard"))

@app.route("/hr_dashboard")
def hr_dashboard():
    if "hr_id" not in session:
        return redirect(url_for("hr_login_page"))
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    
    c.execute("SELECT employee_id, name, department, role FROM employees WHERE added_by_hr=?", (session["hr_id"],))
    employees = c.fetchall()
    c.execute("SELECT * FROM projects")
    projects = c.fetchall()
    c.execute("SELECT * FROM meetings")
    meetings = c.fetchall()
    c.execute('''SELECT l.id, e.name, l.type, l.start_date, l.end_date, l.status 
                 FROM leave_requests l JOIN employees e ON l.employee_id = e.employee_id 
                 WHERE l.status = 'Pending' ''')
    leaves = c.fetchall()
    c.execute('''SELECT p.month, e.name, p.salary, p.bonus, p.status 
                 FROM payroll p JOIN employees e ON p.employee_id = e.employee_id 
                 ORDER BY p.id DESC''')
    payroll = c.fetchall()

    conn.close()
    return render_template("hr_dashboard.html", hr_name=session["hr_name"], 
                           employees=employees, projects=projects, 
                           meetings=meetings, leaves=leaves, payroll=payroll)
@app.route("/get_calendar_events")
def get_calendar_events():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    # Fetch Meetings
    c.execute("SELECT title, date, time FROM meetings")
    meetings = [{"title": f"Meeting: {m[0]}", "start": f"{m[1]}T{m[2]}", "color": "#2563eb"} for m in c.fetchall()]
    # Fetch Project Deadlines
    c.execute("SELECT name, end_date FROM projects")
    deadlines = [{"title": f"Deadline: {p[0]}", "start": p[1], "color": "#ef4444"} for p in c.fetchall()]
    conn.close()
    return jsonify(meetings + deadlines)@app.route("/add_employee", methods=["GET", "POST"])

@app.route("/add_employee", methods=["POST"])
def add_employees():
    if "hr_id" not in session:
        return redirect(url_for("hr_login_page"))
    if request.method == "POST":
        emp_id = request.form["employee_id"]
        name = request.form["name"]
        email = request.form["email"]
        mobile = request.form["mobile_number"]
        password = request.form["password"]
        department = request.form["department"]
        role = request.form["role"]
        date = request.form["date_of_joining"]

        conn = sqlite3.connect(DB)
        c = conn.cursor()
        try:
            c.execute('''
                INSERT INTO employees 
                (employee_id, name, email, mobile, password, department, role, date, added_by_hr)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (emp_id, name, email, mobile, password, department, role, date, session["hr_id"]))
            conn.commit()
            send_employee_email(email, emp_id, name, password, role, department)
            flash("Employee added successfully!", "success")
            return redirect(url_for("hr_dashboard"))
        except sqlite3.IntegrityError:
            flash("Employee ID or Email already exists!", "danger")
        finally:
            conn.close()
    return render_template("add_employee_form.html")

@app.route("/chat", methods=["POST"])
def chat():
    message = request.json.get("message")
    if not message:
        return jsonify({"response": "No message received"})
    
    ints = predict_class(message, model)
    res = get_response(ints, intents)
    
    user_id = session.get("hr_id") or session.get("emp_id")
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("INSERT INTO chatlogs (user_id, message, bot_response, timescamp) VALUES (?, ?, ?, ?)",
              (user_id, message, res, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    
    return jsonify({"response": res})

# -----------------------------
# EMPLOYEE LOGIN
# -----------------------------
@app.route("/employee_login", methods=["POST"])
def employee_login():
    login_input = request.form["emp_id"]
    password = request.form["password"]

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT * FROM employees WHERE (employee_id=? OR email=?) AND password=?", 
              (login_input, login_input, password))
    emp = c.fetchone()
    conn.close()

    if emp:
        session["emp_id"] = emp[1]
        session["emp_name"] = emp[2]
        flash("Login Successful!", "success")
        return redirect(url_for("employee_dashboard"))
    else:
        flash("Invalid Employee ID/Email or Password", "danger")
        return redirect(url_for("index"))

@app.route("/employee_dashboard")
def employee_dashboard():
    if "emp_id" not in session:
        return redirect(url_for("index"))

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    
    # Dynamic: Only fetch projects assigned to this specific employee
    c.execute("SELECT * FROM projects WHERE assigned_employees LIKE ?", ('%' + session["emp_name"] + '%',))
    projects = c.fetchall()

    # Dynamic: Only fetch meetings where employee is a participant
    c.execute("SELECT * FROM meetings WHERE participants LIKE ?", ('%' + session["emp_name"] + '%',))
    meetings = c.fetchall()

    # Dynamic: Fetch leave history
    c.execute("SELECT * FROM leave_requests WHERE employee_id=?", (session["emp_id"],))
    leaves = c.fetchall()

    conn.close()
    return render_template("employee_dashboard.html", 
                           emp_id=session["emp_id"], 
                           emp_name=session["emp_name"],
                           projects=projects, 
                           meetings=meetings,
                           leaves=leaves)

@app.route("/api/performance_stats")
def performance_stats():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT status, COUNT(*) FROM leave_requests GROUP BY status")
    data = dict(c.fetchall())
    conn.close()
    return jsonify(data)

# -----------------------------
# PROJECT / MEETING ROUTES
# -----------------------------
@app.route("/create_project", methods=["GET", "POST"])
def create_project():
    if "hr_id" not in session:
        return redirect(url_for("hr_login_page"))
    
    if request.method == "POST":
        name = request.form.get("name")
        description = request.form.get("description")
        progress = request.form.get("progress")
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")
        assigned_employees = request.form.get("assigned_employees")

        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute('''
            INSERT INTO projects (name, project_description, progress, start_date, end_date, assigned_employees)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, description, progress, start_date, end_date, assigned_employees))
        conn.commit()
        conn.close()
        flash("Project created successfully!", "success")
        return redirect(url_for("hr_dashboard"))
    return render_template("create_project_form.html")

@app.route("/schedule_meeting", methods=["GET", "POST"])
def schedule_meeting():
    if "hr_id" not in session:
        return redirect(url_for("hr_login_page"))

    if request.method == "POST":
        title = request.form.get("title")
        date = request.form.get("date")
        time = request.form.get("time")
        participants = request.form.get("participants")
        status = request.form.get("status")

        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute('''
            INSERT INTO meetings (title, date, time, participants, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (title, date, time, participants, status))
        conn.commit()
        conn.close()

        if participants:
            p_list = [p.strip() for p in participants.split(',')]
            for email in p_list:
                send_meeting_invite(email, title, date, time)

        flash("Meeting scheduled successfully!", "success")
        return redirect(url_for("hr_dashboard"))

    return render_template("schedule_meeting_form.html")


def send_otp_email(to_email, name, otp):
    SENDER_EMAIL = "triossoftwaremail@gmail.com"
    SENDER_PASSWORD = "knaxddlwfpkplsik"
    msg = MIMEMultipart()
    msg["From"] = f"HRMate+ Security <{SENDER_EMAIL}>"
    msg["To"] = to_email
    msg["Subject"] = "Secure Access OTP - HRMate+"
    
    body = f"""
    <html>
    <body style="font-family: Arial; border: 1px solid #eee; padding: 20px;">
        <h2 style="color: #2563eb;">Verification Code</h2>
        <p>Hello {name},</p>
        <p>Your one-time password for HR login is:</p>
        <h1 style="background: #f4f4f4; padding: 10px; display: inline-block; letter-spacing: 5px;">{otp}</h1>
        <p>This code is valid for 10 minutes. If you did not request this, please secure your account.</p>
    </body>
    </html>
    """
    msg.attach(MIMEText(body, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print(f"OTP Error: {e}")

def send_employee_email(to_email, employee_id, name, password, role, dept):
    SENDER_EMAIL = "triossoftwaremail@gmail.com"
    SENDER_PASSWORD = "knaxddlwfpkplsik"
    msg = MIMEMultipart()
    msg["From"] = f"HRMate+ Corporate <{SENDER_EMAIL}>"
    msg["To"] = to_email
    msg["Subject"] = f"OFFICIAL: Corporate Access Credentials - {name}"
    year = datetime.now().year
    body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; color: #444; line-height: 1.6; background-color: #f4f4f4; padding: 20px;">
        <div style="max-width: 600px; margin: auto; background: #ffffff; border-radius: 10px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
            <div style="background-color: #000; color: #ffffff; padding: 30px; text-align: center;">
                <h1 style="margin: 0; font-size: 26px; font-weight: 800; letter-spacing: 3px;">HRMATE<span style="color: #2563eb;">+</span></h1>
                <p style="margin: 5px 0 0; font-size: 13px; font-weight: 300; opacity: 0.8;">ENTERPRISE OPERATING SYSTEM</p>
            </div>
            <div style="padding: 40px;">
                <h2 style="color: #111; font-weight: 700;">Welcome to the Infrastructure, {name}</h2>
                <p>Your professional profile has been successfully integrated into our secure infrastructure. Please find your credentials below:</p>
                <div style="background-color: #f9f9f9; border: 1px solid #eee; border-left: 5px solid #2563eb; padding: 25px; margin: 30px 0; border-radius: 5px;">
                    <p style="margin: 5px 0;"><strong>Employee ID:</strong> <span style="color: #2563eb; font-weight: bold;">{employee_id}</span></p>
                    <p style="margin: 5px 0;"><strong>Corporate Role:</strong> {role}</p>
                    <p style="margin: 5px 0;"><strong>Department:</strong> {dept}</p>
                    <hr style="border: 0; border-top: 1px solid #ddd; margin: 15px 0;">
                    <p style="margin: 5px 0;"><strong>Access Password:</strong> <span style="font-family: 'Courier New', Courier, monospace; background: #eee; padding: 3px 7px; border-radius: 3px;">{password}</span></p>
                </div>
            </div>
            <div style="background-color: #fcfcfc; padding: 25px; text-align: center; font-size: 12px; color: #999;">
                <p>&copy; {year} HRMate+ Global Operations. All rights reserved.</p>
            </div>
        </div>
    </body>
    </html>
    """
    msg.attach(MIMEText(body, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print(f"Mail Error: {e}")

def send_meeting_invite(to_email, title, date, time):
    SENDER_EMAIL = "triossoftwaremail@gmail.com"
    SENDER_PASSWORD = "knaxddlwfpkplsik"
    msg = MIMEMultipart()
    msg["From"] = f"HRMate+ Global <{SENDER_EMAIL}>"
    msg["To"] = to_email
    msg["Subject"] = f"STRATEGIC SYNC: {title} - {date}"
    body = f"""
    <html>
    <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f9f9f9; padding: 20px;">
        <div style="max-width: 550px; margin: auto; background: #fff; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
            <div style="background-color: #000; color: #ffffff; padding: 20px; text-align: center;">
                <h1 style="margin: 0; font-size: 24px; letter-spacing: 2px;">HRMATE<span style="color: #2563eb;">+</span></h1>
            </div>
            <div style="padding: 30px;">
                <h2 style="color: #111; margin: 0;">Meeting Notification</h2>
                <div style="background: #f0f7ff; padding: 25px; border-radius: 6px; margin-top: 20px; border: 1px solid #d0e3ff;">
                    <p style="margin: 0; font-size: 20px; font-weight: bold; color: #2563eb;">{title}</p>
                    <p style="margin: 5px 0; color: #444;"><b>Date:</b> {date}</p>
                    <p style="margin: 5px 0; color: #444;"><b>Time:</b> {time}</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    msg.attach(MIMEText(body, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print(f"Meeting Error: {e}")

def send_payslip_email(to_email, name, month, salary, bonus):
    SENDER_EMAIL = "triossoftwaremail@gmail.com"
    SENDER_PASSWORD = "knaxddlwfpkplsik"
    msg = MIMEMultipart()
    msg["From"] = f"HRMate+ Finance <{SENDER_EMAIL}>"
    msg["To"] = to_email
    msg["Subject"] = f"PAY-SLIP: {month} - {name}"
    
    total = float(salary) + float(bonus)
    
    body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; background-color: #f8fafc; padding: 40px;">
        <div style="max-width: 600px; margin: auto; background: #ffffff; border-radius: 20px; border: 1px solid #e2e8f0; padding: 40px;">
            <h2 style="color: #1e293b;">Earnings Statement: {month}</h2>
            <p>Hello {name}, your payment has been disbursed.</p>
            <hr style="border: 0; border-top: 1px solid #f1f5f9; margin: 20px 0;">
            <p><strong>Base Salary:</strong> ${salary}</p>
            <p><strong>Bonus:</strong> ${bonus}</p>
            <h1 style="color: #2563eb;">Total: ${total}</h1>
        </div>
    </body>
    </html>
    """
    msg.attach(MIMEText(body, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print(f"Payslip Mail Error: {e}")

@app.route("/get_stats_json")
def get_stats_json():
    # Example data for the graph - counts per department
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT department, COUNT(*) FROM employees GROUP BY department")
    data = c.fetchall()
    conn.close()
    return jsonify({"labels": [row[0] for row in data], "values": [row[1] for row in data]})

@app.route("/get_meetings_json")
def get_meetings_json():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT title, date, time, status FROM meetings")
    meetings = c.fetchall()
    conn.close()
    return jsonify([{"title": m[0], "start": f"{m[1]}T{m[2]}", "color": "#2563eb" if m[3] != 'Urgent' else "#ef4444"} for m in meetings])
@app.route("/logout")
def logout():
    session.clear()
    flash("Session terminated successfully.", "success")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=False, port=2300 )
