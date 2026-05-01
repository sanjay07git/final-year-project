# ─────────────────────────────────────────────────────────────────────────────
#            HRMate  |  app.py  |  Single-file Flask application
#            Features: HR/Employee/Admin portals, Google Meet, Chatbot,
#            Google Calendar Two-Way Sync, KMeans project assignment
# ─────────────────────────────────────────────────────────────────────────────

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3, smtplib, random, os, uuid, json, urllib.parse, urllib.request, secrets
import numpy as np
import pickle
import nltk

nltk.download('punkt')
nltk.download('wordnet')
import pytz
import requests as _req

from dotenv import load_dotenv
import os

load_dotenv()



from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from datetime             import datetime, timedelta
from collections          import defaultdict

from keras.models         import load_model
from nltk.stem            import WordNetLemmatizer
from sklearn.cluster      import KMeans

from google.oauth2.credentials  import Credentials
from googleapiclient.discovery  import build

# ─── App config ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"]   = False
app.config['SESSION_PERMANENT'] = False
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

DB = "hr_database.db"

# ─── Google OAuth / Calendar constants ────────────────────────────────────────
GOOGLE_CLIENT_SECRETS = "Se.json"
SCOPES       = ["https://www.googleapis.com/auth/calendar"]
REDIRECT_URI = "http://127.0.0.1:2300/google/callback"
HRMATE_TAG   = "[HRMate]"

# ─── Legacy NLP chatbot (kept as fallback if Gemini key not set) ──────────────
lemmatizer = WordNetLemmatizer()
model      = load_model("chatbot_model.h5")
intents    = json.loads(open("intents.json", encoding="utf-8").read())
words      = pickle.load(open("words.pkl", "rb"))
classes    = pickle.load(open("classes.pkl", "rb"))

def clean_up_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in tokens]

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
    results = [[i, r] for i, r in enumerate(res) if r > 0.25]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(ints, intents_json):
    tag = ints[0]["intent"]
    for i in intents_json["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])
    return "I'm sorry, I don't understand that."


# ═══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

def init_db():
    conn = sqlite3.connect(DB)
    c    = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS hr (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE, mobile_number TEXT, password TEXT NOT NULL,
        status TEXT DEFAULT 'pending', otp TEXT,
        login_attempts INTEGER DEFAULT 0, is_blocked INTEGER DEFAULT 0)''')
    c.execute('''CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id TEXT UNIQUE,
        name TEXT, email TEXT UNIQUE, mobile TEXT, password TEXT,
        department TEXT, role TEXT, date TEXT, added_by_hr INTEGER, skills TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
        project_description TEXT, progress TEXT, start_date TEXT, end_date TEXT,
        assigned_employees TEXT, status TEXT DEFAULT "In Progress", gcal_event_id TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS meetings (
        id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL,
        date TEXT NOT NULL, time TEXT NOT NULL, timezone TEXT DEFAULT 'UTC',
        participants TEXT, status TEXT, meet_link TEXT, gcal_event_id TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER,
        task_name TEXT NOT NULL, assigned_to INTEGER, deadline TEXT, status TEXT,
        FOREIGN KEY (project_id) REFERENCES projects(id),
        FOREIGN KEY (assigned_to) REFERENCES employees(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS chatlogs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
        message TEXT, bot_response TEXT, timescamp TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS leave_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id TEXT,
        type TEXT, start_date TEXT, end_date TEXT, reason TEXT,
        status TEXT DEFAULT 'Pending')''')
    c.execute('''CREATE TABLE IF NOT EXISTS payroll (
        id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id TEXT,
        month TEXT, salary REAL, bonus REAL, status TEXT DEFAULT 'Unpaid')''')
    c.execute('''CREATE TABLE IF NOT EXISTS google_tokens (
        hr_id INTEGER PRIMARY KEY, access_token TEXT, refresh_token TEXT,
        client_id TEXT, client_secret TEXT)''')
    conn.commit()
    conn.close()

def migrate_db():
    conn = sqlite3.connect(DB)
    c    = conn.cursor()
    for sql in [
        "ALTER TABLE meetings ADD COLUMN timezone TEXT DEFAULT 'UTC'",
        "ALTER TABLE meetings ADD COLUMN meet_link TEXT",
        "ALTER TABLE meetings ADD COLUMN gcal_event_id TEXT",
        "ALTER TABLE projects ADD COLUMN gcal_event_id TEXT",
    ]:
        try:
            c.execute(sql); conn.commit()
        except sqlite3.OperationalError:
            pass
    conn.close()

init_db()
migrate_db()



# ═══════════════════════════════════════════════════════════════════════════════
#  GOOGLE CALENDAR HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _load_client_info():
    with open(GOOGLE_CLIENT_SECRETS) as f:
        return json.load(f)["web"]

def get_calendar_service():
    hr_id = session.get("hr_id")
    if not hr_id: return None
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("SELECT access_token,refresh_token,client_id,client_secret FROM google_tokens WHERE hr_id=?", (hr_id,))
    row = c.fetchone(); conn.close()
    if not row: print(f"[GCAL] No token for hr_id={hr_id}"); return None
    creds = Credentials(token=row[0], refresh_token=row[1],
                        token_uri="https://oauth2.googleapis.com/token",
                        client_id=row[2], client_secret=row[3], scopes=SCOPES)
    try: return build("calendar", "v3", credentials=creds)
    except Exception as e: print(f"[GCAL] Build error: {e}"); return None

from datetime import datetime, timedelta
import pytz

def create_google_meet(title, date, time, timezone, participants):
    try:
        service = get_calendar_service()
        if not service:
            return None

        tz = pytz.timezone(timezone)

        start_dt = tz.localize(
            datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        )
        end_dt = start_dt + timedelta(hours=1)

        event = {
            'summary': title,
            'start': {
                'dateTime': start_dt.isoformat(),
                'timeZone': timezone
            },
            'end': {
                'dateTime': end_dt.isoformat(),
                'timeZone': timezone
            },
            'attendees': [{'email': p} for p in participants],
            'conferenceData': {
                'createRequest': {
                    'requestId': str(uuid.uuid4()),
                    'conferenceSolutionKey': {'type': 'hangoutsMeet'}
                }
            }
        }

        created_event = service.events().insert(
            calendarId="primary",
            body=event,
            conferenceDataVersion=1,
            sendUpdates="all"
        ).execute()

        meet_link = created_event.get("hangoutLink")
        return meet_link

    except Exception as e:
        print("[MEET ERROR]", e)
        return None

def push_meeting_to_google(meeting_id):
    service = get_calendar_service()
    if not service: return None, "Not connected to Google Calendar"
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("SELECT title,date,time,timezone,participants,status,meet_link,gcal_event_id FROM meetings WHERE id=?", (meeting_id,))
    row = c.fetchone(); conn.close()
    if not row: return None, "Meeting not found"
    title, date, time_str, tz, participants, status, meet_link, existing_id = row
    start_dt  = f"{date}T{time_str}:00"
    end_time  = (datetime.strptime(f"{date} {time_str}", "%Y-%m-%d %H:%M") + timedelta(hours=1)).strftime("%H:%M")
    end_dt    = f"{date}T{end_time}:00"
    attendees = [{"email": e.strip()} for e in (participants or "").split(",") if e.strip()]
    event = {
        "summary": f"{HRMATE_TAG} {title}",
        "description": f"Scheduled via HRMate\nStatus: {status}",
        "start": {"dateTime": start_dt, "timeZone": tz or "UTC"},
        "end":   {"dateTime": end_dt,   "timeZone": tz or "UTC"},
        "attendees": attendees, "location": meet_link or "",
        "reminders": {"useDefault": False, "overrides": [
            {"method": "email", "minutes": 60}, {"method": "popup", "minutes": 10}]}
    }
    try:
        if existing_id:
            updated  = service.events().update(calendarId="primary", eventId=existing_id, body=event, sendUpdates="all").execute()
            event_id = updated["id"]; msg = "Meeting updated on Google Calendar"
        else:
            created  = service.events().insert(calendarId="primary", body=event, conferenceDataVersion=1, sendUpdates="all").execute()
            event_id = created["id"]; msg = "Meeting pushed to Google Calendar"
        conn = sqlite3.connect(DB); c = conn.cursor()
        c.execute("UPDATE meetings SET gcal_event_id=? WHERE id=?", (event_id, meeting_id))
        conn.commit(); conn.close()
        return event_id, msg
    except Exception as e: print(f"[SYNC] Push meeting error: {e}"); return None, str(e)

def push_deadline_to_google(project_id):
    service = get_calendar_service()
    if not service: return None, "Not connected to Google Calendar"
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("SELECT name,end_date,project_description,gcal_event_id FROM projects WHERE id=?", (project_id,))
    row = c.fetchone(); conn.close()
    if not row: return None, "Project not found"
    name, end_date, description, existing_id = row
    if not end_date: return None, "No deadline set"
    event = {
        "summary": f"{HRMATE_TAG} DEADLINE: {name}",
        "description": f"Project deadline from HRMate\n{description or ''}",
        "start": {"date": end_date}, "end": {"date": end_date}, "colorId": "11",
        "reminders": {"useDefault": False, "overrides": [
            {"method": "email", "minutes": 1440}, {"method": "popup", "minutes": 60}]}
    }
    try:
        if existing_id:
            updated  = service.events().update(calendarId="primary", eventId=existing_id, body=event).execute()
            event_id = updated["id"]; msg = "Deadline updated on Google Calendar"
        else:
            created  = service.events().insert(calendarId="primary", body=event).execute()
            event_id = created["id"]; msg = "Deadline pushed to Google Calendar"
        conn = sqlite3.connect(DB); c = conn.cursor()
        c.execute("UPDATE projects SET gcal_event_id=? WHERE id=?", (event_id, project_id))
        conn.commit(); conn.close()
        return event_id, msg
    except Exception as e: print(f"[SYNC] Deadline error: {e}"); return None, str(e)

def pull_google_events():
    service = get_calendar_service()
    if not service: return [], "Not connected"
    now    = datetime.utcnow().isoformat() + "Z"
    future = (datetime.utcnow() + timedelta(days=60)).isoformat() + "Z"
    try:
        result = service.events().list(calendarId="primary", timeMin=now, timeMax=future,
                                        maxResults=50, singleEvents=True, orderBy="startTime").execute()
        events = result.get("items", [])
    except Exception as e: print(f"[SYNC] Pull error: {e}"); return [], str(e)
    imported = []
    conn = sqlite3.connect(DB); c = conn.cursor()
    for event in events:
        summary = event.get("summary", "")
        if summary.startswith(HRMATE_TAG): continue
        gcal_id = event["id"]
        c.execute("SELECT id FROM meetings WHERE gcal_event_id=?", (gcal_id,))
        if c.fetchone(): continue
        start = event.get("start", {})
        if "dateTime" in start:
            dt       = datetime.fromisoformat(start["dateTime"].replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d"); time_str = dt.strftime("%H:%M"); tz = start.get("timeZone", "UTC")
        elif "date" in start:
            date_str = start["date"]; time_str = "09:00"; tz = "UTC"
        else: continue
        participants = ", ".join(a.get("email", "") for a in event.get("attendees", []) if "email" in a)
        meet_link    = event.get("hangoutLink", "")
        c.execute("INSERT INTO meetings (title,date,time,timezone,participants,status,meet_link,gcal_event_id) VALUES (?,?,?,?,?,?,?,?)",
                  (summary, date_str, time_str, tz, participants, "Scheduled", meet_link, gcal_id))
        imported.append({"title": summary, "date": date_str})
        print(f"[SYNC] Imported: {summary} on {date_str}")
    conn.commit(); conn.close()
    return imported, "Success"

def full_sync():
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("SELECT id FROM meetings"); m_ids = [r[0] for r in c.fetchall()]
    c.execute("SELECT id FROM projects WHERE end_date IS NOT NULL AND end_date!=''"); p_ids = [r[0] for r in c.fetchall()]
    conn.close()
    pushed_m = sum(1 for mid in m_ids if push_meeting_to_google(mid)[0])
    pushed_d = sum(1 for pid in p_ids if push_deadline_to_google(pid)[0])
    pulled, _ = pull_google_events()
    return {"pushed_meetings": pushed_m, "pushed_deadlines": pushed_d, "pulled_events": len(pulled)}


# ═══════════════════════════════════════════════════════════════════════════════
#  EMAIL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def send_otp_email(to_email, name, otp):
    SENDER_EMAIL = os.getenv("EMAIL_USER")
    SENDER_PASSWORD = os.getenv("EMAIL_PASSWORD")
    print(f"[OTP] {otp}")
    msg = MIMEMultipart()
    msg["From"] = f"HRMate Security <{SENDER_EMAIL}>"; msg["To"] = to_email
    msg["Subject"] = "Secure Access OTP - HRMate"
    body = f"""<html><body style="font-family:Arial;padding:20px;">
        <h2 style="color:#2563eb;">Verification Code</h2><p>Hello {name},</p>
        <p>Your OTP for HR login:</p>
        <h1 style="background:#f4f4f4;padding:10px;letter-spacing:5px;">{otp}</h1>
        <p>Valid for 10 minutes.</p></body></html>"""
    msg.attach(MIMEText(body, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(SENDER_EMAIL, SENDER_PASSWORD); s.send_message(msg)
    except Exception as e: print(f"[EMAIL] OTP: {e}")

def send_employee_email(to_email, employee_id, name, password, role, dept):
    SENDER_EMAIL = os.getenv("EMAIL_USER")
    SENDER_PASS = os.getenv("EMAIL_PASSWORD")
    year = datetime.now().year
    msg = MIMEMultipart()
    msg["From"] = f"HRMate Corporate <{SENDER_EMAIL}>"; msg["To"] = to_email
    msg["Subject"] = f"OFFICIAL: Corporate Access Credentials - {name}"
    body = f"""<html><body style="font-family:Arial,sans-serif;background:#f4f4f4;padding:20px;">
        <div style="max-width:600px;margin:auto;background:#fff;border-radius:10px;overflow:hidden;">
            <div style="background:#000;color:#fff;padding:30px;text-align:center;">
                <h1 style="margin:0;letter-spacing:3px;">HRMATE<span style="color:#2563eb;">+</span></h1>
            </div>
            <div style="padding:40px;">
                <h2>Welcome to the Infrastructure, {name}</h2>
                <div style="background:#f9f9f9;border-left:5px solid #2563eb;padding:25px;margin:30px 0;">
                    <p><strong>Employee ID:</strong> <span style="color:#2563eb;">{employee_id}</span></p>
                    <p><strong>Role:</strong> {role} | <strong>Department:</strong> {dept}</p>
                    <p><strong>Password:</strong> <code style="background:#eee;padding:3px 7px;">{password}</code></p>
                </div>
            </div>
            <div style="background:#fcfcfc;padding:20px;text-align:center;font-size:12px;color:#999;">
                &copy; {year} HRMate Global Operations</div>
        </div></body></html>"""
    msg.attach(MIMEText(body, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(SENDER_EMAIL, SENDER_PASSWORD); s.send_message(msg)
    except Exception as e: print(f"[EMAIL] Employee: {e}")

def send_meeting_invite(to_email, title, date, time, meet_link=""):
    SENDER_EMAIL = os.getenv("EMAIL_USER")
    SENDER_PASSWORD = os.getenv("EMAIL_PASS")
    meet_section = f"""<div style="background:#e8f5e9;padding:20px;border-radius:6px;margin-top:20px;text-align:center;">
        <p style="font-weight:bold;color:#1b5e20;">🎥 Join via Google Meet</p>
        <a href="{meet_link}" style="display:inline-block;background:#1a73e8;color:#fff;padding:12px 28px;border-radius:6px;text-decoration:none;font-weight:bold;">Join Meeting</a>
        <p style="font-size:12px;color:#1a73e8;">{meet_link}</p></div>""" if meet_link else ""
    msg = MIMEMultipart()
    msg["From"] = f"HRMate Global <{SENDER_EMAIL}>"; msg["To"] = to_email
    msg["Subject"] = f"STRATEGIC SYNC: {title} - {date}"
    body = f"""<html><body style="font-family:'Segoe UI',sans-serif;background:#f9f9f9;padding:20px;">
        <div style="max-width:550px;margin:auto;background:#fff;border:1px solid #ddd;border-radius:8px;overflow:hidden;">
            <div style="background:#000;color:#fff;padding:20px;text-align:center;">
                <h1 style="margin:0;letter-spacing:2px;">HRMATE<span style="color:#2563eb;">+</span></h1>
            </div>
            <div style="padding:30px;">
                <h2>Meeting Notification</h2>
                <div style="background:#f0f7ff;padding:25px;border-radius:6px;border:1px solid #d0e3ff;">
                    <p style="font-size:20px;font-weight:bold;color:#2563eb;">{title}</p>
                    <p><b>Date:</b> {date} | <b>Time:</b> {time}</p>
                </div>{meet_section}
            </div>
        </div></body></html>"""
    msg.attach(MIMEText(body, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(SENDER_EMAIL, SENDER_PASSWORD); s.send_message(msg)
    except Exception as e: print(f"[EMAIL] Meeting: {e}")

def send_payslip_email(to_email, name, month, salary, bonus):
    SENDER_EMAIL = os.getenv("EMAIL_USER")
    SENDER_PASSWORD = os.getenv("EMAIL_PASS")
    total = float(salary) + float(bonus)
    msg = MIMEMultipart()
    msg["From"] = f"HRMate Finance <{SENDER_EMAIL}>"; msg["To"] = to_email
    msg["Subject"] = f"PAY-SLIP: {month} - {name}"
    body = f"""<html><body style="font-family:Arial,sans-serif;background:#f8fafc;padding:40px;">
        <div style="max-width:600px;margin:auto;background:#fff;border-radius:20px;border:1px solid #e2e8f0;padding:40px;">
            <h2 style="color:#1e293b;">Earnings Statement: {month}</h2>
            <p>Hello {name}, your payment has been disbursed.</p>
            <p><strong>Base Salary:</strong> ${salary}</p>
            <p><strong>Bonus:</strong> ${bonus}</p>
            <h1 style="color:#2563eb;">Total: ${total}</h1>
        </div></body></html>"""
    msg.attach(MIMEText(body, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(SENDER_EMAIL, SENDER_PASSWORD); s.send_message(msg)
    except Exception as e: print(f"[EMAIL] Payslip: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYTICS (KMeans clustering for smart project assignment)
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_employees(employee_data):
    feature_matrix = []
    for emp in employee_data:
        total    = int(emp[3])   if emp[3] else 0
        completed= int(emp[4])   if emp[4] else 0
        avg_time = float(emp[5]) if emp[5] else 0
        rate     = completed / total if total > 0 else 0
        feature_matrix.append([total, completed, rate, avg_time])
    X = np.array(feature_matrix)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    return kmeans.labels_

def get_best_cluster(employee_data, labels):
    cluster_scores = defaultdict(list)
    for emp, label in zip(employee_data, labels):
        total     = int(emp[3]) if emp[3] else 0
        completed = int(emp[4]) if emp[4] else 0
        rate      = completed / total if total > 0 else 0
        cluster_scores[label].append(rate)
    return max(cluster_scores, key=lambda k: sum(cluster_scores[k]) / len(cluster_scores[k]))

def calculate_employee_scores():
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("""SELECT e.id,e.name,COUNT(t.id),SUM(CASE WHEN t.status='Completed' THEN 1 ELSE 0 END)
        FROM employees e LEFT JOIN tasks t ON e.id=t.assigned_to GROUP BY e.id""")
    employees = c.fetchall(); conn.close()
    scored = []
    for emp in employees:
        total = emp[2] or 0; completed = emp[3] or 0
        rate  = completed / total if total > 0 else 0
        scored.append((emp[0], emp[1], (rate * 0.7) + (completed * 0.3)))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


# ═══════════════════════════════════════════════════════════════════════════════
#  GOOGLE OAUTH ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/google/auth")
def google_auth():
    hr_id = session.get("hr_id")
    if not hr_id: return redirect(url_for("index"))
    ci          = _load_client_info()
    state       = secrets.token_urlsafe(32)
    redirect_to = request.args.get("next", url_for("hr_dashboard"))
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS oauth_state (state TEXT PRIMARY KEY, hr_id INTEGER, redirect_to TEXT)")
    try: c.execute("ALTER TABLE oauth_state ADD COLUMN code_verifier TEXT")
    except Exception: pass
    c.execute("INSERT OR REPLACE INTO oauth_state VALUES (?,?,?,?)", (state, hr_id, redirect_to, ""))
    conn.commit(); conn.close()
    params = {
        "client_id": ci["client_id"], "redirect_uri": REDIRECT_URI,
        "response_type": "code", "scope": "https://www.googleapis.com/auth/calendar",
        "access_type": "offline", "prompt": "select_account consent",
        "state": state, "include_granted_scopes": "true"
    }
    auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)
    print(f"[AUTH] Redirecting hr_id={hr_id}, state={state[:16]}...")
    return redirect(auth_url)

@app.route("/google/callback")
def google_callback():
    state = request.args.get("state", ""); code = request.args.get("code", "")
    if not code:
        flash("Google auth cancelled or failed.", "danger")
        return redirect(url_for("hr_dashboard"))
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("SELECT hr_id,redirect_to FROM oauth_state WHERE state=?", (state,))
    row = c.fetchone()
    if not row:
        conn.close(); flash("Authentication session expired. Please try again.", "danger")
        return redirect(url_for("hr_dashboard"))
    hr_id, redirect_to = row
    c.execute("DELETE FROM oauth_state WHERE state=?", (state,))
    conn.commit(); conn.close()
    print(f"[CALLBACK] hr_id={hr_id}, exchanging code...")
    ci = _load_client_info()
    resp = _req.post("https://oauth2.googleapis.com/token", data={
        "code": code, "client_id": ci["client_id"], "client_secret": ci["client_secret"],
        "redirect_uri": REDIRECT_URI, "grant_type": "authorization_code"
    })
    token_data = resp.json()
    print(f"[CALLBACK] Token keys: {list(token_data.keys())}")
    if "error" in token_data:
        flash(f"Google auth failed: {token_data.get('error_description', token_data['error'])}", "danger")
        return redirect(url_for("hr_dashboard"))
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("""INSERT INTO google_tokens (hr_id,access_token,refresh_token,client_id,client_secret)
        VALUES (?,?,?,?,?) ON CONFLICT(hr_id) DO UPDATE SET
        access_token=excluded.access_token, refresh_token=excluded.refresh_token,
        client_id=excluded.client_id, client_secret=excluded.client_secret""",
        (hr_id, token_data["access_token"], token_data.get("refresh_token", ""),
         ci["client_id"], ci["client_secret"]))
    conn.commit(); conn.close()
    session["hr_id"] = hr_id; session["google_connected"] = True
    print(f"[CALLBACK] Token saved for hr_id={hr_id}")
    flash("Google Calendar connected! Scheduling your meeting now...", "success")
    return redirect(redirect_to)

@app.route("/complete_pending_meeting")
def complete_pending_meeting():
    if "hr_id" not in session: return redirect(url_for("index"))
    hr_id = session["hr_id"]
    conn = sqlite3.connect(DB); c = conn.cursor()
    try:
        c.execute("""CREATE TABLE IF NOT EXISTS pending_meetings (
            hr_id INTEGER PRIMARY KEY, title TEXT, date TEXT, time TEXT,
            timezone TEXT, participants TEXT, status TEXT)""")
        c.execute("SELECT title,date,time,timezone,participants,status FROM pending_meetings WHERE hr_id=?", (hr_id,))
        row = c.fetchone()
    except Exception as e: print(f"[PENDING] DB error: {e}"); row = None
    if not row:
        flash("No pending meeting found. Please schedule again.", "warning")
        conn.close(); return redirect(url_for("hr_dashboard") + "#meetingModal")
    title, date, time_val, timezone, participants_string, status = row
    p_list = [p.strip() for p in participants_string.split(",") if p.strip()]
    print(f"[PENDING] Found: {title}, {date}, {time_val}, participants={p_list}")
    meet_link = create_google_meet(title, date, time_val, timezone, p_list) or ""
    print(f"[PENDING] Meet link: {repr(meet_link)}")
    c.execute("INSERT INTO meetings (title,date,time,timezone,participants,status,meet_link) VALUES (?,?,?,?,?,?,?)",
              (title, date, time_val, timezone, participants_string, status, meet_link))
    for email in p_list:
        send_meeting_invite(email, title, date, f"{time_val} ({timezone})", meet_link)
    c.execute("DELETE FROM pending_meetings WHERE hr_id=?", (hr_id,))
    conn.commit(); conn.close()
    flash("Meeting scheduled with real Google Meet link!" if meet_link else "Meeting scheduled! (Meet link failed)", "success" if meet_link else "warning")
    return redirect(url_for("hr_dashboard") + "#meetings")

@app.route("/google/disconnect")
def google_disconnect():
    session.pop("google_connected", None)
    hr_id = session.get("hr_id")
    if hr_id:
        conn = sqlite3.connect(DB); c = conn.cursor()
        c.execute("DELETE FROM google_tokens WHERE hr_id=?", (hr_id,))
        conn.commit(); conn.close()
    flash("Google Calendar disconnected.", "info")
    return redirect(url_for("hr_dashboard"))


# ═══════════════════════════════════════════════════════════════════════════════
#  GOOGLE CALENDAR SYNC ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/calendar/sync_all", methods=["POST"])
def calendar_sync_all():
    if "hr_id" not in session: return redirect(url_for("index"))
    result = full_sync()
    flash(f"Sync complete! Pushed {result['pushed_meetings']} meetings, "
          f"{result['pushed_deadlines']} deadlines, imported {result['pulled_events']} new events.", "success")
    return redirect(url_for("hr_dashboard") + "#meetings")

@app.route("/calendar/push_meeting/<int:meeting_id>", methods=["POST"])
def calendar_push_meeting(meeting_id):
    if "hr_id" not in session: return redirect(url_for("index"))
    event_id, msg = push_meeting_to_google(meeting_id)
    flash(f"✅ {msg}" if event_id else f"⚠️ {msg}", "success" if event_id else "warning")
    return redirect(url_for("hr_dashboard") + "#meetings")

@app.route("/calendar/push_deadline/<int:project_id>", methods=["POST"])
def calendar_push_deadline(project_id):
    if "hr_id" not in session: return redirect(url_for("index"))
    event_id, msg = push_deadline_to_google(project_id)
    flash(f"✅ {msg}" if event_id else f"⚠️ {msg}", "success" if event_id else "warning")
    return redirect(url_for("hr_dashboard") + "#projects")

@app.route("/calendar/pull", methods=["POST"])
def calendar_pull():
    if "hr_id" not in session: return redirect(url_for("index"))
    imported, _ = pull_google_events()
    flash(f"Imported {len(imported)} new events from Google Calendar." if imported
          else "No new events to import.", "success" if imported else "info")
    return redirect(url_for("hr_dashboard") + "#meetings")

@app.route("/calendar/sync_status")
def calendar_sync_status():
    if "hr_id" not in session: return jsonify({"error": "Unauthorized"}), 401
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM meetings"); total_m = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM meetings WHERE gcal_event_id IS NOT NULL AND gcal_event_id!=''"); synced_m = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM projects WHERE end_date IS NOT NULL AND end_date!=''"); total_d = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM projects WHERE gcal_event_id IS NOT NULL AND gcal_event_id!=''"); synced_d = c.fetchone()[0]
    conn.close()
    return jsonify({"meetings": {"total": total_m, "synced": synced_m},
                    "deadlines": {"total": total_d, "synced": synced_d}})


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/hr_login")
def hr_login_page():
    return redirect(url_for("index"))

@app.route("/hr_register", methods=["POST"])
def hr_register():
    conn = sqlite3.connect(DB); c = conn.cursor()
    try:
        c.execute("INSERT INTO hr (name,email,mobile_number,password,status) VALUES (?,?,?,?,'pending')",
                  (request.form.get("name"), request.form.get("email"),
                   request.form.get("mobile_number"), request.form.get("password")))
        conn.commit(); return "REGISTRATION_SUCCESSFUL_WAITING_FOR_ADMIN_APPROVAL"
    except sqlite3.IntegrityError: return "EMAIL_ALREADY_EXISTS"
    finally: conn.close()

@app.route("/hr_login", methods=["POST"])
def hr_login_post():
    email = request.form.get("email"); password = request.form.get("password")
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("SELECT id,name,status,is_blocked,login_attempts FROM hr WHERE email=?", (email,))
    hr_check = c.fetchone()
    if hr_check and hr_check[3] == 1:
        conn.close(); flash("Your account is BLOCKED. Contact Admin.", "danger")
        return redirect(url_for("hr_login_page"))
    if password == "RESEND_OTP":
        c.execute("SELECT id,name,status FROM hr WHERE email=?", (email,))
    else:
        c.execute("SELECT id,name,status FROM hr WHERE email=? AND password=?", (email, password))
    hr = c.fetchone()
    if hr:
        if hr[2] != "approved": conn.close(); return "ACCOUNT_PENDING_APPROVAL_CONTACT_ADMIN"
        c.execute("UPDATE hr SET login_attempts=0 WHERE id=?", (hr[0],))
        can_send = True
        if "last_otp_sent" in session:
            if (datetime.now() - datetime.fromisoformat(session["last_otp_sent"])).total_seconds() < 60:
                can_send = False
        if can_send:
            otp = str(random.randint(100000, 999999))
            c.execute("UPDATE hr SET otp=? WHERE id=?", (otp, hr[0]))
            conn.commit(); send_otp_email(email, hr[1], otp)
            session["last_otp_sent"] = datetime.now().isoformat()
        conn.close()
        session["temp_hr_id"] = hr[0]; session["temp_hr_email"] = email; session["otp_tries"] = 0
        return render_template("hr_otp_verify.html", remaining_time=60)
    else:
        if hr_check:
            attempts = hr_check[4] + 1
            if attempts >= 3:
                c.execute("UPDATE hr SET is_blocked=1 WHERE id=?", (hr_check[0],))
                flash("Account Blocked! 3 failed attempts reached.", "danger")
            else:
                c.execute("UPDATE hr SET login_attempts=? WHERE id=?", (attempts, hr_check[0]))
                flash(f"Invalid password. {3-attempts} attempts remaining.", "danger")
            conn.commit()
        else: flash("Invalid HR login credentials", "danger")
        conn.close(); return redirect(url_for("hr_login_page"))

@app.route("/verify_otp", methods=["POST"])
def verify_otp():
    input_otp = request.form.get("otp"); hr_id = session.get("temp_hr_id")
    sent_time_str = session.get("last_otp_sent")
    if not hr_id or not sent_time_str: return redirect(url_for("index"))
    if datetime.now() > datetime.fromisoformat(sent_time_str) + timedelta(minutes=1):
        flash("OTP expired. Login again.", "danger"); return redirect(url_for("hr_login_page"))
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("SELECT otp,name FROM hr WHERE id=?", (hr_id,))
    data = c.fetchone()
    if data and data[0] == input_otp:
        c.execute("UPDATE hr SET otp=NULL,login_attempts=0 WHERE id=?", (hr_id,))
        conn.commit(); conn.close()
        session["hr_id"] = hr_id; session["hr_name"] = data[1]
        session.pop("temp_hr_id", None); session.pop("last_otp_sent", None); session.pop("otp_tries", None)
        return redirect(url_for("hr_dashboard"))
    else:
        session["otp_tries"] = session.get("otp_tries", 0) + 1
        if session["otp_tries"] >= 3:
            c.execute("UPDATE hr SET is_blocked=1,otp=NULL WHERE id=?", (hr_id,))
            conn.commit(); conn.close(); session.clear()
            flash("Account BLOCKED! 3 incorrect OTP attempts.", "danger")
            return redirect(url_for("hr_login_page"))
        conn.close()
        flash(f"Invalid OTP. {3-session['otp_tries']} attempts remaining.", "danger")
        return render_template("hr_otp_verify.html")

@app.route("/logout")
def logout():
    session.clear(); flash("Session terminated successfully.", "success")
    return redirect(url_for("index"))


# ═══════════════════════════════════════════════════════════════════════════════
#  ADMIN ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        if request.form.get("email") == "admin" and request.form.get("password") == "123":
            session["admin_logged_in"] = True; flash("Admin Authenticated", "success")
            return redirect(url_for("admin_dashboard"))
        flash("Invalid Admin Access", "danger"); return redirect(url_for("index"))
    return render_template("admin_login.html")

@app.route("/admin_dashboard")
def admin_dashboard():
    if not session.get("admin_logged_in"): return redirect(url_for("admin_login"))
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("""SELECT h.id,h.name,h.email,h.mobile_number,h.status,
        (SELECT COUNT(*) FROM employees WHERE added_by_hr=h.id),h.is_blocked FROM hr h""")
    hrs = c.fetchall(); conn.close()
    return render_template("admin_dashboard.html", hrs=hrs)

@app.route("/approve_hr/<int:hr_id>")
def approve_hr(hr_id):
    if not session.get("admin_logged_in"): return redirect(url_for("admin_login"))
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("UPDATE hr SET status='approved' WHERE id=?", (hr_id,))
    conn.commit(); conn.close(); flash("HR Account Approved", "success")
    return redirect(url_for("admin_dashboard"))

@app.route("/unblock_hr/<int:hr_id>")
def unblock_hr(hr_id):
    if not session.get("admin_logged_in"): return redirect(url_for("admin_login"))
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("UPDATE hr SET is_blocked=0,login_attempts=0 WHERE id=?", (hr_id,))
    conn.commit(); conn.close(); flash("HR Account Unblocked", "success")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin_view_hr/<int:hr_id>")
def admin_view_hr(hr_id):
    if not session.get("admin_logged_in"): return jsonify({"error": "Unauthorized"}), 401
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("""SELECT e.employee_id,e.name,e.email,e.role,e.department,
        (SELECT COUNT(*) FROM projects WHERE assigned_employees LIKE '%'||e.name||'%')
        FROM employees e WHERE e.added_by_hr=?""", (hr_id,))
    employees = [{"employee_id": r[0], "name": r[1], "email": r[2], "role": r[3],
                  "department": r[4], "project_status": "Active" if r[5]>0 else "Idle",
                  "meeting_status": "Synchronized"} for r in c.fetchall()]
    conn.close(); return jsonify({"employees": employees})

@app.route("/admin_stats_json")
def admin_stats_json():
    if not session.get("admin_logged_in"): return jsonify({"error": "Unauthorized"}), 401
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("SELECT status,COUNT(*) FROM hr GROUP BY status"); hr_status = dict(c.fetchall())
    c.execute("SELECT department,COUNT(*) FROM employees GROUP BY department"); dept_dist = dict(c.fetchall())
    conn.close()
    return jsonify({"hr_status": hr_status, "dept_dist": dept_dist,
                    "telemetry": {"uptime": "99.98%", "active_sessions": random.randint(5,50), "db_size": "1.2 MB"},
                    "growth": {"labels": ["W1","W2","W3","W4"], "data": [12,19,15,28]}})


# ═══════════════════════════════════════════════════════════════════════════════
#  HR DASHBOARD & ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/hr_dashboard")
def hr_dashboard():
    if "hr_id" not in session: return redirect(url_for("hr_login_page"))
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("SELECT employee_id,name,department,role,email FROM employees WHERE added_by_hr=?", (session["hr_id"],))
    employees = c.fetchall()
    c.execute("SELECT * FROM projects"); projects = c.fetchall()
    c.execute("SELECT * FROM meetings"); meetings = c.fetchall()
    c.execute("""SELECT l.id,e.name,l.type,l.start_date,l.end_date,l.status
        FROM leave_requests l JOIN employees e ON l.employee_id=e.employee_id WHERE l.status='Pending'""")
    leaves = c.fetchall()
    c.execute("""SELECT p.month,e.name,p.salary,p.bonus,p.status
        FROM payroll p JOIN employees e ON p.employee_id=e.employee_id ORDER BY p.id DESC""")
    payroll = c.fetchall(); conn.close()
    return render_template("hr_dashboard.html",
                           hr_name=session["hr_name"], employees=employees,
                           projects=projects, meetings=meetings, leaves=leaves,
                           payroll=payroll, timezones=pytz.common_timezones)

@app.route("/get_calendar_events")
def get_calendar_events():
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("SELECT title,date,time FROM meetings")
    meetings  = [{"title": f"Meeting: {m[0]}", "start": f"{m[1]}T{m[2]}", "color": "#2563eb"} for m in c.fetchall()]
    c.execute("SELECT name,end_date FROM projects")
    deadlines = [{"title": f"Deadline: {p[0]}", "start": p[1], "color": "#ef4444"} for p in c.fetchall()]
    conn.close(); return jsonify(meetings + deadlines)

@app.route("/get_stats_json")
def get_stats_json():
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("SELECT department,COUNT(*) FROM employees GROUP BY department")
    data = c.fetchall(); conn.close()
    return jsonify({"labels": [r[0] for r in data], "values": [r[1] for r in data]})

@app.route("/get_meetings_json")
def get_meetings_json():
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("SELECT title,date,time,status,meet_link FROM meetings")
    meetings = c.fetchall(); conn.close()
    return jsonify([{"title": m[0], "start": f"{m[1]}T{m[2]}",
                     "color": "#2563eb" if m[3] != "Urgent" else "#ef4444",
                     "meet_link": m[4] or ""} for m in meetings])

@app.route("/add_employee", methods=["GET", "POST"])
def add_employees():
    if request.method == "GET": return redirect(url_for("hr_dashboard") + "#employeeModal")
    if "hr_id" not in session: return redirect(url_for("hr_login_page"))
    conn = sqlite3.connect(DB); c = conn.cursor()
    try:
        c.execute("""INSERT INTO employees
            (employee_id,name,email,mobile,password,department,role,date,added_by_hr,skills)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (request.form["employee_id"], request.form["name"], request.form["email"],
             request.form["mobile_number"], request.form["password"], request.form["department"],
             request.form["role"], request.form["date_of_joining"], session["hr_id"], request.form["skill"]))
        conn.commit()
        send_employee_email(request.form["email"], request.form["employee_id"], request.form["name"],
                            request.form["password"], request.form["role"], request.form["department"])
        flash("Employee added successfully!", "success"); return redirect(url_for("hr_dashboard"))
    except sqlite3.IntegrityError: flash("Employee ID or Email already exists!", "danger")
    finally: conn.close()
    return redirect(url_for("hr_dashboard") + "#employeeModal")

@app.route("/delete_employee/<int:id>")
def delete_employee(id):
    if "hr_id" not in session: return redirect(url_for("hr_login_page"))
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("DELETE FROM employees WHERE employee_id=?", (id,))
    conn.commit(); conn.close(); flash("Personnel decommissioned.", "success")
    return redirect(url_for("hr_dashboard"))

@app.route("/update_employee", methods=["POST"])
def update_employee():
    if "hr_id" not in session: return redirect(url_for("hr_login_page"))
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("UPDATE employees SET role=?,department=? WHERE id=?",
              (request.form.get("role"), request.form.get("department"), request.form.get("id")))
    conn.commit(); conn.close(); flash("Personnel credentials updated.", "success")
    return redirect(url_for("hr_dashboard"))

@app.route("/create_project", methods=["GET", "POST"])
def create_project():
    if "hr_id" not in session: return redirect(url_for("hr_login_page"))
    if request.method == "GET": return redirect(url_for("hr_dashboard") + "#projectModal")
    required_role = request.form.get("required_role"); required_skill = request.form.get("required_skill")
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("""SELECT e.id,e.name,e.date,COUNT(t.id),
        SUM(CASE WHEN t.status='Completed' THEN 1 ELSE 0 END),
        AVG(CASE WHEN t.status='Completed' THEN julianday(t.deadline)-julianday(e.date) END)
        FROM employees e LEFT JOIN tasks t ON e.id=t.assigned_to
        WHERE LOWER(e.department)=LOWER(?) AND LOWER(e.skills) LIKE LOWER(?) GROUP BY e.id""",
        (required_role, f"%{required_skill}%"))
    employee_data = c.fetchall(); conn.commit(); conn.close()
    if not employee_data:
        flash("No employees with required skill found!", "danger"); return redirect(url_for("hr_dashboard"))
    labels = cluster_employees(employee_data); best_cluster = get_best_cluster(employee_data, labels)
    selected = [emp for emp, lbl in zip(employee_data, labels) if lbl == best_cluster]
    assigned_ids = ",".join(str(emp[0]) for emp in selected[:3])
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("""INSERT INTO projects (name,project_description,progress,start_date,end_date,assigned_employees)
        VALUES (?,?,?,?,?,?)""",
        (request.form.get("name"), request.form.get("description"), request.form.get("progress"),
         request.form.get("start_date"), request.form.get("end_date"), assigned_ids))
    conn.commit(); conn.close(); flash("Project created with smart clustered employees!", "success")
    return redirect(url_for("hr_dashboard"))

@app.route("/delete_project/<int:project_id>")
def delete_project(project_id):
    if "hr_id" not in session: return redirect(url_for("hr_login_page"))
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("DELETE FROM projects WHERE id=?", (project_id,))
    conn.commit(); conn.close(); flash("Workstream terminated.", "success")
    return redirect(url_for("hr_dashboard"))

@app.route("/update_project_progress", methods=["POST"])
def update_project_progress():
    if "hr_id" not in session: return redirect(url_for("hr_login_page"))
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("UPDATE projects SET progress=? WHERE id=?",
              (request.form.get("progress"), request.form.get("project_id")))
    conn.commit(); conn.close(); flash("Project velocity updated.", "success")
    return redirect(url_for("hr_dashboard"))

@app.route("/schedule_meeting", methods=["GET", "POST"])
def schedule_meeting():
    if "hr_id" not in session: return redirect(url_for("hr_login_page"))
    if request.method == "GET": return redirect(url_for("hr_dashboard") + "#meetingModal")
    title = request.form.get("title"); date = request.form.get("date")
    time  = request.form.get("time");  selected_tz = request.form.get("timezone")
    p_list = request.form.getlist("participants"); participants_string = ", ".join(p_list)
    status = request.form.get("status")
    print(f"[SCHEDULE] title={title}, date={date}, time={time}, tz={selected_tz}, participants={p_list}")
    if not selected_tz:
        flash("Please select a timezone.", "danger"); return redirect(url_for("hr_dashboard") + "#meetingModal")
    try: pytz.timezone(selected_tz)
    except Exception as e: print(f"[SCHEDULE] Timezone error: {e}")
    conn_check = sqlite3.connect(DB); c_check = conn_check.cursor()
    try:
        c_check.execute("SELECT access_token FROM google_tokens WHERE hr_id=?", (session.get("hr_id"),))
        has_token = c_check.fetchone()
    except Exception: has_token = None
    conn_check.close()
    if not has_token:
        conn_pm = sqlite3.connect(DB); c_pm = conn_pm.cursor()
        c_pm.execute("""CREATE TABLE IF NOT EXISTS pending_meetings (
            hr_id INTEGER PRIMARY KEY, title TEXT, date TEXT, time TEXT,
            timezone TEXT, participants TEXT, status TEXT)""")
        c_pm.execute("INSERT OR REPLACE INTO pending_meetings VALUES (?,?,?,?,?,?,?)",
                     (session.get("hr_id"), title, date, time, selected_tz, participants_string, status))
        conn_pm.commit(); conn_pm.close()
        flash("Connect your Google account first — meeting details saved.", "warning")
        return redirect(url_for("google_auth") + "?next=/complete_pending_meeting")
    meet_link = create_google_meet(title, date, time, selected_tz, p_list) or ""
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("INSERT INTO meetings (title,date,time,timezone,participants,status,meet_link) VALUES (?,?,?,?,?,?,?)",
              (title, date, time, selected_tz, participants_string, status, meet_link))
    conn.commit(); conn.close()
    for email in p_list: send_meeting_invite(email, title, date, f"{time} ({selected_tz})", meet_link)
    flash("Meeting scheduled with real Google Meet link!" if meet_link
          else "Meeting saved! (Meet link failed — check terminal)", "success" if meet_link else "warning")
    return redirect(url_for("hr_dashboard") + "#meetings")

@app.route("/delete_meeting/<int:meeting_id>")
def delete_meeting(meeting_id):
    if "hr_id" not in session: return redirect(url_for("hr_login_page"))
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("DELETE FROM meetings WHERE id=?", (meeting_id,))
    conn.commit(); conn.close(); flash("Meeting purged.", "success")
    return redirect(url_for("hr_dashboard") + "#meetings")

@app.route("/update_meeting_status/<int:meeting_id>/<string:new_status>")
def update_meeting_status(meeting_id, new_status):
    if "hr_id" not in session: return redirect(url_for("hr_login_page"))
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("UPDATE meetings SET status=? WHERE id=?", (new_status, meeting_id))
    conn.commit(); conn.close(); flash(f"Status updated to {new_status}.", "success")
    return redirect(url_for("hr_dashboard") + "#meetings")

@app.route("/update_leave/<int:leave_id>/<string:status>")
def update_leave(leave_id, status):
    if "hr_id" not in session: return redirect(url_for("index"))
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("UPDATE leave_requests SET status=? WHERE id=?", (status, leave_id))
    conn.commit(); conn.close(); flash(f"Request marked as {status}", "success")
    return redirect(url_for("hr_dashboard"))

@app.route("/process_payroll", methods=["POST"])
def process_payroll():
    emp_id = request.form.get("employee_id"); salary = request.form.get("salary")
    bonus  = request.form.get("bonus", 0);    month  = datetime.now().strftime("%B %Y")
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("SELECT name,email FROM employees WHERE employee_id=?", (emp_id,))
    employee = c.fetchone()
    if employee:
        c.execute("INSERT INTO payroll (employee_id,month,salary,bonus,status) VALUES (?,?,?,?,'Paid')",
                  (emp_id, month, salary, bonus))
        conn.commit(); send_payslip_email(employee[1], employee[0], month, salary, bonus)
    conn.close(); return redirect(url_for("hr_dashboard"))


# ═══════════════════════════════════════════════════════════════════════════════
#  EMPLOYEE ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/employee_login", methods=["POST"])
def employee_login():
    login_input = request.form["emp_id"]; password = request.form["password"]
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("SELECT * FROM employees WHERE (employee_id=? OR email=?) AND password=?",
              (login_input, login_input, password))
    emp = c.fetchone(); conn.close()
    if emp:
        session["emp_id"] = emp[1]; session["emp_name"] = emp[2]
        flash("Login Successful!", "success"); return redirect(url_for("employee_dashboard"))
    flash("Invalid Employee ID/Email or Password", "danger"); return redirect(url_for("index"))

@app.route("/employee_dashboard")
def employee_dashboard():
    if "emp_id" not in session: return redirect(url_for("index"))
    emp_id = session["emp_id"]
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("SELECT id FROM employees WHERE employee_id=?", (emp_id,))
    result = c.fetchone(); employee_pk = str(result[0]) if result else None
    if employee_pk:
        c.execute("""SELECT * FROM projects WHERE assigned_employees=?
            OR assigned_employees LIKE ? OR assigned_employees LIKE ? OR assigned_employees LIKE ?""",
            (employee_pk, employee_pk+",%", "%,"+employee_pk+",%", "%,"+employee_pk))
        all_projects = c.fetchall()
    else: all_projects = []
    active_projects    = [p for p in all_projects if p[7] != "Completed" and p[3] != 100]
    completed_projects = [p for p in all_projects if p[7] == "Completed" or p[3] == 100]
    c.execute("SELECT email FROM employees WHERE employee_id=?", (emp_id,))
    emp_email_row = c.fetchone(); emp_email = emp_email_row[0] if emp_email_row else ""
    c.execute("SELECT * FROM meetings WHERE participants LIKE ? OR participants LIKE ?",
              ('%'+session["emp_name"]+'%', '%'+emp_email+'%'))
    meetings = c.fetchall()
    c.execute("SELECT * FROM leave_requests WHERE employee_id=?", (session["emp_id"],))
    leaves = c.fetchall(); conn.close()
    return render_template("employee_dashboard.html",
                           emp_id=session["emp_id"], emp_name=session["emp_name"],
                           active_projects=active_projects, completed_projects=completed_projects,
                           meetings=meetings, leaves=leaves)

@app.route("/apply_leave", methods=["POST"])
def apply_leave():
    if "emp_id" not in session: return redirect(url_for("index"))
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("INSERT INTO leave_requests (employee_id,type,start_date,end_date,reason) VALUES (?,?,?,?,?)",
              (session["emp_id"], request.form.get("leave_type"),
               request.form.get("start_date"), request.form.get("end_date"), request.form.get("reason")))
    conn.commit(); conn.close(); flash("Leave request submitted!", "success")
    return redirect(url_for("employee_dashboard"))

@app.route("/update_progress", methods=["POST"])
def update_progress():
    project_id = request.form["project_id"]; progress = int(request.form["progress"])
    status = "Completed" if progress >= 100 else "In Progress"; progress = min(progress, 100)
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("UPDATE projects SET progress=?,status=? WHERE id=?", (progress, status, project_id))
    conn.commit(); conn.close(); return redirect(url_for("employee_dashboard"))

@app.route("/api/performance_stats")
def performance_stats():
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute("SELECT status,COUNT(*) FROM leave_requests GROUP BY status")
    data = dict(c.fetchall()); conn.close(); return jsonify(data)


# ═══════════════════════════════════════════════════════════════════════════════
#  CHATBOT ROUTE
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/chat", methods=["POST"])
def chat():
    message = request.json.get("message", "").strip()

    if not message:
        return jsonify({"response": "No message received"})

    # 🔥 NLP intent prediction
    ints = predict_class(message, model)

    if not ints or float(ints[0]['probability']) < 0.3:
        response = "I'm not sure about that. Try asking HR-related questions."
    else:
        response = get_response(ints, intents)

    # Save chat log
    user_id = session.get("hr_id") or session.get("emp_id")

    try:
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute(
            "INSERT INTO chatlogs (user_id, message, bot_response, timescamp) VALUES (?, ?, ?, ?)",
            (user_id, message, response, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[CHAT ERROR] {e}")

    return jsonify({"response": response})


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=True, port=2300)
