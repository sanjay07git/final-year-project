# ─────────────────────────────────────────────────────────────────────────────
#  HRMate  |  app.py  |  Production-ready Flask application
#  Updated for Render deployment — security hardened, env-driven config
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import uuid
import random
import pickle
import secrets
import sqlite3
import smtplib
import urllib.parse
import urllib.request

import nltk
import numpy as np
import pytz
import requests as _req

from dotenv import load_dotenv
load_dotenv()

from collections          import defaultdict
from datetime             import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from functools            import wraps

from flask import (Flask, flash, jsonify, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import generate_password_hash, check_password_hash

# ── Optional heavy deps (graceful degradation if missing) ─────────────────────
try:
    from keras.models import load_model
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

try:
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build as gcal_build
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# ─── NLTK data bootstrap ──────────────────────────────────────────────────────
_nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(_nltk_data_path):
    os.makedirs(_nltk_data_path, exist_ok=True)
for _pkg in ("punkt", "wordnet"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if _pkg == "punkt" else f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, download_dir=_nltk_data_path, quiet=True)
if _nltk_data_path not in nltk.data.path:
    nltk.data.path.append(_nltk_data_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  APP CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

# Secret key — MUST be set in Render environment variables
app.secret_key = os.environ["SECRET_KEY"]

# Cookie security — use Secure cookies in production (HTTPS on Render)
_IS_PROD = os.getenv("RENDER", "") != ""
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=_IS_PROD,   # True on Render (HTTPS), False locally
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_PERMANENT=False,
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
)

# Allow OAuth over plain HTTP only in local dev
if not _IS_PROD:
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# ─── Database ─────────────────────────────────────────────────────────────────
# Render free tier uses an ephemeral filesystem — data is lost on redeploy.
# For persistence, set DATABASE_URL to a PostgreSQL connection string and
# replace sqlite3 calls with psycopg2/SQLAlchemy.  SQLite is kept here as a
# drop-in default for quick demos.
DB = os.getenv("DB_PATH", "hr_database.db")

# ─── Google OAuth ─────────────────────────────────────────────────────────────
GOOGLE_CLIENT_SECRETS = os.getenv("GOOGLE_CLIENT_SECRETS_FILE", "Se.json")
SCOPES       = ["https://www.googleapis.com/auth/calendar"]
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://127.0.0.1:2300/google/callback")
HRMATE_TAG   = "[HRMate]"

# ─── Admin credentials (NEVER hardcode — use env vars) ────────────────────────
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")   # Required — no default in prod


# ═══════════════════════════════════════════════════════════════════════════════
#  NLP CHATBOT  (lazy-loaded to keep startup fast on Render)
# ═══════════════════════════════════════════════════════════════════════════════

_nlp_model   = None
_lemmatizer  = WordNetLemmatizer() if NLTK_AVAILABLE else None

def _load_nlp_assets():
    """Load chatbot assets once, cache globally."""
    global _nlp_model
    if _nlp_model is not None:
        return _nlp_model
    if not KERAS_AVAILABLE:
        return None
    try:
        _nlp_model = load_model("chatbot_model.h5")
    except Exception as e:
        print(f"[NLP] Could not load model: {e}")
    return _nlp_model

try:
    _intents = json.loads(open("intents.json", encoding="utf-8").read())
    _words   = pickle.load(open("words.pkl", "rb"))
    _classes = pickle.load(open("classes.pkl", "rb"))
    NLP_ASSETS_AVAILABLE = True
except Exception as _e:
    print(f"[NLP] Assets not found — chatbot will return fallback: {_e}")
    _intents = {"intents": []}
    _words   = []
    _classes = []
    NLP_ASSETS_AVAILABLE = False


def _clean_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    return [_lemmatizer.lemmatize(w.lower()) for w in tokens]

def _bow(sentence):
    sentence_words = _clean_sentence(sentence)
    bag = [0] * len(_words)
    for s in sentence_words:
        for i, w in enumerate(_words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    model = _load_nlp_assets()
    if not model or not NLP_ASSETS_AVAILABLE:
        return []
    p   = _bow(sentence)
    res = model.predict(np.array([p]), verbose=0)[0]
    results = [[i, r] for i, r in enumerate(res) if r > 0.25]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": _classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(ints):
    if not ints:
        return "I'm sorry, I don't understand that."
    tag = ints[0]["intent"]
    for i in _intents["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])
    return "I'm sorry, I don't understand that."


# ═══════════════════════════════════════════════════════════════════════════════
#  DATABASE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_db():
    """Return a new SQLite connection with row_factory set."""
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        c = conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS hr (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                mobile_number TEXT,
                password TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                otp TEXT,
                otp_expires_at TEXT,
                login_attempts INTEGER DEFAULT 0,
                is_blocked INTEGER DEFAULT 0
            );
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
                added_by_hr INTEGER,
                skills TEXT
            );
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                project_description TEXT,
                progress TEXT,
                start_date TEXT,
                end_date TEXT,
                assigned_employees TEXT,
                status TEXT DEFAULT 'In Progress',
                gcal_event_id TEXT
            );
            CREATE TABLE IF NOT EXISTS meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                timezone TEXT DEFAULT 'UTC',
                participants TEXT,
                status TEXT,
                meet_link TEXT,
                gcal_event_id TEXT
            );
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                task_name TEXT NOT NULL,
                assigned_to INTEGER,
                deadline TEXT,
                status TEXT,
                FOREIGN KEY (project_id) REFERENCES projects(id),
                FOREIGN KEY (assigned_to) REFERENCES employees(id)
            );
            CREATE TABLE IF NOT EXISTS chatlogs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                message TEXT,
                bot_response TEXT,
                timescamp TEXT
            );
            CREATE TABLE IF NOT EXISTS leave_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT,
                type TEXT,
                start_date TEXT,
                end_date TEXT,
                reason TEXT,
                status TEXT DEFAULT 'Pending'
            );
            CREATE TABLE IF NOT EXISTS payroll (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT,
                month TEXT,
                salary REAL,
                bonus REAL,
                status TEXT DEFAULT 'Unpaid'
            );
            CREATE TABLE IF NOT EXISTS google_tokens (
                hr_id INTEGER PRIMARY KEY,
                access_token TEXT,
                refresh_token TEXT,
                client_id TEXT,
                client_secret TEXT
            );
            CREATE TABLE IF NOT EXISTS oauth_state (
                state TEXT PRIMARY KEY,
                hr_id INTEGER,
                redirect_to TEXT,
                code_verifier TEXT
            );
            CREATE TABLE IF NOT EXISTS pending_meetings (
                hr_id INTEGER PRIMARY KEY,
                title TEXT,
                date TEXT,
                time TEXT,
                timezone TEXT,
                participants TEXT,
                status TEXT
            );
        """)
        conn.commit()


def migrate_db():
    """Non-destructive migrations — safe to run on every startup."""
    migrations = [
        "ALTER TABLE meetings  ADD COLUMN timezone TEXT DEFAULT 'UTC'",
        "ALTER TABLE meetings  ADD COLUMN meet_link TEXT",
        "ALTER TABLE meetings  ADD COLUMN gcal_event_id TEXT",
        "ALTER TABLE projects  ADD COLUMN gcal_event_id TEXT",
        "ALTER TABLE hr        ADD COLUMN otp_expires_at TEXT",
    ]
    with get_db() as conn:
        c = conn.cursor()
        for sql in migrations:
            try:
                c.execute(sql)
                conn.commit()
            except sqlite3.OperationalError:
                pass   # Column already exists


init_db()
migrate_db()


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTH DECORATORS
# ═══════════════════════════════════════════════════════════════════════════════

def hr_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "hr_id" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            flash("Admin access required.", "danger")
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated


def employee_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "emp_id" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated


# ═══════════════════════════════════════════════════════════════════════════════
#  EMAIL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_SENDER_EMAIL    = os.getenv("EMAIL_USER", "")
_SENDER_PASSWORD = os.getenv("EMAIL_PASSWORD", "")   # One env var, used everywhere


def _send_email(to_email: str, subject: str, html_body: str):
    """Central SMTP send — all email functions delegate here."""
    if not _SENDER_EMAIL or not _SENDER_PASSWORD:
        print(f"[EMAIL] Skipped (EMAIL_USER/EMAIL_PASSWORD not set) → {subject}")
        return
    msg = MIMEMultipart()
    msg["From"]    = f"HRMate <{_SENDER_EMAIL}>"
    msg["To"]      = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as s:
            s.login(_SENDER_EMAIL, _SENDER_PASSWORD)
            s.send_message(msg)
    except Exception as e:
        print(f"[EMAIL] Failed ({subject}): {e}")


def send_otp_email(to_email, name, otp):
    _send_email(
        to_email,
        "Secure Access OTP - HRMate",
        f"""<html><body style="font-family:Arial;padding:20px;">
        <h2 style="color:#2563eb;">Verification Code</h2><p>Hello {name},</p>
        <p>Your OTP for HR login:</p>
        <h1 style="background:#f4f4f4;padding:10px;letter-spacing:5px;">{otp}</h1>
        <p>Valid for 10 minutes.</p></body></html>"""
    )


def send_employee_email(to_email, employee_id, name, role, dept):
    """Sends welcome credentials — password is NOT included for security."""
    year = datetime.now().year
    _send_email(
        to_email,
        f"OFFICIAL: Corporate Access Credentials - {name}",
        f"""<html><body style="font-family:Arial,sans-serif;background:#f4f4f4;padding:20px;">
        <div style="max-width:600px;margin:auto;background:#fff;border-radius:10px;overflow:hidden;">
            <div style="background:#000;color:#fff;padding:30px;text-align:center;">
                <h1 style="margin:0;letter-spacing:3px;">HRMATE<span style="color:#2563eb;">+</span></h1>
            </div>
            <div style="padding:40px;">
                <h2>Welcome to the team, {name}!</h2>
                <div style="background:#f9f9f9;border-left:5px solid #2563eb;padding:25px;margin:30px 0;">
                    <p><strong>Employee ID:</strong> <span style="color:#2563eb;">{employee_id}</span></p>
                    <p><strong>Role:</strong> {role} | <strong>Department:</strong> {dept}</p>
                    <p>Your temporary password has been set. Please change it on first login.</p>
                </div>
            </div>
            <div style="background:#fcfcfc;padding:20px;text-align:center;font-size:12px;color:#999;">
                &copy; {year} HRMate Global Operations</div>
        </div></body></html>"""
    )


def send_meeting_invite(to_email, title, date, time_display, meet_link=""):
    meet_section = f"""<div style="background:#e8f5e9;padding:20px;border-radius:6px;margin-top:20px;text-align:center;">
        <p style="font-weight:bold;color:#1b5e20;">Join via Google Meet</p>
        <a href="{meet_link}" style="display:inline-block;background:#1a73e8;color:#fff;padding:12px 28px;border-radius:6px;text-decoration:none;font-weight:bold;">Join Meeting</a>
        <p style="font-size:12px;color:#1a73e8;">{meet_link}</p></div>""" if meet_link else ""
    _send_email(
        to_email,
        f"Meeting Invite: {title} — {date}",
        f"""<html><body style="font-family:'Segoe UI',sans-serif;background:#f9f9f9;padding:20px;">
        <div style="max-width:550px;margin:auto;background:#fff;border:1px solid #ddd;border-radius:8px;overflow:hidden;">
            <div style="background:#000;color:#fff;padding:20px;text-align:center;">
                <h1 style="margin:0;letter-spacing:2px;">HRMATE<span style="color:#2563eb;">+</span></h1>
            </div>
            <div style="padding:30px;">
                <h2>Meeting Notification</h2>
                <div style="background:#f0f7ff;padding:25px;border-radius:6px;border:1px solid #d0e3ff;">
                    <p style="font-size:20px;font-weight:bold;color:#2563eb;">{title}</p>
                    <p><b>Date:</b> {date} | <b>Time:</b> {time_display}</p>
                </div>{meet_section}
            </div>
        </div></body></html>"""
    )


def send_payslip_email(to_email, name, month, salary, bonus):
    total = float(salary) + float(bonus)
    _send_email(
        to_email,
        f"Pay Slip: {month} - {name}",
        f"""<html><body style="font-family:Arial,sans-serif;background:#f8fafc;padding:40px;">
        <div style="max-width:600px;margin:auto;background:#fff;border-radius:20px;border:1px solid #e2e8f0;padding:40px;">
            <h2 style="color:#1e293b;">Earnings Statement: {month}</h2>
            <p>Hello {name}, your payment has been processed.</p>
            <p><strong>Base Salary:</strong> ${salary:,.2f}</p>
            <p><strong>Bonus:</strong> ${float(bonus):,.2f}</p>
            <h1 style="color:#2563eb;">Total: ${total:,.2f}</h1>
        </div></body></html>"""
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  GOOGLE CALENDAR HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _load_client_info():
    if not os.path.exists(GOOGLE_CLIENT_SECRETS):
        # Fall back to env vars (better for Render where files aren't committed)
        return {
            "client_id":     os.environ.get("GOOGLE_CLIENT_ID", ""),
            "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET", ""),
        }
    with open(GOOGLE_CLIENT_SECRETS) as f:
        return json.load(f)["web"]


def get_calendar_service():
    if not GOOGLE_AVAILABLE:
        return None
    hr_id = session.get("hr_id")
    if not hr_id:
        return None
    with get_db() as conn:
        row = conn.execute(
            "SELECT access_token,refresh_token,client_id,client_secret FROM google_tokens WHERE hr_id=?",
            (hr_id,)
        ).fetchone()
    if not row:
        return None
    creds = Credentials(
        token=row["access_token"], refresh_token=row["refresh_token"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=row["client_id"], client_secret=row["client_secret"],
        scopes=SCOPES
    )
    try:
        return gcal_build("calendar", "v3", credentials=creds)
    except Exception as e:
        print(f"[GCAL] Build error: {e}")
        return None


def create_google_meet(title, date, time_str, timezone, participants):
    try:
        service = get_calendar_service()
        if not service:
            return None
        tz       = pytz.timezone(timezone)
        start_dt = tz.localize(datetime.strptime(f"{date} {time_str}", "%Y-%m-%d %H:%M"))
        end_dt   = start_dt + timedelta(hours=1)
        event = {
            "summary": title,
            "start": {"dateTime": start_dt.isoformat(), "timeZone": timezone},
            "end":   {"dateTime": end_dt.isoformat(),   "timeZone": timezone},
            "attendees": [{"email": p} for p in participants],
            "conferenceData": {
                "createRequest": {
                    "requestId": str(uuid.uuid4()),
                    "conferenceSolutionKey": {"type": "hangoutsMeet"}
                }
            }
        }
        created = service.events().insert(
            calendarId="primary", body=event,
            conferenceDataVersion=1, sendUpdates="all"
        ).execute()
        return created.get("hangoutLink")
    except Exception as e:
        print(f"[MEET ERROR] {e}")
        return None


def push_meeting_to_google(meeting_id):
    service = get_calendar_service()
    if not service:
        return None, "Not connected to Google Calendar"
    with get_db() as conn:
        row = conn.execute(
            "SELECT title,date,time,timezone,participants,status,meet_link,gcal_event_id FROM meetings WHERE id=?",
            (meeting_id,)
        ).fetchone()
    if not row:
        return None, "Meeting not found"
    title, date, time_str, tz, participants, status, meet_link, existing_id = \
        row["title"], row["date"], row["time"], row["timezone"], row["participants"], \
        row["status"], row["meet_link"], row["gcal_event_id"]
    end_time = (datetime.strptime(f"{date} {time_str}", "%Y-%m-%d %H:%M") + timedelta(hours=1)).strftime("%H:%M")
    attendees = [{"email": e.strip()} for e in (participants or "").split(",") if e.strip()]
    event = {
        "summary":     f"{HRMATE_TAG} {title}",
        "description": f"Scheduled via HRMate\nStatus: {status}",
        "start":       {"dateTime": f"{date}T{time_str}:00", "timeZone": tz or "UTC"},
        "end":         {"dateTime": f"{date}T{end_time}:00", "timeZone": tz or "UTC"},
        "attendees":   attendees,
        "location":    meet_link or "",
        "reminders":   {"useDefault": False, "overrides": [
            {"method": "email",  "minutes": 60},
            {"method": "popup",  "minutes": 10},
        ]},
    }
    try:
        if existing_id:
            result   = service.events().update(calendarId="primary", eventId=existing_id, body=event, sendUpdates="all").execute()
            event_id = result["id"]; msg = "Meeting updated on Google Calendar"
        else:
            result   = service.events().insert(calendarId="primary", body=event, conferenceDataVersion=1, sendUpdates="all").execute()
            event_id = result["id"]; msg = "Meeting pushed to Google Calendar"
        with get_db() as conn:
            conn.execute("UPDATE meetings SET gcal_event_id=? WHERE id=?", (event_id, meeting_id))
            conn.commit()
        return event_id, msg
    except Exception as e:
        print(f"[SYNC] Push meeting error: {e}")
        return None, str(e)


def push_deadline_to_google(project_id):
    service = get_calendar_service()
    if not service:
        return None, "Not connected to Google Calendar"
    with get_db() as conn:
        row = conn.execute(
            "SELECT name,end_date,project_description,gcal_event_id FROM projects WHERE id=?",
            (project_id,)
        ).fetchone()
    if not row or not row["end_date"]:
        return None, "Project not found or no deadline set"
    name, end_date, description, existing_id = \
        row["name"], row["end_date"], row["project_description"], row["gcal_event_id"]
    event = {
        "summary":     f"{HRMATE_TAG} DEADLINE: {name}",
        "description": f"Project deadline from HRMate\n{description or ''}",
        "start":       {"date": end_date},
        "end":         {"date": end_date},
        "colorId":     "11",
        "reminders":   {"useDefault": False, "overrides": [
            {"method": "email",  "minutes": 1440},
            {"method": "popup",  "minutes": 60},
        ]},
    }
    try:
        if existing_id:
            result   = service.events().update(calendarId="primary", eventId=existing_id, body=event).execute()
            event_id = result["id"]; msg = "Deadline updated on Google Calendar"
        else:
            result   = service.events().insert(calendarId="primary", body=event).execute()
            event_id = result["id"]; msg = "Deadline pushed to Google Calendar"
        with get_db() as conn:
            conn.execute("UPDATE projects SET gcal_event_id=? WHERE id=?", (event_id, project_id))
            conn.commit()
        return event_id, msg
    except Exception as e:
        print(f"[SYNC] Deadline error: {e}")
        return None, str(e)


def pull_google_events():
    service = get_calendar_service()
    if not service:
        return [], "Not connected"
    now    = datetime.utcnow().isoformat() + "Z"
    future = (datetime.utcnow() + timedelta(days=60)).isoformat() + "Z"
    try:
        result = service.events().list(
            calendarId="primary", timeMin=now, timeMax=future,
            maxResults=50, singleEvents=True, orderBy="startTime"
        ).execute()
        events = result.get("items", [])
    except Exception as e:
        print(f"[SYNC] Pull error: {e}")
        return [], str(e)
    imported = []
    with get_db() as conn:
        for event in events:
            summary = event.get("summary", "")
            if summary.startswith(HRMATE_TAG):
                continue
            gcal_id = event["id"]
            if conn.execute("SELECT id FROM meetings WHERE gcal_event_id=?", (gcal_id,)).fetchone():
                continue
            start = event.get("start", {})
            if "dateTime" in start:
                dt       = datetime.fromisoformat(start["dateTime"].replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d")
                time_str = dt.strftime("%H:%M")
                tz       = start.get("timeZone", "UTC")
            elif "date" in start:
                date_str = start["date"]; time_str = "09:00"; tz = "UTC"
            else:
                continue
            participants = ", ".join(
                a.get("email", "") for a in event.get("attendees", []) if "email" in a
            )
            meet_link = event.get("hangoutLink", "")
            conn.execute(
                "INSERT INTO meetings (title,date,time,timezone,participants,status,meet_link,gcal_event_id) VALUES (?,?,?,?,?,?,?,?)",
                (summary, date_str, time_str, tz, participants, "Scheduled", meet_link, gcal_id)
            )
            imported.append({"title": summary, "date": date_str})
        conn.commit()
    return imported, "Success"


def full_sync():
    with get_db() as conn:
        m_ids = [r[0] for r in conn.execute("SELECT id FROM meetings").fetchall()]
        p_ids = [r[0] for r in conn.execute(
            "SELECT id FROM projects WHERE end_date IS NOT NULL AND end_date!=''"
        ).fetchall()]
    pushed_m = sum(1 for mid in m_ids if push_meeting_to_google(mid)[0])
    pushed_d = sum(1 for pid in p_ids if push_deadline_to_google(pid)[0])
    pulled, _ = pull_google_events()
    return {"pushed_meetings": pushed_m, "pushed_deadlines": pushed_d, "pulled_events": len(pulled)}


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYTICS  (KMeans clustering for smart project assignment)
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_employees(employee_data):
    if not SKLEARN_AVAILABLE:
        return [0] * len(employee_data)
    feature_matrix = []
    for emp in employee_data:
        total     = int(emp[3])   if emp[3] else 0
        completed = int(emp[4])   if emp[4] else 0
        avg_time  = float(emp[5]) if emp[5] else 0.0
        rate      = completed / total if total > 0 else 0
        feature_matrix.append([total, completed, rate, avg_time])
    X = np.array(feature_matrix)
    n_clusters = min(3, len(employee_data))   # Can't have more clusters than samples
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
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
    with get_db() as conn:
        employees = conn.execute("""
            SELECT e.id, e.name, COUNT(t.id),
                   SUM(CASE WHEN t.status='Completed' THEN 1 ELSE 0 END)
            FROM employees e
            LEFT JOIN tasks t ON e.id = t.assigned_to
            GROUP BY e.id
        """).fetchall()
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
@hr_required
def google_auth():
    ci    = _load_client_info()
    state = secrets.token_urlsafe(32)
    redirect_to = request.args.get("next", url_for("hr_dashboard"))
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO oauth_state VALUES (?,?,?,?)",
            (state, session["hr_id"], redirect_to, "")
        )
        conn.commit()
    params = {
        "client_id": ci["client_id"], "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/calendar",
        "access_type": "offline", "prompt": "select_account consent",
        "state": state, "include_granted_scopes": "true",
    }
    return redirect("https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params))


@app.route("/google/callback")
def google_callback():
    state = request.args.get("state", "")
    code  = request.args.get("code", "")
    if not code:
        flash("Google auth cancelled or failed.", "danger")
        return redirect(url_for("hr_dashboard"))
    with get_db() as conn:
        row = conn.execute(
            "SELECT hr_id,redirect_to FROM oauth_state WHERE state=?", (state,)
        ).fetchone()
        if not row:
            flash("Authentication session expired. Please try again.", "danger")
            return redirect(url_for("hr_dashboard"))
        hr_id, redirect_to = row["hr_id"], row["redirect_to"]
        conn.execute("DELETE FROM oauth_state WHERE state=?", (state,))
        conn.commit()
    ci   = _load_client_info()
    resp = _req.post("https://oauth2.googleapis.com/token", data={
        "code": code, "client_id": ci["client_id"], "client_secret": ci["client_secret"],
        "redirect_uri": REDIRECT_URI, "grant_type": "authorization_code",
    }, timeout=10)
    token_data = resp.json()
    if "error" in token_data:
        flash(f"Google auth failed: {token_data.get('error_description', token_data['error'])}", "danger")
        return redirect(url_for("hr_dashboard"))
    with get_db() as conn:
        conn.execute("""
            INSERT INTO google_tokens (hr_id,access_token,refresh_token,client_id,client_secret)
            VALUES (?,?,?,?,?)
            ON CONFLICT(hr_id) DO UPDATE SET
                access_token=excluded.access_token,
                refresh_token=excluded.refresh_token,
                client_id=excluded.client_id,
                client_secret=excluded.client_secret
        """, (hr_id, token_data["access_token"], token_data.get("refresh_token", ""),
              ci["client_id"], ci["client_secret"]))
        conn.commit()
    session["hr_id"]           = hr_id
    session["google_connected"] = True
    flash("Google Calendar connected!", "success")
    return redirect(redirect_to)


@app.route("/complete_pending_meeting")
@hr_required
def complete_pending_meeting():
    hr_id = session["hr_id"]
    with get_db() as conn:
        row = conn.execute(
            "SELECT title,date,time,timezone,participants,status FROM pending_meetings WHERE hr_id=?",
            (hr_id,)
        ).fetchone()
    if not row:
        flash("No pending meeting found. Please schedule again.", "warning")
        return redirect(url_for("hr_dashboard") + "#meetingModal")
    title, date, time_val, timezone, participants_string, status = \
        row["title"], row["date"], row["time"], row["timezone"], row["participants"], row["status"]
    p_list    = [p.strip() for p in participants_string.split(",") if p.strip()]
    meet_link = create_google_meet(title, date, time_val, timezone, p_list) or ""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO meetings (title,date,time,timezone,participants,status,meet_link) VALUES (?,?,?,?,?,?,?)",
            (title, date, time_val, timezone, participants_string, status, meet_link)
        )
        conn.execute("DELETE FROM pending_meetings WHERE hr_id=?", (hr_id,))
        conn.commit()
    for email in p_list:
        send_meeting_invite(email, title, date, f"{time_val} ({timezone})", meet_link)
    flash(
        "Meeting scheduled with Google Meet link!" if meet_link else "Meeting saved! (Meet link unavailable)",
        "success" if meet_link else "warning"
    )
    return redirect(url_for("hr_dashboard") + "#meetings")


@app.route("/google/disconnect")
@hr_required
def google_disconnect():
    session.pop("google_connected", None)
    with get_db() as conn:
        conn.execute("DELETE FROM google_tokens WHERE hr_id=?", (session["hr_id"],))
        conn.commit()
    flash("Google Calendar disconnected.", "info")
    return redirect(url_for("hr_dashboard"))


# ═══════════════════════════════════════════════════════════════════════════════
#  CALENDAR SYNC ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/calendar/sync_all", methods=["POST"])
@hr_required
def calendar_sync_all():
    result = full_sync()
    flash(
        f"Sync complete! Pushed {result['pushed_meetings']} meetings, "
        f"{result['pushed_deadlines']} deadlines, imported {result['pulled_events']} new events.",
        "success"
    )
    return redirect(url_for("hr_dashboard") + "#meetings")


@app.route("/calendar/push_meeting/<int:meeting_id>", methods=["POST"])
@hr_required
def calendar_push_meeting(meeting_id):
    event_id, msg = push_meeting_to_google(meeting_id)
    flash(f"✅ {msg}" if event_id else f"⚠️ {msg}", "success" if event_id else "warning")
    return redirect(url_for("hr_dashboard") + "#meetings")


@app.route("/calendar/push_deadline/<int:project_id>", methods=["POST"])
@hr_required
def calendar_push_deadline(project_id):
    event_id, msg = push_deadline_to_google(project_id)
    flash(f"✅ {msg}" if event_id else f"⚠️ {msg}", "success" if event_id else "warning")
    return redirect(url_for("hr_dashboard") + "#projects")


@app.route("/calendar/pull", methods=["POST"])
@hr_required
def calendar_pull():
    imported, _ = pull_google_events()
    flash(
        f"Imported {len(imported)} new events from Google Calendar." if imported else "No new events to import.",
        "success" if imported else "info"
    )
    return redirect(url_for("hr_dashboard") + "#meetings")


@app.route("/calendar/sync_status")
@hr_required
def calendar_sync_status():
    with get_db() as conn:
        total_m  = conn.execute("SELECT COUNT(*) FROM meetings").fetchone()[0]
        synced_m = conn.execute("SELECT COUNT(*) FROM meetings WHERE gcal_event_id IS NOT NULL AND gcal_event_id!=''").fetchone()[0]
        total_d  = conn.execute("SELECT COUNT(*) FROM projects WHERE end_date IS NOT NULL AND end_date!=''").fetchone()[0]
        synced_d = conn.execute("SELECT COUNT(*) FROM projects WHERE gcal_event_id IS NOT NULL AND gcal_event_id!=''").fetchone()[0]
    return jsonify({
        "meetings":  {"total": total_m,  "synced": synced_m},
        "deadlines": {"total": total_d, "synced": synced_d},
    })


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
    name     = request.form.get("name", "").strip()
    email    = request.form.get("email", "").strip().lower()
    mobile   = request.form.get("mobile_number", "").strip()
    password = request.form.get("password", "")
    if not all([name, email, password]):
        return "MISSING_REQUIRED_FIELDS"
    hashed = generate_password_hash(password)
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO hr (name,email,mobile_number,password,status) VALUES (?,?,?,?,'pending')",
                (name, email, mobile, hashed)
            )
            conn.commit()
        return "REGISTRATION_SUCCESSFUL_WAITING_FOR_ADMIN_APPROVAL"
    except sqlite3.IntegrityError:
        return "EMAIL_ALREADY_EXISTS"


@app.route("/hr_login", methods=["POST"])
def hr_login_post():
    email    = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    with get_db() as conn:
        hr = conn.execute(
            "SELECT id,name,status,is_blocked,login_attempts,password FROM hr WHERE email=?",
            (email,)
        ).fetchone()
    if not hr:
        flash("Invalid HR login credentials.", "danger")
        return redirect(url_for("hr_login_page"))
    if hr["is_blocked"]:
        flash("Your account is BLOCKED. Contact the administrator.", "danger")
        return redirect(url_for("hr_login_page"))
    if hr["status"] != "approved":
        return "ACCOUNT_PENDING_APPROVAL_CONTACT_ADMIN"

    # Resend OTP shortcut (password field contains sentinel value)
    resend = (password == "RESEND_OTP")
    if not resend and not check_password_hash(hr["password"], password):
        attempts = hr["login_attempts"] + 1
        with get_db() as conn:
            if attempts >= 3:
                conn.execute("UPDATE hr SET is_blocked=1 WHERE id=?", (hr["id"],))
                flash("Account BLOCKED after 3 failed attempts. Contact admin.", "danger")
            else:
                conn.execute("UPDATE hr SET login_attempts=? WHERE id=?", (attempts, hr["id"]))
                flash(f"Invalid password. {3 - attempts} attempt(s) remaining.", "danger")
            conn.commit()
        return redirect(url_for("hr_login_page"))

    # Throttle OTP resends to once per 60 s
    can_send = True
    if "last_otp_sent" in session:
        elapsed = (datetime.now() - datetime.fromisoformat(session["last_otp_sent"])).total_seconds()
        if elapsed < 60:
            can_send = False

    if can_send:
        otp     = str(random.randint(100000, 999999))
        expires = (datetime.now() + timedelta(minutes=10)).isoformat()
        with get_db() as conn:
            conn.execute(
                "UPDATE hr SET otp=?,otp_expires_at=?,login_attempts=0 WHERE id=?",
                (otp, expires, hr["id"])
            )
            conn.commit()
        send_otp_email(email, hr["name"], otp)
        session["last_otp_sent"] = datetime.now().isoformat()

    session["temp_hr_id"]    = hr["id"]
    session["temp_hr_email"] = email
    session["otp_tries"]     = 0
    return render_template("hr_otp_verify.html", remaining_time=60)


@app.route("/verify_otp", methods=["POST"])
def verify_otp():
    input_otp = request.form.get("otp", "").strip()
    hr_id     = session.get("temp_hr_id")
    if not hr_id:
        return redirect(url_for("index"))
    with get_db() as conn:
        data = conn.execute(
            "SELECT otp,otp_expires_at,name FROM hr WHERE id=?", (hr_id,)
        ).fetchone()
    if not data:
        return redirect(url_for("index"))

    # Check expiry
    if data["otp_expires_at"] and datetime.now() > datetime.fromisoformat(data["otp_expires_at"]):
        flash("OTP expired. Please log in again.", "danger")
        return redirect(url_for("hr_login_page"))

    if data["otp"] == input_otp:
        with get_db() as conn:
            conn.execute("UPDATE hr SET otp=NULL,otp_expires_at=NULL,login_attempts=0 WHERE id=?", (hr_id,))
            conn.commit()
        session["hr_id"]   = hr_id
        session["hr_name"] = data["name"]
        for k in ("temp_hr_id", "temp_hr_email", "last_otp_sent", "otp_tries"):
            session.pop(k, None)
        return redirect(url_for("hr_dashboard"))

    session["otp_tries"] = session.get("otp_tries", 0) + 1
    if session["otp_tries"] >= 3:
        with get_db() as conn:
            conn.execute("UPDATE hr SET is_blocked=1,otp=NULL WHERE id=?", (hr_id,))
            conn.commit()
        session.clear()
        flash("Account BLOCKED after 3 incorrect OTP attempts. Contact admin.", "danger")
        return redirect(url_for("hr_login_page"))

    flash(f"Invalid OTP. {3 - session['otp_tries']} attempt(s) remaining.", "danger")
    return render_template("hr_otp_verify.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("index"))


# ═══════════════════════════════════════════════════════════════════════════════
#  ADMIN ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("email", "")
        password = request.form.get("password", "")
        # ADMIN_PASSWORD must be set as an env var — no fallback in production
        if not ADMIN_PASSWORD:
            flash("Admin credentials not configured. Set ADMIN_PASSWORD env var.", "danger")
            return redirect(url_for("index"))
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            flash("Admin authenticated.", "success")
            return redirect(url_for("admin_dashboard"))
        flash("Invalid admin credentials.", "danger")
        return redirect(url_for("index"))
    return render_template("admin_login.html")


@app.route("/admin_dashboard")
@admin_required
def admin_dashboard():
    with get_db() as conn:
        hrs = conn.execute("""
            SELECT h.id, h.name, h.email, h.mobile_number, h.status,
                   (SELECT COUNT(*) FROM employees WHERE added_by_hr=h.id),
                   h.is_blocked
            FROM hr h
        """).fetchall()
    return render_template("admin_dashboard.html", hrs=hrs)


@app.route("/approve_hr/<int:hr_id>")
@admin_required
def approve_hr(hr_id):
    with get_db() as conn:
        conn.execute("UPDATE hr SET status='approved' WHERE id=?", (hr_id,))
        conn.commit()
    flash("HR account approved.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/unblock_hr/<int:hr_id>")
@admin_required
def unblock_hr(hr_id):
    with get_db() as conn:
        conn.execute("UPDATE hr SET is_blocked=0,login_attempts=0 WHERE id=?", (hr_id,))
        conn.commit()
    flash("HR account unblocked.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin_view_hr/<int:hr_id>")
@admin_required
def admin_view_hr(hr_id):
    with get_db() as conn:
        employees = conn.execute("""
            SELECT e.employee_id, e.name, e.email, e.role, e.department,
                   (SELECT COUNT(*) FROM projects WHERE assigned_employees LIKE '%'||e.id||'%')
            FROM employees e WHERE e.added_by_hr=?
        """, (hr_id,)).fetchall()
    return jsonify({"employees": [
        {"employee_id": r[0], "name": r[1], "email": r[2], "role": r[3],
         "department": r[4], "project_status": "Active" if r[5] > 0 else "Idle"}
        for r in employees
    ]})


@app.route("/admin_stats_json")
@admin_required
def admin_stats_json():
    with get_db() as conn:
        hr_status = dict(conn.execute("SELECT status,COUNT(*) FROM hr GROUP BY status").fetchall())
        dept_dist = dict(conn.execute("SELECT department,COUNT(*) FROM employees GROUP BY department").fetchall())
    return jsonify({
        "hr_status": hr_status,
        "dept_dist": dept_dist,
        "telemetry": {
            "uptime": "99.98%",
            "active_sessions": random.randint(5, 50),
            "db_size": "1.2 MB",
        },
        "growth": {"labels": ["W1", "W2", "W3", "W4"], "data": [12, 19, 15, 28]},
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  HR DASHBOARD & ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/hr_dashboard")
@hr_required
def hr_dashboard():
    with get_db() as conn:
        employees = conn.execute(
            "SELECT employee_id,name,department,role,email FROM employees WHERE added_by_hr=?",
            (session["hr_id"],)
        ).fetchall()
        projects  = conn.execute("SELECT * FROM projects").fetchall()
        meetings  = conn.execute("SELECT * FROM meetings").fetchall()
        leaves    = conn.execute("""
            SELECT l.id, e.name, l.type, l.start_date, l.end_date, l.status
            FROM leave_requests l
            JOIN employees e ON l.employee_id = e.employee_id
            WHERE l.status='Pending'
        """).fetchall()
        payroll   = conn.execute("""
            SELECT p.month, e.name, p.salary, p.bonus, p.status
            FROM payroll p
            JOIN employees e ON p.employee_id = e.employee_id
            ORDER BY p.id DESC
        """).fetchall()
    return render_template(
        "hr_dashboard.html",
        hr_name=session["hr_name"],
        employees=employees,
        projects=projects,
        meetings=meetings,
        leaves=leaves,
        payroll=payroll,
        timezones=pytz.common_timezones,
    )


@app.route("/get_calendar_events")
@hr_required
def get_calendar_events():
    with get_db() as conn:
        meetings  = conn.execute("SELECT title,date,time FROM meetings").fetchall()
        deadlines = conn.execute("SELECT name,end_date FROM projects").fetchall()
    events = [
        {"title": f"Meeting: {m[0]}", "start": f"{m[1]}T{m[2]}", "color": "#2563eb"}
        for m in meetings
    ] + [
        {"title": f"Deadline: {p[0]}", "start": p[1], "color": "#ef4444"}
        for p in deadlines if p[1]
    ]
    return jsonify(events)


@app.route("/get_stats_json")
@hr_required
def get_stats_json():
    with get_db() as conn:
        data = conn.execute("SELECT department,COUNT(*) FROM employees GROUP BY department").fetchall()
    return jsonify({"labels": [r[0] for r in data], "values": [r[1] for r in data]})


@app.route("/get_meetings_json")
@hr_required
def get_meetings_json():
    with get_db() as conn:
        meetings = conn.execute("SELECT title,date,time,status,meet_link FROM meetings").fetchall()
    return jsonify([{
        "title":     m[0],
        "start":     f"{m[1]}T{m[2]}",
        "color":     "#ef4444" if m[3] == "Urgent" else "#2563eb",
        "meet_link": m[4] or "",
    } for m in meetings])


@app.route("/add_employee", methods=["GET", "POST"])
@hr_required
def add_employees():
    if request.method == "GET":
        return redirect(url_for("hr_dashboard") + "#employeeModal")
    hashed_password = generate_password_hash(request.form["password"])
    try:
        with get_db() as conn:
            conn.execute("""
                INSERT INTO employees
                    (employee_id,name,email,mobile,password,department,role,date,added_by_hr,skills)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                request.form["employee_id"], request.form["name"],
                request.form["email"].strip().lower(), request.form["mobile_number"],
                hashed_password, request.form["department"],
                request.form["role"], request.form["date_of_joining"],
                session["hr_id"], request.form["skill"],
            ))
            conn.commit()
        send_employee_email(
            request.form["email"], request.form["employee_id"],
            request.form["name"], request.form["role"], request.form["department"]
        )
        flash("Employee added successfully!", "success")
    except sqlite3.IntegrityError:
        flash("Employee ID or Email already exists!", "danger")
    return redirect(url_for("hr_dashboard"))


@app.route("/delete_employee/<string:employee_id>")
@hr_required
def delete_employee(employee_id):
    with get_db() as conn:
        conn.execute("DELETE FROM employees WHERE employee_id=?", (employee_id,))
        conn.commit()
    flash("Employee removed.", "success")
    return redirect(url_for("hr_dashboard"))


@app.route("/update_employee", methods=["POST"])
@hr_required
def update_employee():
    with get_db() as conn:
        conn.execute(
            "UPDATE employees SET role=?,department=? WHERE id=?",
            (request.form.get("role"), request.form.get("department"), request.form.get("id"))
        )
        conn.commit()
    flash("Employee record updated.", "success")
    return redirect(url_for("hr_dashboard"))


@app.route("/create_project", methods=["GET", "POST"])
@hr_required
def create_project():
    if request.method == "GET":
        return redirect(url_for("hr_dashboard") + "#projectModal")
    required_role  = request.form.get("required_role", "")
    required_skill = request.form.get("required_skill", "")
    with get_db() as conn:
        employee_data = conn.execute("""
            SELECT e.id, e.name, e.date,
                   COUNT(t.id),
                   SUM(CASE WHEN t.status='Completed' THEN 1 ELSE 0 END),
                   AVG(CASE WHEN t.status='Completed' THEN julianday(t.deadline)-julianday(e.date) END)
            FROM employees e
            LEFT JOIN tasks t ON e.id = t.assigned_to
            WHERE LOWER(e.department) = LOWER(?)
              AND LOWER(e.skills) LIKE LOWER(?)
            GROUP BY e.id
        """, (required_role, f"%{required_skill}%")).fetchall()
    if not employee_data:
        flash("No employees found with the required skill in that department.", "danger")
        return redirect(url_for("hr_dashboard"))
    labels      = cluster_employees(employee_data)
    best_cluster = get_best_cluster(employee_data, labels)
    selected    = [emp for emp, lbl in zip(employee_data, labels) if lbl == best_cluster]
    assigned_ids = ",".join(str(emp[0]) for emp in selected[:3])
    with get_db() as conn:
        conn.execute("""
            INSERT INTO projects (name,project_description,progress,start_date,end_date,assigned_employees)
            VALUES (?,?,?,?,?,?)
        """, (
            request.form.get("name"), request.form.get("description"),
            request.form.get("progress", "0"), request.form.get("start_date"),
            request.form.get("end_date"), assigned_ids,
        ))
        conn.commit()
    flash("Project created with AI-clustered employee assignment!", "success")
    return redirect(url_for("hr_dashboard"))


@app.route("/delete_project/<int:project_id>")
@hr_required
def delete_project(project_id):
    with get_db() as conn:
        conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
        conn.commit()
    flash("Project deleted.", "success")
    return redirect(url_for("hr_dashboard"))


@app.route("/update_project_progress", methods=["POST"])
@hr_required
def update_project_progress():
    with get_db() as conn:
        conn.execute(
            "UPDATE projects SET progress=? WHERE id=?",
            (request.form.get("progress"), request.form.get("project_id"))
        )
        conn.commit()
    flash("Project progress updated.", "success")
    return redirect(url_for("hr_dashboard"))


@app.route("/schedule_meeting", methods=["GET", "POST"])
@hr_required
def schedule_meeting():
    if request.method == "GET":
        return redirect(url_for("hr_dashboard") + "#meetingModal")
    title       = request.form.get("title", "").strip()
    date        = request.form.get("date", "")
    time_str    = request.form.get("time", "")
    selected_tz = request.form.get("timezone", "UTC")
    p_list      = request.form.getlist("participants")
    participants_string = ", ".join(p_list)
    status      = request.form.get("status", "Scheduled")
    if not selected_tz:
        flash("Please select a timezone.", "danger")
        return redirect(url_for("hr_dashboard") + "#meetingModal")
    try:
        pytz.timezone(selected_tz)
    except pytz.UnknownTimeZoneError:
        flash("Invalid timezone selected.", "danger")
        return redirect(url_for("hr_dashboard") + "#meetingModal")

    # Check if Google is connected
    with get_db() as conn:
        has_token = conn.execute(
            "SELECT 1 FROM google_tokens WHERE hr_id=?", (session["hr_id"],)
        ).fetchone()

    if not has_token:
        with get_db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO pending_meetings VALUES (?,?,?,?,?,?,?)",
                (session["hr_id"], title, date, time_str, selected_tz, participants_string, status)
            )
            conn.commit()
        flash("Connect your Google account first — meeting details saved.", "warning")
        return redirect(url_for("google_auth") + "?next=/complete_pending_meeting")

    meet_link = create_google_meet(title, date, time_str, selected_tz, p_list) or ""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO meetings (title,date,time,timezone,participants,status,meet_link) VALUES (?,?,?,?,?,?,?)",
            (title, date, time_str, selected_tz, participants_string, status, meet_link)
        )
        conn.commit()
    for email in p_list:
        send_meeting_invite(email, title, date, f"{time_str} ({selected_tz})", meet_link)
    flash(
        "Meeting scheduled with Google Meet link!" if meet_link else "Meeting saved! (Meet link unavailable)",
        "success" if meet_link else "warning"
    )
    return redirect(url_for("hr_dashboard") + "#meetings")


@app.route("/delete_meeting/<int:meeting_id>")
@hr_required
def delete_meeting(meeting_id):
    with get_db() as conn:
        conn.execute("DELETE FROM meetings WHERE id=?", (meeting_id,))
        conn.commit()
    flash("Meeting deleted.", "success")
    return redirect(url_for("hr_dashboard") + "#meetings")


@app.route("/update_meeting_status/<int:meeting_id>/<string:new_status>")
@hr_required
def update_meeting_status(meeting_id, new_status):
    allowed = {"Scheduled", "Urgent", "Completed", "Cancelled"}
    if new_status not in allowed:
        flash("Invalid status.", "danger")
        return redirect(url_for("hr_dashboard") + "#meetings")
    with get_db() as conn:
        conn.execute("UPDATE meetings SET status=? WHERE id=?", (new_status, meeting_id))
        conn.commit()
    flash(f"Meeting status updated to {new_status}.", "success")
    return redirect(url_for("hr_dashboard") + "#meetings")


@app.route("/update_leave/<int:leave_id>/<string:status>")
@hr_required
def update_leave(leave_id, status):
    allowed = {"Approved", "Rejected"}
    if status not in allowed:
        flash("Invalid leave status.", "danger")
        return redirect(url_for("hr_dashboard"))
    with get_db() as conn:
        conn.execute("UPDATE leave_requests SET status=? WHERE id=?", (status, leave_id))
        conn.commit()
    flash(f"Leave request {status.lower()}.", "success")
    return redirect(url_for("hr_dashboard"))


@app.route("/process_payroll", methods=["POST"])
@hr_required
def process_payroll():
    emp_id = request.form.get("employee_id")
    salary = request.form.get("salary", 0)
    bonus  = request.form.get("bonus", 0)
    month  = datetime.now().strftime("%B %Y")
    with get_db() as conn:
        employee = conn.execute(
            "SELECT name,email FROM employees WHERE employee_id=?", (emp_id,)
        ).fetchone()
        if employee:
            conn.execute(
                "INSERT INTO payroll (employee_id,month,salary,bonus,status) VALUES (?,?,?,?,'Paid')",
                (emp_id, month, salary, bonus)
            )
            conn.commit()
    if employee:
        send_payslip_email(employee["email"], employee["name"], month, salary, bonus)
        flash("Payroll processed and payslip sent.", "success")
    else:
        flash("Employee not found.", "danger")
    return redirect(url_for("hr_dashboard"))


# ═══════════════════════════════════════════════════════════════════════════════
#  EMPLOYEE ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/employee_login", methods=["POST"])
def employee_login():
    login_input = request.form.get("emp_id", "").strip()
    password    = request.form.get("password", "")
    with get_db() as conn:
        emp = conn.execute(
            "SELECT * FROM employees WHERE (employee_id=? OR email=?)",
            (login_input, login_input)
        ).fetchone()
    if emp and check_password_hash(emp["password"], password):
        session["emp_id"]   = emp["employee_id"]
        session["emp_name"] = emp["name"]
        flash("Login successful!", "success")
        return redirect(url_for("employee_dashboard"))
    flash("Invalid Employee ID/Email or Password.", "danger")
    return redirect(url_for("index"))


@app.route("/employee_dashboard")
@employee_required
def employee_dashboard():
    emp_id = session["emp_id"]
    with get_db() as conn:
        result = conn.execute("SELECT id FROM employees WHERE employee_id=?", (emp_id,)).fetchone()
        employee_pk = str(result["id"]) if result else None
        all_projects = []
        if employee_pk:
            all_projects = conn.execute("""
                SELECT * FROM projects
                WHERE assigned_employees = ?
                   OR assigned_employees LIKE ?
                   OR assigned_employees LIKE ?
                   OR assigned_employees LIKE ?
            """, (
                employee_pk,
                employee_pk + ",%",
                "%," + employee_pk + ",%",
                "%," + employee_pk,
            )).fetchall()
        emp_email_row = conn.execute("SELECT email FROM employees WHERE employee_id=?", (emp_id,)).fetchone()
        emp_email     = emp_email_row["email"] if emp_email_row else ""
        meetings = conn.execute(
            "SELECT * FROM meetings WHERE participants LIKE ? OR participants LIKE ?",
            ("%" + session["emp_name"] + "%", "%" + emp_email + "%")
        ).fetchall()
        leaves = conn.execute(
            "SELECT * FROM leave_requests WHERE employee_id=?", (emp_id,)
        ).fetchall()
    active_projects    = [p for p in all_projects if dict(p)["status"] != "Completed" and str(dict(p)["progress"]) != "100"]
    completed_projects = [p for p in all_projects if dict(p)["status"] == "Completed" or str(dict(p)["progress"]) == "100"]
    return render_template(
        "employee_dashboard.html",
        emp_id=emp_id, emp_name=session["emp_name"],
        active_projects=active_projects, completed_projects=completed_projects,
        meetings=meetings, leaves=leaves,
    )


@app.route("/apply_leave", methods=["POST"])
@employee_required
def apply_leave():
    with get_db() as conn:
        conn.execute(
            "INSERT INTO leave_requests (employee_id,type,start_date,end_date,reason) VALUES (?,?,?,?,?)",
            (session["emp_id"], request.form.get("leave_type"),
             request.form.get("start_date"), request.form.get("end_date"),
             request.form.get("reason"))
        )
        conn.commit()
    flash("Leave request submitted!", "success")
    return redirect(url_for("employee_dashboard"))


@app.route("/update_progress", methods=["POST"])
@employee_required
def update_progress():
    project_id = request.form["project_id"]
    progress   = min(int(request.form["progress"]), 100)
    status     = "Completed" if progress >= 100 else "In Progress"
    with get_db() as conn:
        conn.execute(
            "UPDATE projects SET progress=?,status=? WHERE id=?",
            (progress, status, project_id)
        )
        conn.commit()
    return redirect(url_for("employee_dashboard"))


@app.route("/api/performance_stats")
def performance_stats():
    with get_db() as conn:
        data = dict(conn.execute("SELECT status,COUNT(*) FROM leave_requests GROUP BY status").fetchall())
    return jsonify(data)


# ═══════════════════════════════════════════════════════════════════════════════
#  CHATBOT ROUTE
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/chat", methods=["POST"])
def chat():
    message = (request.json or {}).get("message", "").strip()
    if not message:
        return jsonify({"response": "Please type a message."})
    ints     = predict_class(message)
    response = get_response(ints)
    user_id  = session.get("hr_id") or session.get("emp_id")
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO chatlogs (user_id,message,bot_response,timescamp) VALUES (?,?,?,?)",
                (user_id, message, response, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
    except Exception as e:
        print(f"[CHAT LOG ERROR] {e}")
    return jsonify({"response": response})


# ═══════════════════════════════════════════════════════════════════════════════
#  HEALTH CHECK  (Render pings this to verify the service is alive)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()}), 200


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT  (Gunicorn is used on Render — this block is for local dev only)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.getenv("PORT", 2300))
    app.run(host="0.0.0.0", port=port, debug=not _IS_PROD)