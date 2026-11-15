"""Per-person access for the preview: emailed codes, transcription quotas, anti-abuse limits.

Anyone can request a code at `/signup`; it is emailed to them (which proves they own the address)
and they log in with that email + code. Every transcription is counted against the person's quota
(5 by default; raise it with `admin.py`). To make farming many accounts unattractive -- distilling
the model needs volume -- accounts are one per email (aliases folded together), disposable email
domains are refused, and people on the default quota also share a per-IP cap.

Codes are stored only as SHA-256 hashes, so the database never holds a usable credential.
Everything lives in one SQLite file (`DEMO_DB`), outside the repository.
"""

import hashlib
import hmac
import html
import ipaddress
import json
import os
import re
import secrets
import smtplib
import socket
import sqlite3
import subprocess
import threading
import time
from contextlib import contextmanager
from email.message import EmailMessage
from pathlib import Path

DB_PATH = os.environ.get("DEMO_DB", os.path.expanduser("~/.config/pandemonium-demo/access.db"))
DEFAULT_QUOTA = int(os.environ.get("DEMO_DEFAULT_QUOTA", "5"))
# Transcriptions per IP per window, summed over every default-quota account seen from that IP.
IP_QUOTA = int(os.environ.get("DEMO_IP_QUOTA", "10"))
IP_WINDOW_DAYS = int(os.environ.get("DEMO_IP_WINDOW_DAYS", "30"))
# Signup form submissions per IP per hour: keeps the form from being used to email-bomb addresses.
SIGNUPS_PER_IP_HOUR = int(os.environ.get("DEMO_SIGNUPS_PER_IP_HOUR", "5"))
# Code emails per address per day, whatever IPs ask: keeps the form from flooding one inbox.
CODES_PER_EMAIL_DAY = int(os.environ.get("DEMO_CODES_PER_EMAIL_DAY", "3"))
SMTP_HOST = os.environ.get("DEMO_SMTP_HOST", "smtp.fit.vutbr.cz")
# FIT's server only accepts mail from a logged-in account (submission port, STARTTLS). The password
# is read from a file only its owner can read, never from the environment or the repository.
SMTP_PORT = int(os.environ.get("DEMO_SMTP_PORT", "587"))
SMTP_USER = os.environ.get("DEMO_SMTP_USER", "")
SMTP_PASSWORD_FILE = os.environ.get("DEMO_SMTP_PASSWORD_FILE",
                                    os.path.expanduser("~/.config/pandemonium-demo/smtp_password"))
# Or hand every email to a trusted host over SSH (`user@host`), which sends it with the password it
# keeps (deploy/pandemonium-mail.py): the password then never exists on this machine. The key is the
# demo's own, restricted on that host to running the sending script.
# Set by `DEMO_MAIL_SSH`, else by the first line of ~/.config/pandemonium-demo/mail_ssh (so admin.py, run
# from a shell, sends the same way as the service).
_MAIL_SSH_FILE = os.path.expanduser("~/.config/pandemonium-demo/mail_ssh")
MAIL_SSH = os.environ.get("DEMO_MAIL_SSH") or (
    open(_MAIL_SSH_FILE).readline().strip() if os.path.exists(_MAIL_SSH_FILE) else "")
MAIL_SSH_KEY = os.environ.get("DEMO_MAIL_SSH_KEY", os.path.expanduser("~/.config/pandemonium-demo/mail_key"))
MAIL_SSH_KNOWN_HOSTS = os.environ.get("DEMO_MAIL_SSH_KNOWN_HOSTS",
                                      os.path.expanduser("~/.config/pandemonium-demo/known_hosts"))
ADMIN_EMAIL = os.environ.get("DEMO_ADMIN_EMAIL", "ipoloka@fit.vut.cz")
MAIL_FROM = os.environ.get("DEMO_MAIL_FROM", f"Pandemonium preview <{ADMIN_EMAIL}>")
# Where people should open the demo; set by the app once the share link exists.
PUBLIC_URL = os.environ.get("DEMO_PUBLIC_URL", "")
# The recording length the pages advertise (app.MAX_SECONDS is the real, somewhat larger, cap).
LIMIT = "30 minutes"
# "Find speakers" runs per account per hour (it costs no transcription, but it does use the GPU), at the
# default quota; an account granted more transcriptions gets proportionally more (`diarize_limit`).
DIARIZE_PER_HOUR = int(os.environ.get("DEMO_DIARIZE_PER_HOUR", "10"))
ADMIN_USER = "admin"  # logs in with the master code (DEMO_ACCESS_CODE[_FILE]) and has no quota

_DISPOSABLE = {
    line.strip().lower()
    for line in (Path(__file__).parent / "disposable_email_domains.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
}
_EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_lock = threading.Lock()


# ---------------------------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------------------------

@contextmanager
def db():
    """One short transaction per call; the lock serializes writers across Gradio's threads."""
    with _lock:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init_db():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            email_norm TEXT PRIMARY KEY, email TEXT NOT NULL, name TEXT, affiliation TEXT, purpose TEXT,
            code_hash TEXT NOT NULL, quota INTEGER NOT NULL, used INTEGER NOT NULL DEFAULT 0,
            revoked INTEGER NOT NULL DEFAULT 0, signup_ip TEXT, created REAL NOT NULL, note TEXT);
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY, email_norm TEXT NOT NULL, ip TEXT, ts REAL NOT NULL,
            seconds REAL, status TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS signup_attempts (ip TEXT, ts REAL, email TEXT, outcome TEXT);
        CREATE INDEX IF NOT EXISTS req_ip ON requests (ip, ts);
        CREATE TABLE IF NOT EXISTS diarizations (email_norm TEXT NOT NULL, ts REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS deletions (token_hash TEXT PRIMARY KEY, email_norm TEXT NOT NULL, expires REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS diar_user ON diarizations (email_norm, ts);
        """)
        # A re-requested code waits here until it is first used, so that asking for a new code -- which
        # anyone who knows the address can do -- never locks the owner out of their current one.
        if "pending_hash" not in {r[1] for r in c.execute("PRAGMA table_info(users)")}:
            c.execute("ALTER TABLE users ADD COLUMN pending_hash TEXT")
    os.chmod(DB_PATH, 0o600)


# ---------------------------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------------------------

def normalize_email(email: str) -> str:
    """Fold the aliases one mailbox can hide behind: case, +tags, and Gmail's dots."""
    local, _, domain = email.strip().lower().partition("@")
    local = local.split("+", 1)[0]
    if domain in ("gmail.com", "googlemail.com"):
        local, domain = local.replace(".", ""), "gmail.com"
    return f"{local}@{domain}"


def _hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def new_code() -> str:
    return secrets.token_urlsafe(9)  # 12 characters, ~72 bits


def is_admin(username: str) -> bool:
    """The login name as typed (Gradio passes it on verbatim), compared as `check_login` does."""
    return (username or "").strip().lower() == ADMIN_USER


def check_login(username: str, password: str, master_code: str = "") -> bool:
    """Gradio auth: email + emailed code, or the admin name + master code."""
    if master_code and is_admin(username):
        return hmac.compare_digest(password.encode(), master_code.encode())
    given = _hash(password.strip())
    with db() as c:
        row = c.execute("SELECT code_hash, pending_hash, revoked FROM users WHERE email_norm = ?",
                        (normalize_email(username),)).fetchone()
        if not row or row["revoked"]:
            return False
        if hmac.compare_digest(given, row["code_hash"]):
            return True
        if row["pending_hash"] and hmac.compare_digest(given, row["pending_hash"]):
            # The owner has the newly requested code: from now on it is the code, the old one is not.
            c.execute("UPDATE users SET code_hash = pending_hash, pending_hash = NULL WHERE email_norm = ?",
                      (normalize_email(username),))
            return True
    return False


def client_ip(headers, peer: str) -> str:
    """The visitor's address.

    Behind the share relay the connection comes from a private/loopback hop and the visitor is in
    X-Forwarded-For. A client can prepend anything to that header itself, so read it from the
    right and take the first public address -- the one the relay appended. A direct connection
    from a public peer is the visitor, and its X-Forwarded-For is not trusted.
    """
    def private(ip):
        try:
            a = ipaddress.ip_address(ip)
            return a.is_private or a.is_loopback
        except ValueError:
            return True

    if peer and not private(peer):
        return peer
    hops = [h.strip() for h in (headers.get("x-forwarded-for") or "").split(",") if h.strip()]
    for hop in reversed(hops):
        if not private(hop):
            return hop
    return peer or "unknown"


# ---------------------------------------------------------------------------------------------
# Quota
# ---------------------------------------------------------------------------------------------

class QuotaError(Exception):
    pass


def reserve(username: str, ip: str, seconds: float) -> int | None:
    """Count one transcription before it runs; returns the request id (None for the admin).

    Counted up front so parallel requests cannot overrun the quota; `refund` gives it back if the
    run fails. Raises QuotaError with a message meant for the user.
    """
    if is_admin(username):
        return None
    email = normalize_email(username)
    since = time.time() - IP_WINDOW_DAYS * 86400
    with db() as c:
        u = c.execute("SELECT * FROM users WHERE email_norm = ?", (email,)).fetchone()
        if not u or u["revoked"]:
            raise QuotaError("This access code is no longer valid.")
        if u["used"] >= u["quota"]:
            raise QuotaError(f"You have used all {u['quota']} transcriptions of this preview. "
                             f"To ask for more, write to {ADMIN_EMAIL}.")
        if u["quota"] <= DEFAULT_QUOTA:  # people you granted more are exempt from the IP cap
            n = c.execute("SELECT COUNT(*) FROM requests r JOIN users u ON u.email_norm = r.email_norm "
                          "WHERE r.ip = ? AND r.ts > ? AND r.status = 'ok' AND u.quota <= ?",
                          (ip, since, DEFAULT_QUOTA)).fetchone()[0]
            if n >= IP_QUOTA:
                raise QuotaError("The preview's limit for your network has been reached. "
                                 f"If you need more, write to {ADMIN_EMAIL}.")
        c.execute("UPDATE users SET used = used + 1 WHERE email_norm = ?", (email,))
        cur = c.execute("INSERT INTO requests (email_norm, ip, ts, seconds, status) VALUES (?, ?, ?, ?, 'ok')",
                        (email, ip, time.time(), seconds))
        return cur.lastrowid


def refund(request_id: int | None):
    if request_id is None:
        return
    with db() as c:
        r = c.execute("SELECT email_norm, status FROM requests WHERE id = ?", (request_id,)).fetchone()
        if r and r["status"] == "ok":
            c.execute("UPDATE requests SET status = 'refunded' WHERE id = ?", (request_id,))
            c.execute("UPDATE users SET used = MAX(used - 1, 0) WHERE email_norm = ?", (r["email_norm"],))


def diarize_limit(quota: int) -> int:
    """Find speakers per hour for an account: DIARIZE_PER_HOUR at the default quota, scaled with the quota."""
    return max(DIARIZE_PER_HOUR, DIARIZE_PER_HOUR * quota // max(DEFAULT_QUOTA, 1))


def reserve_diarization(username: str) -> str | None:
    """Count one "Find speakers" (free, but GPU time): None if allowed, else the message to show.

    Capped per account per hour so the button cannot be used to keep the GPU busy; a revoked account
    gets nothing. The admin is not limited.
    """
    if is_admin(username):
        return None
    email, now = normalize_email(username), time.time()
    with db() as c:
        u = c.execute("SELECT revoked, quota FROM users WHERE email_norm = ?", (email,)).fetchone()
        if not u or u["revoked"]:
            return "This access code is no longer valid."
        limit = diarize_limit(u["quota"])
        n = c.execute("SELECT COUNT(*) FROM diarizations WHERE email_norm = ? AND ts > ?", (email, now - 3600)).fetchone()[0]
        if n >= limit:
            return (f"You have used Find speakers {limit} times in the last hour; please wait a little. "
                    "Transcribe works meanwhile and finds the speakers itself.")
        c.execute("INSERT INTO diarizations VALUES (?, ?)", (email, now))
    return None


def is_active(username: str) -> bool:
    """Whether a logged-in session may continue: the admin, or an account that exists and is not revoked."""
    if is_admin(username):
        return True
    with db() as c:
        u = c.execute("SELECT revoked FROM users WHERE email_norm = ?", (normalize_email(username or ""),)).fetchone()
    return bool(u) and not u["revoked"]


def remaining(username: str) -> tuple[int, int] | None:
    """(left, quota) for a user; None for the admin."""
    if is_admin(username):
        return None
    with db() as c:
        u = c.execute("SELECT used, quota FROM users WHERE email_norm = ?", (normalize_email(username),)).fetchone()
    return (max(u["quota"] - u["used"], 0), u["quota"]) if u else (0, 0)


# ---------------------------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------------------------

def send_mail(to: str, subject: str, body: str, reply_to: str | None = None):
    if MAIL_SSH:  # the mail host sets From and Reply-To itself
        return _send_mail_over_ssh(to, subject, body)
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = MAIL_FROM, to, subject
    if reply_to:
        msg["Reply-To"] = reply_to
    msg.set_content(body)
    if not SMTP_USER or not os.path.exists(SMTP_PASSWORD_FILE):
        raise RuntimeError("email is not configured: set DEMO_SMTP_USER and create DEMO_SMTP_PASSWORD_FILE")
    with open(SMTP_PASSWORD_FILE) as f:
        password = f.readline().rstrip("\n")
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
        s.ehlo()
        s.starttls()
        s.ehlo()
        s.login(SMTP_USER, password)
        s.send_message(msg)


def _send_mail_over_ssh(to: str, subject: str, body: str):
    """One message to the mail host's forced command; raises with its explanation if it refuses."""
    cmd = ["ssh", "-T", "-F", "/dev/null", "-i", MAIL_SSH_KEY, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
           "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={MAIL_SSH_KNOWN_HOSTS}",
           "-o", "ConnectTimeout=10", MAIL_SSH]
    payload = json.dumps({"to": to, "subject": subject, "body": body})
    try:
        out = subprocess.run(cmd, input=payload, capture_output=True, text=True, timeout=45)
    except subprocess.TimeoutExpired:
        raise RuntimeError("mail host did not answer in time")
    if out.returncode != 0 or out.stdout.strip() != "ok":
        raise RuntimeError(f"mail host refused ({out.returncode}): {(out.stderr or out.stdout).strip()[-300:]}")


def mail_code(email: str, name: str, code: str, quota: int, reissued: bool = False):
    link = f"\nOpen the demo here: {PUBLIC_URL}\n" if PUBLIC_URL else ""
    send_mail(email, "Your Pandemonium preview access code", f"""Hello {name or ''},

{'Here is the access code you asked for again for' if reissued else 'Thank you for your interest in'} the Pandemonium research preview.

  Email address:  {email}
  Access code:    {code}

Log in with this email address and access code.{link}
Your access includes {quota} transcriptions of English recordings up to {LIMIT} long.
{'Your earlier code also keeps working until you log in with this one.' if reissued else ''}
By using the preview you agree not to use its outputs to train, fine-tune or distil other models.
This access code is for you only: please do not share it with anyone else.

To ask for more transcriptions, reply to this email.

-- Pandemonium preview, BUT Speech@FIT
""", reply_to=ADMIN_EMAIL)


# ---------------------------------------------------------------------------------------------
# Signup
# ---------------------------------------------------------------------------------------------

def signup(name: str, email: str, affiliation: str, purpose: str, agreed: bool, ip: str) -> tuple[bool, str]:
    """Handle one signup: `(done, message)`. `done` means the form was accepted and the person should go
    and check their inbox (the message is then the same whatever happened, see below); otherwise the
    message says what to fix. Never reveals a code on the page."""
    name, email, affiliation, purpose = (x.strip()[:300] for x in (name, email, affiliation, purpose))
    # One line each: they end up in email headers (the signup notice's subject) and listings.
    name, affiliation = (" ".join(x.split()) for x in (name, affiliation))
    now = time.time()
    norm = normalize_email(email)
    with db() as c:
        recent = c.execute("SELECT COUNT(*) FROM signup_attempts WHERE ip = ? AND ts > ?",
                           (ip, now - 3600)).fetchone()[0]
        mailed = c.execute("SELECT COUNT(*) FROM signup_attempts WHERE email = ? AND outcome = 'mailed' AND ts > ?",
                           (norm, now - 86400)).fetchone()[0]
        c.execute("INSERT INTO signup_attempts VALUES (?, ?, ?, ?)", (ip, now, norm, "attempt"))
    if recent >= SIGNUPS_PER_IP_HOUR:
        return False, "Too many requests from your network. Please try again in an hour."
    if not (name and email and purpose):
        return False, "Please fill in your name, email and how you plan to use the preview."
    if not agreed:
        return False, "Please accept the terms of use."
    if not _EMAIL_RE.match(email):
        return False, "That does not look like a valid email address."
    domain = email.rsplit("@", 1)[1].lower()
    if domain in _DISPOSABLE or any(domain.endswith("." + d) for d in _DISPOSABLE):
        return False, "Please use a permanent email address (disposable addresses are not accepted)."
    try:
        socket.getaddrinfo(domain, None)
    except OSError:
        return False, "That email domain does not exist."

    # From here on every outcome shows the same message, so the page does not tell anyone whether an
    # address is registered (or revoked).
    sent = (True, f"If <b>{html.escape(email)}</b> can receive email, an access code is on its way to it. "
                  "Log in with that email address and the access code. If it does not "
                  "arrive within a few minutes, check your spam folder.")
    if mailed >= CODES_PER_EMAIL_DAY:
        return sent
    code = new_code()
    with db() as c:
        existing = c.execute("SELECT * FROM users WHERE email_norm = ?", (norm,)).fetchone()
        if existing and existing["revoked"]:
            return sent
        if existing:  # asking again: a second code, which takes over only once it is used
            c.execute("UPDATE users SET pending_hash = ? WHERE email_norm = ?", (_hash(code), norm))
            quota = existing["quota"]
        else:
            quota = DEFAULT_QUOTA
            c.execute("INSERT INTO users (email_norm, email, name, affiliation, purpose, code_hash, quota, signup_ip, created) "
                      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (norm, email, name, affiliation, purpose, _hash(code), quota, ip, now))
    try:
        mail_code(email, name, code, quota, reissued=bool(existing))
    except Exception as err:  # do not leave a code nobody received
        with db() as c:
            if existing:
                c.execute("UPDATE users SET pending_hash = NULL WHERE email_norm = ?", (norm,))
            else:
                c.execute("DELETE FROM users WHERE email_norm = ?", (norm,))
        return False, f"Sorry, we could not send the email ({type(err).__name__}). Please try again later."
    with db() as c:
        c.execute("INSERT INTO signup_attempts VALUES (?, ?, ?, ?)", (ip, now, norm, "mailed"))
    if not existing:
        try:
            send_mail(ADMIN_EMAIL, f"[Pandemonium] new signup: {email} ({affiliation or 'no affiliation'})",
                      f"Name: {name}\nEmail: {email}\nAffiliation: {affiliation}\nIP: {ip}\n\nIntended use:\n{purpose}\n\n"
                      f"Default quota: {quota}. Manage with demo/admin.py, e.g.\n"
                      f"  python admin.py grant {email} 50\n  python admin.py revoke {email}\n")
        except Exception:
            pass  # the user's own email went out; the notification is best-effort
    return sent


SIGNUP_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Pandemonium preview: request access</title>
<style>
 :root {{ --bg:#fafafa; --fg:#1f2937; --muted:#6b7280; --line:#e5e7eb; --card:#fff; --accent:#0e9f6e; }}
 @media (prefers-color-scheme: dark) {{ :root {{ --bg:#0b0f17; --fg:#e5e7eb; --muted:#9ca3af; --line:#1f2937; --card:#111827; }} }}
 body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; }}
 main {{ max-width:560px; margin:40px auto; padding:0 16px; }}
 .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:24px; }}
 h1 {{ font-size:22px; margin:0 0 4px; }} p.lead {{ color:var(--muted); margin:0 0 18px; }}
 label {{ display:block; font-weight:600; margin:14px 0 4px; }} small {{ color:var(--muted); font-weight:400; }}
 input[type=text], input[type=email], textarea {{ width:100%; box-sizing:border-box; padding:9px 10px;
   border:1px solid var(--line); border-radius:8px; background:var(--bg); color:var(--fg); font:inherit; }}
 textarea {{ min-height:90px; resize:vertical; }}
 .terms {{ display:flex; gap:8px; align-items:flex-start; font-weight:400; margin-top:16px; }}
 button {{ margin-top:18px; width:100%; padding:11px; border:0; border-radius:8px; background:var(--accent);
   color:#fff; font:600 15px system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; cursor:pointer; }}
 .msg {{ margin-bottom:16px; padding:12px 14px; border-radius:8px; border:1px solid rgba(220,38,38,.45);
   background:rgba(220,38,38,.07); }}
 .notice {{ margin:0 0 16px; padding:10px 12px; border-radius:8px; background:rgba(245,158,11,.12);
   border:1px solid rgba(245,158,11,.5); }}
 a {{ color:var(--accent); }}
</style></head><body><main><div class="card">
<h1>Pandemonium preview</h1>
<p class="lead">Request a personal access code. It will be emailed to you.</p>
{message}
<p class="notice"><b>English only</b> &middot; recordings up to {limit} &middot; {quota} transcriptions per person.</p>
<form method="post" action="signup">
 <label>Name<input type="text" name="name" required maxlength="200" value="{name}"></label>
 <label>Email <small>(work or university address preferred; disposable addresses are refused)</small>
   <input type="email" name="email" required maxlength="200" value="{email}"></label>
 <label>Affiliation <small>(optional)</small><input type="text" name="affiliation" maxlength="200" value="{affiliation}"></label>
 <label>How do you plan to use it?<textarea name="purpose" required maxlength="2000">{purpose}</textarea></label>
 <label class="terms"><input type="checkbox" name="agree" value="1" required>
   <span>I will not use the outputs to train, fine-tune or distil other models, I will not share my access code with anyone else,
   and I have read the <a href="privacy" target="_blank">privacy notice</a>.</span></label>
 <button type="submit">Email me an access code</button>
</form>
<p style="margin-top:18px"><a href="./?login">Back to login</a> &middot; <a href="example/">See an example</a> &middot; <a href="privacy">Privacy notice</a></p>
</div></main></body></html>"""


def signup_page(message: str = "", form: dict | None = None) -> str:
    """The form; `message` says what to fix, and `form` refills what was typed."""
    form = form or {}
    typed = {k: html.escape(str(form.get(k, ""))[:2000], quote=True) for k in ("name", "email", "affiliation", "purpose")}
    return SIGNUP_PAGE.format(message=f"<div class='msg'>{message}</div>" if message else "", quota=DEFAULT_QUOTA,
                              limit=LIMIT, **typed)


# After a request is accepted: say so plainly, then take them to the login form.
SIGNUP_DONE_SECONDS = 15
SIGNUP_DONE_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{seconds};url=./?login">
<title>Pandemonium preview: check your email</title>
<style>
 :root {{ --bg:#fafafa; --fg:#1f2937; --muted:#6b7280; --line:#e5e7eb; --card:#fff; --accent:#0e9f6e; }}
 @media (prefers-color-scheme: dark) {{ :root {{ --bg:#0b0f17; --fg:#e5e7eb; --muted:#9ca3af; --line:#1f2937; --card:#111827; }} }}
 body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; }}
 main {{ max-width:560px; margin:12vh auto 40px; padding:0 16px; }}
 .card {{ background:var(--card); border:1px solid rgba(14,159,110,.5); border-radius:12px; padding:28px; text-align:center; }}
 .tick {{ width:52px; height:52px; margin:0 auto 12px; border-radius:50%; background:var(--accent); color:#fff;
   font-size:30px; line-height:52px; }}
 h1 {{ font-size:22px; margin:0 0 10px; }}
 p {{ margin:8px 0; }} .muted {{ color:var(--muted); font-size:13px; }}
 .btn {{ display:inline-block; margin-top:16px; padding:11px 26px; border-radius:8px; background:var(--accent); color:#fff;
   font-weight:600; text-decoration:none; }}
 a {{ color:var(--accent); }}
</style></head><body><main><div class="card">
<div class="tick">&#10003;</div>
<h1>Check your email</h1>
<p>{message}</p>
<a class="btn" href="./?login">Continue to login</a>
<p class="muted">Taking you to the login page in <span id="n">{seconds}</span> s &middot; <a href="example/">Back to the example</a></p>
</div></main>
<script>
 let n = {seconds};
 const el = document.getElementById('n');
 setInterval(() => {{ if (n > 0) el.textContent = --n; }}, 1000);
</script></body></html>"""


def signup_done_page(message: str) -> str:
    return SIGNUP_DONE_PAGE.format(message=message, seconds=SIGNUP_DONE_SECONDS)


PRIVACY_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Pandemonium preview: privacy notice</title>
<style>
 :root {{ --bg:#fafafa; --fg:#1f2937; --muted:#6b7280; --line:#e5e7eb; --card:#fff; --accent:#0e9f6e; }}
 @media (prefers-color-scheme: dark) {{ :root {{ --bg:#0b0f17; --fg:#e5e7eb; --muted:#9ca3af; --line:#1f2937; --card:#111827; }} }}
 body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.6 system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; }}
 main {{ max-width:680px; margin:40px auto; padding:0 16px; }}
 .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:24px 28px; }}
 h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:16px; margin:22px 0 6px; }}
 p, li {{ margin:6px 0; }} .muted {{ color:var(--muted); }} a {{ color:var(--accent); }}
</style></head><body><main><div class="card">
<h1>Privacy notice</h1>
<p class="muted">Pandemonium research preview &middot; BUT Speech@FIT, Faculty of Information Technology,
Brno University of Technology</p>

<h2>Who is responsible</h2>
<p>The preview is run by the Speech@FIT group at the Faculty of Information Technology, Brno University
of Technology (Bo&zcaron;et&ecaron;chova 2, 612 66 Brno, Czech Republic). Contact:
<a href="mailto:{admin}">{admin}</a>.</p>

<h2>What we collect, and why</h2>
<ul>
 <li><b>When you request access:</b> your name, email address, affiliation, what you plan to use the
 preview for, and the IP address and time of the request. We use them to send you your access code, to
 keep one account per person, and to prevent abuse of the service.</li>
 <li><b>When you use the preview:</b> for each transcription, your account, your IP address, the time and
 the length of the recording, to count your quota and prevent abuse; for each "Find speakers", your account
 and the time, for its hourly limit. Failed login attempts are counted per IP address for 15 minutes, in
 memory only, to stop code guessing. Like any web server, ours also keeps technical logs of requests (IP
 address, time, the address requested), for operation and security.</li>
 <li><b>Your recordings and transcripts:</b> the audio you upload or record, and the transcripts produced
 from it, are processed on the university's own servers only to produce the transcription you asked for.
 Only you can access them: they are not shared with anyone, and recordings are never offered for download.
 Uploads and results are deleted automatically after 7 days at the latest (and whenever the service
 restarts). Your 5 most recent results are also kept in the server's memory, so that transcribing the same
 recording again shows them without using a transcription; this too is cleared on restart.</li>
 <li><b>Speaker names</b> you type stay with your current session and your downloads; they are not stored.</li>
</ul>
<p>Nothing is shared with third parties or used for advertising. Email is sent through the university's
mail server. The preview uses one cookie, to keep you logged in; it is strictly necessary and is not used
for tracking.</p>

<h2>Legal basis</h2>
<p>Your consent, given when you request access (Art. 6(1)(a) GDPR), and our legitimate interest in
keeping the service secure and preventing abuse (Art. 6(1)(f) GDPR) for the IP addresses and usage records.</p>

<h2>How long we keep it</h2>
<p>Recordings and results: at most 7 days (see above). Account and usage data, and the server's technical
logs: for as long as the preview runs, and then deleted. You can delete your account and all data about
you yourself at any time on the <a href="delete">delete my account</a> page, or ask us to; we will do so
without undue delay.</p>

<h2>Your rights</h2>
<p>You can ask for access to your data, its correction or deletion, restriction of processing, or a copy of
it, and you can withdraw your consent at any time, by writing to <a href="mailto:{admin}">{admin}</a>.</p>

<p style="margin-top:22px"><a href="signup">Request access</a> &middot; <a href="./?login">Log in</a>
&middot; <a href="delete">Delete my account</a></p>
</div></main></body></html>"""


# ---------------------------------------------------------------------------------------------
# Deleting one's own account: email + access code, then a link sent to that address, then a button
# ---------------------------------------------------------------------------------------------

DELETION_LINK_HOURS = 1
DELETIONS_PER_IP_HOUR = 5
_DELETION_SENT = ("If that email address and access code are correct, we have emailed a confirmation link to the "
                  "address. Open it within an hour to delete your account and all data about you.")


def delete_account(email_norm: str):
    """Erase an account and every record of it (also `admin.py delete`)."""
    with db() as c:
        for table, column in (("requests", "email_norm"), ("signup_attempts", "email"), ("diarizations", "email_norm"),
                              ("deletions", "email_norm"), ("users", "email_norm")):
            c.execute(f"DELETE FROM {table} WHERE {column} = ?", (email_norm,))


def request_deletion(email: str, code: str, ip: str) -> str:
    """Check the email + access code and, if they match, email a one-time confirmation link.

    The reply is the same either way, so the form does not tell anyone whether a code is right (and it is
    rate-limited per network, like signups): only the mailbox owner learns, through the link.
    """
    norm, now = normalize_email(email or ""), time.time()
    with db() as c:
        recent = c.execute("SELECT COUNT(*) FROM signup_attempts WHERE ip = ? AND outcome = 'delete' AND ts > ?",
                           (ip, now - 3600)).fetchone()[0]
        c.execute("INSERT INTO signup_attempts VALUES (?, ?, ?, ?)", (ip, now, "", "delete"))
        if recent >= DELETIONS_PER_IP_HOUR:
            return "Too many requests from your network. Please try again in an hour."
        u = c.execute("SELECT email, name, code_hash, pending_hash FROM users WHERE email_norm = ?", (norm,)).fetchone()
        given = _hash((code or "").strip())
        if not u or not (hmac.compare_digest(given, u["code_hash"])
                         or (u["pending_hash"] and hmac.compare_digest(given, u["pending_hash"]))):
            return _DELETION_SENT
        token = secrets.token_urlsafe(24)
        c.execute("DELETE FROM deletions WHERE email_norm = ? OR expires < ?", (norm, now))
        c.execute("INSERT INTO deletions VALUES (?, ?, ?)", (_hash(token), norm, now + DELETION_LINK_HOURS * 3600))
    link = f"{PUBLIC_URL.rstrip('/')}/delete/confirm?token={token}"
    try:
        send_mail(u["email"], "[Pandemonium] confirm deleting your account", f"""Hello {u['name'] or ''},

Someone (hopefully you) asked to delete your Pandemonium preview account and all data about it.
To confirm, open this link within an hour and press the button on the page:

  {link}

If you did not ask for this, ignore this email: nothing will be deleted.

-- Pandemonium preview, BUT Speech@FIT
""")
    except Exception as err:
        return f"Sorry, we could not send the confirmation email ({type(err).__name__}). Please try again later."
    return _DELETION_SENT


def deletion_email(token: str) -> str | None:
    """The account a still-valid confirmation link belongs to."""
    with db() as c:
        row = c.execute("SELECT email_norm FROM deletions WHERE token_hash = ? AND expires > ?",
                        (_hash(token or ""), time.time())).fetchone()
    return row["email_norm"] if row else None


def confirm_deletion(token: str) -> str | None:
    """Use a confirmation link: delete its account; returns the account, or None if the link is not valid."""
    email = deletion_email(token)
    if email:
        delete_account(email)
    return email


def _page(title: str, inner: str) -> str:
    """A small standalone page in the signup page's style."""
    head = SIGNUP_PAGE.split("</style>")[0].replace("{{", "{").replace("}}", "}")
    head = head.replace("<title>Pandemonium preview: request access</title>", f"<title>{html.escape(title)}</title>")
    return f"{head}</style></head><body><main><div class='card'>{inner}</div></main></body></html>"


def delete_page(message: str = "", email: str = "") -> str:
    note = f"<div class='msg'>{message}</div>" if message else ""
    return _page("Pandemonium preview: delete my account", f"""<h1>Delete my account</h1>
<p class="lead">Delete your account and all data about it: your details, usage records, uploads and results.
This cannot be undone.</p>{note}
<form method="post" action="delete">
 <label>Email address<input type="email" name="email" required maxlength="200" value="{html.escape(email, quote=True)}"></label>
 <label>Access code<input type="text" name="code" required maxlength="200" autocomplete="off"></label>
 <button type="submit">Email me a confirmation link</button>
</form>
<p style="margin-top:18px"><a href="./?login">Back to login</a> &middot; <a href="privacy">Privacy notice</a></p>""")


def delete_confirm_page(token: str, email: str | None) -> str:
    if not email:
        return _page("Pandemonium preview: link expired", """<h1>This link is not valid</h1>
<p>It has expired (links work for an hour), or it has been used already. You can
<a href="../delete">ask for a new one</a>.</p>""")
    return _page("Pandemonium preview: confirm deletion", f"""<h1>Delete {html.escape(email)}?</h1>
<p class="lead">This deletes your account and all data about it, for good. Your access code stops working.</p>
<form method="post" action="confirm"><input type="hidden" name="token" value="{html.escape(token, quote=True)}">
 <button type="submit" style="background:#dc2626">Delete my account and data</button></form>
<p style="margin-top:18px"><a href="../?login">Keep my account</a></p>""")


def delete_done_page(done: bool) -> str:
    if not done:
        return delete_confirm_page("", None)
    page = _page("Pandemonium preview: account deleted", """<h1>Your account has been deleted</h1>
<p>Your account and all data about it are gone. Thank you for trying the preview.</p>
<p style="margin-top:18px"><a href="../example/">Back to the example</a>
<span style="color:var(--muted)">&middot; taking you there in 6 s</span></p>""")
    return page.replace("</head>", '<meta http-equiv="refresh" content="6;url=../example/"></head>', 1)


def privacy_page() -> str:
    return PRIVACY_PAGE.format(admin=html.escape(ADMIN_EMAIL))
