#!/usr/bin/env python3
"""Send the Pandemonium demo's emails for it, so the SMTP password never reaches the demo server.

Runs on the mail host (pcspeech4) as the SSH forced command of the demo's dedicated key, one message
per connection: the demo pipes a JSON object `{to, subject, body}` to `ssh`, and this script checks
it and sends it through FIT's server with the account owner's login. Whoever holds the demo's key
(including root on the demo server) can therefore only send messages that look like the demo's own:
one recipient, the demo's two subjects, a bounded body, at a bounded rate. Never a shell, never the
password.

Install (on the mail host, as the account owner):

    install -D -m 700 pandemonium-mail.py ~/bin/pandemonium-mail
    install -d -m 700 ~/.config/pandemonium-mail       # on LOCAL disk, see the password note below
    install -m 600 /dev/null ~/.config/pandemonium-mail/smtp_password   # then write the password in

and in ~/.ssh/authorized_keys, one line (see deploy/README or the setup notes):

    restrict,command="$HOME/bin/pandemonium-mail",from="147.229.8.178,2001:67c:1220:808::93e5:8b2" ssh-ed25519 AAAA... pandemonium-demo mail

Check without sending: `echo '{"to": "you@fit.vut.cz", "subject": "[Pandemonium] test", "body": "x"}' |
pandemonium-mail --dry-run`.
"""

import argparse
import fcntl
import json
import os
import re
import smtplib
import stat
import sys
import time
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

SMTP_HOST, SMTP_PORT, SMTP_USER = "smtp.fit.vutbr.cz", 587, "ipoloka"
ADMIN = "ipoloka@fit.vut.cz"
FROM = f"Pandemonium preview <{ADMIN}>"
# The only messages the demo sends: an access code to a person, and a signup notice to the admin.
SUBJECTS = ("Your Pandemonium preview access code", "[Pandemonium] ")
PER_HOUR, PER_DAY = 30, 200
MAX_BODY, MAX_INPUT = 8_000, 64_000
EMAIL_RE = re.compile(r"^[^@\s<>,;\"]{1,64}@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def fail(message: str, code: int = 1):
    print(message, file=sys.stderr)
    sys.exit(code)


def read_password(path: str) -> str:
    try:
        st = os.stat(path)
    except FileNotFoundError:
        fail(f"no password file at {path}")
    if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        fail(f"{path} is readable by others; chmod 600 it first")
    with open(path) as f:
        return f.readline().rstrip("\n")


def check(message: dict) -> dict:
    if not isinstance(message, dict) or set(message) - {"to", "subject", "body"}:
        fail("expected a JSON object with exactly: to, subject, body")
    to, subject, body = (str(message.get(k, "")) for k in ("to", "subject", "body"))
    if not EMAIL_RE.match(to):
        fail("bad recipient")
    if not subject.startswith(SUBJECTS) or len(subject) > 200 or "\n" in subject or "\r" in subject:
        fail("subject not allowed")
    if not body or len(body) > MAX_BODY:
        fail("body empty or too long")
    return {"to": to, "subject": subject, "body": body}


def within_rate(state_dir: str, to: str, subject: str):
    """Append to the audit log under a lock, refusing once the hourly or daily budget is spent."""
    os.makedirs(state_dir, mode=0o700, exist_ok=True)
    path = os.path.join(state_dir, "sent.log")
    with open(path, "a+") as log:
        fcntl.flock(log, fcntl.LOCK_EX)
        log.seek(0)
        now = time.time()
        stamps = [float(line.split("\t", 1)[0]) for line in log if line.strip()]
        if sum(t > now - 3600 for t in stamps) >= PER_HOUR or sum(t > now - 86400 for t in stamps) >= PER_DAY:
            fail("rate limit reached", 2)
        log.write(f"{now:.0f}\t{time.strftime('%Y-%m-%d %H:%M:%S')}\t{to}\t{subject}\n")


def main():
    p = argparse.ArgumentParser(description="Send one Pandemonium demo email read as JSON from stdin.")
    p.add_argument("--password-file", default=os.path.expanduser("~/.config/pandemonium-mail/smtp_password"))
    p.add_argument("--state-dir", default=os.path.expanduser("~/.local/state/pandemonium-mail"))
    p.add_argument("--dry-run", action="store_true", help="check and print the message, send nothing")
    args = p.parse_args()

    raw = sys.stdin.read(MAX_INPUT + 1)
    if len(raw) > MAX_INPUT:
        fail("input too large")
    try:
        message = check(json.loads(raw))
    except json.JSONDecodeError:
        fail("input is not JSON")

    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = FROM, message["to"], message["subject"]
    msg["Reply-To"], msg["Date"] = ADMIN, formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain="fit.vut.cz")
    msg.set_content(message["body"])

    if args.dry_run:
        print(msg)
        print("ok (dry run, nothing sent)")
        return

    password = read_password(args.password_file)
    within_rate(args.state_dir, message["to"], message["subject"])
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
        s.ehlo()
        s.starttls()
        s.ehlo()
        s.login(SMTP_USER, password)
        s.send_message(msg)
    print("ok")


if __name__ == "__main__":
    main()
