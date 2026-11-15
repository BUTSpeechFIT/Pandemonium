"""Manage preview users from the server's shell.

    python admin.py list                    # everyone, with usage
    python admin.py show  EMAIL             # one user, with their requests
    python admin.py grant EMAIL N           # set their quota to N transcriptions
    python admin.py add   EMAIL N           # give N more
    python admin.py revoke  EMAIL           # their code stops working immediately
    python admin.py restore EMAIL
    python admin.py create  EMAIL [--name NAME] [--quota N] [--print]   # add someone yourself; emails
                                            # them a code, or prints it with --print
    python admin.py resend  EMAIL           # issue and email a new code (the old one stops working)
    python admin.py ip      IP              # what one address has been doing
    python admin.py delete  EMAIL           # erase the account and every record of it (GDPR requests)
"""

import argparse
import sys
import time

import access


def _user(c, email):
    u = c.execute("SELECT * FROM users WHERE email_norm = ?", (access.normalize_email(email),)).fetchone()
    if not u:
        sys.exit(f"No user {email!r}.")
    return u


def _when(ts):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["list", "show", "grant", "add", "revoke", "restore", "create", "resend", "ip", "delete"])
    p.add_argument("target", nargs="?")
    p.add_argument("n", nargs="?", type=int)
    p.add_argument("--name", default="")
    p.add_argument("--quota", type=int, default=access.DEFAULT_QUOTA)
    p.add_argument("--print", action="store_true", help="create: show the code here instead of emailing it")
    a = p.parse_args()
    access.init_db()

    if a.cmd == "list":
        with access.db() as c:
            rows = c.execute("SELECT * FROM users ORDER BY created").fetchall()
        print(f"{'email':38s} {'used/quota':>10s}  {'created':16s}  {'signup ip':15s}  affiliation")
        for u in rows:
            flag = "  [revoked]" if u["revoked"] else ""
            print(f"{u['email']:38s} {u['used']:>4d}/{u['quota']:<5d}  {_when(u['created'])}  "
                  f"{u['signup_ip'] or '':15s}  {u['affiliation'] or ''}{flag}")
        print(f"{len(rows)} users")
        return

    if a.cmd == "ip":
        with access.db() as c:
            rows = c.execute("SELECT * FROM requests WHERE ip = ? ORDER BY ts", (a.target,)).fetchall()
            accts = c.execute("SELECT email FROM users WHERE signup_ip = ?", (a.target,)).fetchall()
        print(f"accounts created from {a.target}: {[r['email'] for r in accts]}")
        for r in rows:
            print(f"  {_when(r['ts'])}  {r['email_norm']:34s} {r['seconds'] or 0:7.0f}s  {r['status']}")
        return

    if not a.target:
        p.error("this command needs an EMAIL")

    if a.cmd == "create":
        norm, code = access.normalize_email(a.target), access.new_code()
        with access.db() as c:
            if c.execute("SELECT 1 FROM users WHERE email_norm = ?", (norm,)).fetchone():
                sys.exit(f"{a.target} already exists; use grant/resend.")
            c.execute("INSERT INTO users (email_norm, email, name, affiliation, purpose, code_hash, quota, created, note) "
                      "VALUES (?, ?, ?, '', '', ?, ?, ?, 'created by admin')",
                      (norm, a.target, a.name, access._hash(code), a.quota, time.time()))
        if a.print:
            print(f"Created {a.target} with quota {a.quota}.\n  username: {a.target}\n  code:     {code}")
        else:
            access.mail_code(a.target, a.name, code, a.quota)
            print(f"Created {a.target} with quota {a.quota}; code emailed.")
        return

    with access.db() as c:
        u = _user(c, a.target)
        key = u["email_norm"]
        if a.cmd == "show":
            for k in ("email", "name", "affiliation", "purpose", "used", "quota", "revoked", "signup_ip", "note"):
                print(f"{k:12s} {u[k]}")
            print(f"{'created':12s} {_when(u['created'])}")
            for r in c.execute("SELECT * FROM requests WHERE email_norm = ? ORDER BY ts", (key,)):
                print(f"  {_when(r['ts'])}  {r['ip']:15s} {r['seconds'] or 0:7.0f}s  {r['status']}")
        elif a.cmd in ("grant", "add"):
            if a.n is None:
                p.error(f"{a.cmd} needs a number")
            quota = a.n if a.cmd == "grant" else u["quota"] + a.n
            c.execute("UPDATE users SET quota = ? WHERE email_norm = ?", (quota, key))
            print(f"{u['email']}: quota {u['quota']} -> {quota} (used {u['used']})")
        elif a.cmd in ("revoke", "restore"):
            c.execute("UPDATE users SET revoked = ? WHERE email_norm = ?", (int(a.cmd == "revoke"), key))
            print(f"{u['email']}: {'revoked' if a.cmd == 'revoke' else 'restored'}")
        elif a.cmd == "resend":  # an explicit reset by you: the old code stops working now
            code = access.new_code()
            c.execute("UPDATE users SET code_hash = ?, pending_hash = NULL WHERE email_norm = ?",
                      (access._hash(code), key))
        elif a.cmd == "delete":
            pass  # below, outside this transaction (delete_account opens its own)
    if a.cmd == "delete":
        access.delete_account(key)
        print(f"{u['email']}: account, requests and signup records deleted")
    if a.cmd == "resend":
        access.mail_code(u["email"], u["name"], code, u["quota"], reissued=True)
        print(f"New code emailed to {u['email']}.")


if __name__ == "__main__":
    main()
