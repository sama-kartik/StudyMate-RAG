"""Single-user login.

Credentials come from environment variables, never from the code (the repo is public):
  APP_USER_EMAIL       the login email
  APP_PASSWORD_HASH    a werkzeug password hash  (preferred; see README)
  APP_PASSWORD         or the plain password, e.g. as an encrypted Vercel env var
  SECRET_KEY           optional; signs the session cookie. If unset, a key is derived
                       from the other secrets so all serverless instances agree.
"""
import hashlib
import hmac
import os
import time
from functools import wraps

from flask import jsonify, redirect, request, session, url_for
from werkzeug.security import check_password_hash

USER_EMAIL = os.getenv("APP_USER_EMAIL", "").strip().lower()
_PASSWORD_HASH = os.getenv("APP_PASSWORD_HASH", "")
_PASSWORD = os.getenv("APP_PASSWORD", "")


def is_configured() -> bool:
    return bool(USER_EMAIL and (_PASSWORD_HASH or _PASSWORD))


def secret_key() -> str:
    explicit = os.getenv("SECRET_KEY")
    if explicit:
        return explicit
    # Stable across serverless instances, and changes when the password changes
    # (which signs everyone out).
    material = "|".join(["studymate-session-v1", USER_EMAIL, _PASSWORD_HASH, _PASSWORD,
                         os.getenv("GEMINI_API_KEY", "")])
    return hashlib.sha256(material.encode()).hexdigest()


def check_credentials(email: str, password: str) -> bool:
    if not is_configured():
        return False
    email_ok = hmac.compare_digest((email or "").strip().lower(), USER_EMAIL)
    if _PASSWORD_HASH:
        pw_ok = check_password_hash(_PASSWORD_HASH, password or "")
    else:
        pw_ok = hmac.compare_digest((password or "").encode(), _PASSWORD.encode())
    if not (email_ok and pw_ok):
        time.sleep(1)  # slow down password guessing
        return False
    return True


def current_user():
    return session.get("user")


def login_required(view):
    """Pages redirect to /login; API calls get 401 JSON (the browser then redirects)."""
    @wraps(view)
    def wrapper(*args, **kwargs):
        if current_user():
            return view(*args, **kwargs)
        if request.path.startswith("/api/"):
            return jsonify(error="Please sign in again.", login_required=True), 401
        return redirect(url_for("login"))
    return wrapper
