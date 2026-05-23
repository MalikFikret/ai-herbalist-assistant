"""Authentication, user registration, and session management.

Handles admin credential verification, user login/registration,
password hashing (PBKDF2-SHA256), and session state initialization.
"""

import hashlib
import logging
import os
import secrets
import time
from typing import Dict, List

import streamlit as st

from herbalist_assistant.db import repository as db_repository

from .cookies import (
    _REMEMBER_COOKIE_NAME,
    _forget_remembered_password,
    _get_cookie_manager,
)
from .i18n import get_string
from .state import start_new_chat

_logger = logging.getLogger("herbalist_assistant.ui.auth")

# Admin credentials are sourced from environment variables so they are NOT
# committed in source. Supported variables (first match wins):
#   HA_ADMIN_USERNAME          - custom admin username (default: "admin")
#   HA_ADMIN_PASSWORD_HASH +
#   HA_ADMIN_PASSWORD_SALT     - PBKDF2-SHA256 hash (hex) and salt (hex)
#                                -- most secure, recommended for production
#   HA_ADMIN_PASSWORD          - plaintext fallback, hashed in-memory only
#
# If NONE of these are set, we fall back to the legacy dev default ("1234")
# and log a loud warning. Never leave this unset in a deployment.
ADMIN_USERNAME = os.environ.get("HA_ADMIN_USERNAME", "admin").strip() or "admin"
_ADMIN_PASSWORD_HASH_ENV = os.environ.get("HA_ADMIN_PASSWORD_HASH", "").strip()
_ADMIN_PASSWORD_SALT_ENV = os.environ.get("HA_ADMIN_PASSWORD_SALT", "").strip()
_ADMIN_PASSWORD_ENV = os.environ.get("HA_ADMIN_PASSWORD", "")
_ADMIN_DEFAULT_PASSWORD = "1234"
_ADMIN_DEFAULT_WARNING_EMITTED = False
_ADMIN_USERNAME_SETTING_KEY = "admin_username"
_ADMIN_PASSWORD_HASH_SETTING_KEY = "admin_password_hash"
_ADMIN_PASSWORD_SALT_SETTING_KEY = "admin_password_salt"

AVAILABLE_MODELS: List[str] = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "gemini-2.5-flash",
    "deepseek-chat",
]
DEFAULT_MODEL = "gemini-2.5-flash"
AVAILABLE_WEB_SEARCH_PROVIDERS: List[str] = ["Tavily", "DuckDuckGo"]
DEFAULT_WEB_SEARCH_PROVIDER = "Tavily"


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    ).hex()


def _verify_password(password: str, password_hash: str, salt: str) -> bool:
    return _hash_password(password, salt) == password_hash


def _admin_credentials_locked_by_env() -> bool:
    return bool(
        _ADMIN_PASSWORD_ENV
        or (_ADMIN_PASSWORD_HASH_ENV and _ADMIN_PASSWORD_SALT_ENV)
    )


def _load_runtime_admin_credentials() -> Dict[str, str]:
    username = db_repository.get_app_setting(_ADMIN_USERNAME_SETTING_KEY) or ADMIN_USERNAME
    password_hash = db_repository.get_app_setting(_ADMIN_PASSWORD_HASH_SETTING_KEY) or ""
    salt = db_repository.get_app_setting(_ADMIN_PASSWORD_SALT_SETTING_KEY) or ""
    return {
        "username": username.strip() or ADMIN_USERNAME,
        "password_hash": password_hash.strip(),
        "salt": salt.strip(),
    }


def _register_user(username: str, password: str, confirm_password: str) -> str:
    username = username.strip()
    if not username or not password or not confirm_password:
        return "All fields are required."
    if password != confirm_password:
        return "Passwords do not match."
    admin_username = _load_runtime_admin_credentials()["username"]
    if username == admin_username:
        return "This username is reserved."

    if db_repository.user_exists(username):
        return "User already exists."

    salt = secrets.token_hex(16)
    created = db_repository.create_user(
        username=username,
        password_hash=_hash_password(password, salt),
        salt=salt,
        role="user",
    )
    if not created:
        return "User already exists."
    return "Account created successfully. You can now log in."


def _reset_user_password(username: str, new_password: str, confirm_password: str) -> str:
    username = username.strip()
    if not username or not new_password or not confirm_password:
        return "All fields are required."
    if new_password != confirm_password:
        return "Passwords do not match."
    if len(new_password) < 4:
        return "Password must be at least 4 characters."
    admin_username = _load_runtime_admin_credentials()["username"]
    if username == admin_username:
        return "Admin password reset is disabled in this screen."

    salt = secrets.token_hex(16)
    updated = db_repository.reset_user_password(
        username=username,
        password_hash=_hash_password(new_password, salt),
        salt=salt,
    )
    if not updated:
        return "User not found."
    # Eğer kullanıcı daha önce Remember me ile şifresini hatırlatmışsa,
    # eski (artık geçersiz) entry'yi temizliyoruz. Yeni şifreyi otomatik
    # KAYDETMİYORUZ — bu davranış login formundaki Remember me'ye bağlı.
    _forget_remembered_password(username)
    return "Password reset successful. You can now log in."


def _verify_admin_password(password: str) -> bool:
    """Verify the admin password against the most secure source available.

    Order of precedence:
      1. HA_ADMIN_PASSWORD_HASH + HA_ADMIN_PASSWORD_SALT (PBKDF2-SHA256).
      2. HA_ADMIN_PASSWORD (plaintext in env, compared with constant time).
      3. Hardcoded dev default "1234" (with a WARNING log).
    """
    global _ADMIN_DEFAULT_WARNING_EMITTED

    if _ADMIN_PASSWORD_HASH_ENV and _ADMIN_PASSWORD_SALT_ENV:
        expected = _ADMIN_PASSWORD_HASH_ENV
        actual = _hash_password(password, _ADMIN_PASSWORD_SALT_ENV)
        return secrets.compare_digest(expected, actual)

    if _ADMIN_PASSWORD_ENV:
        return secrets.compare_digest(_ADMIN_PASSWORD_ENV, password)

    runtime_creds = _load_runtime_admin_credentials()
    runtime_hash = runtime_creds.get("password_hash", "")
    runtime_salt = runtime_creds.get("salt", "")
    if runtime_hash and runtime_salt:
        actual = _hash_password(password, runtime_salt)
        return secrets.compare_digest(runtime_hash, actual)

    if not _ADMIN_DEFAULT_WARNING_EMITTED:
        _logger.warning(
            "Admin is using the insecure DEFAULT password. "
            "Set HA_ADMIN_PASSWORD (or HA_ADMIN_PASSWORD_HASH + "
            "HA_ADMIN_PASSWORD_SALT) in your environment before deploying."
        )
        _ADMIN_DEFAULT_WARNING_EMITTED = True
    return secrets.compare_digest(_ADMIN_DEFAULT_PASSWORD, password)


def _authenticate_user(username: str, password: str) -> Dict[str, str]:
    username = username.strip()
    if not username or not password:
        return {"status": "error", "message": "Username and password are required."}

    admin_username = _load_runtime_admin_credentials()["username"]
    if username == admin_username:
        if _verify_admin_password(password):
            return {"status": "ok", "role": "admin"}
        return {"status": "error", "message": "Wrong password."}

    record = db_repository.get_user_auth(username)
    if not record:
        return {"status": "error", "message": "User not found."}

    password_hash = record.get("password_hash", "")
    salt = record.get("salt", "")
    if not password_hash or not salt or not _verify_password(password, password_hash, salt):
        return {"status": "error", "message": "Wrong password."}

    return {"status": "ok", "role": record.get("role", "user")}


def _init_auth_state() -> None:
    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "is_logged_in" not in st.session_state:
        st.session_state.is_logged_in = False
    if "role" not in st.session_state:
        st.session_state.role = "user"
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL
    elif st.session_state.selected_model not in AVAILABLE_MODELS:
        st.session_state.selected_model = DEFAULT_MODEL
    if "web_search_provider" not in st.session_state:
        st.session_state.web_search_provider = DEFAULT_WEB_SEARCH_PROVIDER
    elif st.session_state.web_search_provider not in AVAILABLE_WEB_SEARCH_PROVIDERS:
        st.session_state.web_search_provider = DEFAULT_WEB_SEARCH_PROVIDER
    if "last_index_time" not in st.session_state:
        st.session_state.last_index_time = None
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Chat"
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {
            "name": "",
            "age": "",
            "gender": "",
            "allergies": "",
            "conditions": "",
        }


def _normalize_active_page() -> None:
    """Keep ``active_page`` valid for the current auth role."""
    page = st.session_state.get("active_page", "Chat")
    if st.session_state.get("is_logged_in"):
        if st.session_state.get("role") == "admin":
            allowed = {"Admin Panel", "Chat"}
        else:
            allowed = {"Chat", "Profile"}
    else:
        allowed = {"Chat", "Login", "Profile"}
    if page not in allowed:
        st.session_state.active_page = "Chat"


def _logout() -> None:
    username = st.session_state.get("username", "").strip()
    role = st.session_state.get("role", "user")
    if username and role == "user":
        # Ensure next login starts in a fresh chat for the user.
        start_new_chat(username)
    cookie_mgr = _get_cookie_manager()
    cookie_deleted = False
    if cookie_mgr is not None:
        try:
            cookie_mgr.delete(cookie=_REMEMBER_COOKIE_NAME, key="ha_cookie_logout_del")
            cookie_deleted = True
        except Exception:
            _logger.debug("Remember-me cookie delete failed", exc_info=True)
    st.session_state.clear()
    # Logout sonrası Misafir Chat yerine doğrudan Login ekranı açılsın.
    st.session_state["active_page"] = "Login"
    st.session_state["auth_mode"] = "Login"
    # Bu session için auto-login'i kapat: cookie henüz tarayıcıdan silinmemiş
    # olsa bile kullanıcı çıkış yapmayı seçtiği için tekrar oturum açılmasın.
    st.session_state["ha_remember_consumed"] = True
    # Cookie delete komutunun WebSocket üzerinden tarayıcıya iletilmesi için
    # kısa süre bekle (Login set/delete'le aynı race condition'ı önler).
    if cookie_deleted:
        time.sleep(0.4)
    st.rerun()


def _try_auto_login_from_cookie() -> None:
    """If a valid HMAC-signed remember cookie is present, log the user in.

    Runs once per Streamlit session: after a successful auto-login we set
    ``ha_remember_consumed`` so subsequent reruns skip the cookie roundtrip.
    """
    from .cookies import _verify_remember_token

    if st.session_state.get("is_logged_in"):
        return
    if st.session_state.get("ha_remember_consumed"):
        return
    cookie_mgr = _get_cookie_manager()
    if cookie_mgr is None:
        return

    # Track cookie check attempts to prevent infinite iframe reruns on startup
    if "ha_cookie_check_attempts" not in st.session_state:
        st.session_state.ha_cookie_check_attempts = 0

    try:
        token = cookie_mgr.get(cookie=_REMEMBER_COOKIE_NAME)
    except Exception:
        _logger.debug("Remember-me cookie read failed", exc_info=True)
        st.session_state["ha_remember_consumed"] = True
        return

    st.session_state.ha_cookie_check_attempts += 1

    # The cookie manager returns None on the first run before the JS round-trip
    # completes; mark consumed only when we actually saw a token (or the cookie
    # is conclusively absent on a later rerun).
    if not token:
        if st.session_state.ha_cookie_check_attempts > 1:
            st.session_state["ha_remember_consumed"] = True
        return

    username = _verify_remember_token(str(token))
    if not username:
        try:
            cookie_mgr.delete(cookie=_REMEMBER_COOKIE_NAME, key="ha_cookie_invalid_clear")
        except Exception:
            pass
        st.session_state["ha_remember_consumed"] = True
        return
    profile = db_repository.get_user_profile(username)
    if not profile:
        try:
            cookie_mgr.delete(cookie=_REMEMBER_COOKIE_NAME, key="ha_cookie_unknown_clear")
        except Exception:
            pass
        st.session_state["ha_remember_consumed"] = True
        return
    st.session_state.is_logged_in = True
    st.session_state.username = username
    st.session_state.role = (
        "admin" if username == ADMIN_USERNAME else "user"
    )
    st.session_state.user_profile = profile
    if st.session_state.role == "user":
        from .pages.chat import _sync_conversations_to_session

        start_new_chat(username)
        _sync_conversations_to_session(username)
    st.session_state.active_page = (
        "Admin Panel" if st.session_state.role == "admin" else "Chat"
    )
    st.session_state["ha_remember_consumed"] = True
    st.rerun()
