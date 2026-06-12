"""'Remember me' cookie support and stored credential helpers.

Persists an HMAC-signed (username, expiry) tuple in a browser cookie.
Verification happens server-side using HA_REMEMBER_SECRET (or a derived
fallback) so a tampered cookie cannot impersonate another user.

Also manages per-machine password auto-fill: a lightly obfuscated
username→password mapping stored in ``data/.remembered_logins.json``.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Optional

import streamlit as st

from herbalist_assistant import config

_logger = logging.getLogger("herbalist_assistant.ui.cookies")

# ----- "Remember me" cookie support --------------------------------------
# We persist a HMAC-signed (username, expiry) tuple in a browser cookie.
# Verification happens server-side using HA_REMEMBER_SECRET (or a derived
# fallback) so a tampered cookie cannot impersonate another user.
_REMEMBER_COOKIE_NAME = "ha_remember"
_REMEMBER_TTL_DAYS = 30

# Generated once per process. If HA_REMEMBER_SECRET is not set, "remember me"
# sessions expire on app restart — an acceptable trade-off vs. using GROQ_API_KEY.
_EPHEMERAL_SECRET: bytes = secrets.token_bytes(32)


def _remember_secret() -> bytes:
    """Return the secret used to sign remember-me tokens.

    Priority:
    1. ``HA_REMEMBER_SECRET`` env var  — stable across restarts (recommended).
    2. ``_EPHEMERAL_SECRET``           — random bytes generated at startup.
       Tokens remain valid for the lifetime of the process only; "remember me"
       sessions expire on app restart. This is intentional: it avoids tying
       cookie security to GROQ_API_KEY or any other unrelated credential.
    """
    env_secret = os.environ.get("HA_REMEMBER_SECRET", "").strip()
    if env_secret:
        return env_secret.encode("utf-8")
    return _EPHEMERAL_SECRET


def _make_remember_token(username: str, ttl_days: int = _REMEMBER_TTL_DAYS) -> str:
    expires = int(time.time()) + ttl_days * 86400
    payload = f"{username}|{expires}"
    sig = hmac.new(_remember_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}|{sig}"


def _verify_remember_token(token: str) -> Optional[str]:
    if not token or token.count("|") != 2:
        return None
    try:
        username, expires_str, sig = token.split("|")
    except ValueError:
        return None
    payload = f"{username}|{expires_str}"
    expected = hmac.new(_remember_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        if int(expires_str) < int(time.time()):
            return None
    except ValueError:
        return None
    if not username:
        return None
    return username


# ----- Stored credentials (auto-fill password by username) ----------------
# Dosya: ``data/.remembered_logins.json``. İçinde her kullanıcı adı için XOR-
# obfuscate edilmiş şifre saklanır. Bu PBKDF2 hash'in YERİNE değil, login
# formundaki "isim yazınca şifre kendiliğinden gelsin" UX akışı için.
_REMEMBERED_LOGINS_FILENAME = ".remembered_logins.json"


def _remembered_logins_path() -> Path:
    from herbalist_assistant.settings_manager import get_setting
    p = Path(get_setting("DATA_DIR")) / _REMEMBERED_LOGINS_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _xor_keystream(secret: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(
            hmac.new(
                secret,
                nonce + counter.to_bytes(4, "big"),
                hashlib.sha256,
            ).digest()
        )
        counter += 1
    return bytes(out[:length])


def _obscure_password(plain: str) -> str:
    """Hafif obfuscation. Casual disk inspection'a karşı koruma; gerçek
    güvenlik PBKDF2 hash tarafında zaten var."""
    if not plain:
        return ""
    nonce = secrets.token_bytes(16)
    plain_b = plain.encode("utf-8")
    keystream = _xor_keystream(_remember_secret(), nonce, len(plain_b))
    cipher = bytes(p ^ k for p, k in zip(plain_b, keystream))
    return base64.urlsafe_b64encode(nonce + cipher).decode("ascii")


def _unobscure_password(token: str) -> Optional[str]:
    if not token:
        return None
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
    except Exception:
        return None
    if len(raw) < 16:
        return None
    nonce, cipher = raw[:16], raw[16:]
    keystream = _xor_keystream(_remember_secret(), nonce, len(cipher))
    try:
        return bytes(c ^ k for c, k in zip(cipher, keystream)).decode("utf-8")
    except Exception:
        return None


def _load_remembered_logins() -> dict:
    p = _remembered_logins_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_remembered_logins(data: dict) -> None:
    p = _remembered_logins_path()
    try:
        p.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        _logger.debug("Saving remembered logins failed", exc_info=True)


def _store_remembered_password(username: str, password: str) -> None:
    """Kayıt veya şifre değişimi sonrası username -> şifre eşlemesini saklar."""
    uname = (username or "").strip().lower()
    if not uname or not password:
        return
    data = _load_remembered_logins()
    data[uname] = _obscure_password(password)
    _save_remembered_logins(data)


def _lookup_remembered_password(username: str) -> Optional[str]:
    uname = (username or "").strip().lower()
    if not uname:
        return None
    data = _load_remembered_logins()
    token = data.get(uname)
    if not token:
        return None
    return _unobscure_password(str(token))


def _forget_remembered_password(username: str) -> None:
    uname = (username or "").strip().lower()
    if not uname:
        return
    data = _load_remembered_logins()
    if data.pop(uname, None) is not None:
        _save_remembered_logins(data)


# ----- Cookie manager (extra-streamlit-components) ------------------------

try:
    import extra_streamlit_components as _stx  # type: ignore

    _COOKIES_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _stx = None
    _COOKIES_AVAILABLE = False
    _logger.warning(
        "extra-streamlit-components not installed; 'Remember me' on login will be disabled. "
        "Run: pip install extra-streamlit-components"
    )


_COOKIE_MGR_SESSION_KEY = "_ha_cookie_mgr"


def _initialize_cookie_manager() -> None:
    """Instantiate the CookieManager only when needed.

    To prevent infinite iframe mounts and WebSocket reruns during an active
    session, we skip rendering the iframe if:
    1. The user is logged in AND we don't have a pending logout/cookie action.
    2. We have already checked the remember cookie on startup (consumed = True)
       AND the current page is not 'Login' (where we might need to set/clear cookies).
    """
    if not _COOKIES_AVAILABLE:
        st.session_state[_COOKIE_MGR_SESSION_KEY] = None
        return

    # Check if we actually need the cookie manager
    logged_in = st.session_state.get("is_logged_in", False)
    remember_consumed = st.session_state.get("ha_remember_consumed", False)
    active_page = st.session_state.get("active_page", "Chat")

    # We need it if:
    # - We haven't checked the remember cookie yet on startup (auto-login phase)
    # - OR the user is on the Login page (where they might check "Remember me")
    # - OR the user clicked logout (which will trigger cookie deletion)
    needs_cookies = (not logged_in and not remember_consumed) or (active_page == "Login") or st.session_state.get("ha_logging_out", False)

    if not needs_cookies:
        st.session_state[_COOKIE_MGR_SESSION_KEY] = None
        return

    try:
        st.session_state[_COOKIE_MGR_SESSION_KEY] = _stx.CookieManager(
            key="ha_cookies"
        )
    except Exception:
        _logger.debug("CookieManager init failed", exc_info=True)
        st.session_state[_COOKIE_MGR_SESSION_KEY] = None


def _get_cookie_manager():
    """Return the CookieManager initialized for this run (or None)."""
    return st.session_state.get(_COOKIE_MGR_SESSION_KEY)


def _on_login_username_change() -> None:
    """Login formu UX: kullanıcı adı yazılıp commit edildiğinde (Tab/Enter),
    daha önce bu makinada kayıt olmuş ya da giriş yapmış kullanıcı için
    hatırlanan şifre ``login_password`` session_state alanına yazılır. Bir
    sonraki rerun'da password input bu değerle pre-fill olarak render edilir.

    Eşleşen kayıt yoksa, önceki auto-fill artığı bırakmamak için password
    alanı temizlenir — böylece farklı bir hesaba geçildiğinde yanlış şifre
    formda kalmaz.
    """
    user = str(st.session_state.get("login_username", "")).strip()
    if not user:
        return
    pwd = _lookup_remembered_password(user)
    last_user = st.session_state.get("_ha_autofill_last_user")
    if pwd:
        st.session_state["login_password"] = pwd
        st.session_state["_ha_autofill_last_user"] = user
    elif last_user is not None and last_user != user:
        # Önceki user için autofill yapılmıştı, bu user için yok → temizle.
        st.session_state["login_password"] = ""
        st.session_state["_ha_autofill_last_user"] = user