#!/usr/bin/env python3
"""Generate HA_ADMIN_PASSWORD_HASH and HA_ADMIN_PASSWORD_SALT values.

Usage:
  python scripts/generate_admin_password_hash.py
  python scripts/generate_admin_password_hash.py --password "MyStrongPassword123"
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import secrets


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    ).hex()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate admin password hash/salt for HA_ADMIN_PASSWORD_HASH and HA_ADMIN_PASSWORD_SALT.",
    )
    parser.add_argument(
        "--password",
        default="",
        help="Admin password (if omitted, prompts securely).",
    )
    parser.add_argument(
        "--salt",
        default="",
        help="Optional existing salt (hex). If omitted, a new random salt is generated.",
    )
    args = parser.parse_args()

    password = args.password or getpass.getpass("Admin password: ")
    if not password:
        raise SystemExit("Password cannot be empty.")

    salt = args.salt.strip() or secrets.token_hex(16)
    password_hash = _hash_password(password, salt)

    print("Use these in your .env:")
    print(f"HA_ADMIN_PASSWORD_HASH={password_hash}")
    print(f"HA_ADMIN_PASSWORD_SALT={salt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
