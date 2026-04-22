# Security Policy

This document describes how to report security issues in the **AI Herbalist
Assistant** repository and the operational guarantees (and limits) of the
application as shipped. This file is repository documentation only; it does
**not** change the chatbot's system prompts or user-facing wording.

---

## 1. Reporting a vulnerability

If you believe you have found a security issue, please do **not** open a
public GitHub issue. Instead:

1. Contact the maintainers directly (Malik Fikret, Ebru Tuğçe Polat,
   Melisa Yıldırım, or Prof. Dr. Ramazan KATIRCI) through a private channel.
2. Include:
   - A clear description of the issue and potential impact.
   - Steps to reproduce, ideally with the smallest possible example.
   - Your name/handle if you would like to be credited.
3. Please allow a reasonable window for triage and a coordinated disclosure
   before sharing the details publicly.

We aim to respond to any credible report within 7 days.

---

## 2. Scope

### In scope
- Code under `src/herbalist_assistant/`.
- Deployment guidance in `README.md`, `SECURITY.md`, the `Dockerfile`, and
  the CI workflow under `.github/workflows/`.
- The local persistence layer (`.herbalist.db`, via SQLAlchemy),
  admin credential handling, and any legacy `.users.json` /
  `.chat_history.json` files still present during the first-run
  migration.
- The LangGraph / Groq / Chroma integration configured in
  `herbalist_assistant.graph.advanced_graph`.

### Out of scope
- The accuracy, completeness, or medical correctness of herbal answers.
  This application is an **educational tool** over a local corpus; it is
  **not** a medical device and it provides no diagnostic or treatment
  advice.
- Vulnerabilities in third-party services we call (Groq, LangSmith) or in
  upstream libraries (LangChain, LangGraph, Chroma, Streamlit,
  sentence-transformers). Please report those upstream.
- Denial-of-service from operating the app without an LLM rate limit in
  front of it.

---

## 3. Credentials and secrets

### 3.1 Groq / LangSmith API keys
- Keys belong in `.env`, which is already listed in `.gitignore`.
- If a key is ever committed, it must be considered compromised. **Rotate
  immediately** in the Groq console and the LangSmith settings page.
- The repository ships without any embedded API keys.

### 3.2 Admin credentials
Admin credentials are resolved at startup, in the following order of
preference:

1. `HA_ADMIN_PASSWORD_HASH` + `HA_ADMIN_PASSWORD_SALT` &mdash; PBKDF2-SHA256
   hash comparison. Recommended for any non-local deployment.
2. `HA_ADMIN_PASSWORD` &mdash; plaintext env var, compared with
   `secrets.compare_digest`. Acceptable for local, discouraged for shared
   deployments.
3. The legacy development default `1234` &mdash; used only when none of
   the above is set. The application logs a loud `WARNING` when this
   fallback is used. **Never leave this in place in a deployment.**

`HA_ADMIN_USERNAME` can be used to move the admin account off the default
`admin` name.

### 3.3 Generating a PBKDF2 admin hash
```python
import secrets, hashlib
salt = secrets.token_hex(16)
hash_hex = hashlib.pbkdf2_hmac(
    "sha256", b"your-new-strong-password", salt.encode("utf-8"), 100_000,
).hex()
print("HA_ADMIN_PASSWORD_HASH=" + hash_hex)
print("HA_ADMIN_PASSWORD_SALT=" + salt)
```
Copy the two lines into `.env`, remove `HA_ADMIN_PASSWORD` if present,
and restart the app.

---

## 4. Authentication and authorization model

- Regular users are authenticated against rows in the local SQLite
  database (`.herbalist.db`, schema managed by SQLAlchemy) using
  PBKDF2-SHA256 password hashes and per-user salts.
- The admin account uses the env-var flow described above. It is not
  stored in the database.
- There is **no** password strength enforcement, account lockout, rate
  limiting, or session timeout in the app itself. If you deploy publicly,
  put the app behind a reverse proxy that provides at least rate limiting
  and TLS.

---

## 5. Data protection

- Per-user chat history, source metadata, and 👍 / 👎 feedback live in
  the SQLite database (`.herbalist.db`). On the first run after the
  P1 migration, any legacy `.users.json` / `.chat_history.json` files
  are imported into the DB and then renamed to
  `*.migrated-backup-<timestamp>`. All of these paths are gitignored
  and intended to stay on the machine running the app.
- The Chroma vector database lives in `.chroma_db/` (also gitignored).
  It contains embeddings derived from PDFs under `data/`.
- Do not place sensitive PDFs, personal medical records, or regulated
  data into `data/` without first clearing it with your data-governance
  owner. The app has no row-level access control over indexed content.

---

## 6. Hardening recommendations for deployment

- Set `HA_ADMIN_PASSWORD_HASH` + `HA_ADMIN_PASSWORD_SALT` in the runtime
  environment; do **not** bake them into the container image.
- Run the container as a non-root user (the provided `Dockerfile` does
  this by default).
- Restrict network egress to only `api.groq.com` (and `api.smith.langchain.com`
  if tracing is enabled).
- Run behind an HTTPS-terminating reverse proxy (nginx, Caddy, Traefik).
- Enable LangSmith tracing (`LANGSMITH_TRACING=true`) so you can audit
  prompts and tool calls after the fact.
- Keep backups of `.herbalist.db` (and `.chroma_db/` if expensive to
  rebuild) outside the container — mount them as volumes in production
  and include them in your regular backup rotation.

---

## 7. Non-goals

This section is reinforced to avoid confusion:

- The app is **not** HIPAA / GDPR / KVKK compliant out of the box.
- The app is **not** a substitute for medical advice, diagnosis, or
  treatment.
- The system prompts intentionally do **not** emit generic AI disclaimers
  on every answer. The UI already carries the educational-use notice; any
  medical or legal warnings need to live in your deployment's front-end
  copy, not inside the LLM responses.

---

## 8. Changelog

- 2026-04-21: Initial version written at project-hardening time; covers
  admin env-based credentials, LangSmith key handling, and the data-at-rest
  layout.
