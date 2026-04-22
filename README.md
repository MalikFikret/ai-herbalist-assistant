# 🌿 AI Herbalist Assistant

An AI-powered academic retrieval system (RAG) designed to provide accurate, context-based botanical and herbal information exclusively from provided medical and herbal texts.

## 🚀 Overview
This project utilizes a Retrieval-Augmented Generation (RAG) architecture to answer user queries about herbs, natural remedies, and common ailments. It strictly answers based on the local PDF documents provided in the system and is engineered to ensure a safe, non-diagnostic user experience.

---

## ⚙️ Prerequisites
Before you begin, ensure you have the following installed on your machine:
* **Python 3.10+**
* **Git**

---

## 🛠️ Installation & Setup

Follow these steps carefully to run the project on your local machine.

### 1. Clone the Repository
Open your terminal and clone this repository:
```bash
git clone https://github.com/MalikFikret/ai-herbalist-assistant.git
cd ai-herbalist-assistant
```

### 2. Create a Virtual Environment (Recommended)
It is highly recommended to use a virtual environment to prevent library conflicts.
```bash
python -m venv venv

# To activate it on Windows:
venv\Scripts\activate

# To activate it on Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Install all the required Python libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables ⚠️
The application uses various LLM providers (Groq, DeepSeek, Google) and web search APIs (Tavily).

1. Acquire your API keys from the respective providers (e.g., [Groq Console](https://console.groq.com/), DeepSeek, Google AI Studio, Tavily).
2. In the root directory of this project, create a new file and name it exactly `.env`
3. Open the `.env` file and add your API keys (and, for deployments, admin
   credentials) in the following format:
```env
GROQ_API_KEY=your_groq_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Admin credentials (recommended for any non-local deployment).
# If HA_ADMIN_PASSWORD is not set, the app falls back to the insecure
# default "1234" and emits a warning in the server log.
HA_ADMIN_USERNAME=admin
HA_ADMIN_PASSWORD=change-me-before-deploying

# Or use hashed mode (recommended):
# HA_ADMIN_PASSWORD_HASH=<generated-hex-hash>
# HA_ADMIN_PASSWORD_SALT=<generated-hex-salt>

# Optional: LangSmith tracing
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=lsv2_...
# LANGSMITH_PROJECT=AI-Herbalist-Assistant

# Optional: override the SQLite file location.
# Defaults to `.herbalist.db` in the repo root.
# HA_DB_PATH=/var/data/herbalist.db
```

To generate secure hash/salt values for admin password, run:
```bash
python scripts/generate_admin_password_hash.py
```
Then copy the printed `HA_ADMIN_PASSWORD_HASH` and `HA_ADMIN_PASSWORD_SALT`
into your `.env` (and remove `HA_ADMIN_PASSWORD`).

---

## 🗄️ Data persistence
All users, chat sessions, messages, sources and 👍 / 👎 feedback are
stored in a local SQLite database (`.herbalist.db` by default). The
schema is managed by SQLAlchemy and created automatically on first
startup. If you are upgrading from an older build that used
`.users.json` / `.chat_history.json`, those files are imported into
SQLite on the next run and then renamed to
`*.migrated-backup-<timestamp>` — keep them around until you've
verified everything, then delete at your leisure.

For production deployments, mount `.herbalist.db` on a persistent
volume so chats and users survive container restarts (see the
`Dockerfile` and `SECURITY.md`).

---

## ▶️ Running the Application
Once the setup is complete, you can start the Streamlit server:
```bash
streamlit run src/herbalist_assistant/ui/streamlit_app.py
```
(Note: Adjust the path above if your main execution file is named differently).

The application will open automatically in your default web browser at `http://localhost:8501`.

---

## 🧑🏻‍💻 Developers & Team
This project was developed as a demonstration of applying AI and RAG architectures in domain-specific applications by computer engineering students under the guidance of **Prof. Dr. Ramazan KATIRCI**:
* **Malik Fikret** (Tech Lead & AI Integration)
* **Ebru Tuğçe Polat** (UI/UX & System Design)
* **Melisa Yıldırım** (Data Pipeline & Testing)
