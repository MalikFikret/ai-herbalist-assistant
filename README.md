# 🌿 AI Herbalist Assistant

An AI-powered academic retrieval system (RAG) designed to provide accurate, context-based botanical and herbal information exclusively from provided medical and herbal texts.

## 🚀 Overview
This project utilizes a Retrieval-Augmented Generation (RAG) architecture to answer user queries about herbs, natural remedies, and common ailments. It strictly answers based on the local PDF documents provided in the system and is engineered to ensure a safe, non-diagnostic user experience.

---

## ⚙️ Prerequisites
Before you begin, ensure you have the following installed on your machine:
* **Python 3.9+**
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
The application uses the Groq API for its language model processing. **The application will fail to run without a valid API key.**

1. Go to the [Groq Console](https://console.groq.com/) and create a free account.
2. Generate a new API Key.
3. In the root directory of this project, create a new file and name it exactly `.env`
4. Open the `.env` file and add your API key in the following format:
```env
GROQ_API_KEY=your_api_key_here
```

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
