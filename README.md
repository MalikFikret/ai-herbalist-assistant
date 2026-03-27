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
git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
cd YourRepoName
2. Create a Virtual Environment (Recommended)
It is highly recommended to use a virtual environment to prevent library conflicts.

Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
3. Install Dependencies
Install all the required Python libraries using the requirements.txt file:

Bash
pip install -r requirements.txt
4. Setup Environment Variables ⚠️
The application uses the Groq API for its language model processing. The application will fail to run without a valid API key.

Go to the Groq Console and create a free account.

Generate a new API Key.

In the root directory of this project, create a new file and name it exactly .env

Open the .env file and add your API key in the following format:

Code snippet
GROQ_API_KEY=your_api_key_here
5. Add Reference Data
The AI requires source documents to function as a retrieval system.

Locate the data/ folder in the project directory.

Place your reference PDF files (e.g., The Green Pharmacy) inside this directory. The system will automatically index them upon startup or when triggered via the admin panel.

▶️ Running the Application
Once the setup is complete, you can start the Streamlit server:

Bash
streamlit run src/herbalist_assistant/ui/streamlit_app.py
(Note: Adjust the path above if your main execution file is named differently).

The application will open automatically in your default web browser at http://localhost:8501.

🛑 Troubleshooting
Error: chromadb.errors.InternalError: Database error (code: 14)
If you encounter this error when trying to re-index PDFs, it means the vector database file is currently locked by the active server process.
Solution: Stop the Streamlit server (Ctrl+C in the terminal), delete the chroma_db folder located in your project directory, and restart the server to allow the system to build a fresh database instance.

👥 Developers & Team
This project was developed as a demonstration of applying AI and RAG architectures in domain-specific applications by computer engineering students under the guidance of Professor Ramazan:

Malik Fikret (Tech Lead & AI Integration)

Ebru (UI/UX & System Design)

Melisa (Data Pipeline & Testing)