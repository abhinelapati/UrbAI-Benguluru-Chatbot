<<<<<<< HEAD
# UrbAI-Benguluru-Chatbot
=======
# UrbAI - Bengaluru Chatbot 🤖🌆

UrbAI is a multilingual, AI-powered smart city assistant designed for Bengaluru.  
It delivers **real-time, localized information** using LLMs and RAG architecture.

---

## 🔧 Features

- 🗣️ Voice-based multilingual query support
- 🚇 Metro status and schedule updates
- 🎉 Events discovery across Bengaluru
- 🌦️ Real-time weather info
- 🧠 Localized Retrieval-Augmented Generation (RAG)
- 📊 Feedback logging and performance evaluation (RAGAS)

---

## 🛠️ Tech Stack

- **LLM:** Phi-1.5 + LoRA fine-tuning  
- **Retriever:** FAISS + BM25 + Cross-encoder re-ranking  
- **Frontend:** Streamlit (with `@st.cache_resource` optimization)  
- **Deployment:** Docker + Google Cloud Run  
- **APIs:** Google Translate, Speech-to-Text, Weather API  

---

## 📂 Structure

```bash
.
├── events_app.py         # Streamlit app with all modules
├── events.db             # Event data SQLite database
├── requiremens.txt       # Required Python packages
├── .gitignore
└── README.md             # You're reading it :)
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/JakkiRajasekharRamana/UrbAI-Bangaluru-Chatbot.git
cd UrbAI-Bangaluru-Chatbot
pip install -r requiremens.txt
streamlit run events_app.py
```

---

## 📌 Future Scope

- Live integration with BMTC bus tracking
- Integration with Namma Yatri or Metro card APIs
- Local business discovery with reviews
- Offline support for low-data environments

---

## 📬 Contact

Created by [Jakki Rajasekhar Ramana](https://github.com/JakkiRajasekharRamana)  
For any queries or collaboration ideas, feel free to open an issue or connect!

---

> *Empowering cities through AI.* 🌐
>>>>>>> 948c4b4 (Initial commit of my project)
