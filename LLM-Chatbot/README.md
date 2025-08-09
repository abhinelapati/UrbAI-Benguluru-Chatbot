
# 🏙️ UrbAI – Bengaluru Bot 🤖

**UrbAI-Bengaluru Bot** is a multilingual, AI-powered smart city assistant designed to serve real-time public service information for Bengaluru. The chatbot integrates cutting-edge LLM technologies, Retrieval-Augmented Generation (RAG), and voice/multilingual capabilities to deliver metro/bus schedules, weather updates, events, and more.

---

## 🔥 Features

- 🗺️ **Real-Time Public Transport**: Provides metro and BMTC bus route info, schedules, and fare details.
- 🌦️ **Weather Module**: Real-time weather updates based on the user’s location using OpenWeatherMap API.
- 🎉 **Events Module**: Lists upcoming city events (cultural, tech, entertainment) scraped from public sources.
- 🗣️ **Voice Input Support**: Users can speak their queries (via mic input), transcribed using Google Speech-to-Text API.
- 🌐 **Multilingual Support**: Automatically detects and responds in English or Kannada using Google Translate API.
- 📚 **RAG-Based QA**: Uses FAISS + BM25 and cross-encoder reranking to answer FAQs and knowledge-grounded queries.
- ⚙️ **Optimized Deployment**: Streamlit-based UI deployed using Docker on Google Cloud Run.
- 📉 **Performance Monitoring**: RAGAS-based answer evaluation, latency tracking, and user feedback logging.

---

## 🧠 Tech Stack

| Layer         | Tools & Frameworks |
|---------------|--------------------|
| **LLM**       | Microsoft Phi-1.5 fine-tuned with QLoRA |
| **RAG**       | FAISS, BM25, Cross-Encoder Re-Ranking |
| **Frontend**  | Streamlit (with @st.cache_resource), Voice Input |
| **Backend**   | Python, Flask APIs |
| **Cloud**     | Google Cloud (Cloud Run, Translate API, Speech-to-Text) |
| **Optimization** | 4-bit Quantization, `merge_and_unload()` |
| **Evaluation**| RAGAS (faithfulness, relevance), feedback logging |
| **Deployment**| Docker, Gunicorn, Cloud Run |

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/urbai-bengaluru-bot.git
cd urbai-bengaluru-bot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file with the following keys:

```env
GCP_PROJECT_ID=your-gcp-project
GCP_SPEECH_KEY=your-speech-api-key
GCP_TRANSLATE_KEY=your-translate-api-key
OPENWEATHER_API_KEY=your-weather-api-key
```

### 4. Run Locally

```bash
streamlit run app.py
```

### 5. Docker Build & Deploy (Cloud Run)

```bash
docker build -t urbai-bot .
docker run -p 8501:8501 urbai-bot
```

For GCP Cloud Run deployment, refer to `gcloud` CLI deployment steps.

---

## 🧪 Evaluation

- **LLM**: Fine-tuned on Bengaluru FAQs and metro info using LoRA.
- **Quantized**: 4-bit model for faster response.
- **RAG Evaluation**: Uses `ragas.evaluate()` to score:
  - Faithfulness
  - Answer correctness
  - Latency
- **User Feedback**: Thumbs up/down tracking for answer quality.

---

## 📁 Project Structure

```plaintext
urbai-bengaluru-bot/
│
├── app.py                  # Streamlit app interface
├── rag_engine.py           # RAG setup with FAISS + re-ranking
├── voice_input.py          # Speech-to-text handling
├── translate.py            # Multilingual translation
├── metro_module.py         # Metro + Bus planner
├── weather_module.py       # Weather info fetcher
├── events_module.py        # Event scraper & formatter
├── model_utils.py          # LoRA loading & 4-bit merging
├── feedback_logger.py      # Feedback capture and export
├── Dockerfile              # For containerizing the app
└── requirements.txt        # Python dependencies
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change. Contributions in extending to other cities (Delhi, Mumbai, etc.) or improving re-ranking logic are especially appreciated.

---

## 📜 License

MIT License. See `LICENSE` file for more info.

---

## ✨ Credits

Built with ❤️ by [Your Name], integrating Google Cloud, HuggingFace, and open city APIs to empower smart living in Bengaluru.

---

## 🌐 Demo

🚀 Hosted demo: **[Link available upon request or add your link here]**
