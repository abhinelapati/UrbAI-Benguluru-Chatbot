import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import streamlit as st
import speech_recognition as sr
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rank_bm25 import BM25Okapi
import nltk
import os
import logging
from fuzzywuzzy import fuzz
import re
from commute_recommendations import calculate_commute_recommendations
from datetime import datetime
import requests

# Set up logging
logging.basicConfig(filename='error_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure NLTK resources are downloaded
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
try:
    nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
    nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)
except Exception as e:
    logging.error(f"Failed to download NLTK resources: {str(e)}")
    st.error("Failed to download required NLTK packages (punkt and punkt_tab). Please ensure you have an internet connection and try running: python -m nltk.downloader punkt punkt_tab -d C:\\Users\\DELL\\nltk_data in your terminal.")
    st.stop()

# Verify tokenization works
try:
    nltk.word_tokenize("Test sentence.")
except Exception as e:
    logging.error(f"Tokenization test failed: {str(e)}")
    st.error("NLTK tokenization failed. Please ensure punkt and punkt_tab are downloaded correctly. Check error_log.txt for details.")
    st.stop()

# Enable wide mode for full landscape utilization
st.set_page_config(layout="wide")

# Function to load Torch
def load_torch():
    import torch
    return torch

# Load Torch
try:
    torch = load_torch()
    torch.set_num_threads(8)  # Optimize for CPU
except Exception as e:
    logging.error(f"Failed to load Torch: {str(e)}")
    st.error("Failed to load PyTorch. Please ensure it is installed correctly.")
    st.stop()

# Load dataset
try:
    df = pd.read_csv("knowledge_base.csv")
except FileNotFoundError:
    st.error("knowledge_base.csv not found. Please create it in the project folder.")
    st.stop()
except Exception as e:
    logging.error(f"Error loading knowledge_base.csv: {str(e)}")
    st.error("Error loading dataset. Check error_log.txt for details.")
    st.stop()

# Segment documents into sentences for better granularity
texts = []
metadata = []
locations = []
for idx, row in df.iterrows():
    sentences = row['answer'].split(". ")
    for sentence in sentences:
        if sentence.strip():
            texts.append(sentence.strip() + ".")
            metadata.append(row['type'])
            locations.append(row.get('location', ''))

# Load real-time data
try:
    with open("realtime_data.json", "r") as f:
        realtime_data = json.load(f)
except FileNotFoundError:
    st.warning("realtime_data.json not found. Real-time data integration will be skipped.")
    realtime_data = {}
except Exception as e:
    logging.error(f"Error loading realtime_data.json: {str(e)}")
    realtime_data = {}

# Initialize feedback file if it doesn't exist
if not os.path.exists("feedback.json"):
    with open("feedback.json", "w") as f:
        json.dump([], f)

# Initialize retrieval feedback file
if not os.path.exists("retrieval_feedback.json"):
    with open("retrieval_feedback.json", "w") as f:
        json.dump({}, f)

# Load retrieval feedback
try:
    with open("retrieval_feedback.json", "r") as f:
        retrieval_feedback = json.load(f)
except Exception as e:
    logging.error(f"Error loading retrieval_feedback.json: {str(e)}")
    retrieval_feedback = {}

# Set up BM25 for keyword search
try:
    tokenized_texts = [nltk.word_tokenize(text.lower()) for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
except Exception as e:
    logging.error(f"Error setting up BM25: {str(e)}")
    st.error("Error initializing keyword search. Check error_log.txt for details.")
    st.stop()

# Create embeddings with a lightweight model
try:
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    embeddings_np = embeddings.cpu().numpy()
except Exception as e:
    logging.error(f"Error creating embeddings: {str(e)}")
    st.error("Error creating embeddings. Check error_log.txt for details.")
    st.stop()

# Set up FAISS index
try:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
except Exception as e:
    logging.error(f"Error setting up FAISS index: {str(e)}")
    st.error("Error initializing FAISS index. Check error_log.txt for details.")
    st.stop()

# Load cross-encoder for re-ranking
try:
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
except Exception as e:
    logging.error(f"Error loading cross-encoder: {str(e)}")
    st.error("Error loading cross-encoder. Check error_log.txt for details.")
    st.stop()

# Load the language model
model_name = "google/flan-t5-base"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cpu")
except Exception as e:
    logging.error(f"Error loading language model: {str(e)}")
    st.error("Error loading language model. Check error_log.txt for details.")
    st.stop()

# Domain-specific query expansion with negation handling
def expand_query(query, query_type):
    query_lower = query.lower()
    expansion_dict = {
        "traffic": ["congestion", "road conditions", "speed"],
        "events": ["festival", "market", "concert"],
        "services": ["report", "apply", "permit", "bbmp", "bwssb"],
        "transit": ["subway", "bus", "delay", "bmtc", "metro"],
        "weather": ["temperature", "forecast", "conditions"],
        "general": ["tourist", "food", "history", "attractions"]
    }
    # Detect negation
    negation_words = ["not", "don't", "doesn't", "avoid"]
    negations = [word for word in negation_words if word in query_lower]
    tokens = nltk.word_tokenize(query_lower)
    expanded_tokens = set(tokens)
    excluded_tokens = set()
    
    # Handle negations
    if negations:
        for token in tokens:
            if any(token.startswith(neg) for neg in negations):
                continue
            idx = tokens.index(token)
            if idx > 0 and tokens[idx-1] in negations:
                excluded_tokens.add(token)
    
    # Expand based on query type
    if query_type.lower() in expansion_dict:
        expanded_tokens.update(expansion_dict[query_type.lower()])
    if "bangalore" not in query_lower:
        expanded_tokens.add("bangalore")
    if any(word in query_lower for word in ["food", "eat", "restaurant"]):
        expanded_tokens.update(["street food", "vidyarthi bhavan", "south indian"])
    if any(word in query_lower for word in ["tourist", "visit", "place"]):
        expanded_tokens.update(["lalbagh", "cubbon park"])
    if any(word in query_lower for word in ["hospital", "doctor"]):
        expanded_tokens.update(["manipal", "fortis"])
    if any(word in query_lower for word in ["college", "university"]):
        expanded_tokens.update(["rv college", "pes university"])
    
    return " ".join(expanded_tokens), excluded_tokens

# Query intent classification with multi-intent detection
def classify_intent(query, query_type):
    query_lower = query.lower()
    if query_type != "General":
        return [query_type.lower()]
    
    intents = []
    # Split query into parts based on conjunctions
    parts = re.split(r'\band\b|\bthen\b', query_lower)
    for part in parts:
        part = part.strip()
        if any(word in part for word in ["how do i", "where can i", "report", "apply", "pay", "bbmp", "bwssb", "police"]):
            intents.append("services")
        elif any(word in part for word in ["traffic", "road", "closure"]):
            intents.append("traffic")
        elif any(word in part for word in ["event", "festival", "weekend", "karaga"]):
            intents.append("events")
        elif any(word in part for word in ["subway", "bus", "delay", "bmtc", "metro", "commute"]):
            intents.append("transit")
        elif "weather" in part:
            intents.append("weather")
        elif any(word in part for word in ["food", "restaurant", "dosa", "eat"]):
            intents.append("general_food")
        elif any(word in part for word in ["hospital", "doctor"]):
            intents.append("general_hospital")
        else:
            intents.append("general")
    
    return intents if intents else ["general"]

# Function to normalize text for matching
def normalize_text(text):
    text = text.lower()
    replacements = {
        "st": "street",
        "ave": "avenue",
        "rd": "road",
        "blvd": "boulevard"
    }
    tokens = nltk.word_tokenize(text)
    normalized_tokens = [replacements.get(token, token) for token in tokens]
    return " ".join(normalized_tokens)

# Function to retrieve real-time data with improved matching
def get_realtime_data(query_type, query):
    query_type_lower = query_type.lower()
    if query_type_lower not in realtime_data:
        logging.info(f"No real-time data for query type: {query_type_lower}")
        return None
    
    data = realtime_data.get(query_type_lower, {})
    normalized_query = normalize_text(query)
    logging.info(f"Query: {query}, Normalized Query: {normalized_query}, Query Type: {query_type_lower}")
    
    for key, value in data.items():
        normalized_key = normalize_text(key)
        logging.info(f"Checking key: {key}, Normalized Key: {normalized_key}")
        similarity = fuzz.partial_ratio(normalized_key, normalized_query)
        if similarity >= 80:
            logging.info(f"Match found! Key: {key}, Similarity: {similarity}")
            if query_type_lower == "traffic":
                return f"Real-time update: Traffic on {key} is {value['status']} with an average speed of {value['speed']} (last updated: {value['last_updated']})."
            elif query_type_lower == "events":
                return f"Real-time update: {value['event']} at {key} on {value['date']} at {value['time']}."
            elif query_type_lower == "transit":
                return f"Real-time update: {key} is {value['status']} with a delay of {value['delay']} (last updated: {value['last_updated']})."
            elif query_type_lower == "weather":
                return f"Real-time update: The weather in Bangalore is {value['current']} with a temperature of {value['temperature']} (last updated: {value['last_updated']})."
    
    logging.info("No match found in real-time data.")
    return None

# RAG
retrieval_cache = {}
MAX_CACHE_SIZE = 100
def retrieve_context(query, query_type, filtered_texts, filtered_metadata, filtered_locations, filtered_index, tokenized_filtered_texts, excluded_tokens, k_initial=10, k_final=3):
    cache_key = (query, query_type, tuple(filtered_texts))
    if cache_key in retrieval_cache:
        return retrieval_cache[cache_key]

    try:
        expanded_query, _ = expand_query(query, query_type)
        bm25_filtered = BM25Okapi(tokenized_filtered_texts)
        tokenized_query = nltk.word_tokenize(expanded_query.lower())
        bm25_scores = bm25_filtered.get_scores(tokenized_query)
        
        query_embedding = embed_model.encode([expanded_query], convert_to_tensor=True)
        query_embedding = query_embedding.cpu().numpy()
        D, I = filtered_index.search(query_embedding, k_initial)
        
        retrieved_docs = []
        for idx in range(len(filtered_texts)):
            # Skip documents containing excluded tokens
            if any(token in filtered_texts[idx].lower() for token in excluded_tokens):
                continue
            faiss_rank = np.where(I[0] == idx)[0]
            faiss_score = D[0][faiss_rank[0]] if len(faiss_rank) > 0 else float('inf')
            bm_score = bm25_scores[idx]
            combined_score = (1 / (faiss_score + 1e-5) if faiss_score != float('inf') else 0) * 0.5 + bm_score * 0.5
            if combined_score > 0:
                retrieved_docs.append((filtered_texts[idx], filtered_metadata[idx], filtered_locations[idx], combined_score, idx))
        
        if not retrieved_docs:
            logging.info(f"No relevant documents found for query: {query}")
            return "No relevant documents found."
        
        pairs = [(expanded_query, doc[0]) for doc in retrieved_docs]
        scores = cross_encoder.predict(pairs)
        final_scores = []
        for idx, (doc, doc_type, location, combined_score, doc_idx) in enumerate(retrieved_docs):
            cross_score = scores[idx]
            metadata_boost = 2.0 if (query_type.lower() == doc_type and query_type != "General") else 1.0
            # Location boost: if query mentions a location (e.g., "MG Road") and document has matching location
            location_boost = 1.5 if location and location.lower() in query.lower() else 1.0
            # Feedback boost: adjust score based on user feedback
            feedback_key = f"doc_{doc_idx}"
            feedback_boost = retrieval_feedback.get(feedback_key, {"positive": 0, "negative": 0})
            feedback_score = 1.0 + (feedback_boost["positive"] * 0.1) - (feedback_boost["negative"] * 0.1)
            final_score = cross_score * 0.7 + combined_score * 0.3 * metadata_boost * location_boost * feedback_score
            final_scores.append((doc, final_score, doc_type, doc_idx))
        
        final_scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = final_scores[:k_final]
        confidence = top_docs[0][1] if top_docs else 0.0
        if confidence < 0.3:
            logging.info(f"Low confidence for query: {query}, confidence: {confidence}")
            return "Low confidence in retrieved documents. Please rephrase your query or provide more details."
        
        summarized_context = []
        retrieved_doc_indices = []
        for doc, score, _, doc_idx in top_docs:
            summarized_context.append(doc)
            retrieved_doc_indices.append(doc_idx)
        
        context = " ".join(summarized_context)
        if len(context) > 1000:
            context = context[:1000] + "..."
        
        # Log retrieved documents for evaluation
        logging.info(f"Query: {query}, Retrieved Docs: {[texts[idx] for idx in retrieved_doc_indices]}, Scores: {[score for _, score, _, _ in top_docs]}")
        
        if len(retrieval_cache) >= MAX_CACHE_SIZE:
            retrieval_cache.pop(next(iter(retrieval_cache)))
        retrieval_cache[cache_key] = (context, retrieved_doc_indices)
        return context, retrieved_doc_indices
    except Exception as e:
        logging.error(f"Error in retrieval: {str(e)}")
        return f"Error in retrieval: {str(e)}", []

# Function to generate an answer with improved prompt
def generate_answer(query, context, query_type, intent_type):
    realtime_info = get_realtime_data(intent_type, query)
    additional_context = f"\nReal-time data: {realtime_info}" if realtime_info else ""
    
    if isinstance(context, tuple):
        context, retrieved_doc_indices = context
    else:
        retrieved_doc_indices = []
    
    if "No relevant" in context or "Low confidence" in context or "Error" in context:
        instruction = "I don't have sufficient information to answer that accurately. Please provide more details or try a different question."
        prompt = f"""
        You are a helpful assistant for smart city services in Bangalore. {instruction}

        Question: {query}

        Answer: {instruction}
        """
    else:
        instruction = f"Analyze the context and any real-time data to provide an accurate, concise answer about Bangalore. Reason step-by-step to ensure clarity. If real-time data conflicts with static data, prioritize the real-time data. Use bullet points for lists (e.g., top attractions, steps to file a complaint). If the context is insufficient, state: 'I don't have enough information to answer that accurately.' If unsure, explain your reasoning and provide the best possible answer."
        if intent_type != "general":
            instruction += f" Focus on {intent_type} aspects of Bangalore city services."
        prompt = f"""
        You are a professional assistant for smart city services in Bangalore. {instruction}

        Context: {context}{additional_context}

        Question: {query}

        Query Type: {query_type}

        Answer:
        """
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cpu")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=200, num_beams=5)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer if answer.strip() else "I don't have enough information to answer that.", retrieved_doc_indices
    except Exception as e:
        logging.error(f"Error generating answer: {str(e)}")
        return f"Error generating answer: {str(e)}", retrieved_doc_indices

# Function for voice input
def transcribe_audio():
    placeholder = st.empty()
    placeholder.markdown('<div class="listening">🎙 Listening...</div>', unsafe_allow_html=True)
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = r.listen(source, timeout=5)
            text = r.recognize_google(audio)
            return text
    except sr.WaitTimeoutError:
        return "No audio detected. Please try again."
    except sr.UnknownValueError:
        return "Could not understand audio."
    except Exception as e:
        logging.error(f"Voice input error: {str(e)}")
        return f"Voice input error: {str(e)}"
    finally:
        placeholder.empty()

# Custom CSS
st.markdown("""
<style>
/* Base Theme */
.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #e0e0e0;
    font-family: 'Poppins', sans-serif;
}

/* Header Styling */
.header {
    background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url('https://images.unsplash.com/photo-1596176530529-78163a4f7af2?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80');
    background-size: cover;
    background-position: center;
    padding: 30px;
    border-radius: 15px;
    color: #ffffff;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
}

.header h1 {
    font-size: 2.5em;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    background: linear-gradient(120deg, #e6b17e, #ffd700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

.header p {
    font-size: 1.1em;
    margin: 10px 0;
    color: #ffd700;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
}

/* Top Navigation Bar */
.top-nav {
    background: rgba(22, 33, 62, 0.8);
    padding: 15px 25px;
    border-radius: 12px;
    margin-bottom: 20px;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

/* Chat Container */
.chat-container {
    max-height: 600px;
    overflow-y: auto;
    padding: 20px;
    border-radius: 15px;
    background: rgba(26, 26, 46, 0.9);
    margin-bottom: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.user-message {
    background: linear-gradient(135deg, #4a90e2, #357abd);
    color: #ffffff;
    padding: 12px 18px;
    margin: 10px 0;
    border-radius: 20px 20px 5px 20px;
    max-width: 70%;
    margin-left: 30%;
    text-align: right;
    animation: slideInRight 0.3s ease-out;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
}

.bot-message {
    background: linear-gradient(135deg, #e6b17e, #d4af37);
    color: #1a1a2e;
    padding: 12px 18px;
    margin: 10px 0;
    border-radius: 20px 20px 20px 5px;
    max-width: 70%;
    margin-right: 30%;
    animation: slideInLeft 0.3s ease-out;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
}

@keyframes slideInRight {
    from { opacity: 0; transform: translateX(50px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-50px); }
    to { opacity: 1; transform: translateX(0); }
}

/* Input Area */
.input-container {
    background: rgba(22, 33, 62, 0.8);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.stTextInput > div > input {
    border-radius: 25px;
    padding: 12px 20px;
    border: 2px solid #e6b17e;
    background: rgba(26, 26, 46, 0.9);
    color: #ffffff;
    font-size: 1.1em;
    transition: all 0.3s ease;
}

.stTextInput > div > input:focus {
    border-color: #ffd700;
    box-shadow: 0 0 15px rgba(230, 177, 126, 0.3);
    transform: scale(1.01);
}

/* Buttons */
.stButton > button {
    border-radius: 25px;
    background: linear-gradient(135deg, #e6b17e, #d4af37);
    color: #1a1a2e;
    padding: 10px 20px;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(230, 177, 126, 0.4);
    background: linear-gradient(135deg, #d4af37, #e6b17e);
}

/* Commute Form */
.commute-form {
    background: rgba(22, 33, 62, 0.8);
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.commute-form h3 {
    color: #e6b17e;
    font-size: 1.4em;
    margin-bottom: 15px;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
}

/* Route Recommendations */
.route-card {
    background: rgba(26, 26, 46, 0.9);
    padding: 15px;
    border-radius: 12px;
    margin: 10px 0;
    border-left: 4px solid #e6b17e;
    transition: transform 0.3s ease;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

.route-card:hover {
    transform: translateX(5px);
}

.route-card h4 {
    color: #e6b17e;
    font-size: 1.2em;
    margin-bottom: 10px;
}

/* Feedback Buttons */
.feedback-button {
    background: rgba(26, 26, 46, 0.9);
    color: #e6b17e;
    border: 1px solid #e6b17e;
    border-radius: 20px;
    padding: 5px 12px;
    margin: 0 5px;
    transition: all 0.3s ease;
}

.feedback-button:hover {
    background: #e6b17e;
    color: #1a1a2e;
}

/* Listening Animation */
.listening {
    color: #e6b17e;
    font-size: 1.2em;
    padding: 10px;
    border-radius: 10px;
    background: rgba(26, 26, 46, 0.9);
    animation: pulseGlow 1.5s infinite;
}

@keyframes pulseGlow {
    0% { box-shadow: 0 0 5px #e6b17e; }
    50% { box-shadow: 0 0 20px #e6b17e; }
    100% { box-shadow: 0 0 5px #e6b17e; }
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(26, 26, 46, 0.9);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: #e6b17e;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #d4af37;
}

/* Select Box Styling */
.stSelectbox > div > div {
    background: rgba(26, 26, 46, 0.9);
    border: 2px solid #e6b17e;
    border-radius: 12px;
    color: #ffffff;
}

.stSelectbox > div > div:hover {
    border-color: #ffd700;
}

/* Message Timestamp */
.message-timestamp {
    font-size: 0.8em;
    opacity: 0.8;
    margin-top: 5px;
    font-style: italic;
}

/* Bangalore-specific Elements */
.header::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 5px;
    background: linear-gradient(90deg, #e6b17e, #ffd700, #e6b17e);
    border-radius: 15px 15px 0 0;
}

</style>
""", unsafe_allow_html=True)

# Header with Bangalore-specific content
st.markdown("""
    <div class="header">
        <h1>ನಮಸ್ಕಾರ Bengaluru!</h1>
        <p>Your Smart City Guide to the Silicon Valley of India</p>
        <p style="font-size: 0.9em; color: #e6b17e;">Connecting Citizens • Empowering Communities • Building Tomorrow</p>
    </div>
""", unsafe_allow_html=True)

# Top Navigation Bar
with st.container():
    st.markdown('<div class="top-nav">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown('<div class="nav-left">', unsafe_allow_html=True)
        query_type = st.selectbox("Query Type", ("General", "Traffic", "Events", "Services", "Transit", "Weather"), label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="nav-right" style="display: flex; align-items: center; justify-content: flex-end;">', unsafe_allow_html=True)
        col3, col4, col5, col6 = st.columns([1, 1, 1, 2])
        with col3:
            if st.button("🗑 Clear", help="Clear the current chat history"):
                logging.info("Clear History button clicked.")
                if st.session_state.chat_history:
                    st.session_state.chat_history = []
                    st.success("Chat history cleared!")
                    st.rerun()
                else:
                    st.info("No chat history to clear.")
        with col4:
            if st.button("💾 Save", help="Save the current chat history to a file"):
                logging.info("Save History button clicked.")
                if not st.session_state.chat_history:
                    st.warning("No chat history to save.")
                else:
                    try:
                        if not os.access(os.getcwd(), os.W_OK):
                            raise PermissionError("No write permission in the current directory.")
                        with open("chat_history.json", "w") as f:
                            json.dump(st.session_state.chat_history, f)
                        st.success("Chat history saved to chat_history.json!")
                        logging.info("Chat history saved successfully.")
                    except Exception as e:
                        logging.error(f"Error saving chat history: {str(e)}")
                        st.error("Error saving chat history. Check error_log.txt for details.")
        with col5:
            if st.button("📂 Load", help="Load chat history from a file"):
                logging.info("Load History button clicked.")
                try:
                    if os.path.exists("chat_history.json"):
                        with open("chat_history.json", "r") as f:
                            loaded_history = json.load(f)
                        if isinstance(loaded_history, list) and all(len(item) == 3 for item in loaded_history):
                            MAX_HISTORY_SIZE = 50
                            st.session_state.chat_history = loaded_history[-MAX_HISTORY_SIZE:]
                            st.success("Chat history loaded from chat_history.json!")
                            logging.info("Chat history loaded successfully.")
                            st.rerun()
                        else:
                            st.error("Invalid chat history format in chat_history.json.")
                            logging.error("Invalid chat history format in chat_history.json.")
                    else:
                        st.warning("No saved chat history found.")
                        logging.info("No chat_history.json file found.")
                except Exception as e:
                    logging.error(f"Error loading chat history: {str(e)}")
                    st.error("Error loading chat history. Check error_log.txt for details.")
        with col6:
            st.markdown('''<a href="https://astute-strategy-463309-f9.wm.r.appspot.com/" target="_blank" style="text-decoration:none;display:inline-block;">\
                <button style="border-radius:30px;background:linear-gradient(135deg,#ffd700,#e6b17e);color:#1a1a2e;padding:16px 36px;font-size:1.1em;font-weight:700;border:none;box-shadow:0 2px 8px rgba(230,177,126,0.25);transition:all 0.3s;text-transform:uppercase;letter-spacing:1px;cursor:pointer;">\
                🚗 Commute\
                </button>\
            </a>''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Filter dataset based on query type
try:
    if query_type != "General":
        filtered_indices = [i for i, m in enumerate(metadata) if m == query_type.lower()]
        if not filtered_indices:
            st.warning(f"No data available for {query_type}.")
            filtered_texts = texts
            filtered_metadata = metadata
            filtered_locations = locations
            filtered_index = index
            tokenized_filtered_texts = tokenized_texts
        else:
            filtered_texts = [texts[i] for i in filtered_indices]
            filtered_metadata = [metadata[i] for i in filtered_indices]
            filtered_locations = [locations[i] for i in filtered_indices]
            filtered_embeddings = embed_model.encode(filtered_texts, convert_to_tensor=True)
            filtered_embeddings_np = filtered_embeddings.cpu().numpy()
            filtered_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
            filtered_index.add(filtered_embeddings_np)
            tokenized_filtered_texts = [tokenized_texts[i] for i in filtered_indices]
    else:
        filtered_texts = texts
        filtered_metadata = metadata
        filtered_locations = locations
        filtered_index = index
        tokenized_filtered_texts = tokenized_texts
except Exception as e:
    logging.error(f"Error filtering dataset: {str(e)}")
    st.error("Error filtering dataset. Check error_log.txt for details.")
    st.stop()

# Initialize chat history and feedback
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'feedback' not in st.session_state:
    try:
        with open("feedback.json", "r") as f:
            st.session_state.feedback = json.load(f)
    except Exception as e:
        logging.error(f"Error loading feedback.json: {str(e)}")
        st.session_state.feedback = []

# Main Area
# Input Section
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input("Ask a question:", key="text_input", placeholder="Type your question here...", label_visibility="collapsed")
    with col2:
        if st.button("🎙 Speak"):
            user_input = transcribe_audio()
            if not user_input.startswith("Error") and user_input != "Could not understand audio." and user_input != "No audio detected. Please try again.":
                st.write("You said:", user_input)
            else:
                st.error(user_input)
    st.markdown('</div>', unsafe_allow_html=True)

# Process user input
if user_input and user_input.strip():
    try:
        intents = classify_intent(user_input, query_type)
        answers = []
        doc_indices = []
        
        for intent in intents:
            expanded_query, excluded_tokens = expand_query(user_input, intent)
            context, retrieved_doc_indices = retrieve_context(user_input, intent, filtered_texts, filtered_metadata, filtered_locations, filtered_index, tokenized_filtered_texts, excluded_tokens)
            answer, intent_doc_indices = generate_answer(user_input, (context, retrieved_doc_indices), query_type, intent)
            answers.append(answer)
            doc_indices.extend(intent_doc_indices)
        
        # Combine answers for multi-intent queries
        if len(answers) > 1:
            combined_answer = "\n\n".join([f"*Regarding {intent.replace('general_', '')}:*\n{answer}" for intent, answer in zip(intents, answers)])
        else:
            combined_answer = answers[0]
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.chat_history.append((user_input, combined_answer, timestamp, doc_indices))
        st.write("*Answer:*", combined_answer)
    except Exception as e:
        logging.error(f"Error processing user input: {str(e)}")
        st.error("Error processing your query. Check error_log.txt for details.")
elif user_input and not user_input.strip():
    st.warning("Please enter a non-empty query.")

# Chat History
st.subheader("Chat History")
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    if st.session_state.chat_history:
        for query, response, timestamp, doc_indices in st.session_state.chat_history:
            st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong> {query}
                    <div class="message-timestamp">{timestamp}</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
                <div class="bot-message">
                    <strong>Bot:</strong> {response}
                    <div class="message-timestamp">{timestamp}</div>
                </div>
            """, unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("👍", key=f"like_{query}_{timestamp}", help="Like this response"):
                    st.session_state.feedback.append((query, response, "positive"))
                    # Update retrieval feedback
                    for doc_idx in doc_indices:
                        feedback_key = f"doc_{doc_idx}"
                        feedback_data = retrieval_feedback.get(feedback_key, {"positive": 0, "negative": 0})
                        feedback_data["positive"] += 1
                        retrieval_feedback[feedback_key] = feedback_data
                    try:
                        with open("feedback.json", "w") as f:
                            json.dump(st.session_state.feedback, f)
                        with open("retrieval_feedback.json", "w") as f:
                            json.dump(retrieval_feedback, f)
                        st.write("Thanks for your feedback!")
                    except Exception as e:
                        logging.error(f"Error saving feedback: {str(e)}")
                        st.error("Error saving feedback. Check error_log.txt for details.")
            with col2:
                if st.button("👎", key=f"dislike_{query}_{timestamp}", help="Dislike this response"):
                    st.session_state.feedback.append((query, response, "negative"))
                    # Update retrieval feedback
                    for doc_idx in doc_indices:
                        feedback_key = f"doc_{doc_idx}"
                        feedback_data = retrieval_feedback.get(feedback_key, {"positive": 0, "negative": 0})
                        feedback_data["negative"] += 1
                        retrieval_feedback[feedback_key] = feedback_data
                    try:
                        with open("feedback.json", "w") as f:
                            json.dump(st.session_state.feedback, f)
                        with open("retrieval_feedback.json", "w") as f:
                            json.dump(retrieval_feedback, f)
                        st.write("Sorry, I'll try to improve!")
                    except Exception as e:
                        logging.error(f"Error saving feedback: {str(e)}")
                        st.error("Error saving feedback. Check error_log.txt for details.")
            with col3:
                if st.button("Copy", key=f"copy_{query}_{timestamp}", help="Copy this response"):
                    st.write("Response copied!")
    else:
        st.markdown('<p style="color: #a0a0a0;">No chat history yet.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
#Loading events 
def load_events():
    try:
        with open("events.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("events.json not found.")
        return []

GOOGLE_CALENDAR_API_KEY = "Your-Api-key"
# User-provided public calendar for Karnataka events
KARNATAKA_CALENDAR_ID = "a1c104988594d2f675c572bb4fde0f6f6a6ba14db661be1ca76464396be000c5@group.calendar.google.com"

# Function to fetch Karnataka events from Google Calendar
from datetime import datetime, timedelta

def fetch_karnataka_events():
    now = datetime.utcnow().isoformat() + 'Z'
    url = (
        f"https://www.googleapis.com/calendar/v3/calendars/{KARNATAKA_CALENDAR_ID}/events"
        f"?key={GOOGLE_CALENDAR_API_KEY}"
        f"&timeMin={now}"
        f"&singleEvents=true"
        f"&orderBy=startTime"
        f"&maxResults=20"
    )
    try:
        response = requests.get(url)
        if response.status_code == 401:
            st.error("Google Calendar API returned 401 Unauthorized.\n"
                     "- Make sure your API key is enabled for the Calendar API.\n"
                     "- Make sure the calendar is public.\n"
                     "- Double-check the calendar ID.")
            return []
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            st.error(f"Google Calendar API error: {data['error'].get('message', 'Unknown error')}")
            return []
        return data.get('items', [])
    except Exception as e:
        st.error(f"Could not fetch events from Google Calendar. Error: {e}")
        return []

# Update show_events to use Karnataka events

def show_events():
    st.subheader("Karnataka Events Calendar (Live from Google Calendar)")
    events = fetch_karnataka_events()
    today = datetime.now().date()
    upcoming_events = []
    for event in events:
        # Parse event start date
        start = event.get('start', {}).get('dateTime') or event.get('start', {}).get('date')
        if start:
            try:
                event_date = datetime.fromisoformat(start.replace('Z', '+00:00')).date() if 'T' in start else datetime.strptime(start, "%Y-%m-%d").date()
                if event_date >= today:
                    upcoming_events.append((event, event_date))
            except Exception as e:
                continue
    if not upcoming_events:
        st.write("No upcoming events found.")
    for event, event_date in sorted(upcoming_events, key=lambda x: x[1]):
        st.write(f"**{event.get('summary', 'No Title')}** - {event_date}")
        location = event.get('location', 'No location specified')
        st.write(f"Location: {location}")
        description = event.get('description', 'No description provided')
        st.write(description)
        st.write("---")

# Existing app setup (e.g., st.title, other features) remains unchanged
st.sidebar.title("Options")
option = st.sidebar.selectbox("Choose a feature", ["Q&A", "Commute", "Events"])  # Adjust options based on your existing features

if option == "Events":
    show_events()
# Existing conditionals for other features (e.g., Q&A, Commute) remain below
    
#     # Extract unique categories
#     categories = sorted(set(event.get('category', 'Uncategorized') for event in upcoming_events))
    
#     # Add a dropdown for categories
#     selected_category = st.selectbox("Filter by Category", ["All"] + categories)
    
#     # Filter events based on selected category
#     if selected_category != "All":
#         filtered_events = [event for event in upcoming_events if event.get('category') == selected_category]
#     else:
#         filtered_events = upcoming_events
    
#     st.subheader(f"Upcoming Events ({selected_category})")
#     for event in filtered_events:
#         st.write(f"{event['name']}** - {event['date']}")
#         st.write(f"Location: {event['location']}")
#         st.write(event['description'])
#         st.write("---")

# # App setup
# st.sidebar.title("Options")
# option = st.sidebar.selectbox("Choose a feature", ["Q&A", "Commute", "Events"])

# if option == "Events":
#     st.write("Browse upcoming events in Bangalore!")
#     show_events()

## API & DATASET VERSTION ##
# # Initialize the database
# events_app.init_db()

# # Initialize session state for conversation
# if "conversation_step" not in st.session_state:
#     st.session_state.conversation_step = None
# if "event_data" not in st.session_state:
#     st.session_state.event_data = {}

# # Chat input
# user_input = st.text_input("Ask a question or type a command:", key="text_input")
# if st.button("Send"):
#     if st.session_state.conversation_step is None:
#         if user_input == "/events":
#             events = events_app.get_upcoming_events()
#             if events:
#                 for event in events:
#                     st.write(f"{event['name']}** - {event['date']}")
#                     st.write(f"Location: {event['location']}")
#                     st.write(event['description'])
#                     st.write("---")
#             else:
#                 st.write("No upcoming events found.")
#         elif user_input == "/add_event":
#             st.session_state.conversation_step = "name"
#             st.write("Please provide the event name.")
#         else:
#             st.write("Command not recognized. Try /events or /add_event.")
#     elif st.session_state.conversation_step == "name":
#         st.session_state.event_data["name"] = user_input
#         st.session_state.conversation_step = "date"
#         st.write("Please provide the event date (YYYY-MM-DD).")
#     elif st.session_state.conversation_step == "date":
#         st.session_state.event_data["date"] = user_input
#         st.session_state.conversation_step = "location"
#         st.write("Please provide the event location.")
#     elif st.session_state.conversation_step == "location":
#         st.session_state.event_data["location"] = user_input
#         st.session_state.conversation_step = "description"
#         st.write("Please provide a brief description of the event.")
#     elif st.session_state.conversation_step == "description":
#         st.session_state.event_data["description"] = user_input
#         events_app.add_event(**st.session_state.event_data)
#         st.write("Event added successfully!")
#         st.session_state.conversation_step = None
#         st.session_state.event_data = {}
