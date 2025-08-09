import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
from rank_bm25 import BM25Okapi
import nltk
import os
import logging
from fuzzywuzzy import fuzz
import re
import json

# --- Global Data and Feedback (Loaded directly or passed in) ---
# These variables will be loaded by core_llm_logic when imported
# or you can pass them as arguments to functions if you prefer.

# NLTK setup (ensure punkt is downloaded)
try:
    nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
    nltk.data.path.append(nltk_data_path) # Add to NLTK path
    
    # Check if 'punkt' is already downloaded. Download quietly if not.
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK 'punkt' tokenizer data...")
        nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
        print("NLTK 'punkt' download complete.")
    
except Exception as e:
    logging.error(f"Failed to set up NLTK resources in core_llm_logic: {str(e)}")
    # If NLTK setup fails, functions using it will fail.

# Load dataset (df_kb is not directly used by functions, but `texts`, `metadata`, `locations` are)
try:
    df_kb = pd.read_csv("knowledge_base.csv")
    texts = []
    metadata = []
    locations = []
    for idx, row in df_kb.iterrows():
        # Using .strip() to remove leading/trailing whitespace
        # Ensure sentences are well-formed in KB for splitting accuracy
        sentences = row['answer'].split(". ") 
        for sentence in sentences:
            if sentence.strip(): # Only add non-empty sentences
                texts.append(sentence.strip() + ".") # Add back the period
                metadata.append(row['type'].strip()) # Ensure no whitespace in type
                locations.append(row.get('location', '').strip()) # Get location and strip
except FileNotFoundError:
    logging.error("knowledge_base.csv not found in core_llm_logic. Make sure it's in the same directory.")
    texts, metadata, locations = [], [], [] # Initialize empty to avoid crashes
except Exception as e:
    logging.error(f"Error loading knowledge_base.csv in core_llm_logic: {str(e)}")
    texts, metadata, locations = [], [], []

# Load real-time data
try:
    with open("realtime_data.json", "r") as f:
        realtime_data = json.load(f)
except FileNotFoundError:
    logging.warning("realtime_data.json not found in core_llm_logic. Real-time data integration will be skipped.")
    realtime_data = {}
except Exception as e:
    logging.error(f"Error loading realtime_data.json in core_llm_logic: {str(e)}")
    realtime_data = {}

# Load retrieval feedback (used by RAG)
try:
    if os.path.exists("retrieval_feedback.json"):
        with open("retrieval_feedback.json", "r") as f:
            retrieval_feedback = json.load(f)
    else:
        retrieval_feedback = {}
except Exception as e:
    logging.error(f"Error loading retrieval_feedback.json in core_llm_logic: {str(e)}")
    retrieval_feedback = {}


# --- Model Loading Functions (WITHOUT @st.cache_resource) ---

def load_retrieval_models_core():
    """Loads all models needed for retrieval and reranking."""
    print("Loading retrieval models (SentenceTransformer, CrossEncoder)...")
    try:
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("Retrieval models loaded.")
        return embed_model, cross_encoder
    except Exception as e:
        print(f"Error: Failed to load retrieval models: {e}")
        logging.error(f"Failed to load retrieval models: {str(e)}")
        raise # Re-raise to stop execution

def load_llm_components_core():
    """Loads the fine-tuned LLM and tokenizer."""
    base_model_local_path = "models/phi-1_5" 
    adapter_path = "bangalore-phi-lora-adapter" 
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print(f"Loading base model from: {base_model_local_path} with 4-bit quantization...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_local_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
            local_files_only=True
        )
        print("Base model loaded.")

        print(f"Loading tokenizer from: {base_model_local_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_local_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token 
        print("Tokenizer loaded.")

        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("Fine-tuned model successfully loaded!")
        
        print("Attempting to merge adapter into base model for faster inference (if GPU available)...")
        try:
            model = model.merge_and_unload()
            print("Adapter merged successfully.")
        except Exception as merge_e:
            print(f"Warning: Failed to merge adapter: {merge_e}")
            logging.warning(f"Failed to merge adapter: {merge_e}")

        return model, tokenizer
    except FileNotFoundError as fnf_e:
        print(f"Error: Model file not found in core_llm_logic. Ensure paths are correct: {fnf_e}")
        logging.error(f"FileNotFoundError in core_llm_logic: {fnf_e}")
        raise # Re-raise to stop execution if models aren't found
    except Exception as e:
        print(f"Error: Critical error loading LLM components in core_llm_logic: {e}")
        logging.error(f"Fatal error loading LLM components in core_llm_logic: {str(e)}")
        raise # Re-raise to stop execution


# --- Pre-computation of RAG Assets ---
def precompute_filtered_rag_assets_core(_texts, _metadata, _locations, _embed_model):
    """
    Pre-computes FAISS & BM25 indexes for each query type from global data.
    """
    print("Pre-computing filtered RAG assets for each query type...")
    asset_dict = {}
    
    # Ensure all possible categories from your KB are included, plus standard ones.
    all_query_types_from_metadata = set(m for m in _metadata)
    predefined_types = {"General", "Traffic", "Events", "Services", "Transit", "Weather", "general_food", "general_tourist", "general_hospital", "general_education"}
    all_query_types = sorted(list(all_query_types_from_metadata.union(predefined_types))) 
    
    # Tokenize all texts once for efficiency (BM25 needs tokenized texts)
    tokenized_texts_full = [nltk.word_tokenize(text.lower()) for text in _texts]

    for q_type in all_query_types:
        if q_type == "General":
            filtered_indices = list(range(len(_texts)))
        else:
            type_lower = q_type.lower()
            filtered_indices = [i for i, m in enumerate(_metadata) if m == type_lower]
            if not filtered_indices:
                print(f"Warning: No data found for category '{q_type}' in knowledge_base.csv. Skipping pre-computation for this type.")
                continue 

        filtered_texts_subset = [_texts[i] for i in filtered_indices]
        
        embeddings = _embed_model.encode(filtered_texts_subset, convert_to_tensor=True, show_progress_bar=False).cpu().numpy()
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings)
        
        # Tokenize subset explicitly for BM25
        tokenized_filtered_texts_subset = [nltk.word_tokenize(text.lower()) for text in filtered_texts_subset] 
        bm25_index = BM25Okapi(tokenized_filtered_texts_subset)
        
        asset_dict[q_type] = {
            "texts": filtered_texts_subset,
            "metadata": [_metadata[i] for i in filtered_indices],
            "locations": [_locations[i] for i in filtered_indices],
            "faiss_index": faiss_index,
            "bm25_index": bm25_index,
            "doc_indices_map": filtered_indices # Map from local subset index back to global 'texts' index
        }
    print("All filtered RAG assets pre-computed.")
    return asset_dict

# --- Core RAG & LLM Functions ---

def handle_chitchat(query):
    query_lower = query.lower().strip()
    
    chitchat_greetings = ["hi", "hello", "hey", "yo", "good morning", "good afternoon", "good evening"]
    chitchat_thanks = ["thanks", "thank you", "thankyou", "thx", "appreciate it", "cheers"]
    chitchat_howareyou = ["how are you", "how are you doing", "how's it going", "how are u"]
    chitchat_about_bot = ["who are you", "what are you", "what can you do", "tell me about yourself", "who made you"]

    if query_lower in chitchat_greetings:
        return "Hello! How can I help you with information about Bangalore today?"
    if query_lower in chitchat_thanks:
        return "You're welcome! Is there anything else I can help with?"
    if query_lower in chitchat_howareyou:
        return "I'm just a bot, but I'm ready to help you! What would you like to know?"
    if any(phrase in query_lower for phrase in chitchat_about_bot):
        return "I am Bengaluru Bot, your smart city guide. I can answer questions about Bangalore, give recommendations, and provide real-time updates."
        
    return None

def expand_query(query, query_type):
    query_lower = query.lower()
    expansion_dict = {
        "traffic": ["congestion", "road conditions", "speed", "traffic jam", "commute time"],
        "events": ["festival", "market", "concert", "show", "exhibition", "event happening"],
        "services": ["report", "apply", "permit", "bbmp", "bwssb", "complaint", "pay bill", "tax", "license"],
        "transit": ["subway", "bus", "delay", "bmtc", "metro", "train", "route", "public transport"],
        "weather": ["temperature", "forecast", "conditions", "climate", "rain", "sunny", "humidity"],
        "general": ["tourist", "food", "history", "attractions", "places", "visit", "sightseeing"]
    }
    negation_words = ["not", "don't", "doesn't", "avoid", "exclude"]
    negations = [word for word in negation_words if word in query_lower]
    tokens = nltk.word_tokenize(query_lower)
    expanded_tokens = set(tokens)
    excluded_tokens = set()
    
    if negations:
        for i, token in enumerate(tokens):
            if token in negation_words:
                if i + 1 < len(tokens):
                    excluded_tokens.add(tokens[i+1])
                continue
            if i > 0 and tokens[i-1] in negation_words:
                excluded_tokens.add(token)

    if query_type.lower() in expansion_dict:
        expanded_tokens.update(expansion_dict[query_type.lower()])
    if "bangalore" not in query_lower:
        expanded_tokens.add("bangalore")
    
    if any(word in query_lower for word in ["food", "eat", "restaurant", "cuisine", "dishes", "menu", "dining", "cafe", "pub"]):
        expanded_tokens.update(["street food", "vidyarthi bhavan", "south indian", "north indian", "biryani", "dosa", "idli", "thali", "pizza", "burger", "coffee", "brewery"])
    if any(word in query_lower for word in ["tourist", "visit", "place", "attractions", "sightseeing", "explore"]):
        expanded_tokens.update(["lalbagh", "cubbon park", "bangalore palace", "tipu sultan's summer palace", "iskcon temple", "bull temple", "museum", "art gallery", "historical site"])
    if any(word in query_lower for word in ["hospital", "doctor", "medical", "clinic", "health", "emergency"]):
        expanded_tokens.update(["manipal", "fortis", "apollo", "narayana hrudayalaya", "hospital near me", "clinic opening hours"])
    if any(word in query_lower for word in ["college", "university", "education", "institute", "school", "study"]):
        expanded_tokens.update(["rv college", "pes university", "iisc", "christ university", "rvce", "vit", "jain university", "admission"])
    
    expanded_tokens = expanded_tokens - excluded_tokens # Apply exclusions after expansions
    
    return " ".join(expanded_tokens), excluded_tokens

def classify_intent(query, query_type):
    query_lower = query.lower()
    if query_type != "General":
        return [query_type.lower()]
    
    intents = []
    
    if any(word in query_lower for word in ["report", "apply", "pay", "bbmp", "bwssb", "police", "complaint", "service", "how do i", "register"]):
        intents.append("services")
    if any(word in query_lower for word in ["traffic", "road", "closure", "congestion", "jam", "commute time", "speed"]):
        intents.append("traffic")
    if any(word in query_lower for word in ["event", "festival", "weekend", "concert", "market", "happening", "karaga", "show", "exhibition", "fair"]):
        intents.append("events")
    if any(word in query_lower for word in ["subway", "bus", "delay", "bmtc", "metro", "train", "route", "public transport", "fare"]):
        intents.append("transit")
    if any(word in query_lower for word in ["weather", "temperature", "forecast", "climate", "rain", "sunny", "humidity", "hot", "cold"]):
        intents.append("weather")
    
    if any(word in query_lower for word in ["food", "eat", "restaurant", "cuisine", "dosa", "cafe", "pub", "dining", "breakfast", "lunch", "dinner"]):
        intents.append("general_food")
    if any(word in query_lower for word in ["what to do", "places to visit", "tourist", "attractions", "sightseeing", "explore", "museum", "park", "palace", "history", "landmarks"]):
        intents.append("general_tourist")
    if any(word in query_lower for word in ["hospital", "doctor", "medical", "clinic", "health", "emergency", "treatment"]):
        intents.append("general_hospital")
    if any(word in query_lower for word in ["college", "university", "education", "institute", "school", "study", "admission"]):
        intents.append("general_education")
    
    if not intents:
        intents.append("general")
    
    return intents

def normalize_text(text):
    text = text.lower()
    replacements = {
        "st": "street", "ave": "avenue", "rd": "road", "blvd": "boulevard"
    }
    tokens = nltk.word_tokenize(text)
    normalized_tokens = [replacements.get(token, token) for token in tokens]
    return " ".join(normalized_tokens)

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
        logging.info(f"Checking real-time key: {key}, Normalized Key: {normalized_key}")
        similarity = fuzz.partial_ratio(normalized_key, normalized_query)
        if similarity >= 80:
            logging.info(f"Match found in real-time data! Key: {key}, Similarity: {similarity}")
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

retrieval_cache = {}
MAX_CACHE_SIZE = 100
def retrieve_context(query, query_type, filtered_texts, filtered_metadata, filtered_locations, filtered_faiss_index, filtered_bm25_index, doc_indices_map, excluded_tokens, k_initial=10, k_final=3, embed_model=None, cross_encoder=None):
    if embed_model is None or cross_encoder is None:
        logging.error("embed_model and cross_encoder must be provided to retrieve_context.")
        return "Error in retrieval: Retrieval models not provided.", []

    cache_key = (query, query_type, tuple(filtered_texts))
    if cache_key in retrieval_cache:
        return retrieval_cache[cache_key]

    try:
        expanded_query, _ = expand_query(query, query_type)
        
        tokenized_query = nltk.word_tokenize(expanded_query.lower())
        bm25_scores = filtered_bm25_index.get_scores(tokenized_query)
        bm25_top_indices = set(np.argsort(bm25_scores)[::-1][:k_initial]) 

        query_embedding = embed_model.encode([expanded_query], convert_to_tensor=True, show_progress_bar=False).cpu().numpy()
        D, I = filtered_faiss_index.search(query_embedding, k_initial)
        faiss_top_indices = set(I[0])

        combined_candidates_local_indices = list(faiss_top_indices.union(bm25_top_indices))
        
        if not combined_candidates_local_indices:
            logging.info(f"No initial candidates from FAISS or BM25 for query: '{query}'")
            return "No relevant documents found.", []

        filtered_candidates_for_reranking = []
        for local_idx in combined_candidates_local_indices:
            if 0 <= local_idx < len(filtered_texts) and not any(token in filtered_texts[local_idx].lower() for token in excluded_tokens):
                filtered_candidates_for_reranking.append((local_idx, filtered_texts[local_idx]))

        if not filtered_candidates_for_reranking:
            logging.info(f"No valid candidates after filtering for excluded tokens for query: '{query}'")
            return "No relevant documents found.", []

        pairs_for_cross_encoder = [(expanded_query, text) for local_idx, text in filtered_candidates_for_reranking]
        cross_scores = cross_encoder.predict(pairs_for_cross_encoder)
        
        final_scored_candidates = []
        for i, (local_idx, doc_text) in enumerate(filtered_candidates_for_reranking):
            cross_score = cross_scores[i]
            
            doc_type = filtered_metadata[local_idx]
            location = filtered_locations[local_idx]

            metadata_boost = 2.0 if (query_type.lower() == doc_type and query_type != "General") else 1.0
            location_boost = 1.5 if location and location.lower() in query.lower() else 1.0
            
            global_doc_idx = doc_indices_map[local_idx]
            feedback_key = f"doc_{global_doc_idx}"
            feedback_data = retrieval_feedback.get(feedback_key, {"positive": 0, "negative": 0})
            feedback_score = 1.0 + (feedback_data["positive"] * 0.1) - (feedback_data["negative"] * 0.1)
            
            final_weighted_score = cross_score * metadata_boost * location_boost * feedback_score
            
            final_scored_candidates.append((doc_text, final_weighted_score, doc_type, global_doc_idx))
        
        final_scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_docs_with_scores = final_scored_candidates[:k_final]
        
        logging.info(f"Query: '{query}', Top RAG Documents (Scores, Global Indices):")
        for doc_text, score, doc_type, global_idx in top_docs_with_scores:
            logging.info(f"  - Score: {score:.4f}, Type: {doc_type}, Global Index: {global_idx}, Text: '{doc_text[:100]}...'")

        confidence = top_docs_with_scores[0][1] if top_docs_with_scores else 0.0
        
        RAG_CONFIDENCE_THRESHOLD = 0.65 
        if confidence < RAG_CONFIDENCE_THRESHOLD:
            logging.info(f"Low confidence RAG retrieval for query: '{query}', confidence: {confidence:.4f} < {RAG_CONFIDENCE_THRESHOLD}. Returning no context to LLM.")
            return "Low confidence in retrieved documents. Please rephrase your query or provide more details.", []
        
        summarized_context = []
        retrieved_global_doc_indices = [global_idx for _, _, _, global_idx in top_docs_with_scores]
        for doc_text, _, _, _ in top_docs_with_scores:
            summarized_context.append(doc_text)
        
        context = " ".join(summarized_context)
        if len(context) > 1000:
            context = context[:1000] + "..."
        
        if len(retrieval_cache) >= MAX_CACHE_SIZE:
            retrieval_cache.pop(next(iter(retrieval_cache)))
        retrieval_cache[cache_key] = (context, retrieved_global_doc_indices)
        return context, retrieved_global_doc_indices
    except Exception as e:
        logging.error(f"Error in retrieval: {str(e)}")
        return f"Error in retrieval: {str(e)}", []


def generate_answer(query, context, query_type, intent_type, model=None, tokenizer=None):
    if model is None or tokenizer is None:
        logging.error("Model and tokenizer must be provided to generate_answer.")
        return "Error: AI model not properly loaded.", []

    retrieved_doc_indices = []
    context_str = ""
    
    if isinstance(context, tuple):
        temp_context_str, retrieved_doc_indices = context
        context_str = temp_context_str

    rag_failure_messages = [
        "No relevant documents found.",
        "Low confidence in retrieved documents. Please rephrase your query or provide more details.",
        "Error in retrieval"
    ]

    is_rag_failed = any(msg in context_str for msg in rag_failure_messages) or not context_str.strip()

    if is_rag_failed:
        prompt_text = (
            f"Instruction: You are Bengaluru Bot. The requested information is not available or could not be found. "
            f"Politely state that you cannot answer the user's question based on the provided information. "
            f"Do NOT use any outside knowledge or hallucinate.\n"
            f"Input:\n"
            f"Answer:"
        )
        retrieved_doc_indices = []
        logging.info(f"LLM Prompt (RAG Failed - Strict Decline): '{prompt_text.replace('\n', ' ')[:200]}...'")

    else:
        prompt_text = (
            f"Instruction: You are Bengaluru Bot. Answer the user's question accurately and concisely, "
            f"drawing ONLY from the provided context. Do NOT use any outside knowledge or hallucinate. "
            f"If the exact answer is not present in the context, explicitly state 'I cannot answer this question based on the provided information.'\n"
            f"Input: {context_str}\n"
            f"Answer: {query} "
        )
        logging.info(f"LLM Prompt (With Usable RAG Context - Strict Adherence): '{prompt_text.replace('\n', ' ')[:200]}...'")

    try:
        inputs = tokenizer(
            prompt_text, 
            return_tensors="pt", 
            return_attention_mask=True, 
            truncation=True, 
            max_length=1024 
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.5,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated_token_ids = outputs[0][inputs['input_ids'].shape[1]:]
        final_answer = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        
        final_answer = final_answer.strip()
        if final_answer.lower().startswith(query.lower()):
            final_answer = final_answer[len(query):].strip()
        if final_answer.lower().startswith("answer:"):
             final_answer = final_answer[len("answer:"):].strip()
        
        if not final_answer:
            final_answer = "I apologize, I couldn't generate a clear answer."

        logging.info(f"LLM Raw Decoded Answer: '{final_answer}'")

        return final_answer, retrieved_doc_indices

    except Exception as e:
        logging.error(f"Error generating answer with fine-tuned model: {str(e)}")
        return f"An error occurred during AI response generation: {str(e)}", retrieved_doc_indices
