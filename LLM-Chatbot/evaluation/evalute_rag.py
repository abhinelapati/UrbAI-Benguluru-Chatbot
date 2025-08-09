import pandas as pd
import numpy as np
import faiss
import nltk
import os
import logging
from datetime import datetime

# Import core LLM/RAG logic directly from your new module
from core_llm_logic import (
    load_retrieval_models_core,
    load_llm_components_core,
    precompute_filtered_rag_assets_core,
    handle_chitchat,
    expand_query,
    classify_intent,
    get_realtime_data, # Include if your test queries might hit real-time data
    retrieve_context,
    generate_answer,
    # Also import the global data lists that are loaded in core_llm_logic
    texts, metadata, locations, realtime_data, retrieval_feedback
)

# Ragas specific imports (CORRECTED FOR RAGAS 1.0.X+ API)
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,        
    AnswerRelevancy,     
    ContextRelevance,    
    AnswerCorrectness,   
)

# --- Set up basic logging for the evaluation script ---
logging.basicConfig(filename='rag_evaluation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
print("RAG Evaluation Script Started. Check rag_evaluation.log for detailed output.")

# --- Main execution block of the script ---
# This ensures all the model loading and evaluation logic runs only when this script is executed directly.
if __name__ == "__main__":
    # --- Load Models and Assets (These will run once when __main__ is executed) ---
    try:
        print("\n--- Initializing models and RAG assets ---")
        embed_model, cross_encoder = load_retrieval_models_core()
        model, tokenizer = load_llm_components_core()
        # Pass the newly loaded embed_model to precompute_filtered_rag_assets_core
        filtered_assets = precompute_filtered_rag_assets_core(texts, metadata, locations, embed_model)
        print("--- Models and RAG assets initialized. ---")
    except Exception as e:
        logging.critical(f"FATAL: Failed to load models or precompute assets for RAG evaluation: {e}")
        print(f"FATAL: Failed to load models or precompute assets for RAG evaluation: {e}")
        exit(1) # Exit if models can't load

    # --- Define Your Test Questions ---
    test_questions_data = {
        'question': [
            "Hi",
            "What are some popular places to visit in Bangalore?",
            "Tell me about the traffic situation in Electronic City.",
            "How do I report a civic issue to BBMP?",
            "Food to eat in Bangalore?", # This is your critical test query
            "What is the meaning of life?", # Out of scope
            "How can I find good hospitals in Bangalore and what is the best way to commute?", # Complex query
            "Where can I find South Indian breakfast in Bangalore?", # Specific food query
            "What cultural events are happening this month?", # Event query
            "Thank you for your help!", # Another chitchat
            "What is the weather like in Bangalore?", # Weather query
            "Tell me about Bangalore Metro.", # Transit query
            "What is IISc famous for?", # Education query
            "Who built Bangalore Palace?", # Historical (might require specific KB entry)
            "Are there any parks near Cubbon Park?" # Specific location query
        ],
        'ground_truths': [
            ["Hello! How can I help you with information about Bangalore today?"], # Chitchat expectation
            ["Lalbagh Botanical Garden, Cubbon Park, and Bangalore Palace are popular attractions."],
            ["Traffic in Electronic City is typically heavy, especially during peak hours. Expect slow speeds."],
            ["You can report civic issues to BBMP via their website, the 'Sahaaya' app, or by visiting a local ward office."],
            ["Bangalore offers diverse food, try crispy dosas at Vidyarthi Bhavan or street food on VV Puram."], # More specific ground truth
            ["I cannot answer that question based on the information provided in my knowledge base."], # For out of scope
            ["For hospitals, consider Manipal or Fortis. For commute, Namma Metro is efficient."], # Complex answer
            ["Vidyarthi Bhavan is famous for dosas, and VV Puram offers various street food options."],
            ["You can check local listings for concerts and cultural events at venues like Palace Grounds or Chowdiah Memorial Hall. Many events happen on weekends."],
            ["You're welcome! Is there anything else I can help with?"], # Chitchat expectation
            ["The weather in Bangalore is partly cloudy with a temperature of 28°C. Monsoon is from June to October."],
            ["Namma Metro is an efficient way to travel in Bangalore with Green and Purple lines."],
            ["IISc is a premier institute for scientific research and higher education."],
            ["Bangalore Palace is a historical royal palace in Bangalore, India, built by Rev. J. Garrett."],
            ["Cubbon Park is a large park in the heart of Bangalore. Lalbagh Botanical Garden is also nearby."]
        ]
    }

    # --- Simulate Chatbot's Logic to Generate Data for Ragas ---
    generated_data = {
        'question': [],
        'answer': [],
        'contexts': [], # List of lists of strings
        'ground_truths': [] # List of lists of strings
    }

    print("\n--- Generating data for Ragas evaluation ---")
    for i, q in enumerate(test_questions_data['question']):
        print(f"Processing question {i+1}/{len(test_questions_data['question'])}: '{q}'")
        
        current_question = q
        current_ground_truth = test_questions_data['ground_truths'][i]

        chitchat_response_en = handle_chitchat(current_question)
        
        if chitchat_response_en:
            final_answer = chitchat_response_en
            retrieved_docs_list = [] # No documents for chitchat
            logging.info(f"Test '{current_question}' (Chitchat): Answered directly.")
        else:
            english_query = current_question 
            detected_intents = classify_intent(english_query, "General") 
            
            all_retrieved_contexts_str = []
            all_retrieved_indices = []
            combined_llm_answers = []

            for intent in detected_intents:
                selected_assets_for_rag = filtered_assets.get(intent)
                if not selected_assets_for_rag:
                    selected_assets_for_rag = filtered_assets["General"]
                    logging.warning(f"Intent '{intent}' had no pre-computed assets. Falling back to General for retrieval.")

                expanded_query_str, excluded_tokens_list = expand_query(english_query, intent)
                
                context_str, retrieved_indices = retrieve_context(
                    english_query, intent, 
                    selected_assets_for_rag["texts"], selected_assets_for_rag["metadata"], selected_assets_for_rag["locations"], 
                    selected_assets_for_rag["faiss_index"], selected_assets_for_rag["bm25_index"],
                    selected_assets_for_rag["doc_indices_map"], excluded_tokens_list,
                    embed_model=embed_model, cross_encoder=cross_encoder 
                )
                
                answer_part, intent_doc_indices = generate_answer(
                    english_query, (context_str, retrieved_indices), "General", intent, 
                    model=model, tokenizer=tokenizer 
                )
                
                combined_llm_answers.append(answer_part)
                if not ("No relevant documents found." in context_str or "Low confidence" in context_str or "Error" in context_str):
                    all_retrieved_contexts_str.append(context_str) 
                all_retrieved_indices.extend(intent_doc_indices)

            if len(combined_llm_answers) > 1:
                final_answer = "\n\n".join([f"Regarding {intent.replace('_', ' ')}: {ans}" for intent, ans in zip(detected_intents, combined_llm_answers)])
            else:
                final_answer = combined_llm_answers[0] if combined_llm_answers else "I could not find an answer."

            retrieved_docs_list = []
            if all_retrieved_indices:
                retrieved_docs_list = [texts[idx] for idx in list(set(all_retrieved_indices))]


        generated_data['question'].append(current_question)
        generated_data['answer'].append(final_answer)
        generated_data['contexts'].append(retrieved_docs_list)
        generated_data['ground_truths'].append(current_ground_truth)

    print("\n--- Running Ragas Evaluation ---")
    ragas_dataset = Dataset.from_dict(generated_data)

    faithfulness_metric = Faithfulness()
    answer_relevancy_metric = AnswerRelevancy()
    context_relevance_metric = ContextRelevance()
    answer_correctness_metric = AnswerCorrectness()

    score = evaluate(
        ragas_dataset,
        metrics=[
            faithfulness_metric,
            answer_relevancy_metric,
            context_relevance_metric,
            answer_correctness_metric, 
        ],
    )

    print("\n--- Ragas Evaluation Scores ---")
    print(score)

    df_score = score.to_dataframe()
    print("\nDetailed Ragas Scores (DataFrame):")
    print(df_score)

    output_csv_path = "ragas_evaluation_results.csv"
    df_score.to_csv(output_csv_path, index=False)
    print(f"\nResults saved to '{output_csv_path}'")

    # --- Optional: Basic Visualization (requires matplotlib and seaborn) ---
    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # print("\n--- Generating Basic Plots (Check pop-up windows) ---")
    # metrics_to_plot = [col for col in df_score.columns if col not in ['question', 'answer', 'contexts', 'ground_truths']]
    # for col in metrics_to_plot:
    #     plt.figure(figsize=(6, 4))
    #     sns.histplot(df_score[col].dropna(), kde=True)
    #     plt.title(f'Distribution of {col}')
    #     plt.xlabel(col)
    #     plt.ylabel('Count')
    #     plt.show()

    print("\nRAG Evaluation Script Finished.")
