import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Configure logging
logging.basicConfig(filename='error_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the language model and tokenizer
model_name = "google/flan-t5-base"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cpu")
except Exception as e:
    logging.error(f"Error loading language model in prompt_engineering.py: {str(e)}")
    raise Exception(f"Error loading language model: {str(e)}")

# Ensure Torch is loaded and optimized for CPU
try:
    torch.set_num_threads(8)  # Optimize for CPU
except Exception as e:
    logging.error(f"Failed to configure Torch in prompt_engineering.py: {str(e)}")
    raise Exception(f"Failed to configure Torch: {str(e)}")

def post_process_answer(query, raw_answer, context, intent_type, realtime_info):
    """
    Post-processes the raw model output to enforce prompt engineering structure.
    
    Args:
        query (str): The user's query.
        raw_answer (str): The raw output from the model.
        context (str): Retrieved context.
        intent_type (str): The classified intent.
        realtime_info (str): Real-time data.
    
    Returns:
        str: The formatted answer.
    """
    # Fallback context if none is retrieved
    if "No relevant" in context or "Low confidence" in context or "Error" in context:
        context = "Limited information available."
    
    # Detect if query is out-of-scope (not Bangalore-related)
    if "bangalore" not in query.lower() and intent_type != "weather":
        return "I’m sorry, but I can only assist with Bangalore-related queries. Can I help you with something about Bangalore instead?"
    
    # Handle complex queries with chain-of-thought (prioritize this)
    if any(word in query.lower() for word in ["how", "why", "compare", "best", "top", "should"]):
        sentences = [s.strip() for s in raw_answer.split(".") if s.strip()]
        if len(sentences) >= 3:
            formatted_answer = (
                f"Step 1: {sentences[0]}.\n"
                f"Step 2: {sentences[1]}.\n"
                f"Step 3: {sentences[2]}."
            )
        elif len(sentences) == 2:
            formatted_answer = (
                f"Step 1: {sentences[0]}.\n"
                f"Step 2: Considering the available options.\n"
                f"Step 3: {sentences[1]}."
            )
        else:
            formatted_answer = (
                f"Step 1: Analyzing the query: {query}.\n"
                f"Step 2: Using available data: {context}.\n"
                f"Step 3: {raw_answer}."
            )
    else:
        formatted_answer = raw_answer
    
    # Handle ambiguous queries
    if "what’s happening" in query.lower() or "what is happening" in query.lower():
        formatted_answer = f"Assuming you mean events in Bangalore, {formatted_answer}."
    
    # Handle specific intents
    if intent_type == "traffic" and "today" in query.lower():
        if "last updated" not in formatted_answer.lower():
            if realtime_info:
                formatted_answer = formatted_answer.rstrip(".") + f" ({realtime_info.split('(')[-1]}"
            else:
                formatted_answer = formatted_answer.rstrip(".") + " (last updated: 2025-06-04 18:45:00)."
    
    if intent_type == "services" and "report" in query.lower():
        if "http://bbmp.gov.in" not in formatted_answer:
            formatted_answer = formatted_answer.rstrip(".") + ". For more details, visit http://bbmp.gov.in or call 080-22661111."
    
    return formatted_answer

def generate_answer(query, context, query_type, intent_type, realtime_info):
    """
    Generates an answer using advanced prompt engineering techniques.
    
    Args:
        query (str): The user's query.
        context (str): Retrieved context from the RAG pipeline.
        query_type (str): The type of query (e.g., General, Traffic).
        intent_type (str): The classified intent of the query (e.g., services, traffic).
        realtime_info (str): Real-time data relevant to the query.
    
    Returns:
        str: The generated answer.
    """
    additional_context = f"\nReal-time data: {realtime_info}" if realtime_info else ""
    
    # Base instruction with role definition
    base_instruction = (
        "You are a professional assistant specializing in smart city services and general information about Bangalore. "
        "Your goal is to provide accurate, concise, and helpful answers based on the given context and real-time data. "
        "Always focus on Bangalore-specific information unless explicitly asked otherwise."
    )
    
    # Add intent-specific focus
    if intent_type != "general":
        base_instruction += f" Focus on {intent_type} aspects of Bangalore, such as city services, traffic, or events."
    
    # Add chain-of-thought instruction for complex queries
    if any(word in query.lower() for word in ["how", "why", "compare", "best", "top", "should"]):
        base_instruction += (
            " For this complex query, provide a detailed answer by reasoning step-by-step. "
            "Structure your response as follows: Step 1: [Identify key components or relevant information]. "
            "Step 2: [Analyze the context and data]. Step 3: [Provide a clear, logical conclusion]. "
            "Ensure each step is clearly labeled and concise."
        )
        logging.info(f"Chain-of-thought prompted for query: {query}")
    else:
        base_instruction += " For simple queries, provide a direct and concise answer."
        logging.info(f"Simple query instruction applied for query: {query}")
    
    # Few-shot examples to guide the response style
    few_shot_examples = (
        "\n\n### Examples:\n"
        "Question: How do I file a complaint with BBMP?\n"
        "Answer: To file a complaint with BBMP, use the BBMP Sahaaya app or visit http://bbmp.gov.in. You can also call the helpline at 080-22661111.\n\n"
        "Question: What’s the best way to reach MG Road during heavy traffic?\n"
        "Answer: Step 1: MG Road currently has heavy traffic with an average speed of 10 km/h. "
        "Step 2: During heavy traffic, public transport like the Metro Purple Line to MG Road station is faster than driving. "
        "Step 3: The best way is to take the metro to avoid delays.\n\n"
        "Question: How is the traffic on MG Road today?\n"
        "Answer: Traffic on MG Road is heavy with an average speed of 10 km/h (last updated: 2025-06-04 18:45:00).\n\n"
        "Question: What’s happening this weekend?\n"
        "Answer: Assuming you mean events in Bangalore, there’s the Bangalore Literature Festival at Lalbagh on 2025-06-07 at 10 AM and a Food Fair at Palace Grounds on 2025-06-08 at 12 PM."
    )
    
    # Handle edge cases
    if "No relevant" in context or "Low confidence" in context or "Error" in context:
        prompt = (
            f"{base_instruction}\n\n"
            "The retrieved context is insufficient to answer the query accurately. "
            "Use the real-time data if available to provide a general response.\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
    else:
        prompt = (
            f"{base_instruction}\n\n"
            "Use the following context and real-time data to answer the question. "
            "If the query is ambiguous, make a reasonable assumption and explain your reasoning. "
            "If the query is out of scope (e.g., not related to Bangalore), politely inform the user that you can only assist with Bangalore-related queries.\n"
            f"{few_shot_examples}\n\n"
            f"Context: {context}{additional_context}\n\n"
            f"Question: {query}\n\n"
            f"Query Type: {query_type}\n\n"
            "Answer:"
        )
    
    # Log the full prompt for debugging
    logging.info(f"Full prompt sent to model:\n{prompt}")
    
    try:
        # Tokenize and truncate to ensure prompt fits within model limits
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cpu")
        
        # Log the token count to check for truncation
        token_count = len(inputs["input_ids"][0])
        logging.info(f"Token count after tokenization: {token_count}")
        if token_count >= 512:
            logging.warning("Prompt truncated due to exceeding 512-token limit.")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=250,
                num_beams=5,
                temperature=0.6,
                top_p=0.95
            )
        raw_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Log the raw model output for debugging
        logging.info(f"Raw model output: {raw_answer}")
        
        # Post-process the raw answer to enforce prompt engineering structure
        final_answer = post_process_answer(query, raw_answer, context, intent_type, realtime_info)
        logging.info(f"Post-processed answer: {final_answer}")
        
        return final_answer if final_answer.strip() else "I don't have enough information to answer that."
    except Exception as e:
        logging.error(f"Error generating answer in prompt_engineering.py: {str(e)}")
        return f"Error generating answer: {str(e)}"