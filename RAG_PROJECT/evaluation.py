import os
import json
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def evaluate_response(query, context, answer):
    """
    Acts as a 'Judge LLM' with robust JSON parsing to handle LLM chatter.
    """
    api_key = os.getenv("GROQ_API_KEY")
    evaluator_llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key, temperature=0)

    eval_prompt = f"""
    You are an expert Legal Document Auditor. Evaluate this RAG response.
    
    [QUERY]: {query}
    [CONTEXT]: {context}
    [ANSWER]: {answer}
    
    Return the result STRICTLY as a JSON object with these keys:
    "faithfulness_score", "relevance_score", "precision_score", "feedback"
    
    Scores are 1-5. Ensure the JSON is the ONLY thing in your response.
    """

    try:
        response = evaluator_llm.invoke(eval_prompt)
        content = response.content.strip()

        # --- ROBUST JSON EXTRACTION ---
        # This regex finds the first { and the last } and takes everything in between
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            json_str = match.group(0)
            eval_data = json.loads(json_str)
        else:
            # Fallback if no JSON braces are found
            raise ValueError("No JSON block found in LLM response")

        return eval_data

    except Exception as e:
        return {
            "faithfulness_score": 0,
            "relevance_score": 0,
            "precision_score": 0,
            "feedback": f"Parsing Error: {str(e)}"
        }