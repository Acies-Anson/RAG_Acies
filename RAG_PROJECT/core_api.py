import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Force load the .env file at the top of the module
load_dotenv()

def call_llm(prompt):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Check your .env file.")
    
    # Pass the key explicitly to the constructor
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        groq_api_key=api_key, 
        temperature=0
    )
    return llm.invoke(prompt)