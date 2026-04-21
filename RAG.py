import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_xai import ChatXAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. Setup
load_dotenv()
api_key = os.getenv("XAI_API_KEY")

# 2. Dynamic File Loading
documents = []
data_folder = Path("data")  

if not data_folder.exists():
    raise FileNotFoundError(f"The folder '{data_folder}' does not exist.")

for file_path in data_folder.glob("*.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        # Create Document object with content and source filename metadata
        documents.append(Document(
            page_content=content, 
            metadata={"source": file_path.name}
        ))

print(f"Loaded {len(documents)} documents from {data_folder}")

# 3. Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
chunks = text_splitter.split_documents(documents)

# 4. Embedding & Vector Store (BGE-Small + Cosine Similarity)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vector_db = FAISS.from_documents(
    chunks, 
    embeddings, 
    distance_strategy="COSINE" 
)

# 5. Retrieval with Ranking and Scores
query = "What is the termination policy"
results = vector_db.similarity_search_with_relevance_scores(query, k=3)

print(f"\n--- Semantic Ranking for: '{query}' ---\n")

context_chunks = []
for rank, (doc, score) in enumerate(results, 1):
    score_pct = score * 100
    print(f"Rank {rank} | Match: {score_pct:.2f}% | Source: {doc.metadata['source']}")
    print(f"Snippet: {doc.page_content[:100]}...")
    print("-" * 50)
    context_chunks.append(f"Source: {doc.metadata['source']}\nContent: {doc.page_content}")

# 6. Grok API Synthesis
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=api_key,
    temperature=0
)

context_text = "\n\n".join(context_chunks)
prompt = f"""You are a Data Protection Officer. 
Answer the question accurately using ONLY the provided context. 
Cite which document the information came from in your answer.

Context:
{context_text}

Question: {query}
Answer:"""

response = llm.invoke(prompt)

print("\n--- Grok's Final Answer ---")
print(response.content)