import streamlit as st
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

st.title("🏗️ Construction RAG Assistant")

# -------------------------
# Load Documents
# -------------------------
def load_documents(folder_path):
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".md"):
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as f:
                documents.append(f.read())
    return documents


# -------------------------
# Chunk Documents
# -------------------------
def chunk_documents(documents, chunk_size=300):
    chunks = []
    for doc in documents:
        words = doc.split()
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks


# -------------------------
# Setup RAG (Embeddings + FAISS)
# -------------------------
@st.cache_resource
def setup_rag():
    documents = load_documents("data")
    chunks = chunk_documents(documents)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return model, index, chunks


model, index, chunks = setup_rag()


# -------------------------
# Retrieval Function
# -------------------------
def retrieve(query, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]


# -------------------------
# Answer Generation (FREE HuggingFace API)
# -------------------------
def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a construction assistant.

Answer ONLY using the context below.
If not found, say: Not available in documents.

Context:
{context}

Question:
{query}
"""

    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

    headers = {
        "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.5
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    data = response.json()

    # Handle response properly
    if isinstance(data, list):
        return data[0].get("generated_text", "No response")

    elif isinstance(data, dict) and "error" in data:
        return f"API Error: {data['error']}"

    else:
        return str(data)


# -------------------------
# UI
# -------------------------
query = st.text_input("Ask a question:")

if query:
    retrieved_chunks = retrieve(query)

    st.subheader("📄 Retrieved Context")
    for chunk in retrieved_chunks:
        st.write(chunk[:300] + "...")

    answer = generate_answer(query, retrieved_chunks)

    st.subheader("🤖 Answer")
    st.write(answer)