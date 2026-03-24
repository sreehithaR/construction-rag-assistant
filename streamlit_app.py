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
    # Simple answer generation from retrieved context
    if not retrieved_chunks:
        return "No relevant information found."

    answer = "Based on the documents:\n\n"

    for i, chunk in enumerate(retrieved_chunks, 1):
        answer += f"{i}. {chunk[:300]}...\n\n"

    return answer


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