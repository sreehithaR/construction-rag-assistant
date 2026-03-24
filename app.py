print("Running app...")

import os

# -------------------------
# STEP 2: Load Documents
# -------------------------
def load_documents(folder_path):
    documents = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".md"):
            file_path = os.path.join(folder_path, file_name)
            
            with open(file_path, "r", encoding="utf-8") as file:
                documents.append(file.read())

    return documents


# -------------------------
# STEP 2: Chunk Documents
# -------------------------
def chunk_documents(documents, chunk_size=300):
    chunks = []

    for doc in documents:
        words = doc.split()

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)

    return chunks


print("Loading documents...")

documents = load_documents("data")
print("Documents loaded:", len(documents))

chunks = chunk_documents(documents)
print("Chunks created:", len(chunks))


# -------------------------
# STEP 3: Embeddings + FAISS
# -------------------------
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

print("\nCreating embeddings...")

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(chunks)

print("Embeddings created:", len(embeddings))

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

print("FAISS index created and embeddings stored!")


# -------------------------
# STEP 4: Retrieval
# -------------------------
def retrieve(query, k=3):
    print("\nSearching for:", query)

    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)

    results = [chunks[i] for i in indices[0]]
    return results


# -------------------------
# STEP 5: Answer Generation
# -------------------------
import requests

def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a construction assistant.

Answer ONLY using the context below.
If the answer is not present, say: "Not available in documents."

Context:
{context}

Question:
{query}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]





# -------------------------
# TEST RUN
# -------------------------
query = "What factors affect construction delays?"

retrieved_chunks = retrieve(query)

print("\nRetrieved Context:")
for chunk in retrieved_chunks:
    print("-", chunk[:150], "...")

answer = generate_answer(query, retrieved_chunks)

print("\nFinal Answer:")
print(answer)