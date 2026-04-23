from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import os

# -------------------------
# APP
# -------------------------
app = FastAPI(title="RAG SAF API")

# -------------------------
# PATHS
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHUNKS_FILE = os.path.join(BASE_DIR, "data", "saf_chunks_otimizados.json")
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

# -------------------------
# DEBUG (agora no lugar certo)
# -------------------------
print("FAISS PATH REAL:", os.path.join(FAISS_PATH, "index.faiss"))
print("EXISTE?:", os.path.exists(os.path.join(FAISS_PATH, "index.faiss")))

# -------------------------
# INPUT
# -------------------------
class Question(BaseModel):
    question: str

# -------------------------
# MODELO
# -------------------------
model = SentenceTransformer("intfloat/multilingual-e5-large")

# -------------------------
# CHUNKS
# -------------------------
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# -------------------------
# FAISS
# -------------------------
index = faiss.read_index(os.path.join(FAISS_PATH, "index.faiss"))

if index.d != model.get_sentence_embedding_dimension():
    raise ValueError(
        f"Dim mismatch: index={index.d}, model={model.get_sentence_embedding_dimension()}"
    )

# -------------------------
# SEARCH
# -------------------------
def search(query, k=3):
    # 🔥 prefixo obrigatório do E5
    query_text = "query: " + query

    query_vec = model.encode(
        [query_text],
        normalize_embeddings=True
    )
    query_vec = np.array(query_vec).astype("float32")

    distances, indices = index.search(query_vec, k)

    # 🔒 proteção contra índice inválido
    results = []
    for i in indices[0]:
        if 0 <= i < len(chunks):
            results.append(chunks[i])

    return results

# -------------------------
# ENDPOINT
# -------------------------
@app.post("/ask")
def ask(q: Question):
    results = search(q.question)

    return {
        "question": q.question,
        "results": results
    }