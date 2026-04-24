from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import os
from dotenv import load_dotenv

load_dotenv()

from groq import Groq

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
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.post("/ask")
def ask(q: Question):
    results = search(q.question)

    # -----------------------------
    # 📦 CONTEXTO RAG
    # -----------------------------
    context = "\n\n".join(
    f"[{i+1}] FONTE: {r.get('titulo','')}\n{r.get('conteudo','').strip()}"
    for i, r in enumerate(results[:3])
    if r.get("conteudo")
)

    # -----------------------------
    # 🧠 PROMPT LIMPO (IMPORTANTE)
    # -----------------------------
    prompt = f"""
Você é um assistente de suporte interno baseado em documentos.

REGRAS OBRIGATÓRIAS:
- Use SOMENTE o contexto abaixo
- Nunca invente informação
- Sempre cite a fonte entre colchetes [1], [2], [3] quando usar um trecho
- Cada informação importante deve ter uma fonte

FORMATO DE RESPOSTA:
- Resposta clara e operacional
- Passo a passo quando necessário
- No final: liste as fontes usadas

Se não houver informação suficiente no contexto, diga:
"Não encontrei essa informação no sistema."

CONTEXTO:
{context}

PERGUNTA:
{q.question}
"""
    # -----------------------------
    # ⚡ GROQ
    # -----------------------------
    completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-8b-instant"
    )

    answer = completion.choices[0].message.content

    # -----------------------------
    # 📤 RESPOSTA FINAL
    # -----------------------------
    return {
        "question": q.question,
        "answer": answer,
        "results": results
    }