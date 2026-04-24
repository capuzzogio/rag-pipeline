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
# DEBUG
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
    query_text = "query: " + query
 
    query_vec = model.encode(
        [query_text],
        normalize_embeddings=True
    )
    query_vec = np.array(query_vec).astype("float32")
 
    distances, indices = index.search(query_vec, k)
 
    results = []
    for i in indices[0]:
        if 0 <= i < len(chunks):
            results.append(chunks[i])
 
    return results
 
# -------------------------
# HELPER: extrai texto do chunk
# -------------------------
CAMPOS_TEXTO = [
    "conteudo", "solucao", "fluxo_resolucao", "como_verificar",
    "como_reativar", "passos", "etapas_sequenciais", "regras",
    "condicoes", "descricao", "instrucoes_fora_horario",
]
 
def extrair_texto_chunk(chunk: dict) -> str:
    partes = []
    if chunk.get("titulo"):
        partes.append(f"Tópico: {chunk['titulo']}")
    for campo in CAMPOS_TEXTO:
        val = chunk.get(campo)
        if not val:
            continue
        if isinstance(val, list):
            partes.append("\n".join(str(v) for v in val))
        elif isinstance(val, dict):
            partes.append("\n".join(str(v) for v in val.values()))
        else:
            partes.append(str(val))
    return "\n".join(p for p in partes if p.strip())
 
# -------------------------
# GROQ CLIENT
# -------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
 
# -------------------------
# ENDPOINT
# -------------------------
@app.post("/ask")
def ask(q: Question):
    results = search(q.question)
 
    context_parts = []
    for i, r in enumerate(results[:3]):
        texto = extrair_texto_chunk(r)
        if texto.strip():
            context_parts.append(f"[{i+1}] {texto}")
 
    context = "\n\n".join(context_parts)
 
    if not context.strip():
        return {
            "question": q.question,
            "answer": "Não encontrei informações suficientes no sistema para responder.",
            "results": results
        }
 
    system_prompt = """Você é um assistente de suporte interno da empresa.
Responda SEMPRE em português, de forma clara e operacional.
Use SOMENTE as informações do contexto fornecido.
Nunca invente informações que não estejam no contexto.
Cite as fontes entre colchetes [1], [2] ou [3] ao usar cada informação.
Se o contexto não contiver a resposta, diga exatamente: "Não encontrei essa informação no sistema." """
 
    user_prompt = f"""CONTEXTO:
{context}
 
PERGUNTA:
{q.question}
 
Responda de forma direta e operacional, com passo a passo se necessário. Cite as fontes usadas."""
 
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=1024,
    )
 
    answer = completion.choices[0].message.content.strip()
 
    return {
        "question": q.question,
        "answer": answer,
        "results": results
    }