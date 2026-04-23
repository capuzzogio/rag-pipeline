"""
rebuild_index.py — Reconstrói o índice FAISS a partir do JSON de chunks.

Execute:
    python rebuild_index.py

Rode sempre que o arquivo saf_chunks_otimizados.json for atualizado.
"""

import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


# ── Caminhos ─────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
FAISS_DIR  = BASE_DIR / "faiss_index"
CHUNKS_FILE = DATA_DIR / "saf_chunks_otimizados.json"

# Cria a pasta faiss_index se não existir (corrige erro 2)
FAISS_DIR.mkdir(parents=True, exist_ok=True)


# ── Campos que podem conter texto útil para embedding ────────────────────────
CAMPOS_TEXTO = [
    "conteudo", "titulo", "caminho_sistema", "solucao", "fluxo_resolucao",
    "como_verificar", "como_reativar", "passos", "etapas_sequenciais",
    "regras", "condicoes", "formas_encerramento", "fluxo_por_periodo",
    "checklist_verificacao", "solucao_cache", "regra_ambiguidade",
    "causa_raiz", "instrucoes_fora_horario", "mapeamento_chamados",
    "mapeamento_acoes", "como_localizar", "quando_usar", "restricoes",
    "descricao",
]


def extrair_texto(chunk: dict) -> str:
    """
    Monta o texto de embedding de um chunk JSON.
    Não depende de nenhum campo específico — lê o que existir.
    Corrige o KeyError: 'conteudo'.
    """
    partes = []

    # título sempre em primeiro (ancoragem semântica)
    if chunk.get("titulo"):
        partes.append(f"TÓPICO: {chunk['titulo']}")

    # itera pelos campos de conteúdo que existirem no chunk
    for campo in CAMPOS_TEXTO:
        val = chunk.get(campo)
        if not val:
            continue
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    partes.append(" ".join(str(v) for v in item.values()))
                else:
                    partes.append(str(item))
        elif isinstance(val, dict):
            partes.append(" ".join(str(v) for v in val.values()))
        else:
            partes.append(str(val))

    # FAQ aumenta o recall: o modelo vai casar perguntas do usuário
    if chunk.get("faq"):
        partes.append("PERGUNTAS: " + " | ".join(chunk["faq"]))

    return "\n".join(p for p in partes if p.strip())


def main():
    # ── 1. Carregar chunks ────────────────────────────────────────────────────
    print("📂 Carregando chunks...")
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"✅ {len(chunks)} chunks carregados")

    # ── 2. Montar textos ──────────────────────────────────────────────────────
    print("📝 Extraindo textos dos chunks...")
    textos = [extrair_texto(c) for c in chunks]

    # debug: mostra os primeiros 120 chars de cada texto
    for i, t in enumerate(textos[:3]):
        print(f"   chunk {i+1}: {t[:120].replace(chr(10), ' ')}...")

    # ── 3. Gerar embeddings ───────────────────────────────────────────────────
    print("\n🧠 Carregando modelo intfloat/multilingual-e5-large...")
    print("   (560MB — pode demorar ~2 min na primeira vez)")

    model = SentenceTransformer("intfloat/multilingual-e5-large")

    # O e5-large exige prefixo "query: " na busca e "passage: " na indexação
    textos_indexacao = [f"passage: {t}" for t in textos]

    print("   Gerando embeddings (pode levar 1–3 min no CPU)...")
    embeddings = model.encode(
        textos_indexacao,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=8,        # reduz uso de RAM no CPU
    )
    embeddings = np.array(embeddings, dtype="float32")
    print(f"✅ Embeddings gerados: shape {embeddings.shape}")
    assert embeddings.shape[1] == 1024, (
        f"Dimensão esperada: 1024, obtida: {embeddings.shape[1]}\n"
        "Verifique se o modelo multilingual-e5-large foi carregado corretamente."
    )

    # ── 4. Criar e salvar índice FAISS ────────────────────────────────────────
    print("\n🗄️  Criando índice FAISS...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product = cosseno (vetores normalizados)
    index.add(embeddings)
    print(f"✅ {index.ntotal} vetores indexados (dim={dim})")

    import os

    os.makedirs(FAISS_DIR, exist_ok=True)

    output_path = os.path.join(str(FAISS_DIR), "index.faiss")
    faiss.write_index(index, output_path)

    print(f"✅ index.faiss salvo em: {FAISS_DIR}")

       # ── 5. Salvar metadados dos chunks (para o chat recuperar título, sistema etc.)
    metadados = []
    for i, chunk in enumerate(chunks):
        metadados.append({
            "id":           chunk.get("id", ""),
            "titulo":       chunk.get("titulo", ""),
            "sistema":      ", ".join(chunk.get("sistema", [])) if isinstance(chunk.get("sistema"), list) else str(chunk.get("sistema", "")),
            "categoria":    chunk.get("categoria", ""),
            "subcategoria": chunk.get("subcategoria", ""),
            "acoes_zeev":   ", ".join(chunk.get("acoes_zeev", [])) if isinstance(chunk.get("acoes_zeev"), list) else str(chunk.get("acoes_zeev", "")),
            "restricoes":   " | ".join(chunk.get("restricoes", [])) if isinstance(chunk.get("restricoes"), list) else str(chunk.get("restricoes", "")),
            "texto":        textos[i],
        })

    meta_path = FAISS_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadados, f, ensure_ascii=False, indent=2)
    print(f"✅ metadata.json salvo em: {meta_path}")

    # ── 6. Salvar texto bruto para recuperação no chat ─────────────────────────
    textos_path = FAISS_DIR / "textos.json"
    with open(textos_path, "w", encoding="utf-8") as f:
        json.dump(textos, f, ensure_ascii=False, indent=2)
    print(f"✅ textos.json salvo em: {textos_path}")

    # ── Resumo ────────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("BUILD CONCLUÍDO")
    print(f"  Chunks indexados : {index.ntotal}")
    print(f"  Dimensão vetor   : {dim}")
    print(f"  Modelo           : intfloat/multilingual-e5-large")
    print(f"  Arquivos salvos  : {FAISS_DIR}/")
    print("    ├── index.faiss")
    print("    ├── metadata.json")
    print("    └── textos.json")
    print("─" * 50)


if __name__ == "__main__":
    main()