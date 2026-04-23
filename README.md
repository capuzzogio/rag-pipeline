## 🔄 Evolução do Projeto

Refatorado de uma versão inicial com LangChain para uma arquitetura direta com FAISS + embeddings,
visando maior controle e consistência no pipeline de RAG.


# 🔍 RAG SAF — Busca Semântica para Suporte Operacional

Sistema de **Retrieval-Augmented Generation (RAG)** desenvolvido para consulta inteligente de conteúdos operacionais do SAF (Serviço de Apoio à Franquia).

O projeto permite realizar perguntas em linguagem natural e recuperar automaticamente os trechos mais relevantes da base de conhecimento, utilizando busca semântica com embeddings.

---

## 🔄 Evolução do Projeto

Este projeto foi inicialmente desenvolvido com LangChain + LLM externo.
Posteriormente, foi refatorado para uma arquitetura direta com **FAISS + embeddings locais**, visando maior controle, consistência e performance no pipeline de recuperação.

---

## 🧠 Tecnologias

* Python
* FAISS (busca vetorial)
* Sentence Transformers
* Modelo: `intfloat/multilingual-e5-large`
* FastAPI

---

## ⚙️ Como funciona

1. Os dados são estruturados em chunks JSON
2. Cada chunk é convertido em embedding semântico
3. Os vetores são indexados com FAISS
4. A API recebe uma pergunta e retorna os chunks mais relevantes

### Pipeline:

```id="p7p5rk"
Chunks JSON
   │
   ▼
extrair_texto()
   │
   ▼
"passage: texto"
   │
   ▼
Embeddings (E5 - 1024 dim)
   │
   ▼
FAISS Index (cosine similarity)
   │
   ▼
Query → "query: pergunta"
   │
   ▼
Busca Top-K
   │
   ▼
Resultados relevantes
```

---

## 📁 Estrutura do Projeto

```id="o3bc5v"
RAG_SAF/
│
├── api.py                # API FastAPI
├── rebuild_index.py      # Geração de embeddings + FAISS
├── data/
│   └── saf_chunks_otimizados.json
├── faiss_index/
│   ├── index.faiss
│   ├── metadata.json
│   └── textos.json
├── requirements.txt
└── README.md
```

---

## 🚀 Como executar

### 1. Instalar dependências

```bash id="wq3vni"
pip install -r requirements.txt
```

---

### 2. Gerar índice FAISS

```bash id="mgsn56"
python rebuild_index.py
```

---

### 3. Rodar API

```bash id="v1p7nq"
uvicorn api:app --reload
```

Acesse:
👉 http://127.0.0.1:8000/docs

---

## 🧪 Exemplo de uso

### Requisição:

```json id="r0e0zb"
{
  "question": "Como abrir agenda de profissional?"
}
```

---

### ✅ Resposta (exemplo real):

```json id="0h4m2c"
{
  "titulo": "Abertura de agenda de profissional",
  "conteudo": "Para abrir a agenda (grade de horários) de um profissional médico no sistema Amei, acessar: Cadastro > Lista de Profissionais > selecionar o profissional > Horários de Atendimento > clicar em Incluir."
}
```

---

## 💬 Exemplos de Consultas

```python id="r6v9o2"
"Como abrir agenda de profissional?"
"Qual sistema usar para cadastro de profissional?"
"Quando usar Zeev, Octadesk ou Yungas?"
"Como priorizar chamados no SAF?"
"Qual o fluxo correto para abertura de chamados?"
```

---

## 💡 Diferenciais

* Uso de modelo **multilíngue otimizado para português**
* Pipeline consistente (chunks + embeddings + FAISS alinhados)
* Validação de dimensão entre modelo e índice (evita erros silenciosos)
* Estrutura pronta para evolução com LLM

---

## 🧠 Conceitos Demonstrados

* Retrieval-Augmented Generation (RAG)
* Embeddings semânticos
* Busca vetorial com FAISS
* Engenharia de dados para IA
* APIs com FastAPI

---

## 🚧 Próximos passos

* Implementar reranking dos resultados
* Gerar respostas finais com LLM
* Adicionar filtros por sistema (Amei, WebDental, Zeev)
* Deploy em ambiente cloud

---

## 📌 Objetivo

Projeto desenvolvido para demonstrar habilidades em:

* NLP aplicado
* Sistemas de busca semântica
* Arquitetura de aplicações com IA
* Construção de pipelines RAG

---

🚀 Projeto de portfólio focado em aplicações práticas de IA para suporte operacional.
