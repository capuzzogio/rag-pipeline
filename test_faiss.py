import faiss
import os

path = r"C:\Users\giovanna.ribeiro\OneDrive - AmorSaúde\Área de Trabalho\RAG SAF\faiss_index\index.faiss"

print("EXISTE:", os.path.exists(path))

index = faiss.read_index(path)

print("INDEX CARREGADO COM SUCESSO")
print("TOTAL VETORES:", index.ntotal)