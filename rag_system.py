# pip install faiss-cpu sentence-transformers


import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGSystem:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())

    def index_documents(self, documents):
        embeddings = self.model.encode(documents)
        self.index.add(np.array(embeddings, dtype=np.float32))

    def retrieve(self, query, top_k=5):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)
        return indices.flatten().tolist()

# Ejemplo de uso
if __name__ == "__main__":
    rag = RAGSystem()
    docs = [
        "El rápido zorro marrón salta sobre el perro perezoso.",
        "La inteligencia artificial está transformando el mundo.",
        "Python es un lenguaje de programación muy popular."
    ]
    rag.index_documents(docs)
    query = "¿Qué es Python?"
    results = rag.retrieve(query)
    print("Documentos relevantes:", results)
