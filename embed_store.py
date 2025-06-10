# embed_store.py

import pandas as pd
import faiss
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

FACTS_CSV_PATH = "facts.csv"
EMBEDDINGS_PATH = "facts_embeddings.pkl"
FAISS_INDEX_PATH = "faiss.index"

class EmbedStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.facts = []
        self.embeddings = None
        self.index = None

    def load_facts(self):
        df = pd.read_csv(FACTS_CSV_PATH)
        self.facts = df["text"].tolist()
        return self.facts

    def embed_facts(self):
        print("🔄 Embedding facts...")
        self.embeddings = self.model.encode(self.facts, convert_to_numpy=True, show_progress_bar=True)
        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump((self.facts, self.embeddings), f)
        return self.embeddings

    def build_index(self):
        print("📦 Building FAISS index...")
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.embeddings)
        faiss.write_index(self.index, FAISS_INDEX_PATH)

    def load_index(self):
        print("📥 Loading FAISS index and embeddings...")
        if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(FAISS_INDEX_PATH):
            with open(EMBEDDINGS_PATH, "rb") as f:
                self.facts, self.embeddings = pickle.load(f)
            self.index = faiss.read_index(FAISS_INDEX_PATH)
        else:
            self.load_facts()
            self.embed_facts()
            self.build_index()

    def get_top_k_facts(self, query, k=3):
        if self.index is None:
            raise ValueError("FAISS index is not loaded. Please call 'load_index()' first.")

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_embedding, k)
        return [self.facts[i] for i in I[0]]


# Test script (only runs if this file is executed directly)
if __name__ == "__main__":
    retriever = EmbedStore()
    retriever.load_facts()
    retriever.embed_facts()
    retriever.build_index()
    retriever.load_index()

    test_query = "Government to give free electricity to farmers"
    top_facts = retriever.get_top_k_facts(test_query, k=3)

    print("\n🔍 Top matching facts:")
    for fact in top_facts:
        print("-", fact)
