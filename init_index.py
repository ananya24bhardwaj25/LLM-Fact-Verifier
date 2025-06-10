# init_index.py
from embed_store import EmbedStore

retriever = EmbedStore()
retriever.load_facts()
retriever.embed_facts()
retriever.build_index()
print("✅ Index and embeddings built and saved.")
