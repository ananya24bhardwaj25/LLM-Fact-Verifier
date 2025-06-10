# LLM-Fact-Verifier
About Section:
This project is a local fact-checking assistant powered by the Mistral language model (via Ollama), FAISS vector search, and an intuitive Streamlit interface. It enables users to input claims and retrieve relevant factual evidence from a custom dataset.

🔍 Key Features:

Embeds and indexes custom factual data with SentenceTransformers and FAISS

Runs completely offline with Mistral LLM via Ollama — no external APIs required

Provides an easy-to-use web interface for fact verification via Streamlit

Supports classification of claims as Supported, Contradicted, or Unverifiable

# Project Structure

├── appdemo.py            #  UI

├── embed_store.py        # Embedding + FAISS storage logic

├── verifier.py           # Command-line interface (optional)

├── facts.csv             # Local factual data

├── facts_embeddings.pkl  # Pickled embeddings

├── faiss.index           # FAISS index for fact search

├── requirements.txt      # Python dependencies

└── README.md             # You're here!


# How it Works

The model embeds factual data from facts.csv using SentenceTransformers.

FAISS creates a vector index to allow similarity-based retrieval.

A user's claim is embedded and compared to the index.

The top-k similar facts are passed to the LLM (mistral) via Ollama.

The model returns a verdict: SUPPORTED, CONTRADICTED, or UNVERIFIABLE.
