# Renewable Energy Chat

This repository contains a Retrieval-Augmented Generation (RAG) chatbot from renewable-energy corpus. The articles are already scraped and stored in `data/scraped_articles.json`. The project uses LangChain for orchestration, ChromaDB for dense retrieval, a BM25 retriever for lexical recall, and Groq’s hosted LLM for responses.

## Key Libraries
- `langchain` – RAG chain composition and retrievers
- `chromadb` – persisted vector store for dense similarity search
- `sentence-transformers` – embedding + cross-encoder reranker models
- `langchain-community` / `rank-bm25` – BM25 hybrid retriever 
- `langchain-groq` – Groq LLM client (llama-3.3-70b)
- `python-dotenv` – environment variable loading
- `loguru` – structured logging

## Project Layout
- `main.py` – CLI entrypoint that builds/loads stores and starts the chat loop.
- `config.py` – model selections, file paths, and logging config.
- `data/scraped_articles.json` – renewable-energy article corpus (already crawled).
- `chroma_db/` – persisted Chroma database (created after first run).
- `logs/` – application logs.
- `src/`
  - `preprocessing.py` – text cleanup before chunking.
  - `embeddings.py` – chunking helpers plus Chroma/BM25 builders.
  - `chatbot.py` – hybrid retriever, reranker, and chat interface.
  - `reranker.py` – cross-encoder reranking wrapper.

## Quick Start
1. **Create and activate a virtual environment (recommended).**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies.**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure secrets.**
   Copy `.env.example` to `.env` (if present) or create `.env`, then set:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
4. **Run the chatbot.**
   ```bash
   python main.py
   ```

## Managing the Knowledge Base
- The first run embeds all documents and persists them to `chroma_db/`. Subsequent runs load that store immediately.
- To rebuild the embeddings (e.g., after changing preprocessing, chunk sizes, or the corpus), delete `chroma_db/` or run:
  ```bash
  python main.py --rebuild
  ```
- To update the knowledge base contents, replace `data/scraped_articles.json` with your new renewable-energy corpus and rebuild.

## Notes
- BM25 chunks are rebuilt from the JSON file on startup to keep dense and lexical stores in sync.

