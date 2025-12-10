import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

BASE_DIR = Path(__file__).parent
SCRAPED_DATA_FILE = BASE_DIR / 'data' / 'scraped_articles.json'
CHROMA_DB_PATH = BASE_DIR / 'chroma_db'

Path('logs').mkdir(exist_ok=True)
logger.add('logs/app.log')

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L6-v2'
LLM_MODEL = 'llama-3.3-70b-versatile'
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

RETRIEVAL_K = 10
RERANK_K = 4
