import json
import shutil
import sys

from config import CHROMA_DB_PATH, SCRAPED_DATA_FILE, logger
from src.chatbot import create_rag_chain, format_sources, query_chatbot
from src.embeddings import chunk_documents, create_bm25_retriever, create_vector_store, load_vector_store


def initialize_chatbot(force_rebuild=False):
    logger.info('Starting RAG chatbot')

    if force_rebuild and CHROMA_DB_PATH.exists():
        logger.info('Removing old vector store')
        shutil.rmtree(CHROMA_DB_PATH)

    if CHROMA_DB_PATH.exists() and not force_rebuild:
        logger.info('Loading existing vector store')
        vectorstore = load_vector_store(CHROMA_DB_PATH)
        with open(SCRAPED_DATA_FILE, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        chunks = chunk_documents(articles)
    else:
        logger.info('Building knowledge base from scratch')
        if not SCRAPED_DATA_FILE.exists():
            logger.error(f'Data file not found: {SCRAPED_DATA_FILE}')
            raise FileNotFoundError(f'Data file not found: {SCRAPED_DATA_FILE}')

        with open(SCRAPED_DATA_FILE, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        logger.info(f'Loaded {len(articles)} articles')

        if not articles:
            logger.error('No articles in JSON file')
            raise ValueError('No articles in JSON file')

        chunks = chunk_documents(articles)
        vectorstore = create_vector_store(chunks, CHROMA_DB_PATH)

    bm25_retriever = create_bm25_retriever(chunks)
    chain, retriever = create_rag_chain(vectorstore, bm25_retriever)
    logger.info('RAG chain initialized')

    return chain, retriever


def chat_loop(chain, retriever):
    logger.info('Starting chat loop')

    while True:
        try:
            question = input('You: ').strip()
        except (KeyboardInterrupt, EOFError):
            logger.info('User interrupted')
            break

        if question.lower() in ['exit', 'quit']:
            logger.info('User exited')
            break

        if not question:
            continue

        try:
            logger.info(f'Query: {question}')
            answer, sources = query_chatbot(chain, retriever, question)

            print(f'Bot: {answer}\n')

            source_urls = format_sources(sources)
            if source_urls:
                print('Sources:')
                for url in source_urls:
                    print(f'  - {url}')
            print()
        except Exception as e:
            logger.error(f'Query failed: {e}')
            print(f'Error: {e}\n')


def run_chatbot(force_rebuild=False):
    chain, retriever = initialize_chatbot(force_rebuild)
    chat_loop(chain, retriever)


if __name__ == '__main__':
    run_chatbot('--rebuild' in sys.argv)
