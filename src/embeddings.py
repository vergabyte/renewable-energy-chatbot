from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, RETRIEVAL_K, logger
from src.preprocessing import preprocess_text


def chunk_documents(articles):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )

    chunks = [
        Document(page_content=split, metadata={'source': article['url']})
        for article in articles
        for split in splitter.split_text(preprocess_text(article['content']))
    ]

    logger.info(f'Created {len(chunks)} chunks from {len(articles)} articles')
    return chunks


def create_vector_store(chunks, persist_directory):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory),
    )
    logger.info(f'Vector store created at {persist_directory}')
    return vectorstore


def load_vector_store(persist_directory):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=str(persist_directory), embedding_function=embeddings)


def create_bm25_retriever(chunks):
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = RETRIEVAL_K
    logger.info(f'BM25 retriever created with {len(chunks)} documents')
    return retriever
