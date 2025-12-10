from langchain.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, LLM_MODEL, RETRIEVAL_K, RERANK_K, logger
from src.reranker import rerank_documents


def create_rag_chain(vectorstore, bm25_retriever):
    llm = ChatGroq(model=LLM_MODEL, groq_api_key=GROQ_API_KEY, temperature=0.5)

    dense_retriever = vectorstore.as_retriever(search_kwargs={'k': RETRIEVAL_K})
    retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    logger.info(f'Hybrid retrieval: dense + BM25, k={RETRIEVAL_K}')

    template = """Answer the question using only the information from the context below. If you cannot answer the question based on the context, say "I cannot find that information in the knowledge base."

Context:
{context}

Question: {question}

Answer based only on the context above:"""

    prompt = PromptTemplate.from_template(template)
    format_docs = lambda docs: '\n\n'.join([doc.page_content for doc in docs])

    chain = (
        {'context': retriever | format_docs, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def query_chatbot(chain, retriever, question):
    retrieved_docs = retriever.invoke(question)
    sources = rerank_documents(question, retrieved_docs, RERANK_K)

    print('\n--- Retrieved Chunks (Debug) ---')
    for i, doc in enumerate(sources, 1):
        print(f'\nChunk {i}:')
        print(doc.page_content[:200])
        print('...')
    print('--- End Chunks ---\n')

    answer = chain.invoke(question)
    return answer, sources


def format_sources(sources):
    return list({doc.metadata['source'] for doc in sources if 'source' in doc.metadata})
