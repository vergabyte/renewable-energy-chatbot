from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL, logger

_reranker = None


def rerank_documents(query, documents, top_k):
    global _reranker
    if _reranker is None:
        logger.info(f'Loading cross-encoder: {RERANKER_MODEL}')
        _reranker = CrossEncoder(RERANKER_MODEL)

    if not documents:
        return []

    pairs = [(query, doc.page_content) for doc in documents]
    scores = _reranker.predict(pairs)
    doc_score_pairs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)[:top_k]
    reranked_docs = [doc for doc, score in doc_score_pairs]

    logger.info(f'Reranked {len(documents)} -> {len(reranked_docs)} documents')
    return reranked_docs
