from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME

_model_cache = {}


def _get_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible Embeddings wrapper around a sentence-transformers model,
    so ingestion (batch document embedding) and retrieval (query embedding) share the
    exact same model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return _get_model(self.model_name).encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return _get_model(self.model_name).encode([text], show_progress_bar=False)[0].tolist()


def embed(chunks: List[str], model_name: str = EMBEDDING_MODEL_NAME):
    """
    @param chunks Chunks of text
    @return List / Numpy array of embeddings
    """
    return _get_model(model_name).encode(chunks, show_progress_bar=True)
