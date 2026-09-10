from typing import List, Optional

from chromadb.utils.batch_utils import create_batches
from langchain_chroma import Chroma

from config import VECTOR_DB_DIR
from ingestion.embed import SentenceTransformerEmbeddings

COLLECTION_NAME = "ro_tax_code"


def _get_store(store_name: str, collection_name: str = COLLECTION_NAME) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=SentenceTransformerEmbeddings(),
        persist_directory=store_name,
    )


def save_embeddings(
    texts: List[str],
    embeddings,
    metadatas: Optional[List[dict]] = None,
    store_name: str = VECTOR_DB_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """
    @param texts Chunk texts, aligned with embeddings/metadatas
    @param embeddings List / Numpy array of precomputed embeddings, aligned with texts
    @param metadatas Optional per-chunk metadata dicts, aligned with texts
    @param store_name Directory to persist the Chroma collection to
    @return the Chroma vector store
    """
    store = _get_store(store_name, collection_name)

    ids = [
        f"{(metadatas[i] if metadatas else {}).get('chunk_id', i)}"
        for i in range(len(texts))
    ]
    vectors = [e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings]

    documents = list(texts)
    for batch_ids, batch_vectors, batch_metadatas, batch_documents in create_batches(
        api=store._client,
        ids=ids,
        embeddings=vectors,
        metadatas=metadatas,
        documents=documents,
    ):
        store._collection.upsert(
            ids=batch_ids,
            embeddings=batch_vectors,
            documents=batch_documents,
            metadatas=batch_metadatas,
        )

    return store


def load_vectorstore(store_name: str = VECTOR_DB_DIR, collection_name: str = COLLECTION_NAME) -> Chroma:
    """Load an existing persisted Chroma collection for retrieval."""
    return _get_store(store_name, collection_name)
