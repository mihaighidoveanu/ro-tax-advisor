from ingestion.embed import SentenceTransformerEmbeddings, embed


def test_embed_returns_one_vector_per_chunk():
    texts = ["Articolul 8 vorbeste despre sediul permanent.", "Articolul 13 vorbeste despre contribuabili."]
    embeddings = embed(texts)

    assert len(embeddings) == len(texts)
    assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 output dimension


def test_sentence_transformer_embeddings_document_and_query_share_dimension():
    embeddings = SentenceTransformerEmbeddings()

    doc_vectors = embeddings.embed_documents(["Un text oarecare despre impozite."])
    query_vector = embeddings.embed_query("Ce impozite exista?")

    assert len(doc_vectors[0]) == len(query_vector)
