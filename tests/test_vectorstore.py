from ingestion.vectorstore import load_vectorstore


def test_similarity_search_finds_relevant_article(fixture_vectorstore):
    store = load_vectorstore(fixture_vectorstore)

    results = store.similarity_search("Ce este sediul permanent?", k=1)

    assert len(results) == 1
    assert results[0].metadata.get("article_no") == 8


def test_similarity_search_returns_metadata(fixture_vectorstore):
    store = load_vectorstore(fixture_vectorstore)

    results = store.similarity_search("impozitul pe profit", k=3)

    assert all("chunk_id" in doc.metadata for doc in results)
