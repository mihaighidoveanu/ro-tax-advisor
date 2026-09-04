import os
import sys

# Set before any src import triggers config.py's load_dotenv() (which never overrides an
# already-set env var), so tests never spin up a real/local Phoenix tracing session.
os.environ.setdefault("ENABLE_TRACING", "false")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

FIXTURE_HTML = os.path.join(os.path.dirname(__file__), "fixtures", "sample_fiscal_code.html")


@pytest.fixture(scope="session")
def fixture_html():
    return FIXTURE_HTML


@pytest.fixture(scope="session")
def parsed_chunks(fixture_html, tmp_path_factory):
    from ingestion.parse import chunk

    cache_dir = tmp_path_factory.mktemp("docling_cache")
    return chunk(fixture_html, html_dir=str(cache_dir / "html"), docling_dir=str(cache_dir / "docling"))


@pytest.fixture(scope="session")
def fixture_embeddings(parsed_chunks):
    from ingestion.embed import embed

    texts, _ = parsed_chunks
    return embed(texts)


@pytest.fixture(scope="session")
def fixture_vectorstore(parsed_chunks, fixture_embeddings, tmp_path_factory):
    from ingestion.vectorstore import save_embeddings

    texts, metadatas = parsed_chunks
    store_dir = str(tmp_path_factory.mktemp("vector_db"))
    save_embeddings(texts, fixture_embeddings, metadatas, store_name=store_dir)
    return store_dir
