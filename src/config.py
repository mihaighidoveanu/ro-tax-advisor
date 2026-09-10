import os
from dotenv import load_dotenv

load_dotenv()


def _use_system_trust_store() -> None:
    """Verify TLS against the OS trust store instead of certifi's frozen bundle.

    Corporate networks often terminate TLS at a proxy that presents a certificate
    signed by an internal root CA. That CA lives in the OS store (so ``curl`` and
    browsers trust it) but not in the ``certifi`` bundle the OpenAI SDK and httpx
    default to, so every model call fails with ``APIConnectionError``. ``truststore``
    patches ``ssl`` globally to use the OS store, fixing OpenAI and Phoenix export.

    Best-effort: if ``truststore`` is unavailable the app runs with certifi as before.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        pass


_use_system_trust_store()


def _bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("API_KEY")
# Point at an OpenAI-compatible provider other than OpenAI itself (e.g. Groq's
# https://api.groq.com/openai/v1). Leave unset to use OpenAI's default endpoint.
BASE_URL = os.getenv("BASE_URL") or None

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

INPUT_FILE = os.getenv("INPUT_FILE", "input/Legea nr.227_2015.html")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "tmp/vector_db")
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))

PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or None
PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY") or None
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "ro-tax-advisor")
ENABLE_TRACING = _bool(os.getenv("ENABLE_TRACING"), default=True)

SOURCE_URL = "https://static.anaf.ro/static/10/Anaf/legislatie/Cod_fiscal_norme_2016.htm"
