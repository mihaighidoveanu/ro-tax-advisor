import os
from dotenv import load_dotenv

load_dotenv()


def _bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("API_KEY")

# Phoenix's and RAGAS's OpenAI-backed judges read the SDK-standard env var, not our
# custom API_KEY name, so bridge it once here.
if API_KEY and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = API_KEY

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

INPUT_FILE = os.getenv("INPUT_FILE", "input/Legea nr.227_2015.html")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "tmp/vector_db")
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))

PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or None
PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY") or None
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "ro-tax-advisor")
ENABLE_TRACING = _bool(os.getenv("ENABLE_TRACING"), default=True)

SOURCE_URL = "https://static.anaf.ro/static/10/Anaf/legislatie/Cod_fiscal_norme_2016.htm"
