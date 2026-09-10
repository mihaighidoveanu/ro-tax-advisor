from langchain_openai import ChatOpenAI

from config import API_KEY, BASE_URL, MODEL_NAME

_llm_cache = {}


def get_llm(model_name: str = MODEL_NAME) -> ChatOpenAI:
    if model_name not in _llm_cache:
        _llm_cache[model_name] = ChatOpenAI(
            model_name=model_name, openai_api_key=API_KEY, openai_api_base=BASE_URL, temperature=0
        )
    return _llm_cache[model_name]
