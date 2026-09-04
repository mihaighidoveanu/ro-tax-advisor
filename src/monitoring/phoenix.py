import logging

from config import (
    ENABLE_TRACING,
    PHOENIX_API_KEY,
    PHOENIX_COLLECTOR_ENDPOINT,
    PHOENIX_PROJECT_NAME,
)

logger = logging.getLogger(__name__)

_initialized = False
_session = None


def init_tracing():
    """
    Wire up Arize Phoenix tracing/monitoring for the LangGraph/LangChain RAG pipeline.

    If PHOENIX_COLLECTOR_ENDPOINT is set (e.g. Phoenix Cloud or a remote collector),
    traces are sent there. Otherwise a local Phoenix instance is launched in-process,
    with its UI reachable at http://localhost:6006. Idempotent and a no-op when
    ENABLE_TRACING is false. Tracing setup failures are logged and swallowed so
    observability issues never take down the RAG app itself.
    """
    global _initialized, _session
    if _initialized or not ENABLE_TRACING:
        return

    from phoenix.otel import register

    try:
        if not PHOENIX_COLLECTOR_ENDPOINT:
            import phoenix as px

            _session = px.launch_app()

        register(
            project_name=PHOENIX_PROJECT_NAME,
            endpoint=PHOENIX_COLLECTOR_ENDPOINT,
            api_key=PHOENIX_API_KEY,
            auto_instrument=True,
        )
        _initialized = True
    except Exception:
        logger.warning("Arize Phoenix tracing could not be initialized; continuing without tracing.", exc_info=True)
