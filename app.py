import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st

from config import VECTOR_DB_DIR
from monitoring.phoenix import init_tracing
from rag.rag import answer_question

st.set_page_config(page_title="Consilier fiscal RO", page_icon="\U0001f4d6")


@st.cache_resource
def _setup():
    init_tracing()


def _citation_label(metadata: dict) -> str:
    parts = []
    if metadata.get("title"):
        parts.append(f"Titlul {metadata.get('title_no', '')} - {metadata['title']}".strip())
    if metadata.get("chapter"):
        parts.append(f"Capitolul {metadata.get('chapter_no', '')} - {metadata['chapter']}".strip())
    if metadata.get("article"):
        parts.append(f"Articolul {metadata.get('article_no', '')} - {metadata['article']}".strip())
    return " | ".join(parts) or metadata.get("headings", "sursa necunoscuta")


def _render_sources(citations, key_prefix: str):
    if not citations:
        return
    with st.popover("\U0001f4c4 Vezi sursele"):
        for i, citation in enumerate(citations):
            st.markdown(f"**{_citation_label(citation['metadata'])}**")
            st.caption(citation["text"])
            if i < len(citations) - 1:
                st.divider()


_setup()

st.title("Consilier fiscal - Codul fiscal romanesc")
st.caption("Raspunsuri cu citatii, pe baza Codului fiscal (ANAF).")

if not os.path.isdir(VECTOR_DB_DIR):
    st.warning(
        f"Nu am gasit un index in `{VECTOR_DB_DIR}`. Ruleaza mai intai "
        "`python src/cli.py ingest --download` pentru a indexa Codul fiscal."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_sources(message.get("citations", []), key_prefix=f"history-{i}")

question = st.chat_input("Intreaba ceva despre Codul fiscal...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Caut in Codul fiscal..."):
            try:
                result = answer_question(question)
            except Exception as exc:
                result = {"answer": f"A aparut o eroare la generarea raspunsului: {exc}", "citations": []}
        st.markdown(result["answer"])
        _render_sources(result["citations"], key_prefix="latest")

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "citations": result["citations"]}
    )
