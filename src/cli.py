import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import INPUT_FILE, VECTOR_DB_DIR
from ingestion.download import download_source
from ingestion.embed import embed
from ingestion.parse import chunk
from ingestion.vectorstore import save_embeddings
from monitoring.phoenix import init_tracing


def run_ingest(input_file: str, store_name: str, auto_download: bool) -> None:
    if auto_download:
        input_file = download_source(input_file)
    elif not os.path.exists(input_file):
        raise FileNotFoundError(
            f"{input_file!r} not found. Re-run with --download to fetch it from ANAF, "
            "or place the HTML file at that path yourself."
        )

    print(f"Parsing and chunking {input_file}...")
    texts, metadatas = chunk(input_file)
    print(f"Produced {len(texts)} chunks. Embedding...")
    embeddings = embed(texts)
    print(f"Saving embeddings to {store_name}...")
    save_embeddings(texts, embeddings, metadatas, store_name=store_name)
    print("Ingestion complete.")


def run_ask(question: str, store_name: str) -> None:
    from rag.rag import answer_question

    init_tracing()
    result = answer_question(question, store_name=store_name)
    print(result["answer"])
    print("\nSurse:")
    for citation in result["citations"]:
        metadata = citation["metadata"]
        label = metadata.get("headings") or metadata.get("article") or "sursa necunoscuta"
        print(f"  - {label}")


def main():
    parser = argparse.ArgumentParser(description="Romanian Tax Advisor CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Parse, chunk, embed and index the fiscal code")
    ingest_parser.add_argument("--input", default=INPUT_FILE, help="Path to the input HTML file")
    ingest_parser.add_argument("--store", default=VECTOR_DB_DIR, help="Vector store directory")
    ingest_parser.add_argument("--download", action="store_true", help="Download the source HTML from ANAF if missing")

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the indexed fiscal code")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("--store", default=VECTOR_DB_DIR, help="Vector store directory")

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingest(args.input, args.store, args.download)
    elif args.command == "ask":
        run_ask(args.question, args.store)


if __name__ == "__main__":
    main()
