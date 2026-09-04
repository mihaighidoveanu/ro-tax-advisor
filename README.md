# Romanian Tax Advisor

This chatbot enables queries on romanian fiscal code, with citations.

## Input file

https://static.anaf.ro/static/10/Anaf/legislatie/Cod_fiscal_norme_2016.htm#A139

## Setup

```bash
python -m venv .venv
```

### Windows
```
.venv\Scripts\activate.ps1
pip install --ugprade pip
pip install -r requirements.txt
```

### Unix
```
source .venv/bin/activate
pip install --ugprade pip
pip install -r requirements.txt
```

### Configure secrets

```bash
cp .env.example .env
# then fill in API_KEY (and anything else you want to override) in .env
```

## Ingest the fiscal code

Downloads the source HTML from ANAF (if not already present under `input/`), parses and
chunks it with Docling, embeds each chunk, and saves everything to a local Chroma vector
store under `tmp/vector_db`:

```bash
python src/cli.py ingest --download
```

## Ask a question from the terminal

```bash
python src/cli.py ask "Ce este sediul permanent?"
```

## Run the chatbot UI

```bash
streamlit run app.py
```

Opens a chat UI at http://localhost:8501 with a "Vezi sursele" (See sources) button next
to each answer, showing the fiscal-code articles used to generate it.

## Monitoring

The RAG pipeline is instrumented with [Arize Phoenix](https://docs.arize.com/phoenix).
By default (no `PHOENIX_COLLECTOR_ENDPOINT` set) a local Phoenix instance launches
automatically the first time the pipeline runs, with its UI at http://localhost:6006. To
point at Phoenix Cloud or a remote collector instead, set `PHOENIX_COLLECTOR_ENDPOINT`
(and `PHOENIX_API_KEY` if required) in `.env`.

## Tests

```bash
pytest
```

## Evaluation

```bash
python src/evaluation/evaluate.py
```

Runs the small hand-curated question set in `src/evaluation/dataset.py` against the
indexed fiscal code and reports retrieval hit rate and answer keyword coverage.

Add `--llm-judges` to also score each answer with Phoenix's Faithfulness (is the answer
grounded in its retrieved context?) and Context Relevance (were the retrieved chunks
actually useful?) LLM-judge evaluators. This makes real LLM calls, so it's opt-in:

```bash
python src/evaluation/evaluate.py --llm-judges
```

You can also run those two evaluators directly, without the heuristic scoring:

```bash
python src/evaluation/phoenix_evals.py
```

### RAGAS benchmark

```bash
python src/evaluation/ragas_benchmark.py
```

Generates a synthetic QA testset from the already-indexed fiscal-code chunks with
[RAGAS](https://docs.ragas.io/), runs the RAG pipeline over it, and scores the results
with RAGAS's Faithfulness, Context Precision, Context Recall and Answer Relevancy
metrics. This is a heavier benchmark than `evaluate.py` - it makes an LLM call to
generate each synthetic question plus several more per question to score it, so expect
it to take a while and to use real API quota. Run `ingest` first so there's an index to
generate a testset from and to answer against.

**Dependency note:** `ragas`'s testset generator depends on `scikit-network`, whose
newer releases require compiling a C extension; `requirements.txt` pins
`scikit-network==0.12.1` (last pure-Python wheel) and `langchain-community==0.3.31`
(current `ragas` still imports a submodule that later `langchain-community` releases
removed) to avoid both issues. If you bump either package, re-verify `import ragas`
still works.

## Development Roadmap

The development roadmap is available [here](DEVELOPMENT.md).
