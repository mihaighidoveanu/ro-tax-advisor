"""Small hand-curated QA evaluation set for the Romanian fiscal code RAG pipeline.

Each item's `expected_keywords` are terms that a correct answer should contain
(case-insensitive substring match); `expected_article` is the article number a
correct retrieval should surface among its citations.
"""

EVAL_DATASET = [
    {
        "question": "Ce este sediul permanent conform Codului fiscal?",
        "expected_keywords": ["sediu", "permanent"],
        "expected_article": 8,
    },
    {
        "question": "Cine sunt contribuabilii obligati la plata impozitului pe profit?",
        "expected_keywords": ["persoane juridice", "impozit"],
        "expected_article": 13,
    },
    {
        "question": "Ce inseamna santier de constructii in contextul sediului permanent?",
        "expected_keywords": ["santier", "constructii"],
        "expected_article": 8,
    },
]
