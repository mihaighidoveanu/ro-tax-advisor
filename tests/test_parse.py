from ingestion.parse import extract_legal_metadata, _roman_to_int


def test_extract_legal_metadata_full_trail():
    headings = ["Titlul I", "Capitolul III - Definitii", "Articolul 8 Definitia sediului permanent"]
    metadata = extract_legal_metadata(headings)

    assert metadata["title_no"] == 1
    assert metadata["chapter"] == "Definitii"
    assert metadata["chapter_no"] == 3
    assert metadata["article"] == "Definitia sediului permanent"
    assert metadata["article_no"] == 8


def test_extract_legal_metadata_arabic_numbers():
    headings = ["Titlul II Impozitul pe profit", "Sectiunea 2", "Articolul 21"]
    metadata = extract_legal_metadata(headings)

    assert metadata["title"] == "Impozitul pe profit"
    assert metadata["title_no"] == 2
    assert metadata["section_no"] == 2
    assert "article" not in metadata  # no descriptive text after the article number
    assert metadata["article_no"] == 21


def test_extract_legal_metadata_ignores_unrelated_headings():
    assert extract_legal_metadata(["Introducere", "Nota redactionala"]) == {}


def test_extract_legal_metadata_empty_input():
    assert extract_legal_metadata([]) == {}
    assert extract_legal_metadata(None) == {}


def test_roman_to_int():
    assert _roman_to_int("I") == 1
    assert _roman_to_int("IV") == 4
    assert _roman_to_int("IX") == 9
    assert _roman_to_int("XL") == 40


def test_chunk_produces_metadata(parsed_chunks):
    texts, metadatas = parsed_chunks

    assert len(texts) == len(metadatas) > 0
    assert any(m.get("article_no") == 8 for m in metadatas)
    article_8 = next(m for m in metadatas if m.get("article_no") == 8)
    assert article_8["article"] == "Definitia sediului permanent"
    assert article_8["chapter_no"] == 1
