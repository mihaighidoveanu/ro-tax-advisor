import os
import re
from typing import List, Tuple

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from config import EMBEDDING_MODEL_NAME

# Matches Romanian fiscal-code heading levels, e.g. "Titlul I", "Capitolul 3 - Impozite",
# "Sectiunea 1", "Articolul 8 Definitia sediului permanent" (diacritics optional).
_HEADING_PATTERN = re.compile(
    r"^\s*(Titlul?|Capitolul|Sec[țt]iunea|Articolul)\s+([IVXLCDM]+|\d+)\s*[-–:.]?\s*(.*)$",
    re.IGNORECASE,
)
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _roman_to_int(value: str) -> int:
    total = 0
    for i, char in enumerate(value.upper()):
        digit = _ROMAN_VALUES.get(char)
        if digit is None:
            raise ValueError(f"Not a roman numeral: {value}")
        if i + 1 < len(value) and _ROMAN_VALUES.get(value[i + 1].upper(), 0) > digit:
            total -= digit
        else:
            total += digit
    return total


def _to_number(value: str):
    if value.isdigit():
        return int(value)
    try:
        return _roman_to_int(value)
    except ValueError:
        return value


_LEVEL_KEYS = {
    "titlu": ("title", "title_no"),
    "capitolul": ("chapter", "chapter_no"),
    "sectiunea": ("section", "section_no"),
    "articolul": ("article", "article_no"),
}


def _normalize_level(word: str) -> str:
    word = word.lower().rstrip("l")  # "titlul" -> "titlu"
    for key in _LEVEL_KEYS:
        if word.startswith(key.rstrip("l")):
            return key
    return word


def extract_legal_metadata(headings: List[str]) -> dict:
    """
    Derive title/chapter/section/article metadata from a Docling chunk's heading
    trail (outermost heading first) using Romanian fiscal-code heading conventions.

    @param headings Chunk heading trail, e.g. ["Titlul I", "Capitolul III", "Articolul 8 Definitia sediului permanent"]
    @return dict with any of title/title_no, chapter/chapter_no, section/section_no, article/article_no found
    """
    metadata = {}
    for heading in headings or []:
        match = _HEADING_PATTERN.match(heading)
        if not match:
            continue
        level_word, number, rest = match.groups()
        level = _normalize_level(level_word)
        name_key, no_key = _LEVEL_KEYS.get(level, (level, f"{level}_no"))
        rest = rest.strip()
        if rest:
            metadata[name_key] = rest
        metadata[no_key] = _to_number(number)
    return metadata


def _get_tokenizer(embedding_model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceTokenizer:
    return HuggingFaceTokenizer.from_pretrained(embedding_model_name, max_tokens=256)


def convert(input_file: str, html_dir: str = "tmp/html", docling_dir: str = "tmp/docling"):
    """
    Parse an HTML input file with Docling and cache the result for reuse/inspection.

    @param input_file HTML input file
    @param html_dir Directory to save the DoclingDocument as HTML for visual inspection
    @param docling_dir Directory to save the DoclingDocument as JSON for reuse without re-parsing
    @return the parsed DoclingDocument
    """
    os.makedirs(html_dir, exist_ok=True)
    os.makedirs(docling_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(input_file))[0]
    html_out = os.path.join(html_dir, f"{stem}.html")
    json_out = os.path.join(docling_dir, f"{stem}.json")

    converter = DocumentConverter()
    result = converter.convert(input_file)
    document = result.document

    document.save_as_html(html_out)
    document.save_as_json(json_out)

    return document


def chunk(input_file: str, html_dir: str = "tmp/html", docling_dir: str = "tmp/docling") -> Tuple[List[str], List[dict]]:
    """
    @param input_file HTML input file
    @return Tuple of (chunk texts, chunk metadata dicts)
    """
    document = convert(input_file, html_dir=html_dir, docling_dir=docling_dir)

    chunker = HybridChunker(tokenizer=_get_tokenizer())

    texts: List[str] = []
    metadatas: List[dict] = []
    for i, doc_chunk in enumerate(chunker.chunk(document)):
        texts.append(chunker.contextualize(doc_chunk))
        metadata = extract_legal_metadata(doc_chunk.meta.headings)
        metadata["chunk_id"] = i
        metadata["source"] = input_file
        metadata["headings"] = " > ".join(doc_chunk.meta.headings or [])
        metadatas.append(metadata)

    return texts, metadatas
