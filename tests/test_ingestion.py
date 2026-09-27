"""Tests for loaders, metadata extraction, chunking and keyword filtering (no services needed)."""

from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from app.ingestion.chunking import chunk_document, pack_blocks, split_by_headers
from app.ingestion.loaders import (
    LoadedDocument,
    UnsupportedFileTypeError,
    clean_markdown,
    html_to_markdown,
    load_document,
)
from app.ingestion.metadata import code_block_metadata, detect_category, detect_product, detect_version
from app.memory.keyword_index import matches_filter, tokenize

SAMPLE_DOCS = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


# --- Markdown cleaning ------------------------------------------------------

def test_clean_markdown_strips_site_syntax_but_not_code():
    md = (
        "# Title { #title }\n\n"
        "/// tip | Pro tip\nUse it.\n///\n\n"
        "{{< note >}}\n<Info>Shown</Info>\n"
        "```python theme={\"x\": 1}\n<Info> stays in code\n```\n"
    )
    out = clean_markdown(md)
    assert "# Title\n" in out
    assert "**Tip:** Pro tip" in out and "///" not in out
    assert "{{<" not in out and "Shown" in out and "<Info>Shown" not in out
    assert "```python\n<Info> stays in code\n```" in out


def test_load_markdown_reads_front_matter_title(tmp_path):
    f = tmp_path / "guide.md"
    f.write_text("---\ntitle: My Guide\ndescription: About it\n---\n\nBody text here.\n")
    doc = load_document(f)
    assert doc.title == "My Guide"
    assert doc.description == "About it"
    assert doc.text == "Body text here."


def test_unsupported_extension(tmp_path):
    f = tmp_path / "notes.docx"
    f.write_bytes(b"x")
    with pytest.raises(UnsupportedFileTypeError):
        load_document(f)


# --- HTML -------------------------------------------------------------------

def test_html_to_markdown_keeps_structure():
    html = """<div>
      <h2>Install</h2><p>Run <code>pip</code> now.</p>
      <ul><li>First</li><li>Second</li></ul>
      <pre><code class="language-bash">pip install x
  --upgrade</code></pre>
      <table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>
    </div>"""
    md = html_to_markdown(BeautifulSoup(html, "html.parser"))
    assert "## Install" in md
    assert "Run pip now." in md
    assert "- First" in md and "- Second" in md
    assert "```bash\npip install x\n  --upgrade\n```" in md
    assert "A | B\n1 | 2" in md


# --- Chunking ---------------------------------------------------------------

def test_split_by_headers_ignores_comments_in_code():
    md = "# Doc\nintro\n## Part\n```python\n# not a heading\nx = 1\n```\n### Sub\ndetail"
    sections = split_by_headers(md)
    assert [s.headers for s in sections] == [
        {"h1": "Doc"},
        {"h1": "Doc", "h2": "Part"},
        {"h1": "Doc", "h2": "Part", "h3": "Sub"},
    ]
    assert "# not a heading" in sections[1].body


def test_pack_blocks_keeps_code_blocks_whole():
    code = "```python\ndef f():\n    return 1\n```"
    body = "\n\n".join(["para " * 30, code, "para " * 30])
    chunks = pack_blocks(body, chunk_size=200, chunk_overlap=0)
    assert any(code in c for c in chunks)
    assert all(c.count("```") % 2 == 0 for c in chunks)


def test_pack_blocks_refences_oversized_code():
    code = "```yaml\n" + "\n".join(f"key{i}: value{i}" for i in range(100)) + "\n```"
    chunks = pack_blocks(code, chunk_size=300, chunk_overlap=0)
    assert len(chunks) > 1
    assert all(c.startswith("```yaml\n") and c.endswith("\n```") for c in chunks)


def test_chunk_document_adds_breadcrumb_and_metadata():
    doc = LoadedDocument(
        source="d.md", doc_type="markdown", title="Doc", content_hash="h",
        text="# Doc\n## Setup\nInstall the package with pip before using it.\n\n```bash\npip install x\n```",
    )
    chunks = chunk_document(doc, {"source": "d.md"}, chunk_size=1000, chunk_overlap=100)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.page_content.startswith("Doc > Setup\n\n")
    assert c.metadata["section"] == "Setup"
    assert c.metadata["has_code"] is True and c.metadata["code_languages"] == "bash"
    assert c.metadata["chunk_id"] == "d.md::0"


# --- Metadata ---------------------------------------------------------------

@pytest.mark.parametrize(
    "source,title,expected",
    [
        ("python_3_12_whats_new.pdf", "What's New In Python 3.12", "3.12"),
        ("postgresql_16_index_types.pdf", "PostgreSQL: Documentation: 16: Index Types", "16"),
        ("pydantic_v2_migration_guide.md", "Pydantic V2 Migration Guide", "2"),
        ("kubernetes_deployments.html", "Deployments", None),
        ("report_2024.md", "Annual report", None),
    ],
)
def test_detect_version(source, title, expected):
    assert detect_version(source, title) == expected


def test_detect_product_and_category():
    assert detect_product("fastapi_first_steps.md", "First Steps") == "fastapi"
    assert detect_category("fastapi_first_steps.md", "First Steps") == "tutorial"
    assert detect_category("pydantic_v2_migration_guide.md", "") == "migration_guide"
    assert detect_category("notes.md", "Architecture") == "concept"


def test_code_block_metadata():
    meta = code_block_metadata("text\n```python\nx\n```\n```yaml\ny: 1\n```")
    assert meta == {"has_code": True, "code_block_count": 2, "code_languages": "python,yaml"}


# --- Keyword index helpers --------------------------------------------------

def test_tokenize_keeps_identifiers_and_drops_stopwords():
    assert tokenize("How do I use model_validate in V2?") == ["use", "model_validate", "v2"]


def test_tokenize_stems_word_forms_consistently():
    assert tokenize("suppression") == tokenize("suppress") == ["suppress"]
    assert tokenize("predictions") == tokenize("prediction") == ["predict"]
    assert tokenize("deployments") == tokenize("deployment") == ["deployment"]
    assert tokenize("dependencies") == ["dependency"]
    assert tokenize("uses") == ["uses"]  # too short to stem


def test_matches_filter():
    meta = {"product": "pydantic", "version": "2", "has_code": True}
    assert matches_filter(meta, None)
    assert matches_filter(meta, {"product": "pydantic"})
    assert matches_filter(meta, {"$and": [{"product": "pydantic"}, {"version": {"$in": ["1", "2"]}}]})
    assert matches_filter(meta, {"$or": [{"product": "redis"}, {"has_code": True}]})
    assert not matches_filter(meta, {"version": {"$ne": "2"}})


# --- Sample corpus smoke test -----------------------------------------------

@pytest.mark.skipif(not SAMPLE_DOCS.exists(), reason="sample docs not present")
def test_all_sample_docs_load_and_chunk():
    for path in SAMPLE_DOCS.iterdir():
        doc = load_document(path)
        chunks = chunk_document(doc, {"source": doc.source}, chunk_size=1000, chunk_overlap=150)
        assert chunks, path.name
        assert all(c.page_content.count("```") % 2 == 0 for c in chunks), path.name
