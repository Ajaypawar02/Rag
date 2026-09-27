"""Structure-aware chunking.

1. Markdown/HTML (normalized to Markdown) is first split on `#`/`##`/`###`
   headings, so a chunk never straddles two sections. The splitter is
   fence-aware: `# comment` lines inside code blocks are not headings.
2. Each section is cut into blocks (whole code blocks, prose paragraphs) that
   are packed into chunks of up to `chunk_size` characters, so code blocks
   stay intact; trailing prose overlaps into the next chunk.
3. PDFs are split per page, then recursively with `chunk_overlap`.

Each chunk's text is prefixed with a breadcrumb ("Title > Section > Sub")
so the embedding and the LLM both see where the chunk came from.

LangChain's MarkdownHeaderTextSplitter is not used because it strips leading
whitespace from every line, which destroys indentation in Python/YAML code;
its Markdown recursive splitter cuts at code fences, orphaning closing fences.
"""

import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.loaders import FENCE_RE, LoadedDocument
from app.ingestion.metadata import code_block_metadata, drop_none

MAX_HEADER_LEVEL = 3
MIN_CHUNK_CHARS = 40  # drop fragments too small to answer anything

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE_OPEN_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")


@dataclass
class Section:
    headers: dict[str, str]  # {"h1": "...", "h2": "...", "h3": "..."}
    body: str


def split_by_headers(markdown: str, max_level: int = MAX_HEADER_LEVEL) -> list[Section]:
    """Split Markdown into sections at headings up to `max_level`, preserving code verbatim."""
    sections: list[Section] = []
    headers: dict[int, str] = {}
    lines: list[str] = []
    fence: str | None = None

    def flush() -> None:
        body = "\n".join(lines).strip("\n")
        if body.strip():
            sections.append(Section({f"h{lvl}": text for lvl, text in sorted(headers.items())}, body))
        lines.clear()

    for line in markdown.split("\n"):
        if fence:
            if line.lstrip().startswith(fence):
                fence = None
            lines.append(line)
            continue
        if m := _FENCE_OPEN_RE.match(line):
            fence = m.group(1)
            lines.append(line)
            continue
        if (m := _HEADING_RE.match(line)) and len(m.group(1)) <= max_level:
            flush()
            level = len(m.group(1))
            headers = {lvl: text for lvl, text in headers.items() if lvl < level}
            headers[level] = m.group(2).strip()
            continue
        lines.append(line)
    flush()
    return sections


def chunk_document(
    doc: LoadedDocument, doc_metadata: dict, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Turn a loaded document into embeddable chunks with full metadata."""
    if doc.doc_type == "pdf":
        pieces = _split_pdf(doc.pages, chunk_size, chunk_overlap)
    else:
        pieces = _split_markdown(doc.text, doc.title, chunk_size, chunk_overlap)

    chunks: list[Document] = []
    for text, extra in pieces:
        if len(text.strip()) < MIN_CHUNK_CHARS:
            continue
        index = len(chunks)
        breadcrumb = " > ".join(x for x in (doc.title, extra.get("section")) if x)
        metadata = drop_none(
            {
                **doc_metadata,
                **extra,
                **code_block_metadata(text),
                "chunk_index": index,
                "chunk_id": f"{doc.source}::{index}",
            }
        )
        chunks.append(Document(page_content=f"{breadcrumb}\n\n{text}", metadata=metadata))
    return chunks


def _split_markdown(text: str, title: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, dict]]:
    pieces = []
    for section in split_by_headers(text):
        # The H1 usually repeats the document title; keep it out of the breadcrumb.
        path = [h for h in section.headers.values() if h.lower() != title.lower()]
        extra = {**section.headers, "section": " > ".join(path) or None}
        pieces.extend((piece, extra) for piece in pack_blocks(section.body, chunk_size, chunk_overlap))
    return pieces


def split_blocks(body: str) -> list[tuple[str, bool]]:
    """Split a section into (text, is_code) blocks: whole fenced code blocks and prose paragraphs."""
    blocks: list[tuple[str, bool]] = []

    def add_prose(prose: str) -> None:
        blocks.extend((p.strip("\n"), False) for p in re.split(r"\n[ \t]*\n", prose) if p.strip())

    last = 0
    for m in FENCE_RE.finditer(body):
        add_prose(body[last : m.start()])
        blocks.append((m.group(0).strip("\n"), True))
        last = m.end()
    add_prose(body[last:])
    return blocks


def pack_blocks(body: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily pack blocks into chunks of at most ~`chunk_size` characters.

    Code blocks are never split unless a single block exceeds `chunk_size`, in
    which case it is split on line boundaries and each part is re-fenced.
    Overlap carries trailing prose paragraphs (never code) into the next chunk.
    """
    blocks: list[tuple[str, bool]] = []
    for text, is_code in split_blocks(body):
        if len(text) <= chunk_size:
            blocks.append((text, is_code))
        elif is_code:
            blocks.extend((part, True) for part in _split_code_block(text, chunk_size))
        else:
            prose_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            blocks.extend((part, False) for part in prose_splitter.split_text(text))

    chunks: list[str] = []
    current: list[tuple[str, bool]] = []

    def size(items: list[tuple[str, bool]]) -> int:
        return sum(len(t) for t, _ in items) + 2 * max(len(items) - 1, 0)

    for block in blocks:
        if current and size(current + [block]) > chunk_size:
            chunks.append("\n\n".join(t for t, _ in current))
            carry: list[tuple[str, bool]] = []
            for prev in reversed(current):
                if prev[1] or size([prev] + carry) > chunk_overlap:
                    break
                carry.insert(0, prev)
            current = carry if size(carry + [block]) <= chunk_size else []
        current.append(block)
    if current:
        chunks.append("\n\n".join(t for t, _ in current))
    return chunks


def _split_code_block(block: str, chunk_size: int) -> list[str]:
    lines = block.split("\n")
    opening, closing = lines[0], lines[-1]
    inner = "\n".join(lines[1:-1])
    budget = max(chunk_size - len(opening) - len(closing) - 2, 100)
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""], chunk_size=budget, chunk_overlap=0, strip_whitespace=False
    )
    return [f"{opening}\n{part.strip(chr(10))}\n{closing}" for part in splitter.split_text(inner) if part.strip()]


def _split_pdf(pages: list[str], chunk_size: int, chunk_overlap: int) -> list[tuple[str, dict]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [
        (piece, {"page": page_no})
        for page_no, page_text in enumerate(pages, start=1)
        for piece in splitter.split_text(page_text)
    ]
