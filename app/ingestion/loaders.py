"""Loaders for Markdown, HTML and PDF documentation.

Markdown and HTML are both normalized to clean Markdown so they share one
header-aware chunking path. PDFs keep their page structure instead, because
extracted PDF text has no reliable heading markup.
"""

import hashlib
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import yaml
from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from pypdf import PdfReader

DocType = Literal["markdown", "html", "pdf"]

SUPPORTED_EXTENSIONS: dict[str, DocType] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".mdx": "markdown",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
}


class IngestionError(Exception):
    """Raised when a document cannot be loaded or yields no content."""


class UnsupportedFileTypeError(IngestionError):
    pass


@dataclass
class LoadedDocument:
    source: str  # file name; the document's key in the vector store
    doc_type: DocType
    title: str
    content_hash: str
    text: str = ""  # normalized Markdown (markdown / html)
    pages: list[str] = field(default_factory=list)  # page texts (pdf)
    description: str | None = None


def load_document(path: Path) -> LoadedDocument:
    """Load a supported file into a normalized `LoadedDocument`."""
    doc_type = SUPPORTED_EXTENSIONS.get(path.suffix.lower())
    if doc_type is None:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{path.suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    raw = path.read_bytes()
    content_hash = hashlib.sha256(raw).hexdigest()[:16]

    if doc_type == "markdown":
        doc = _load_markdown(raw.decode("utf-8", errors="replace"), path)
    elif doc_type == "html":
        doc = _load_html(raw.decode("utf-8", errors="replace"), path)
    else:
        doc = _load_pdf(path)
    doc.content_hash = content_hash

    if not doc.text.strip() and not any(p.strip() for p in doc.pages):
        raise IngestionError(f"No extractable text in '{path.name}'.")
    return doc


def humanize_filename(path: Path) -> str:
    """`pydantic_v2_migration_guide.md` -> `Pydantic V2 Migration Guide`."""
    return re.sub(r"[_\-]+", " ", path.stem).strip().title()


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #

# A fenced code block, matched as a whole so cleanup rules never touch code.
FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?^[ \t]*\1[ \t]*$", re.M | re.S)

_FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n", re.S)
_FENCE_INFO_RE = re.compile(r"\A([ \t]*(?:`{3,}|~{3,}))[ \t]*([\w+#.-]*)[^\n]*")
_HEADING_ANCHOR_RE = re.compile(r"^(#{1,6} .*?)\s*\{\s*#[\w-]+\s*\}[ \t]*$", re.M)
# FastAPI `/// tip | Title` ... `///`, MkDocs `!!! note "Title"` / `??? note`
_ADMONITION_RE = re.compile(r"^(?:///|!!!|\?\?\?\+?)[ \t]*(\w+)(?:[ \t]*\|[ \t]*(.*?)|[ \t]+\"(.*?)\")?[ \t]*$", re.M)
_ADMONITION_CLOSE_RE = re.compile(r"^///[ \t]*$", re.M)
_HUGO_SHORTCODE_RE = re.compile(r"\{\{[<%].*?[%>]\}\}", re.S)
_MDX_COMMENT_RE = re.compile(r"\{/\*.*?\*/\}", re.S)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_HTML_ANCHOR_RE = re.compile(r"<a\s+id=\"[^\"]*\"\s*/?>(?:</a>)?")
_JSX_TAG_RE = re.compile(r"</?[A-Z][\w.]*(?:\s[^<>]*)?/?>")  # <Info>, <Callout icon="x">, </Tip>
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def map_outside_code(text: str, fn: Callable[[str], str]) -> str:
    """Apply `fn` to every part of `text` that is not a fenced code block."""
    parts, last = [], 0
    for m in FENCE_RE.finditer(text):
        parts.append(fn(text[last : m.start()]))
        parts.append(_normalize_fence(m.group(0)))
        last = m.end()
    parts.append(fn(text[last:]))
    return "".join(parts)


def _normalize_fence(block: str) -> str:
    """Reduce a fence's info string to just the language (drops `theme={...}` etc.)."""
    return _FENCE_INFO_RE.sub(lambda m: m.group(1) + m.group(2), block, count=1)


def _clean_prose(text: str) -> str:
    text = _HEADING_ANCHOR_RE.sub(r"\1", text)
    text = _ADMONITION_RE.sub(
        lambda m: f"**{m.group(1).capitalize()}:** {m.group(2) or m.group(3) or ''}".rstrip(), text
    )
    text = _ADMONITION_CLOSE_RE.sub("", text)
    for pattern in (_HUGO_SHORTCODE_RE, _MDX_COMMENT_RE, _HTML_COMMENT_RE, _HTML_ANCHOR_RE, _JSX_TAG_RE):
        text = pattern.sub("", text)
    return text


def clean_markdown(text: str) -> str:
    """Strip doc-site tooling syntax (front matter excluded) while keeping code intact."""
    text = map_outside_code(text, _clean_prose)
    return _BLANK_LINES_RE.sub("\n\n", text).strip()


def _load_markdown(text: str, path: Path) -> LoadedDocument:
    front: dict = {}
    if m := _FRONT_MATTER_RE.match(text):
        try:
            front = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            front = {}
        text = text[m.end() :]

    text = clean_markdown(text)
    first_h1 = re.search(r"^# (.+)$", FENCE_RE.sub("", text), re.M)  # ignore `# comments` in code
    title = front.get("title") or (first_h1.group(1).strip() if first_h1 else humanize_filename(path))
    return LoadedDocument(
        source=path.name,
        doc_type="markdown",
        title=str(title),
        content_hash="",
        text=text,
        description=front.get("description"),
    )


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

# Tried in order; the first match with real content is treated as the article body.
_MAIN_CONTENT_SELECTORS = (".td-content", ".prose", "article", "[role=main]", "main")
_NOISE_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "aside", "form", "button", "svg", "iframe")
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_BLOCK_TAGS = _HEADINGS | {
    "p", "div", "section", "ul", "ol", "li", "dl", "dt", "dd", "blockquote",
    "pre", "table", "figure", "figcaption", "details", "summary", "hr", "br",
}


def _load_html(html: str, path: Path) -> LoadedDocument:
    soup = BeautifulSoup(html, "html.parser")
    title = _html_title(soup) or humanize_filename(path)
    desc_tag = soup.find("meta", attrs={"name": "description"})
    description = desc_tag.get("content") if desc_tag else None

    root = _main_content(soup)
    for tag in root.find_all(_NOISE_TAGS):
        tag.decompose()
    text = _BLANK_LINES_RE.sub("\n\n", html_to_markdown(root)).strip()
    return LoadedDocument(
        source=path.name, doc_type="html", title=title, content_hash="", text=text, description=description
    )


def _html_title(soup: BeautifulSoup) -> str | None:
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return " ".join(og["content"].split())
    if soup.title and soup.title.string:
        return " ".join(soup.title.string.split("|")[0].split())
    h1 = soup.find("h1")
    return h1.get_text(" ", strip=True) if h1 else None


def _main_content(soup: BeautifulSoup) -> Tag:
    for selector in _MAIN_CONTENT_SELECTORS:
        el = soup.select_one(selector)
        if el and len(el.get_text(strip=True)) > 200:
            return el
    return soup.body or soup


def html_to_markdown(root: Tag) -> str:
    """Convert an HTML subtree to Markdown: headings, paragraphs, lists, tables and code."""
    out: list[str] = []
    _walk(root, out)
    return "\n\n".join(out)


def _walk(node: Tag, out: list[str]) -> None:
    inline: list[str] = []

    def flush() -> None:
        text = re.sub(r"\s+", " ", " ".join(inline)).strip()
        if text:
            out.append(text)
        inline.clear()

    for child in node.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            inline.append(str(child))
            continue
        if not isinstance(child, Tag):
            continue
        if child.name not in _BLOCK_TAGS:
            inline.append(child.get_text(" "))
            continue

        flush()
        if child.name in _HEADINGS:
            if heading := child.get_text(" ", strip=True):
                out.append(f"{'#' * int(child.name[1])} {heading}")
        elif child.name == "pre":
            out.append(f"```{_code_language(child)}\n{child.get_text().strip(chr(10))}\n```")
        elif child.name == "table":
            out.append(_table_to_text(child))
        elif child.name == "li":
            start = len(out)
            _walk(child, out)
            if len(out) > start:
                out[start] = f"- {out[start]}"
        else:
            _walk(child, out)
    flush()


def _code_language(pre: Tag) -> str:
    for el in (pre.find("code"), pre):
        if el is None:
            continue
        if lang := el.get("data-lang"):
            return lang
        for cls in el.get("class", []):
            if cls.startswith(("language-", "lang-")):
                return cls.split("-", 1)[1]
    return ""


def _table_to_text(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #

def _load_pdf(path: Path) -> LoadedDocument:
    try:
        reader = PdfReader(path)
    except Exception as exc:  # pypdf raises several unrelated types for corrupt files
        raise IngestionError(f"Could not read PDF '{path.name}': {exc}") from exc

    # Layout mode keeps code blocks and their indentation; the default mode drops
    # text drawn in some monospace fonts (e.g. code in browser-printed docs).
    pages = [_clean_pdf_page(page.extract_text(extraction_mode="layout") or "") for page in reader.pages]
    meta_title = (reader.metadata.title if reader.metadata else None) or ""
    return LoadedDocument(
        source=path.name,
        doc_type="pdf",
        title=" ".join(meta_title.split()) or humanize_filename(path),
        content_hash="",
        pages=pages,
    )


_INNER_SPACES_RE = re.compile(r"(?<=\S) {2,}")


def _clean_pdf_page(text: str) -> str:
    """Undo layout-mode padding: shared left margin and column-alignment gaps, keeping indentation."""
    lines = [_INNER_SPACES_RE.sub(" ", line.rstrip()) for line in textwrap.dedent(text).split("\n")]
    return _BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()
