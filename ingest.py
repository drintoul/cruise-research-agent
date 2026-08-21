"""Periodic local-document ingestion for the travel knowledge base.

Supported files under DOCUMENTS_DIR are discovered recursively:
- PDF: text extracted with pypdf, with an OCR fallback for image-only/scanned pages.
- HTML/HTM: main content extracted locally with BeautifulSoup, preserving
  section/heading provenance.

Every source file is hashed with SHA-256 before extraction. Qdrant payload
metadata acts as the ingestion manifest:

- same document_hash already present -> skip
- same source_path with a different hash -> ingest replacement, then delete
  stale chunks for the prior version
- new hash/path -> ingest

The worker is intentionally separate from Chainlit so document maintenance does
not depend on active chat sessions.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup, Tag
from pypdf import PdfReader
from pdf2image import convert_from_path
from pytesseract import image_to_string
from qdrant_client import models

from tools import QDRANT_COLLECTION, qdrant, upsert_knowledge_chunks


SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm"}
ALLOWED_PROVIDERS = {
    "Princess Cruises",
    "Royal Caribbean",
}
HTML_NOISE_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "header",
    "aside",
    "svg",
    "form",
    "button",
    "iframe",
    "canvas",
)
HTML_CONTENT_TAGS = (
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "li",
    "pre",
    "blockquote",
    "table",
)

DOCUMENTS_DIR = Path(os.environ["DOCUMENTS_DIR"])
INGEST_SCAN_INTERVAL_SECONDS = max(
    10, int(os.environ["INGEST_SCAN_INTERVAL_SECONDS"])
)
DOCUMENT_CHUNK_SIZE = max(500, int(os.environ["DOCUMENT_CHUNK_SIZE"]))
DOCUMENT_CHUNK_OVERLAP = max(0, int(os.environ["DOCUMENT_CHUNK_OVERLAP"]))
INGEST_HEARTBEAT_FILE = Path(os.environ["INGEST_HEARTBEAT_FILE"])
INGEST_HEALTH_MAX_AGE_SECONDS = max(
    30, int(os.environ["INGEST_HEALTH_MAX_AGE_SECONDS"])
)

if DOCUMENT_CHUNK_OVERLAP >= DOCUMENT_CHUNK_SIZE:
    raise ValueError("DOCUMENT_CHUNK_OVERLAP must be smaller than DOCUMENT_CHUNK_SIZE")

logging.basicConfig(
    level=os.environ["LOG_LEVEL"].upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("document-ingester")


@dataclass
class ScanStats:
    discovered: int = 0
    eligible: int = 0
    ignored_provider: int = 0
    ingested: int = 0
    skipped: int = 0
    failed: int = 0
    chunks: int = 0


@dataclass
class HtmlSection:
    heading_path: list[str]
    text: str

    @property
    def section_title(self) -> str | None:
        return self.heading_path[-1] if self.heading_path else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_long_text(text: str, size: int, overlap: int) -> Iterable[str]:
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = max(
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            )
            if boundary > start + size // 2:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            yield chunk

        if end >= len(text):
            break
        start = max(start + 1, end - overlap)


def chunk_text(text: str) -> list[str]:
    """Paragraph-aware chunking shared by PDF and HTML ingestion."""
    text = _normalize_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > DOCUMENT_CHUNK_SIZE:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(
                _split_long_text(
                    paragraph,
                    size=DOCUMENT_CHUNK_SIZE,
                    overlap=DOCUMENT_CHUNK_OVERLAP,
                )
            )
            continue

        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= DOCUMENT_CHUNK_SIZE:
            current = candidate
            continue

        chunks.append(current.strip())
        tail = (
            current[-DOCUMENT_CHUNK_OVERLAP:].strip()
            if DOCUMENT_CHUNK_OVERLAP
            else ""
        )
        current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph

    if current:
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]


async def _collection_exists() -> bool:
    collections = await qdrant.get_collections()
    return any(item.name == QDRANT_COLLECTION for item in collections.collections)


async def _find_by_payload(field: str, value: str, limit: int = 1):
    if not await _collection_exists():
        return []

    points, _ = await qdrant.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key=field,
                    match=models.MatchValue(value=value),
                )
            ]
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return points


async def document_hash_exists(document_hash: str) -> bool:
    return bool(await _find_by_payload("document_hash", document_hash, limit=1))


async def source_path_exists(source_path: str) -> bool:
    return bool(await _find_by_payload("source_path", source_path, limit=1))


async def ensure_payload_indexes() -> None:
    """Create keyword indexes used by ingestion/retrieval filters when absent."""
    if not await _collection_exists():
        return

    info = await qdrant.get_collection(QDRANT_COLLECTION)
    existing = set((info.payload_schema or {}).keys())
    for field in (
        "document_hash",
        "source_path",
        "document_id",
        "provider",
        "source_type",
    ):
        if field in existing:
            continue
        await qdrant.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )


async def delete_old_versions(source_path: str, current_hash: str) -> None:
    """Delete stale chunks only after the replacement version was upserted."""
    if not await _collection_exists():
        return

    await qdrant.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_path",
                        match=models.MatchValue(value=source_path),
                    )
                ],
                must_not=[
                    models.FieldCondition(
                        key="document_hash",
                        match=models.MatchValue(value=current_hash),
                    )
                ],
            )
        ),
        wait=True,
    )


def _provider_from_relative_path(relative_path: Path) -> str | None:
    """Return the top-level provider directory for a document."""
    if len(relative_path.parts) > 1:
        return relative_path.parts[0]
    return None


def _common_file_metadata(path: Path, document_hash: str) -> dict:
    relative_path = path.relative_to(DOCUMENTS_DIR)
    stat = path.stat()
    file_modified_at = datetime.fromtimestamp(
        stat.st_mtime, tz=timezone.utc
    ).isoformat()

    return {
        "provider": _provider_from_relative_path(relative_path),
        "source_path": relative_path.as_posix(),
        "source_url": None,
        "document_id": str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"local-document:{relative_path.as_posix()}")
        ),
        "document_hash": document_hash,
        "hash_algorithm": "sha256",
        "file_name": path.name,
        "file_size_bytes": stat.st_size,
        "file_modified_at": file_modified_at,
        "updated_at": file_modified_at,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def extract_pdf_chunks(path: Path, document_hash: str) -> list[dict]:
    common = _common_file_metadata(path, document_hash)
    source_path = common["source_path"]
    reader = PdfReader(str(path))

    pdf_title = None
    try:
        metadata = reader.metadata
        if metadata and metadata.title:
            pdf_title = str(metadata.title).strip() or None
    except Exception:
        logger.debug("Could not read PDF metadata for %s", source_path, exc_info=True)

    title = pdf_title or path.stem
    records: list[dict] = []
    global_chunk_index = 0

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            logger.warning(
                "Unable to extract page %s from %s",
                page_index,
                source_path,
                exc_info=True,
            )
            page_text = ""

        if not page_text.strip():
            try:
                image = convert_from_path(
                    str(path),
                    first_page=page_index,
                    last_page=page_index,
                    dpi=200,
                )[0]
                page_text = image_to_string(image) or ""
                if page_text.strip():
                    logger.info(
                        "OCR extracted text from page %s of %s",
                        page_index,
                        source_path,
                    )
            except Exception:
                logger.warning(
                    "OCR failed for page %s of %s",
                    page_index,
                    source_path,
                    exc_info=True,
                )

        if not page_text:
            continue

        for page_chunk_index, text in enumerate(chunk_text(page_text)):
            records.append(
                {
                    **common,
                    "text": text,
                    "source_type": "pdf",
                    "mime_type": "application/pdf",
                    "title": title,
                    "chunk_id": (
                        f"{common['document_id']}:p{page_index}:c{page_chunk_index}"
                    ),
                    "chunk_index": global_chunk_index,
                    "page_number": page_index,
                    "page_chunk_index": page_chunk_index,
                    "page_count": len(reader.pages),
                    "section_title": None,
                    "heading_path": [],
                }
            )
            global_chunk_index += 1

    return records


def _html_table_text(table: Tag) -> str:
    rows: list[str] = []
    for row in table.find_all("tr"):
        cells = [
            _normalize_text(cell.get_text(" ", strip=True))
            for cell in row.find_all(["th", "td"])
        ]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _html_main_root(soup: BeautifulSoup) -> Tag | BeautifulSoup:
    return (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.body
        or soup
    )


def _extract_html_sections(root: Tag | BeautifulSoup) -> list[HtmlSection]:
    for element in list(root.find_all(HTML_NOISE_TAGS)):
        element.decompose()

    heading_by_level: dict[int, str] = {}
    current_blocks: list[str] = []
    current_heading_path: list[str] = []
    sections: list[HtmlSection] = []
    previous_block: str | None = None

    def flush() -> None:
        nonlocal current_blocks, current_heading_path
        text = _normalize_text("\n\n".join(current_blocks))
        if text:
            sections.append(
                HtmlSection(
                    heading_path=list(current_heading_path),
                    text=text,
                )
            )
        current_blocks = []

    for element in root.find_all(HTML_CONTENT_TAGS):
        if not isinstance(element, Tag):
            continue

        # Avoid duplicate text from nested block structures.
        if element.name != "table" and element.find_parent("table") is not None:
            continue
        if element.name == "p" and element.find_parent("li") is not None:
            continue

        if element.name and re.fullmatch(r"h[1-6]", element.name):
            heading = _normalize_text(element.get_text(" ", strip=True))
            if not heading:
                continue

            flush()
            level = int(element.name[1])
            heading_by_level[level] = heading
            for stale_level in [key for key in heading_by_level if key > level]:
                del heading_by_level[stale_level]
            current_heading_path = [
                heading_by_level[key] for key in sorted(heading_by_level)
            ]
            previous_block = None
            continue

        if element.name == "table":
            block = _html_table_text(element)
        else:
            block = _normalize_text(element.get_text(" ", strip=True))

        if not block or block == previous_block:
            continue
        previous_block = block
        current_blocks.append(block)

    flush()
    return sections


def extract_html_chunks(path: Path, document_hash: str) -> list[dict]:
    common = _common_file_metadata(path, document_hash)
    raw = path.read_bytes()
    soup = BeautifulSoup(raw, "html.parser")

    html_title = None
    if soup.title:
        html_title = _normalize_text(soup.title.get_text(" ", strip=True)) or None
    if not html_title:
        first_h1 = soup.find("h1")
        if first_h1:
            html_title = _normalize_text(first_h1.get_text(" ", strip=True)) or None

    canonical = soup.find("link", rel=lambda value: value and "canonical" in value)
    if canonical and canonical.get("href"):
        common["source_url"] = str(canonical.get("href")).strip() or None
    else:
        og_url = soup.find("meta", attrs={"property": "og:url"})
        if og_url and og_url.get("content"):
            common["source_url"] = str(og_url.get("content")).strip() or None

    title = html_title or path.stem
    root = _html_main_root(soup)
    sections = _extract_html_sections(root)

    records: list[dict] = []
    global_chunk_index = 0

    for section_index, section in enumerate(sections):
        heading_prefix = (
            " > ".join(section.heading_path) if section.heading_path else ""
        )
        text_for_chunking = (
            f"{heading_prefix}\n\n{section.text}" if heading_prefix else section.text
        )

        for section_chunk_index, text in enumerate(chunk_text(text_for_chunking)):
            records.append(
                {
                    **common,
                    "text": text,
                    "source_type": "html",
                    "mime_type": "text/html",
                    "title": title,
                    "chunk_id": (
                        f"{common['document_id']}:s{section_index}:c{section_chunk_index}"
                    ),
                    "chunk_index": global_chunk_index,
                    "page_number": None,
                    "page_chunk_index": None,
                    "page_count": None,
                    "section_index": section_index,
                    "section_chunk_index": section_chunk_index,
                    "section_title": section.section_title,
                    "heading_path": section.heading_path,
                }
            )
            global_chunk_index += 1

    return records


def extract_document_chunks(path: Path, document_hash: str) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_chunks(path, document_hash)
    if suffix in {".html", ".htm"}:
        return extract_html_chunks(path, document_hash)
    raise ValueError(f"Unsupported document type: {suffix}")


async def ingest_document(path: Path) -> tuple[str, int]:
    """Return (status, chunk_count), where status is ingested/skipped."""
    relative = path.relative_to(DOCUMENTS_DIR)
    provider = _provider_from_relative_path(relative)
    if provider not in ALLOWED_PROVIDERS:
        raise ValueError(
            f"Provider {provider or '<root>'!r} is not allowed for ingestion"
        )

    relative_path = relative.as_posix()
    document_hash = await asyncio.to_thread(sha256_file, path)

    if await document_hash_exists(document_hash):
        logger.info(
            "SKIP unchanged/duplicate document: %s [%s]",
            relative_path,
            document_hash[:12],
        )
        return "skipped", 0

    replacing_existing_path = await source_path_exists(relative_path)
    chunks = await asyncio.to_thread(extract_document_chunks, path, document_hash)

    if not chunks:
        if path.suffix.lower() == ".pdf":
            raise ValueError(
                "No extractable text found. The PDF may be image-only/scanned "
                "and require OCR."
            )
        raise ValueError("No main-content text could be extracted from the HTML file.")

    count = await upsert_knowledge_chunks(chunks)
    await ensure_payload_indexes()

    # Safe update ordering: the replacement exists before stale points are removed.
    if replacing_existing_path:
        await delete_old_versions(relative_path, document_hash)

    logger.info(
        "INGESTED %s: %s chunks [%s]%s",
        relative_path,
        count,
        document_hash[:12],
        " (replaced previous version)" if replacing_existing_path else "",
    )
    return "ingested", count


def write_heartbeat() -> None:
    INGEST_HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
    INGEST_HEARTBEAT_FILE.write_text(str(time.time()), encoding="utf-8")


def check_health() -> int:
    if not INGEST_HEARTBEAT_FILE.exists():
        print("Document ingester heartbeat does not exist.")
        return 1

    try:
        heartbeat = float(INGEST_HEARTBEAT_FILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        print("Document ingester heartbeat is invalid.")
        return 1

    age = time.time() - heartbeat
    if age > INGEST_HEALTH_MAX_AGE_SECONDS:
        print(
            "Document ingester heartbeat is stale: "
            f"{age:.0f}s old; maximum={INGEST_HEALTH_MAX_AGE_SECONDS}s"
        )
        return 1

    print(f"Document ingester healthy: heartbeat age={age:.0f}s")
    return 0


async def scan_once() -> ScanStats:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        path
        for path in DOCUMENTS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    documents: list[Path] = []
    ignored_by_provider: dict[str, int] = {}

    for path in candidates:
        relative_path = path.relative_to(DOCUMENTS_DIR)
        provider = _provider_from_relative_path(relative_path)

        if provider not in ALLOWED_PROVIDERS:
            provider_label = provider or "<root>"
            ignored_by_provider[provider_label] = (
                ignored_by_provider.get(provider_label, 0) + 1
            )
            continue

        documents.append(path)

    stats = ScanStats(
        discovered=len(candidates),
        eligible=len(documents),
        ignored_provider=len(candidates) - len(documents),
    )

    logger.info(
        "Scanning %s: discovered=%s eligible=%s ignored_provider=%s "
        "supported_types=(.pdf,.html,.htm) allowed_providers=%s",
        DOCUMENTS_DIR,
        stats.discovered,
        stats.eligible,
        stats.ignored_provider,
        ", ".join(sorted(ALLOWED_PROVIDERS)),
    )

    for provider, count in sorted(ignored_by_provider.items()):
        logger.warning(
            "Ignoring %s supported document(s) under disallowed provider %r",
            count,
            provider,
        )

    for path in documents:
        try:
            status, count = await ingest_document(path)
            if status == "skipped":
                stats.skipped += 1
            else:
                stats.ingested += 1
                stats.chunks += count
        except Exception:
            stats.failed += 1
            logger.exception("Failed to ingest %s", path)

    logger.info(
        "Scan complete: discovered=%s eligible=%s ignored_provider=%s "
        "ingested=%s skipped=%s failed=%s chunks=%s",
        stats.discovered,
        stats.eligible,
        stats.ignored_provider,
        stats.ingested,
        stats.skipped,
        stats.failed,
        stats.chunks,
    )
    write_heartbeat()
    return stats


async def watch() -> None:
    logger.info(
        "Watching %s every %s seconds for PDF/HTML/HTM documents",
        DOCUMENTS_DIR,
        INGEST_SCAN_INTERVAL_SECONDS,
    )
    while True:
        try:
            await scan_once()
        except Exception:
            # A catastrophic scan failure does not refresh the heartbeat, so the
            # Docker health check will eventually become unhealthy.
            logger.exception("Document scan failed; will retry on next interval")
        await asyncio.sleep(INGEST_SCAN_INTERVAL_SECONDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest PDF, HTML, and HTM files into Qdrant"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Scan once and exit")
    mode.add_argument("--watch", action="store_true", help="Continuously poll")
    mode.add_argument(
        "--healthcheck",
        action="store_true",
        help="Check watcher heartbeat and exit",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    if args.healthcheck:
        sys.exit(check_health())
    if args.once:
        await scan_once()
    else:
        await watch()


if __name__ == "__main__":
    asyncio.run(async_main())
