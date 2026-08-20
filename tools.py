"""Research tools for the Basic Agentic Research Agent.

Tool responsibilities are intentionally separated:
- Qdrant: proprietary/curated knowledge-base retrieval (RAG)
- SearXNG: web source discovery
- Firecrawl: scrape or bounded crawl of selected public sources

The agent should never treat SearXNG snippets as authoritative evidence when the
underlying page can be scraped.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import os
import socket
import uuid
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from qdrant_client import AsyncQdrantClient, models


OLLAMA_BASE_URL = os.environ["OLLAMA_BASE_URL"].rstrip("/")
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]

QDRANT_URL = os.environ["QDRANT_URL"].rstrip("/")
QDRANT_COLLECTION = os.environ["QDRANT_COLLECTION"]
RAG_TOP_K = int(os.environ["RAG_TOP_K"])

SEARXNG_BASE_URL = os.environ["SEARXNG_BASE_URL"].rstrip("/")
SEARXNG_MAX_RESULTS = int(os.environ["SEARXNG_MAX_RESULTS"])
SEARXNG_TIMEOUT_SECONDS = float(os.environ["SEARXNG_TIMEOUT_SECONDS"])

FIRECRAWL_BASE_URL = os.environ["FIRECRAWL_BASE_URL"].rstrip("/")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY", "").strip()
FIRECRAWL_TIMEOUT_SECONDS = float(os.environ["FIRECRAWL_TIMEOUT_SECONDS"])
FIRECRAWL_CRAWL_TIMEOUT_SECONDS = float(
    os.environ["FIRECRAWL_CRAWL_TIMEOUT_SECONDS"]
)
MAX_CRAWL_PAGES = int(os.environ["MAX_CRAWL_PAGES"])
MAX_CRAWL_DEPTH = int(os.environ["MAX_CRAWL_DEPTH"])
MAX_TOOL_OUTPUT_CHARS = int(os.environ["MAX_TOOL_OUTPUT_CHARS"])


qdrant = AsyncQdrantClient(url=QDRANT_URL, timeout=20)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def _json(data: Any) -> str:
    """Serialize tool output consistently."""
    text = json.dumps(data, ensure_ascii=False, default=str, indent=2)
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text
    return text[:MAX_TOOL_OUTPUT_CHARS] + "\n... [tool output truncated]"


def _firecrawl_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if FIRECRAWL_API_KEY:
        headers["Authorization"] = f"Bearer {FIRECRAWL_API_KEY}"
    return headers


async def _validate_public_url(url: str) -> str:
    """Reject non-HTTP and private-network targets to reduce SSRF risk."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only absolute http:// or https:// URLs are allowed.")

    hostname = parsed.hostname.strip("[]").lower()
    if hostname in {"localhost", "localhost.localdomain"}:
        raise ValueError("Localhost URLs are not allowed.")

    addresses: set[str] = set()
    try:
        addresses.add(str(ipaddress.ip_address(hostname)))
    except ValueError:
        try:
            info = await asyncio.to_thread(socket.getaddrinfo, hostname, None)
        except socket.gaierror as exc:
            raise ValueError(f"Could not resolve hostname: {hostname}") from exc
        for row in info:
            addresses.add(row[4][0].split("%", 1)[0])

    for address in addresses:
        ip = ipaddress.ip_address(address)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError(f"Private or non-public target is not allowed: {address}")

    return url


async def _ensure_collection(vector_size: int) -> None:
    collections = await qdrant.get_collections()
    if any(item.name == QDRANT_COLLECTION for item in collections.collections):
        return

    await qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


@tool
async def search_knowledge_base(
    query: str,
    provider: str = "",
    limit: int = RAG_TOP_K,
) -> str:
    """Search the private Qdrant travel knowledge base.

    Use this for curated or proprietary travel information. The optional
    provider can narrow retrieval. This public demonstration knowledge base
    contains Princess Cruises and Royal Caribbean content only. Do not imply
    that other vendors are represented in local RAG, and do not invent
    information when the knowledge base returns no relevant chunks.

    Args:
        query: Natural-language retrieval query.
        provider: Optional provider/brand filter.
        limit: Maximum chunks to return; clamped to 1-10.
    """
    limit = max(1, min(int(limit), 10))
    vector = await embeddings.aembed_query(query)
    await _ensure_collection(len(vector))

    query_filter = None
    if provider.strip():
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="provider",
                    match=models.MatchValue(value=provider.strip()),
                )
            ]
        )

    response = await qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    results: list[dict[str, Any]] = []
    for point in response.points:
        payload = point.payload or {}
        text = (
            payload.get("text")
            or payload.get("content")
            or payload.get("page_content")
            or ""
        )
        results.append(
            {
                "score": round(float(point.score), 5) if point.score is not None else None,
                "provider": payload.get("provider"),
                "title": payload.get("title"),
                "source_url": payload.get("source_url"),
                "document_id": payload.get("document_id"),
                "document_hash": payload.get("document_hash"),
                "source_path": payload.get("source_path"),
                "source_type": payload.get("source_type"),
                "mime_type": payload.get("mime_type"),
                "page_number": payload.get("page_number"),
                "section_title": payload.get("section_title"),
                "heading_path": payload.get("heading_path"),
                "chunk_id": payload.get("chunk_id"),
                "updated_at": payload.get("updated_at"),
                "text": text,
            }
        )

    if not results:
        return _json(
            {
                "query": query,
                "provider": provider or None,
                "results": [],
                "message": "No matching knowledge-base content was found.",
            }
        )

    return _json(
        {
            "query": query,
            "provider": provider or None,
            "citation_instruction": (
                "When using these results, cite the knowledge-base title and "
                "source_url when available."
            ),
            "results": results,
        }
    )


@tool
async def search_web(
    query: str,
    time_range: str = "",
    limit: int = SEARXNG_MAX_RESULTS,
) -> str:
    """Discover public web sources with SearXNG.

    Search results are candidate sources, not authoritative evidence. For
    factual claims, normally follow a promising result with scrape_url.

    Args:
        query: Focused search query. Domain-restricted queries such as
            "site:example.com topic" are encouraged for authoritative sources.
        time_range: Optional SearXNG range: day, month, or year.
        limit: Number of results; clamped to 1-15.
    """
    limit = max(1, min(int(limit), 15))
    params: dict[str, str] = {
        "q": query,
        "format": "json",
        "language": "en",
    }
    if time_range in {"day", "month", "year"}:
        params["time_range"] = time_range

    async with httpx.AsyncClient(timeout=SEARXNG_TIMEOUT_SECONDS) as client:
        response = await client.get(f"{SEARXNG_BASE_URL}/search", params=params)
        response.raise_for_status()
        payload = response.json()

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload.get("results", []):
        url = item.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        results.append(
            {
                "title": item.get("title"),
                "url": url,
                "snippet": item.get("content"),
                "engine": item.get("engine"),
                "engines": item.get("engines"),
                "score": item.get("score"),
                "published_date": item.get("publishedDate")
                or item.get("published_date"),
            }
        )
        if len(results) >= limit:
            break

    return _json(
        {
            "query": query,
            "warning": (
                "These are discovery results. Search snippets should not normally "
                "be cited as evidence; scrape selected source URLs first."
            ),
            "results": results,
        }
    )


@tool
async def scrape_url(url: str) -> str:
    """Scrape one selected public webpage with Firecrawl.

    Use after search_web has identified a relevant page, or when the exact
    authoritative URL is already known.

    Args:
        url: Public http(s) URL to scrape.
    """
    url = await _validate_public_url(url)
    payload = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": True,
    }

    async with httpx.AsyncClient(timeout=FIRECRAWL_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{FIRECRAWL_BASE_URL}/v2/scrape",
            headers=_firecrawl_headers(),
            json=payload,
        )
        response.raise_for_status()
        body = response.json()

    data = body.get("data") or {}
    metadata = data.get("metadata") or {}
    markdown = data.get("markdown") or ""

    # Strip markdown link targets so the model cannot accidentally cite relative
    # or broken links that appear inside the page body instead of the scraped URL.
    markdown = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", markdown)

    return _json(
        {
            "success": body.get("success", True),
            "url": metadata.get("sourceURL")
            or metadata.get("sourceUrl")
            or url,
            "title": metadata.get("title"),
            "description": metadata.get("description"),
            "published_at": metadata.get("publishedTime")
            or metadata.get("published_at"),
            "citation": (
                "Cite this page using the exact URL above. Do not use any URLs "
                "that only appear inside the page body unless scrape_url was also "
                "called on that exact URL."
            ),
            "markdown": markdown,
        }
    )


@tool
async def crawl_site(
    url: str,
    max_pages: int = 6,
    max_depth: int = 1,
) -> str:
    """Run a small bounded Firecrawl crawl of a public site section.

    Prefer scrape_url for known pages. Use crawling only when the requested
    evidence is genuinely distributed across multiple related pages.

    Args:
        url: Public starting URL.
        max_pages: Requested page limit, clamped to configured maximum.
        max_depth: Requested discovery depth, clamped to configured maximum.
    """
    url = await _validate_public_url(url)
    max_pages = max(1, min(int(max_pages), MAX_CRAWL_PAGES))
    max_depth = max(0, min(int(max_depth), MAX_CRAWL_DEPTH))

    request_body = {
        "url": url,
        "limit": max_pages,
        "maxDiscoveryDepth": max_depth,
        "crawlEntireDomain": False,
        "allowExternalLinks": False,
        "allowSubdomains": False,
        "ignoreQueryParameters": True,
        "scrapeOptions": {
            "formats": ["markdown"],
            "onlyMainContent": True,
        },
    }

    async with httpx.AsyncClient(timeout=FIRECRAWL_TIMEOUT_SECONDS) as client:
        start_response = await client.post(
            f"{FIRECRAWL_BASE_URL}/v2/crawl",
            headers=_firecrawl_headers(),
            json=request_body,
        )
        start_response.raise_for_status()
        start_data = start_response.json()

        job_id = start_data.get("id")
        if not job_id:
            return _json(
                {
                    "success": False,
                    "error": "Firecrawl did not return a crawl job id.",
                    "response": start_data,
                }
            )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + FIRECRAWL_CRAWL_TIMEOUT_SECONDS
        status_data: dict[str, Any] = {}

        while loop.time() < deadline:
            status_response = await client.get(
                f"{FIRECRAWL_BASE_URL}/v2/crawl/{job_id}",
                headers=_firecrawl_headers(),
            )
            status_response.raise_for_status()
            status_data = status_response.json()
            status = status_data.get("status")

            if status in {"completed", "failed", "cancelled"}:
                break

            await asyncio.sleep(2)
        else:
            return _json(
                {
                    "success": False,
                    "job_id": job_id,
                    "status": "timeout",
                    "message": (
                        "Crawl is still running or did not finish within the "
                        "configured timeout."
                    ),
                }
            )

    pages: list[dict[str, Any]] = []
    for item in status_data.get("data", [])[:max_pages]:
        metadata = item.get("metadata") or {}
        pages.append(
            {
                "url": metadata.get("sourceURL")
                or metadata.get("sourceUrl")
                or metadata.get("url"),
                "title": metadata.get("title"),
                "description": metadata.get("description"),
                "markdown": item.get("markdown") or "",
            }
        )

    return _json(
        {
            "success": status_data.get("status") == "completed",
            "job_id": job_id,
            "status": status_data.get("status"),
            "completed": status_data.get("completed"),
            "total": status_data.get("total"),
            "pages": pages,
        }
    )


async def upsert_knowledge_chunks(chunks: list[dict[str, Any]]) -> int:
    """Helper for a future ingestion script; this is NOT exposed as an agent tool.

    Expected chunk payload fields:
        text: required
        provider: e.g. Princess Cruises, Royal Caribbean, Disney, Universal
        title: document/page title
        source_url: source URL if one exists
        document_id: stable document identifier
        chunk_id: stable chunk identifier
        updated_at: source update timestamp/date when known

    The same EMBEDDING_MODEL must be used for both ingestion and retrieval.
    """
    if not chunks:
        return 0

    texts = [str(chunk["text"]) for chunk in chunks]
    vectors = await embeddings.aembed_documents(texts)
    await _ensure_collection(len(vectors[0]))

    points: list[models.PointStruct] = []
    for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
        stable_key = "|".join(
            [
                str(chunk.get("document_id") or chunk.get("source_url") or "document"),
                str(chunk.get("chunk_id") or index),
                str(chunk["text"]),
            ]
        )
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))
        payload = dict(chunk)
        payload["text"] = str(chunk["text"])
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        )

    await qdrant.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
        wait=True,
    )
    return len(points)


RESEARCH_TOOLS = [
    search_knowledge_base,
    search_web,
    scrape_url,
    crawl_site,
]
