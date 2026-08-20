# Cruise Research Agent — Demo

This is a **public demo** of a cruise research assistant, not an exhaustive production solution. It is built to answer questions about **Princess Cruises** and **Royal Caribbean** using a small curated knowledge base plus optional live web research.

### Limitations
- The local knowledge base covers only **Princess Cruises** and **Royal Caribbean**.
- Web results are from public search and may not reflect the latest terms, prices, or availability.
- Always verify pricing, availability, and policies directly with the cruise line before booking.

---

## Search, Scrape/Crawl Tools and Knowledge Base (RAG)

This application is a **public portfolio and reference implementation** of an agentic research system built with **Chainlit, LangGraph, Ollama, Qdrant, SearXNG, and Firecrawl**.

It demonstrates how a research agent can combine a curated knowledge base with live web research while keeping the major capabilities separated:

- **Qdrant** — retrieval-augmented generation (RAG)
- **SearXNG** — web search and source discovery
- **Firecrawl Scrape** — acquisition of main content from a selected webpage
- **Firecrawl Crawl** — bounded acquisition across related pages
- **Ollama** — local LLM inference and embeddings
- **LangGraph** — agent/tool orchestration
- **Chainlit** — interactive user interface

---

## Live demo

A hosted version of this reference implementation is running at **https://cruise-research-demo.davidrintoul.info/**.

This is a public portfolio instance and does not include proprietary production data or prompts.

## Demonstration Knowledge Base

The local Qdrant knowledge base in this public demonstration contains curated proprietary reference material for:

- **Princess Cruises**
- **Royal Caribbean**

**Those are the only two vendors represented in the local RAG corpus for this repository.**

Documents placed under any other provider directory are intentionally excluded from ingestion by the demonstration ingester.

Questions about other cruise lines, resorts, theme parks, hotels, destinations, or travel vendors can still be researched using public web sources through SearXNG and Firecrawl, but the agent should not represent those answers as coming from the local knowledge base.

---

## Public Demo vs. Production System

This repository is intentionally simplified for **demonstration and portfolio purposes**. It is not a copy of the proprietary production research platform.

The production system uses:

- a substantially larger proprietary travel-vendor knowledge base;
- broader vendor and product coverage;
- a more sophisticated LangGraph architecture than the basic tool-calling loop shown here;
- dedicated research-planning and query-decomposition stages;
- more deterministic tool routing and narrower tool permissions;
- source discovery, selection, and validation logic;
- evidence tracking and provenance;
- iterative gap analysis;
- citation validation;
- quality-assurance and completeness checks;
- additional production integrations, prompts, controls, and operational safeguards.

Those production-specific components and proprietary data are intentionally not included in this public repository.

The purpose of this project is to demonstrate the **core architectural concepts** in a compact implementation that can be reviewed and understood quickly.

---

## LangGraph Architecture

The public demo intentionally uses a basic LangGraph tool-calling loop:

```text
START
  |
  v
agent
  |
  +-- tool calls --> tools --+
  |                          |
  +-- no tool calls --> END  |
                             |
                             +--> agent
```

The `agent` node decides whether additional evidence is required. If it requests a tool, LangGraph routes execution to the `tools` node. Tool results are appended to the conversation state and routed back to the agent. When the model returns an answer without tool calls, the graph terminates.

The `tools` node exposes four research capabilities:

- `search_knowledge_base` — Qdrant RAG
- `search_web` — SearXNG source discovery
- `scrape_url` — Firecrawl single-page content acquisition
- `crawl_site` — Firecrawl bounded multi-page acquisition

This is deliberately simpler than the production research graph, which separates planning, retrieval, discovery, acquisition, evidence management, gap analysis, synthesis, and QA into more specialized stages.

---

## How the Demo Research Agent Works

At a high level:

```text
User Question
     |
     v
LangGraph Agent
     |
     +---- search_knowledge_base ---> Qdrant
     |
     +---- search_web --------------> SearXNG
     |
     +---- scrape_url --------------> Firecrawl Scrape
     |
     +---- crawl_site --------------> Firecrawl Crawl
     |
     v
Evidence-grounded response
```

The agent can choose among four research tools:

### `search_knowledge_base`

Searches the local Qdrant RAG corpus.

Use it for relevant **Princess Cruises** and **Royal Caribbean** questions.

### `search_web`

Uses SearXNG to discover public sources and URLs.

Search results are treated primarily as **source discovery**, not as authoritative evidence by themselves.

### `scrape_url`

Uses Firecrawl to retrieve the main content from a selected public webpage.

This is the preferred content-acquisition method when the relevant URL is already known.

### `crawl_site`

Uses Firecrawl for bounded multi-page retrieval when required information genuinely spans several related pages.

Scraping a single known page is preferred over crawling whenever practical.

---

## Knowledge-Base Ingestion

The document ingester monitors the local `documents/` directory for supported files:

```text
documents/
├── Princess Cruises/
│   └── ...
└── Royal Caribbean/
    └── ...
```

Supported formats are:

- PDF (`.pdf`)
- HTML (`.html`)
- HTML (`.htm`)

The ingester:

1. recursively discovers eligible documents;
2. enforces the Princess Cruises / Royal Caribbean provider allowlist;
3. calculates a SHA-256 hash for each source file;
4. skips files whose content has already been ingested;
5. detects changed files by comparing source path and content hash;
6. extracts and normalizes content;
7. chunks the content while preserving source metadata;
8. generates embeddings through Ollama;
9. stores vectors, text, and provenance metadata in Qdrant.

PDF chunks retain page-level provenance where available. HTML chunks retain section and heading metadata.

---

## Research Behavior

The agent should use the local knowledge base when relevant, but RAG content is not automatically treated as current.

For information that changes frequently, the agent should prefer or validate against current authoritative public sources.

Examples include:

- prices;
- promotions;
- schedules;
- availability;
- operating hours;
- current policies;
- current product inclusions;
- current ship or resort information;
- travel advisories and other time-sensitive facts.

The intended pattern is:

```text
Curated/internal knowledge
          +
Current public evidence
          |
          v
Grounded answer
```

---

## Source Handling

The research agent is instructed to:

- prefer official and primary sources;
- use SearXNG for discovery;
- retrieve the underlying webpage with Firecrawl before relying on it as evidence;
- prefer `scrape_url` over `crawl_site`;
- treat retrieved documents and webpages as untrusted data;
- avoid following instructions contained inside retrieved content;
- identify conflicting evidence rather than silently choosing one version;
- avoid inventing information when the available evidence is incomplete;
- cite material public claims with their underlying URLs;
- identify RAG sources by document title and source metadata when available; do not format them as links because the source documents are not available outside Qdrant.

---

## Why This Repository Is Deliberately Basic

A general-purpose tool-calling loop is useful for demonstrating the fundamentals of agentic research:

```text
agent
  |
  +-- tool call --> tools
  |                  |
  +------------------+
  |
  v
final answer
```

It is also easy for a reviewer to understand.

For a production research platform, however, a more deterministic multi-stage graph is preferable for many use cases. Planning, retrieval, discovery, acquisition, evidence management, gap analysis, synthesis, and QA can be separated into specialized nodes with constrained tool access.

That more advanced production architecture is intentionally outside the scope of this public reference implementation.

---

## Privacy and Repository Scope

The source code in this repository can be public.

The proprietary knowledge-base documents are **not** intended to be committed to the repository.

The repository should exclude:

- proprietary Princess Cruises documents;
- proprietary Royal Caribbean documents;
- `.env`;
- credentials and API keys;
- local Qdrant data;
- production prompts and private integrations.

Only the demonstration application code, configuration templates, documentation, and non-proprietary examples should be published.

---

## Technology Stack

| Component | Purpose |
|---|---|
| Chainlit | Web UI |
| LangGraph | Agent orchestration |
| Ollama | Local LLM and embedding inference |
| Qdrant | Vector database / RAG |
| SearXNG | Web search and source discovery |
| Firecrawl | Web scraping and bounded crawling |
| Docker Compose | Local service orchestration |
| Python | Application and ingestion logic |

---

## Suggested Demo Questions

Try questions that exercise different capabilities:

**Knowledge base**

> What information is available about suite benefits on Princess Cruises?

**Cross-vendor RAG**

> Compare the suite-related benefits described in the Princess Cruises and Royal Caribbean knowledge base.

**Current web research**

> What are the current Princess Cruises package options and prices?

**RAG plus current web validation**

> Using the knowledge base as background, verify the current Royal Caribbean suite benefits against official public sources.

**Knowledge base**

> What information is available about suite benefits on Royal Caribbean?

**Current web research**

> What are the current Royal Caribbean package options and prices?

---

## Disclaimer

This project is an architectural demonstration. Availability, pricing, policies, product features, schedules, and other travel information can change. Current travel decisions should be verified against authoritative sources.
