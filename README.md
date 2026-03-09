<div align="center">

# OlympiadMind

**Multimodal Physics Reasoning System over International Olympiad Problems**

IPhO · APhO · WoPhO · EPhO · Handwritten Solutions · LaTeX · Knowledge Graphs

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Nougat](https://img.shields.io/badge/Meta_Nougat-LaTeX_OCR-orange?style=flat-square)](https://github.com/facebookresearch/nougat)
[![TrOCR](https://img.shields.io/badge/Microsoft_TrOCR-Handwriting_OCR-blue?style=flat-square)](https://huggingface.co/microsoft/trocr-large-handwritten)
[![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-018bff?style=flat-square&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-agent-green?style=flat-square)](https://langchain.com/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)

</div>

---

## Overview

Physics Olympiad solution PDFs are among the hardest documents to parse programmatically. A single PDF can contain typeset text, rendered LaTeX equations, embedded figures, and handwritten student solutions — all in one file. Standard PDF parsers fail silently on three of those four content types.

OlympiadMind solves this. It processes 1,000+ International Physics Olympiad problem and solution PDFs across all four content modalities, structures the extracted knowledge into a typed graph database, and exposes a reasoning agent that solves new physics problems step by step — retrieving relevant past problems and domain concepts to support each step.

The system is then benchmarked against GPT-4o on unseen olympiad problems, with results reported per topic and per difficulty level.

---

## Current Status

| Phase | Description | Status |
|---|---|---|
| 1 | Dataset audit and content-type classification | Complete |
| 2 | Multimodal PDF extraction pipeline | In progress |
| 3 | External knowledge corpus ingestion | Planned |
| 4 | Neo4j knowledge graph construction | Planned |
| 5 | RAG reasoning pipeline (LangGraph) | Planned |
| 6 | Benchmark: OlympiadMind vs GPT-4o | Planned |
| 7 | FastAPI + Streamlit frontend | Planned |

---

## The Extraction Problem

Standard PDF parsers (pdfplumber, PyPDF2) handle clean typeset text reasonably well. They fail on everything else. This project handles all four content types that appear in physics olympiad PDFs:

| Content Type | Challenge | Tool |
|---|---|---|
| Typeset text | Multi-column layouts, headers, metadata | PyMuPDF |
| Rendered LaTeX equations | Math is encoded as vector graphics, not text | Meta Nougat |
| Embedded figures and diagrams | No text content — needs visual description | Qwen2-VL |
| Handwritten student solutions | Variable handwriting, mixed equations and prose | Microsoft TrOCR + pix2tex |

Each content type has a separate extraction pipeline with its own quality metric. Reporting extraction quality per content type — not just a single aggregate score — is intentional: it shows exactly where the pipeline is strong and where it degrades.

---

## Extraction Quality (Target)

| Content Type | Metric | Target |
|---|---|---|
| Typeset text | Character error rate | < 2% |
| LaTeX equations | BLEU against ground truth | > 0.85 |
| Figure descriptions | Human eval (1-5 scale) | > 4.0 |
| Handwritten OCR | Word error rate | < 15% |

Baselines computed using GPT-4o-Vision on a 50-document held-out set.

---

## Architecture

```
1,000+ Physics Olympiad PDFs
(IPhO, APhO, WoPhO, EPhO — problems and solutions)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 1 — Content-Type Classification                  │
│                                                         │
│  Per-page classifier (PyMuPDF text confidence score):  │
│    text_conf > 0.80 → typeset                           │
│    text_conf < 0.30 → handwritten                       │
│    contains vector bbox → equation region               │
│    contains raster bbox → figure region                 │
│                                                         │
│  Output: page manifest with content type per page      │
│  Stored as: data/manifests/{doc_id}.json               │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 2 — Four-Modality Extraction Pipeline            │
│                                                         │
│  Stream 1 — Typeset text (PyMuPDF)                     │
│    Multi-column layout detection                        │
│    Reading order reconstruction                         │
│    Sub-question segmentation (a. b. c. d. pattern)     │
│    Output: clean UTF-8 text per section                 │
│                                                         │
│  Stream 2 — LaTeX equations (Meta Nougat)              │
│    Nougat: academic PDF → LaTeX (end-to-end)           │
│    Equation bounding boxes → cropped images             │
│    pix2tex fallback for inline equations                │
│    Output: LaTeX string + rendered PNG per equation     │
│                                                         │
│  Stream 3 — Figures (Qwen2-VL)                         │
│    PyMuPDF raster region extraction                     │
│    Qwen2-VL: image → structured description            │
│    Fields: diagram_type, physics_concept,              │
│    axes_labels, annotations, description               │
│    Output: image file + structured JSON per figure     │
│                                                         │
│  Stream 4 — Handwritten pages (TrOCR + pix2tex)       │
│    TrOCR-large-handwritten: prose text OCR             │
│    pix2tex: handwritten equation regions               │
│    Layout analysis: text block vs equation block       │
│    Output: merged text + LaTeX per handwritten page    │
│                                                         │
│  Merger: combine 4 streams → PhysicsDocument JSON     │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 3 — External Knowledge Corpus                    │
│                                                         │
│  Wikipedia Physics articles (API — no scraping)        │
│    Topics: Mechanics, EM, Thermodynamics,              │
│    Optics, Modern Physics, Waves, Fluids               │
│                                                         │
│  Physics Stack Exchange (public data dump)             │
│    Filtered: score > 10, accepted answers only         │
│                                                         │
│  arXiv abstracts (physics.edu-ph category)             │
│    Introductory and olympiad-level content only        │
│                                                         │
│  Processing: chunk → embed (text-embedding-3-small)   │
│  → FAISS index for retrieval                           │
│  Target: ~50k chunks across all sources               │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 4 — Knowledge Graph (Neo4j AuraDB)               │
│                                                         │
│  Nodes: Problem, Solution, Topic, Concept,             │
│         Figure, Equation, Competition, Author          │
│                                                         │
│  Edges:                                                 │
│    Problem -[:BELONGS_TO]→ Topic                       │
│    Problem -[:REQUIRES]→ Concept                       │
│    Problem -[:HAS_SOLUTION]→ Solution                  │
│    Problem -[:HAS_FIGURE]→ Figure                      │
│    Problem -[:SIMILAR_TO]→ Problem                     │
│    Solution -[:USES_CONCEPT]→ Concept                  │
│    Solution -[:USES_EQUATION]→ Equation                │
│                                                         │
│  SIMILAR_TO built from: shared Concept nodes +        │
│  embedding cosine similarity on problem text          │
│                                                         │
│  Deduplication: same problem appears across           │
│  multiple competition PDFs — merged via               │
│  fuzzy title match + year + competition metadata      │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 5 — RAG Reasoning Agent (LangGraph)              │
│                                                         │
│  Input: new physics problem (text + optional image)    │
│                                                         │
│  LangGraph typed state machine:                        │
│    Step 1 — Topic classifier                           │
│             which branch of physics?                   │
│    Step 2 — Concept extractor                          │
│             which specific concepts are required?      │
│    Step 3 — Graph retrieval                            │
│             SIMILAR_TO edges → top-k past problems    │
│    Step 4 — Vector retrieval                           │
│             FAISS → relevant knowledge chunks          │
│    Step 5 — Solution generator                         │
│             problem + retrieved context → step-by-step │
│    Step 6 — Self-validator                             │
│             dimensional analysis + answer form check  │
│    Step 7 — Answer formatter                           │
│             solution + citations + confidence score   │
│                                                         │
│  Fallback node: low-confidence queries → clarification │
│  All retrieved Cypher queries returned with answer     │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 6 — Benchmark: OlympiadMind vs GPT-4o           │
│                                                         │
│  100 held-out problems (never seen in training/graph)  │
│  Split: 20 per topic — Mechanics, EM, Thermo,         │
│  Optics, Modern Physics                                │
│                                                         │
│  Metrics per problem:                                  │
│    final_answer_correct (0/1)                          │
│    method_correct (0/1 — human eval)                  │
│    steps_valid (0/1 — no physics errors in reasoning) │
│    latex_valid (does generated LaTeX render?)         │
│                                                         │
│  Reported: per-topic accuracy, overall accuracy,      │
│  failure mode analysis (wrong concept retrieval,      │
│  correct method wrong arithmetic, hallucination)      │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 7 — API + Frontend                               │
│                                                         │
│  FastAPI — POST /v1/solve, GET /v1/problems,           │
│  GET /v1/concepts, GET /v1/similar                     │
│                                                         │
│  Streamlit — problem solver UI, graph explorer,        │
│  extraction quality dashboard, benchmark results       │
└─────────────────────────────────────────────────────────┘
```

---

## Knowledge Graph Schema

```
(Problem)
  ├── id: String
  ├── competition: String       — IPhO, APhO, WoPhO, EPhO
  ├── year: Integer
  ├── problem_number: Integer
  ├── topic: String
  ├── difficulty: Integer       — 1-5 estimated from competition level
  ├── sub_question_count: Integer
  ├── has_handwritten_solution: Boolean
  └── extraction_quality: Float — avg quality score across content types

(Solution)
  ├── id: String
  ├── method: String            — energy method, force method, etc.
  ├── key_result: String        — final answer in LaTeX
  ├── step_count: Integer
  └── source: String            — typeset / handwritten / both

(Concept)
  ├── name: String              — e.g. "magnetic flux", "harmonic oscillator"
  ├── topic: String
  └── canonical_name: String

(Equation)
  ├── latex: String
  ├── description: String
  └── concept: String

(Figure)
  ├── diagram_type: String      — force diagram, circuit, graph, etc.
  ├── physics_concept: String
  └── description: String
```

---

## Why This Project

**What most NLP projects do:** take a clean dataset, train or fine-tune a model, report accuracy.

**What this project does:** handle a dataset that breaks most pipelines (mixed modalities, handwriting, LaTeX), build a structured knowledge representation from unstructured scientific documents, and evaluate a reasoning system against the strongest available baseline on a hard benchmark.

Every component solves a problem that exists in real enterprise environments — document intelligence, scientific knowledge extraction, structured reasoning over domain-specific corpora. These are active problems at Google DeepMind, Microsoft Research, Bloomberg, and any company processing large volumes of technical documents.

---

## Technical Decisions

**Why Nougat over pdfplumber for equations?**
pdfplumber extracts text tokens — equations rendered as vector graphics have no text tokens. Nougat (Meta, 2023) is trained specifically to convert academic PDFs to LaTeX markup, handling equations, tables, and figures in a single pass. It outperforms every text-extraction approach on math-heavy documents.

**Why TrOCR for handwriting?**
TrOCR-large-handwritten (Microsoft) is the current state of the art for printed and handwritten text recognition, outperforming Tesseract significantly on scientific handwriting. It handles the mixed case (prose + equations on the same page) better than general-purpose OCR models when combined with a layout analysis pre-pass.

**Why a knowledge graph over pure vector RAG?**
Vector RAG retrieves semantically similar chunks — it cannot answer "find all problems that use the conservation of magnetic flux AND harmonic oscillator analysis simultaneously." That's a two-hop graph query. The graph also lets the reasoning agent explain *why* it retrieved a particular past problem, making the system auditable.

**Why LangGraph over a single-prompt RAG?**
A single prompt that does topic classification, concept extraction, retrieval, reasoning, and validation simultaneously breaks on hard problems. LangGraph makes each step independently debuggable — you can see exactly where the pipeline fails on a given problem and fix that step without touching the others.

---

## Project Structure

```
OlympiadMind/
├── data/
│   ├── raw/
│   │   ├── problems/           — raw problem PDFs (1,000+)
│   │   └── solutions/          — raw solution PDFs (1,000+)
│   ├── manifests/              — per-doc content type classification
│   ├── extracted/
│   │   ├── documents.jsonl     — merged PhysicsDocument records
│   │   ├── equations/          — LaTeX strings + PNG renders
│   │   ├── figures/            — extracted images + descriptions
│   │   └── failed.csv          — extraction failures with reason
│   └── corpus/
│       ├── wikipedia.jsonl     — chunked Wikipedia articles
│       ├── stackexchange.jsonl — filtered Physics SE QA pairs
│       └── arxiv.jsonl         — arXiv edu-ph abstracts
│
├── src/
│   ├── extraction/
│   │   ├── classifier.py       — content type detection per page
│   │   ├── text_extractor.py   — PyMuPDF typeset text pipeline
│   │   ├── equation_extractor.py — Nougat + pix2tex pipeline
│   │   ├── figure_extractor.py — Qwen2-VL figure description
│   │   ├── handwriting_ocr.py  — TrOCR + pix2tex pipeline
│   │   ├── merger.py           — 4-stream → PhysicsDocument
│   │   └── quality.py          — per-modality quality scoring
│   │
│   ├── corpus/
│   │   ├── wikipedia_ingest.py — Wikipedia API ingestion
│   │   ├── stackexchange_ingest.py — SE data dump processing
│   │   ├── arxiv_ingest.py     — arXiv API ingestion
│   │   └── embedder.py         — chunk + embed + FAISS index
│   │
│   ├── graph/
│   │   ├── schema.py           — node/edge Pydantic models
│   │   ├── builder.py          — MERGE-based Neo4j writer
│   │   ├── similarity.py       — SIMILAR_TO edge construction
│   │   └── queries.py          — reusable Cypher queries
│   │
│   ├── agent/
│   │   ├── state.py            — QueryState TypedDict
│   │   ├── nodes.py            — one function per LangGraph node
│   │   ├── workflow.py         — graph assembly + edges
│   │   ├── retriever.py        — graph + vector hybrid retrieval
│   │   └── validator.py        — dimensional analysis + form check
│   │
│   └── api/
│       ├── routes/             — FastAPI route handlers
│       └── models.py           — request/response schemas
│
├── evaluation/
│   ├── benchmark.py            — 100-problem evaluation harness
│   ├── gpt4o_baseline.py       — GPT-4o evaluation on same set
│   ├── metrics.py              — answer correctness, method scoring
│   └── failure_analysis.py     — error clustering by failure mode
│
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_extraction_experiments.ipynb
│   ├── 03_graph_exploration.ipynb
│   ├── 04_agent_testing.ipynb
│   └── 05_benchmark_analysis.ipynb
│
├── app.py                      — Streamlit entrypoint
├── main.py                     — FastAPI entrypoint
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| PDF parsing | PyMuPDF | Fastest, most reliable for layout analysis |
| LaTeX OCR | Meta Nougat | State of the art on academic PDFs |
| Handwriting OCR | Microsoft TrOCR | Best published WER on handwritten text |
| Equation OCR | pix2tex | Lightweight, accurate on inline equations |
| Figure understanding | Qwen2-VL | Multimodal, handles scientific diagrams |
| Vector embeddings | OpenAI text-embedding-3-small | Best cost/quality for retrieval |
| Vector index | FAISS | Production-grade, no infra overhead |
| Knowledge graph | Neo4j AuraDB | Industry standard, Cypher query language |
| RAG agent | LangGraph | Typed state machine, debuggable per step |
| API | FastAPI | Async, auto-docs, Pydantic validation |
| Frontend | Streamlit | Fast to build, sufficient for demo |

---

## Dataset

1,000+ PDF files from International Physics Olympiad competitions:
- IPhO (International Physics Olympiad) — 1967 to present
- APhO (Asian Physics Olympiad)
- WoPhO (World Physics Olympiad)
- EPhO (European Physics Olympiad)

Mix of typeset problem sheets and solution files, including scanned handwritten student solutions from top finishers.

---

## Author

[GitHub](#) · [LinkedIn](#)
