# RAG Career Copilot (LangChain + FastAPI + OpenAI)

A Retrieval-Augmented Generation (RAG) API that ingests job descriptions and CVs, answers questions with citations, computes a skill-based match score, and rewrites CV bullets without hallucinating new technologies.

## Features

- **RAG Q&A with citations** (source + page + chunk id)
- **Metadata filtering** (doc_type/company/role)
- **Skill extraction + match score** (deterministic scoring on required skills)
- **ATS bullet rewriting** with guardrails (no new technologies invented)
- **Dockerized** for easy deployment
- **Evaluation folder** with a small benchmark runner

## Tech Stack

- FastAPI
- LangChain
- OpenAI (Chat + Embeddings)
- ChromaDB (local persistent vector store)
- PyPDF (PDF ingestion)

---

## Project Structure

```text
rag-career-copilot/
├─ app/
│  └─ main.py
├─ eval/
│  ├─ questions.jsonl
│  ├─ run_eval.py
│  └─ README.md
├─ vector_db/               # persisted ChromaDB (created at runtime)
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yaml
├─ .env.example
└─ README.md
