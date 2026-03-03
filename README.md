# AI CV–Job Alignment System (LLM + RAG)

An end-to-end backend system that aligns CVs with job descriptions using Large Language Models (LLMs) and a Retrieval-Augmented Generation (RAG) architecture.

The system ingests job descriptions across multiple engineering domains, extracts structured required skills, computes compatibility scores against a given CV, ranks opportunities by alignment, and rewrites CV bullets to improve ATS compatibility.

Designed with a modular, service-oriented architecture and production-oriented backend practices.

---

## 🚀 Features

* 📥 Automatic ingestion of job descriptions (text or JSON)
* 🧠 Structured skill extraction using LLMs
* 📊 CV–job compatibility scoring
* 🏆 Automatic ranking of opportunities by alignment
* ✍️ Context-aware CV bullet rewriting for ATS optimization
* 🗄️ Vector-based retrieval using ChromaDB
* ⚙️ Clean FastAPI-based REST architecture

---

## 🏗️ Architecture Overview

The system follows a modular backend structure:

* **FastAPI** → REST API layer
* **LangChain** → LLM orchestration
* **OpenAI Embeddings** → Semantic representation
* **ChromaDB** → Vector database for retrieval
* **Service Layer** → Matching, ranking, and rewriting logic

Core workflow:

1. Ingest job descriptions and CVs
2. Generate embeddings and store in vector database
3. Extract structured required skills
4. Compute compatibility scores
5. Rank opportunities
6. Rewrite CV bullets aligned to job context

---

## 📊 Example Capabilities

* Differentiates domain proximity (AI/Data vs Mechanical vs Industrial Engineering)
* Identifies missing required skills
* Produces alignment-based job rankings
* Enhances CV phrasing without inventing experience

---

## 🛠 Tech Stack

* Python 3.12
* FastAPI
* LangChain
* OpenAI API
* ChromaDB
* Docker (optional deployment)

---

## 🔧 Running the Project

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file:

```
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
```

3. Start the server:

```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

4. Access API docs:

```
http://localhost:8000/docs
```

---

## 🎯 Motivation

This project explores how LLMs can be applied beyond chat interfaces to build structured, production-ready AI systems.

It focuses on:

* RAG pipelines
* Skill extraction and structured reasoning
* Backend architecture for AI services
* Practical application of embeddings in real-world matching problems

---

## 📌 Future Improvements

* Weighted scoring (required vs nice-to-have skills)
* Market-wide skill frequency analysis
* Semantic skill normalization
* Domain classification module
* Automated skill gap learning recommendations

---

## 📬 Author

Rafael Onieva

Open to collaboration in AI systems, data engineering, and applied LLM architectures.
