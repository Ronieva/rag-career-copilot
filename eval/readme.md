# Evaluation (Mini Benchmark)

This folder contains a tiny evaluation harness for the RAG API.

## What it does

- Loads `questions.jsonl`
- Sends each question to `/query`
- Reports:
  - how many answers returned
  - how many included citations
  - a simple "citation coverage" proxy

## How to run

1) Start the API (locally or docker)
2) Run:

```bash
python eval/run_eval.py --base-url http://127.0.0.1:8000
```
