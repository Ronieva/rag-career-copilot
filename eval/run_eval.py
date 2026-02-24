import argparse
import json
import sys
from pathlib import Path

import requests


def load_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--file", default=str(Path(__file__).parent / "questions.jsonl"), help="Path to questions.jsonl")
    parser.add_argument("--top-k", type=int, default=6, help="Retriever top_k")
    parser.add_argument("--doc-type", default="job", help="Metadata filter doc_type (optional)")
    parser.add_argument("--company", default=None, help="Metadata filter company (optional)")
    parser.add_argument("--role", default=None, help="Metadata filter role (optional)")
    args = parser.parse_args()

    qpath = Path(args.file)
    if not qpath.exists():
        print(f"File not found: {qpath}", file=sys.stderr)
        sys.exit(1)

    questions = load_jsonl(qpath)
    if not questions:
        print("No questions found.", file=sys.stderr)
        sys.exit(1)

    total = 0
    answered = 0
    with_citations = 0
    citation_items = 0

    for q in questions:
        total += 1
        payload = {
            "question": q["question"],
            "top_k": args.top_k,
        }
        # Optional filters
        if args.doc_type:
            payload["doc_type"] = args.doc_type
        if args.company:
            payload["company"] = args.company
        if args.role:
            payload["role"] = args.role

        r = requests.post(f"{args.base_url}/query", json=payload, timeout=120)
        if r.status_code != 200:
            print(f"[{q.get('id','?')}] ERROR {r.status_code}: {r.text[:200]}")
            continue

        data = r.json()
        answer = (data.get("answer") or "").strip()
        citations = data.get("citations") or []

        if answer:
            answered += 1
        if citations:
            with_citations += 1
            citation_items += len(citations)

        print(f"\n[{q.get('id','?')}] {q['question']}")
        print(f"Answer: {answer[:300]}{'…' if len(answer) > 300 else ''}")
        print(f"Citations: {len(citations)}")

    print("\n--- SUMMARY ---")
    print(f"Total questions: {total}")
    print(f"Answered: {answered}")
    print(f"With citations: {with_citations}")
    if total > 0:
        print(f"Citation rate: {with_citations/total:.2%}")
    if with_citations > 0:
        print(f"Avg citations per cited answer: {citation_items/with_citations:.2f}")


if __name__ == "__main__":
    main()
