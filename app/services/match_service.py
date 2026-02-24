import json
from app.core.llm import llm
from app.rag.prompts import JOB_SKILL_EXTRACTION, CV_SKILL_EXTRACTION
from app.rag.vectorstore import vectorstore


def extract_json(prompt, text):
    response = llm.invoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ]).content
    return json.loads(response)


def match(job_id, cv_id):
    job_docs = vectorstore.similarity_search("job", k=20, filter={"doc_id": job_id})
    cv_docs = vectorstore.similarity_search("cv", k=20, filter={"doc_id": cv_id})

    job_text = "\n".join([d.page_content for d in job_docs])
    cv_text = "\n".join([d.page_content for d in cv_docs])

    job_data = extract_json(JOB_SKILL_EXTRACTION, job_text)
    cv_data = extract_json(CV_SKILL_EXTRACTION, cv_text)

    required = set(job_data["required_skills"])
    cv_skills = set(cv_data["skills"])

    score = (len(required & cv_skills) / len(required) * 100) if required else 0

    return {
        "match_score": round(score, 2),
        "required_skills": list(required),
        "cv_skills": list(cv_skills),
        "missing_required_skills": list(required - cv_skills)
    }
