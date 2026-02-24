RAG_SYSTEM = """You are a professional career AI assistant.
Use ONLY the provided context. Do not hallucinate.
If insufficient data, say: 'I don't know based on the available context.'
"""

JOB_SKILL_EXTRACTION = """
Extract required and nice-to-have technical skills.
Return JSON:
{
  "required_skills": [],
  "nice_to_have_skills": []
}
"""

CV_SKILL_EXTRACTION = """
Extract all explicitly mentioned technical skills.
Return JSON:
{
  "skills": []
}
"""

REWRITE_PROMPT = """
Rewrite CV bullets to better match the job.
Do NOT invent technologies.
Return JSON:
{
  "rewritten_bullets": []
}
"""
