from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.core.config import settings

# LLM client (deterministic)
llm = ChatOpenAI(
    model=settings.OPENAI_MODEL,
    temperature=0
)

# Embeddings client
embeddings = OpenAIEmbeddings(
    model=settings.OPENAI_EMBED_MODEL
)
