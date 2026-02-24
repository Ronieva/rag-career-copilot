from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.core.config import settings

llm = ChatOpenAI(
    model=settings.OPENAI_MODEL,
    temperature=0
)

embeddings = OpenAIEmbeddings(
    model=settings.OPENAI_EMBED_MODEL
)
