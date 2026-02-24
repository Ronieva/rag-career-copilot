from langchain_community.vectorstores import Chroma
from app.core.config import settings
from app.core.llm import embeddings

vectorstore = Chroma(
    persist_directory=settings.VECTOR_DIR,
    embedding_function=embeddings,
    collection_name="career_docs"
)
