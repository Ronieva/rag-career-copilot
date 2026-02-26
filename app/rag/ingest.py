import uuid
from typing import Optional, Tuple, List, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from app.rag.vectorstore import vectorstore


def split_documents(docs: List[Document], base_meta: Dict[str, Any]) -> List[Document]:
    """Split PDF pages into chunks and attach consistent metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    output: List[Document] = []
    for i, ch in enumerate(chunks):
        meta = dict(ch.metadata or {})
        meta.update(base_meta)
        meta["chunk_id"] = f"chunk_{i}"
        output.append(Document(page_content=ch.page_content, metadata=meta))
    return output


def ingest_pdf(
    path: str,
    filename: str,
    doc_type: str = "job",
    company: Optional[str] = None,
    role: Optional[str] = None,
) -> Tuple[str, int]:
    """Load a PDF, chunk it, and store it in the vector database."""
    loader = PyPDFLoader(path)
    pages = loader.load()

    doc_id = str(uuid.uuid4())
    base_meta = {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "company": company or "",
        "role": role or "",
        "source": filename,
    }

    chunks = split_documents(pages, base_meta)
    vectorstore.add_documents(chunks)
    vectorstore.persist()

    return doc_id, len(chunks)
