import uuid
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from app.rag.vectorstore import vectorstore

def split_documents(docs, base_meta):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    output = []
    for i, ch in enumerate(chunks):
        meta = dict(ch.metadata or {})
        meta.update(base_meta)
        meta["chunk_id"] = f"chunk_{i}"
        output.append(Document(page_content=ch.page_content, metadata=meta))
    return output


def ingest_pdf(path, filename, doc_type="job", company=None, role=None):
    loader = PyPDFLoader(path)
    pages = loader.load()

    doc_id = str(uuid.uuid4())

    base_meta = {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "company": company or "",
        "role": role or "",
        "source": filename
    }

    chunks = split_documents(pages, base_meta)
    vectorstore.add_documents(chunks)
    vectorstore.persist()

    return doc_id, len(chunks)
