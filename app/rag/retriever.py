def build_retriever(vectorstore, filters=None, k=6):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20,
            "filter": filters or None
        }
    )
