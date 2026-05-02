# retriever.py — with section filtering + MMR
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
    collection_name="agrobot_knowledge"
)

def retrieve(query: str, topic_filter: str = None, k: int = 4):
    # Use MMR (Maximal Marginal Relevance) to avoid duplicate chunks
    search_kwargs = {
        "k": k,
        "fetch_k": 12,   # fetch more candidates, then diversify
    }
    if topic_filter:
        search_kwargs["filter"] = {"topic": topic_filter}

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )
    return retriever.invoke(query)

# Test
docs = retrieve("cotton market export economy")
for i, doc in enumerate(docs, 1):
    print(f"\n--- Result {i} ---")
    print("Section:", doc.metadata.get("section"))
    print(doc.page_content[:200])