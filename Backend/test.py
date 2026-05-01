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

query = " cotton market"
docs = db.similarity_search(query, k=3)

for i, doc in enumerate(docs, 1):
    print(f"\n--- Result {i} ---")
    print("Source:", doc.metadata)
    print(doc.page_content[:300])