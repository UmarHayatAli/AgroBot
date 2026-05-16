from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter  # only this import changes
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

KB_DIR = Path("knowledge-base")
DB_DIR = Path("chroma_db")
COLLECTION_NAME = "agrobot_knowledge"


def load_markdown_documents():
    docs = []
    for path in KB_DIR.rglob("*.md"):
        loader = TextLoader(str(path), encoding="utf-8")
        file_docs = loader.load()

        for doc in file_docs:
            doc.metadata["source"] = str(path.relative_to(KB_DIR))
            doc.metadata["topic"] = path.stem
            docs.append(doc)

    return docs


def build_vector_db():
    if not KB_DIR.exists():
        raise FileNotFoundError(f"{KB_DIR} folder not found")

    raw_docs = load_markdown_documents()
    if not raw_docs:
        raise ValueError("No .md files found in knowledge-base/")

    # CHANGE 1: Split on markdown headers instead of raw character count
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#",  "h1"),
            ("##", "section"),
            ("###","subsection"),
        ],
        strip_headers=False,  # keep "## Market & Economic Notes" inside chunk text
    )

    chunks = []
    for doc in raw_docs:
        # MarkdownHeaderTextSplitter works on raw text, not Documents
        split_chunks = splitter.split_text(doc.page_content)

        for chunk in split_chunks:
            # CHANGE 2: Carry forward your original metadata, then add section
            chunk.metadata["source"] = doc.metadata["source"]
            chunk.metadata["topic"]  = doc.metadata["topic"]
            section = chunk.metadata.get("section", "overview")

            # Prepend section label so the embedding captures it semantically
            chunk.page_content = (
                f"[{doc.metadata['topic']} — {section}]\n\n{chunk.page_content}"
            )
            chunks.append(chunk)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_DIR),
    )

    vectordb.persist()
    print(f"Indexed {len(raw_docs)} docs into {len(chunks)} chunks.")
    print(f"Saved Chroma DB at: {DB_DIR.resolve()}")


if __name__ == "__main__":
    build_vector_db()