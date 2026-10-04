from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------
# 1. Embedding model
# -----------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# -----------------------------
# 2. Load existing FAISS
# -----------------------------

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)


# -----------------------------
# 3. Create retriever
# -----------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)