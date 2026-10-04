from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------
# 1. Load PDF
# -----------------------------

pdf_path = "pdfs/sample.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print("PDF pages:", len(documents))


# -----------------------------
# 2. Split into chunks
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# -----------------------------
# 3. Create embeddings
# -----------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# -----------------------------
# 4. Create FAISS vector store
# -----------------------------

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS created")


# -----------------------------
# 5. Save FAISS to disk
# -----------------------------

vectorstore.save_local("vectorstore")

print("Vectorstore saved!")