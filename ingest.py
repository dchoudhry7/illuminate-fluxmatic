import os
import sys
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from dotenv import load_dotenv
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

def load_text_documents(data_dir):
    documents = []
    if not os.path.exists(data_dir):
        return documents
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                doc = Document(
                    page_content=content,
                    metadata={"source": filename, "category": "General"}
                )
                documents.append(doc)
            except Exception as e:
                pass
    return documents

def load_pdf_documents(uploads_dir):
    documents = []
    if not os.path.exists(uploads_dir):
        return documents
    for filename in os.listdir(uploads_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(uploads_dir, filename)
            category = "General"
            display_name = filename
            if "__" in filename:
                parts = filename.split("__", 1)
                category = parts[0]
                display_name = parts[1]
            try:
                reader = pypdf.PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={"source": display_name, "raw_name": filename, "category": category}
                    )
                    documents.append(doc)
            except Exception as e:
                pass
    return documents

def main():
    txt_docs = load_text_documents(DATA_DIR)
    pdf_docs = load_pdf_documents(UPLOADS_DIR)
    all_raw_docs = txt_docs + pdf_docs
    if not all_raw_docs:
        return
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    chunks = []
    for doc in all_raw_docs:
        doc_chunks = text_splitter.split_text(doc.page_content)
        for chunk_text in doc_chunks:
            meta = doc.metadata.copy()
            chunks.append(Document(page_content=chunk_text, metadata=meta))
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    if os.path.exists(DB_DIR):
        import shutil
        try:
            shutil.rmtree(DB_DIR)
        except Exception as e:
            pass
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    vector_db.persist()

if __name__ == "__main__":
    main()
