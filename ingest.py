import os
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Load and clean all HTML files from corpus/
def load_corpus(folder="corpus"):
    documents = []
    for filename in os.listdir(folder):
        if filename.endswith(".html"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="latin-1") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                documents.append({"text": text, "source": filename})
            print(f"Loaded: {filename}")
    return documents

# Split into chunks
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])
        for split in splits:
            chunks.append({"text": split, "source": doc["source"]})
    return chunks

# Embed and store in ChromaDB
def build_vectorstore(chunks):
    import shutil
    if os.path.exists("vectorstore"):
        shutil.rmtree("vectorstore")
    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"]} for c in chunks]
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory="vectorstore"
    )
    print(f"Stored {len(chunks)} chunks in vectorstore.")
    return vectorstore

if __name__ == "__main__":
    print("Loading corpus...")
    docs = load_corpus()
    print(f"Loaded {len(docs)} files.")
    print("Chunking...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    print("Embedding and storing...")
    build_vectorstore(chunks)
    print("Done! Vectorstore ready.")