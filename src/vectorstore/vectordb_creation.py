# Run this script to create a vector database from markdown files.
# Run from the root directory of the project with: uv run src/rag/vectordb_creation.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain.schema import Document
import re
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Load and chunk documents, adding filename and page number to metadata
docs = []
for p in Path("data/processed/").glob("*.md"):
    text = p.read_text(encoding="utf-8")
    
    print(f"Processing file: {p.name}")
    print(f"File size: {len(text)} characters")
    print(f"First 200 characters: {text[:200]}...\n")
    
    # Extract page numbers using regex
    page_matches = re.finditer(r'<!-- Page (\d+) -->', text)
    last_page = 1
    current_pos = 0
    page_contents = []
    
    for match in page_matches:
        page_num = int(match.group(1))
        next_pos = match.start()
        if current_pos > 0:
            page_contents.append({
                'content': text[current_pos:next_pos].strip(),
                'page': last_page
            })
        last_page = page_num
        current_pos = match.end()
    
    # Add the final segment
    if current_pos < len(text):
        page_contents.append({
            'content': text[current_pos:].strip(),
            'page': last_page
        })
    
    # Create documents with page numbers in metadata
    for content in page_contents:
        if content['content']:
            docs.append(Document(
                page_content=content['content'],
                metadata={
                    "source": str(p.name),
                    "page": content['page']
                }
            ))

if docs:
    print(f"Loaded {len(docs)} documents from markdown files.")
else:
    print("No documents found in markdown files.")
    
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
chunks = splitter.split_documents(docs)

print(f"Total chunks: {len(chunks)}")
print(f"First chunk preview: {chunks[0].page_content[:200]}...")
print(f"First chunk metadata: {chunks[0].metadata}")


# create embeddings and persist in Chroma
embeddings = OllamaEmbeddings(model="jeffh/intfloat-multilingual-e5-large-instruct:f16")

persist_directory = "data/vectorstore_db"

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory,
)