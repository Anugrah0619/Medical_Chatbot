import os
from dotenv import load_dotenv
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not found in environment variables.")

print("Initializing Pinecone client...")
pc = Pinecone(api_key=pinecone_api_key)
print("Pinecone client initialized.")

index_name = "medicalbot"

if index_name not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Index '{index_name}' created successfully.")
else:
    print(f"Index '{index_name}' already exists. Connecting to it.")
print("Pinecone index checked/created.")

print("Loading PDF files from 'data/'...")
extracted_data = load_pdf_file(data='data/')
print("PDF files loaded.")

print("Splitting text into chunks...")
text_chunks = text_split(extracted_data)
print(f"Text split into {len(text_chunks)} chunks.")

print("Downloading/Initializing Hugging Face embeddings model...")
embeddings = download_hugging_face_embeddings()
print("Embeddings model initialized.")

print(f"Embedding {len(text_chunks)} chunks and upserting into Pinecone index '{index_name}'...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
print("Documents successfully embedded and upserted into Pinecone.")

print("\nData indexing process complete. Your Pinecone index is ready.")

