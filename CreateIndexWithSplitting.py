import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Set Pinecone API key
PINECONE_API_KEY = "API_KEY"
PINECONE_ENV = "us-east-1"

pc = Pinecone(
    api_key=PINECONE_API_KEY
)


# Create index (if not created already)
index_name = "learning2040civics"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Must match your embedding model's output
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
# Connect to the index
index = pc.Index(index_name)


def print_retriever_data(results):
    """
    Nicely print the retriever data from a Pinecone query.
    """
    if not results or "matches" not in results or len(results["matches"]) == 0:
        print("No matching results found.")
        return

    for i, match in enumerate(results["matches"]):
        match_id = match.get("id", "N/A")
        score = match.get("score", 0.0)
        text = match.get("metadata", {}).get("text", "No text provided")

        print(f"--- Result {i + 1} ---")
        print(f"ID: {match_id}")
        print(f"Score: {score:.4f}")
        print("Text:")
        print(text)
        print("\n")

pdf_directory = "C:/Users/Administrator/Desktop/Learning2040/CivicsDoc"
# Use glob to grab all PDF file paths in the directory
pdf_paths = glob.glob(os.path.join(pdf_directory, "*.pdf"))

# Initialize an empty list to hold documents
documents = []
for pdf_path in pdf_paths:
    loader = PyPDFLoader(pdf_path)
    # Load returns a list of document pages
    docs = loader.load()
    # Optionally, add the source (filename) to each document's metadata:
    for doc in docs:
        doc.metadata["source"] = os.path.basename(pdf_path)
    documents.extend(docs)


# Option A: Combine all text and then split (if you don't need per-page metadata)
text_data = "\n".join([doc.page_content for doc in documents])
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(text_data)


# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key="API_KEY")

# Generate embeddings
vectors = [embeddings.embed_query(chunk) for chunk in chunks]


# Upload chunks to Pinecone
ids = [f"chunk-{i}" for i in range(len(chunks))]
metadata = [{"text": chunk} for chunk in chunks]

# Upsert data into Pinecone
index.upsert(vectors=zip(ids, vectors, metadata))
print("Data uploaded to Pinecone successfully!")


query_text = "מה תפקידיה של הכנסת?"
query_embedding = embeddings.embed_query(query_text)

# Search Pinecone
results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

# Retrieve the most relevant chunk
retrieved_text = [match["metadata"]["text"] for match in results["matches"]]
print(retrieved_text)
print("Total vectors in Pinecone:", index.describe_index_stats())
print_retriever_data(retrieved_text)


def query_pinecone(user_query, top_k=3):
    """Retrieve relevant text from Pinecone based on user query."""
    query_embedding = embeddings.embed_query(user_query)

    # Search Pinecone
    results = index.query(vector=query_embedding, top_k=top_k, score_threshold=0.2,  include_metadata=True)
    print_retriever_data(results)

query_pinecone("מה תפקידיה של הכנסת?")