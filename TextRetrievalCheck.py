import pinecone
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# ------------------------------
# Configuration
# ------------------------------
PINECONE_API_KEY = "API_KEY"  # Replace with your API key
PINECONE_ENV = "us-east-1"  # Adjust if needed
index_name = "learning2040civics"  # Make sure this matches your index name

# ------------------------------
# Initialize Pinecone and Connect to the Index
# ------------------------------
pc = Pinecone(
    api_key=PINECONE_API_KEY
)
index = pc.Index(index_name)

# ------------------------------
# Load the SentenceTransformer Model
# ------------------------------
# We're using the BAAI/bge-m3 model for generating embeddings
model_name = "BAAI/bge-m3"
model = SentenceTransformer(model_name)

# ------------------------------
# Define the Query
# ------------------------------
query_text = "מהם התפקידים של הכנסת?"  # Hebrew query text

# ------------------------------
# Generate Query Embedding
# ------------------------------
query_embedding = model.encode(query_text).tolist()

# ------------------------------
# Query Pinecone for the Top 3 Matches
# ------------------------------
results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

# ------------------------------
# Print the Query Results
# ------------------------------
def print_results(results):
    if not results or "matches" not in results or len(results["matches"]) == 0:
        print("No matching results found.")
        return

    for i, match in enumerate(results["matches"]):
        match_id = match.get("id", "N/A")
        score = match.get("score", 0.0)
        metadata = match.get("metadata", {})
        print(f"--- Result {i + 1} ---")
        print(f"ID: {match_id}")
        print(f"Score: {score:.4f}")
        print("Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        print("\n")

print("Query:", query_text)
print_results(results)



query_text = "מהי דמוקרטיה?"  # Hebrew query text

query_embedding = model.encode(query_text).tolist()

results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

print("Query:", query_text)
print_results(results)

query_text = "מה קרה במצדה?"  # Hebrew query text

query_embedding = model.encode(query_text).tolist()

results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

print("Query:", query_text)
print_results(results)


query_text = "מהם שלבי יצירת חוק?"  # Hebrew query text

query_embedding = model.encode(query_text).tolist()

results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

print("Query:", query_text)
print_results(results)


query_text = "מה היה יחס היהדות הדתית כלפי הקמת המדינה?"  # Hebrew query text

query_embedding = model.encode(query_text).tolist()

results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

print("Query:", query_text)
print_results(results)