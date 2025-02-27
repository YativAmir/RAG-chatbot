import pinecone
import os
from sentence_transformers import SentenceTransformer
import docx2txt
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone with your API key and environment
PINECONE_API_KEY = "API_KEY"
PINECONE_ENV = "us-east-1"  # Adjust as needed

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

# -----------------------------------------------------------------
# Use BAAI/bge-m3 for embedding generation
# -----------------------------------------------------------------

# Change the model name to BAAI/bge-m3 (or a more specific variant if needed)
model_name = "BAAI/bge-m3"
model = SentenceTransformer(model_name)

test_embedding = model.encode("Test sentence")
print("Embedding dimension:", len(test_embedding))


index_name = "learning2040civics"
embedding_dimension = len(test_embedding)  # dynamically determine dimension

if index_name not in pc.list_indexes().names():
    pinecone.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

# Connect to the index
index = pc.Index(index_name)

# -----------------------------------------------------------------
# Process the documents using docx2txt (adjust paths as needed)
# -----------------------------------------------------------------
KnessetRoles         = "C:/Users/Administrator/Desktop/Learning2040/files from old pc/Civics Material/Word Files/Parliamentary Roles+Representation.docx"
equalityInLaw        = "C:/Users/Administrator/Desktop/Learning2040/files from old pc/Civics Material/Word Files/Equality Before the Law.docx"
Metzada              = "C:/Users/Administrator/Desktop/Learning2040/files from old pc/Civics Material/Word Files/Event Case-Masada.docx"
nations              = "C:/Users/Administrator/Desktop/Learning2040/files from old pc/Civics Material/Word Files/Event Case-National Sovereignty+Ethnic and Political Nation.docx"
Democracy            = "C:/Users/Administrator/Desktop/Learning2040/files from old pc/Civics Material/Word Files/What is Democracy.docx"
MegilatHaatmaot      = "C:/Users/Administrator/Desktop/Learning2040/files from old pc/Civics Material/Word Files/Declaration of Independence.docx"
RelijiosJudahisem    = "C:/Users/Administrator/Desktop/Learning2040/files from old pc/Civics Material/Word Files/The Religious Jewish View on the Establishment of the State.docx"
startUpNation        = "C:/Users/Administrator/Desktop/Learning2040/files from old pc/Civics Material/Word Files/Background to the Establishment of the State.docx"
jujment              = "C:/Users/Administrator/Desktop/Learning2040/files from old pc/Civics Material/Word Files/The Judiciary.docx"
backround            = "C:/Users/Administrator/Desktop/Learning2040/files from old pc/Civics Material/Word Files/Introduction.docx"
agreementsLaws       = "C:/Users/Administrator/Desktop/Learning2040/files from old pc/Civics Material/Word Files/Agreements, Laws, Symbols, and Institutions.docx"
creatingLaws         = "C:/Users/Administrator/Desktop/Learning2040/files from old pc/Civics Material/Word Files/The Lawmaking Process.docx"
basicRights          = "C:/Users/Administrator/Desktop/Learning2040/files from old pc/Civics Material/Word Files/Basic Rights+Shoshi Heller.docx"
majorityDecision     = "C:/Users/Administrator/Desktop/Learning2040/files from old pc/Civics Material/Word Files/Examples of Majority Decision.docx"
govvermentLimitations= "C:/Users/Administrator/Desktop/Learning2040/files from old pc/Civics Material/Word Files/Example of Government Limitation Principle.docx"
govvermentSupervion  = "C:/Users/Administrator/Desktop/Learning2040/files from old pc/Civics Material/Word Files/Supervision and Oversight of Government Work.docx"


# Process the documents
govvermentSupervion = docx2txt.process(govvermentSupervion)
govvermentLimitations = docx2txt.process(govvermentLimitations)
majorityDecision = docx2txt.process(majorityDecision)
basicRights = docx2txt.process(basicRights)
creatingLaws = docx2txt.process(creatingLaws)
agreementsLaws = docx2txt.process(agreementsLaws)
backround = docx2txt.process(backround)
jujment = docx2txt.process(jujment)
startUpNation = docx2txt.process(startUpNation)
RelijiosJudahisem = docx2txt.process(RelijiosJudahisem)
MegilatHaatmaot = docx2txt.process(MegilatHaatmaot)
Democracy = docx2txt.process(Democracy)
nations = docx2txt.process(nations)
Metzada = docx2txt.process(Metzada)
equalityInLaw = docx2txt.process(equalityInLaw)
KnessetRoles = docx2txt.process(KnessetRoles)

# Combine all documents into chunks (each document is one chunk in this example)
chunks = [
    govvermentSupervion,
    govvermentLimitations,
    majorityDecision,
    basicRights,
    creatingLaws,
    agreementsLaws,
    backround,
    jujment,
    startUpNation,
    RelijiosJudahisem,
    MegilatHaatmaot,
    Democracy,
    nations,
    Metzada,
    equalityInLaw,
    KnessetRoles,
]

# Associate metadata (e.g., source names) with each chunk
sources = [
    "govvermentSupervion",
    "govvermentLimitations",
    "majorityDecision",
    "basicRights",
    "creatingLaws",
    "agreementsLaws",
    "backround",
    "jujment",
    "startUpNation",
    "RelijiosJudahisem",
    "MegilatHaatmaot",
    "Democracy",
    "nations",
    "Metzada",
    "equalityInLaw",
    "KnessetRoles",
]
metadata = [{"text": chunk, "source": source} for chunk, source in zip(chunks, sources)]

# Create unique IDs for each chunk
ids = [f"chunk-{i}" for i in range(len(chunks))]

# Generate embeddings for each chunk using the BAAI model
vectors = [model.encode(chunk).tolist() for chunk in chunks]

# Upsert the vectors (with metadata) into the Pinecone index.
index.upsert(vectors=zip(ids, vectors, metadata))
print("Data uploaded to Pinecone successfully!")


# Function to neatly print query results
def print_retriever_data(results):
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

# Define a query (Hebrew text) and embed it with the same model
query_text = "מהו תהליך חקיקת החוק?"
query_embedding = model.encode(query_text).tolist()


# Query the Pinecone index for the top 3 matching vectors
results = index.query(vector=query_embedding, top_k=3, include_metadata=True)



print("Query Results:")
print_retriever_data(results)

# Optionally, print index stats
print("Total vectors in Pinecone:", index.describe_index_stats())


query_text = "מהם התפקידים בגצ?"
query_embedding = model.encode(query_text).tolist()

results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

print("Query Results:")
print_retriever_data(results)

# Optionally, print index stats
print("Total vectors in Pinecone:", index.describe_index_stats())