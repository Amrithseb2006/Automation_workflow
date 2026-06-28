from load import load_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from google import genai

import os
import uuid
from dotenv import load_dotenv

# ----------------------------------------------------
# Load Environment Variables
# ----------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ----------------------------------------------------
# Initialize Gemini
# ----------------------------------------------------
client = genai.Client(api_key=GEMINI_API_KEY)

# ----------------------------------------------------
# Initialize Pinecone
# ----------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "medirag-gemini"

# Gemini embedding dimension
EMBED_DIMENSION = 3072

if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)

# ----------------------------------------------------
# Load PDF
# ----------------------------------------------------
file_path = r"C:\Users\ashwi\Amrith\Seekie\ingestion\Curriculum_BTech_CSE_IoT-R2025.pdf"

all_docs = load_pdf(file_path)

print(f"Loaded {len(all_docs)} pages.")

# ----------------------------------------------------
# Chunking
# ----------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

chunked_docs = []

for doc in all_docs:

    chunks = text_splitter.split_text(doc["text"])

    for chunk in chunks:

        chunked_docs.append(
            {
                "text": chunk,
                "metadata": doc["metadata"]
            }
        )

print(f"Created {len(chunked_docs)} chunks.")

# ----------------------------------------------------
# Save chunks for debugging
# ----------------------------------------------------
with open("chunked_docs_debug.txt", "w", encoding="utf-8") as f:

    for i, chunk in enumerate(chunked_docs):

        f.write("=" * 80 + "\n")
        f.write(f"Chunk {i+1}\n")
        f.write(f"Metadata: {chunk['metadata']}\n")
        f.write("-" * 80 + "\n")
        f.write(chunk["text"])
        f.write("\n\n")

print("Saved chunked_docs_debug.txt")

# ----------------------------------------------------
# Create Embeddings
# ----------------------------------------------------
records = []

for i, doc in enumerate(chunked_docs):

    print(f"Embedding chunk {i+1}/{len(chunked_docs)}")

    embedding = client.models.embed_content(
        model="gemini-embedding-001",
        contents=doc["text"]
    )

    vector = embedding.embeddings[0].values

    metadata = doc["metadata"]

    records.append(
        {
            "id": str(uuid.uuid4()),
            "values": vector,
            "metadata": {
                "text": doc["text"],
                "doc_id": metadata.get("doc_id", ""),
                "page": metadata.get("page", 0),
                "section": metadata.get("section", ""),
                "source": metadata.get("source", ""),
                "type": metadata.get("type", "")
            }
        }
    )

print(f"Generated {len(records)} embeddings.")

# ----------------------------------------------------
# Upload to Pinecone
# ----------------------------------------------------
batch_size = 50

for i in range(0, len(records), batch_size):

    batch = records[i:i + batch_size]

    index.upsert(
        vectors=batch,
        namespace="example-namespace"
    )

    print(
        f"Uploaded batch {(i//batch_size)+1} "
        f"({len(batch)} vectors)"
    )

print("\nIngestion Complete!")
print(f"Total vectors uploaded: {len(records)}")