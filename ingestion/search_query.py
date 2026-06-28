import os
from dotenv import load_dotenv

from pinecone import Pinecone
from google import genai

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------

load_dotenv()

# --------------------------------------------------
# Gemini Client
# --------------------------------------------------

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# --------------------------------------------------
# Pinecone
# --------------------------------------------------

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

index = pc.Index("medirag-gemini")

# --------------------------------------------------
# Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.2
)

# --------------------------------------------------
# Retrieval
# --------------------------------------------------

def retrieve(query: str, top_k: int = 10):

    # Generate Gemini embedding for query
    embedding = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )

    query_vector = embedding.embeddings[0].values

    # Search Pinecone
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace="example-namespace"
    )

    chunks = []

    print("\nRetrieved Chunks\n")

    for match in results.matches:

        print("=" * 80)
        print("Score :", match.score)
        print("Page  :", match.metadata.get("page"))
        print(match.metadata.get("text"))

        chunks.append({
            "text": match.metadata.get("text", ""),
            "source": match.metadata.get("source", ""),
            "page": match.metadata.get("page", ""),
            "section": match.metadata.get("section", ""),
            "score": match.score
        })

    return chunks


# --------------------------------------------------
# Build Context
# --------------------------------------------------

def build_context(chunks):

    context = []

    for i, chunk in enumerate(chunks, 1):

        context.append(

            f"[Chunk {i} | "
            f"Page {chunk['page']} | "
            f"Score {chunk['score']:.4f}]\n\n"

            f"{chunk['text']}"

        )

    return "\n\n-----------------------\n\n".join(context)


# --------------------------------------------------
# Answer
# --------------------------------------------------

def answer(query):

    chunks = retrieve(query)

    context = build_context(chunks)

    messages = [

        SystemMessage(
            content="""
You are an educational institution RAG assistant.

Answer ONLY from the retrieved context.

If the answer is not present in the context,
say you don't have enough information.

Always mention the page number if available.
"""
        ),

        HumanMessage(
            content=f"""
Context:

{context}

Question:

{query}
"""
        )

    ]

    response = llm.invoke(messages)

    return response.content


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    query = input("Enter your query: ")

    answer_text = answer(query)

    print("\nGenerated Answer:\n")
    print(answer_text)