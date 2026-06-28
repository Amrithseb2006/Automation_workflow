import os
import uuid
import shutil
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone, ServerlessSpec
from google.genai import types

load_dotenv()

# ── In-memory job store ───────────────────────────────────────────
# Maps job_id → { status, message, total, processed, error }
ingestion_jobs: dict[str, dict] = {}

# ── App setup ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure temp upload dir exists
    os.makedirs("tmp_uploads", exist_ok=True)
    yield
    # Shutdown: clean up
    shutil.rmtree("tmp_uploads", ignore_errors=True)

app = FastAPI(
    title="MediRAG API",
    description="RAG backend for WHO clinical guideline PDFs",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic schemas ──────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class ChunkResult(BaseModel):
    text: str
    source: str
    page: int | str
    section: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    chunks: list[ChunkResult]

class IngestStatusResponse(BaseModel):
    job_id: str
    status: str          # "pending" | "running" | "done" | "failed"
    message: str
    total: int
    processed: int
    error: Optional[str] = None

# ── Ingestion helpers (imported lazily to avoid slow startup) ─────
def _run_ingestion(job_id: str, file_path: str, use_image_captions: bool):
    """
    Blocking function run in a thread via asyncio.to_thread.
    Updates ingestion_jobs[job_id] throughout.
    """
    try:
        # Lazy imports so startup is fast
        import pymupdf
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from pinecone import Pinecone

        job = ingestion_jobs[job_id]
        job["status"] = "running"

        # ── Step 1: Load text ────────────────────────────────────
        job["message"] = "Extracting text from PDF..."
        from load import load_pdf
        text_docs = load_pdf(file_path)
        job["message"] = f"Extracted {len(text_docs)} text pages."

        # ── Step 2 (optional): Image captions ───────────────────
        caption_docs = []
        if use_image_captions:
            job["message"] = "Extracting images..."
            from multimodal_info import extract_images_from_pdf
            image_docs = extract_images_from_pdf(file_path)
            job["message"] = f"Captioning {len(image_docs)} images with Gemini..."
            from image_to_text import images_to_text_gemini
            caption_docs = images_to_text_gemini(image_docs)
            job["message"] = f"Generated {len(caption_docs)} image captions."

        all_docs = text_docs + caption_docs

        # ── Step 3: Chunk ────────────────────────────────────────
        job["message"] = "Chunking documents..."
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
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
            for chunk in splitter.split_text(doc["text"]):
                chunked_docs.append({"text": chunk, "metadata": doc["metadata"]})

        job["total"] = len(chunked_docs)
        job["message"] = f"Created {len(chunked_docs)} chunks."

        # ── Step 4: Upsert to Pinecone ───────────────────────────
        job["message"] = "Connecting to Pinecone..."
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        index_name = "medirag-gemini"

        if not pc.has_index(index_name):

            job["message"] = "Creating Pinecone Index..."

            pc.create_index(

                name=index_name,

                dimension=3072,

                metric="cosine",

                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        index = pc.Index(index_name)

        records = []

        for doc in chunked_docs:

            embedding = client.models.embed_content(

                model="gemini-embedding-001",

                contents=doc["text"],

                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"
                )

            )

            vector = embedding.embeddings[0].values

            meta = doc["metadata"]

            records.append({

                "id": str(uuid.uuid4()),

                "values": vector,

                "metadata": {

                    "text": doc["text"],

                    "doc_id": str(meta.get("doc_id","")),

                    "page": int(meta.get("page",0)),

                    "section": str(meta.get("section","")),

                    "source": str(meta.get("source","")),

                    "type": str(meta.get("type",""))

                }

            })

        batch_size = 50
        for i in range(0, len(records), batch_size):
            index.upsert(

                vectors=records[i:i+batch_size],

                namespace="example-namespace"

            )
            job["processed"] = min(i + batch_size, len(records))
            job["message"] = f"Upserted {job['processed']}/{len(records)} chunks..."

        job["status"]  = "done"
        job["message"] = f"Ingestion complete. {len(records)} chunks indexed."
        job["processed"] = len(records)

    except Exception as e:
        ingestion_jobs[job_id]["status"] = "failed"
        ingestion_jobs[job_id]["error"]  = str(e)
        ingestion_jobs[job_id]["message"] = "Ingestion failed."

    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)


# ── Routes ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "MediRAG API"}

@app.delete("/clear")
def clear_index():
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        index = pc.Index("medirag-gemini")  
        index.delete(delete_all=True, namespace="example-namespace")  

        return {"status": "ok", "message": "Index cleared."}

    except Exception as e:
        print("CLEAR ERROR:", str(e))  
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", status_code=202)
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    use_image_captions: bool = False,
):
    """
    Upload a PDF and kick off the ingestion pipeline in the background.
    Returns a job_id you can poll via GET /ingest/status/{job_id}.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save upload to disk temporarily
    job_id   = str(uuid.uuid4())
    tmp_path = f"tmp_uploads/{job_id}_{file.filename}"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ingestion_jobs[job_id] = {
        "status":    "pending",
        "message":   "Job queued.",
        "total":     0,
        "processed": 0,
        "error":     None,
    }

    # Run blocking ingestion in a thread so the event loop stays free
    background_tasks.add_task(
        asyncio.to_thread,
        _run_ingestion,
        job_id,
        tmp_path,
        use_image_captions,
    )

    return {"job_id": job_id, "message": "Ingestion started."}


@app.get("/ingest/status/{job_id}", response_model=IngestStatusResponse)
def ingest_status(job_id: str):
    """Poll ingestion progress."""
    job = ingestion_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return IngestStatusResponse(job_id=job_id, **job)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Retrieve relevant chunks from Pinecone and generate an answer via Groq.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        from pinecone import Pinecone
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage

        # ── Retrieve ─────────────────────────────────────────────
        pc    = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("medirag-gemini")

        client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY")
        )

        embedding = client.models.embed_content(
            model="gemini-embedding-001",
            contents=req.query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY"
            )
        )

        query_vector = embedding.embeddings[0].values

        results = index.query(
            vector=query_vector,
            top_k=10,
            include_metadata=True,
            namespace="example-namespace"
        )

        chunks = []
        for match in results.matches:
            meta = match.metadata
            chunks.append(ChunkResult(
            text=meta.get("text",""),
            source=meta.get("source",""),
            page=meta.get("page",""),
            section=meta.get("section",""),
            score=match.score
        ))

        if not chunks:
            return QueryResponse(
                answer="No relevant context found in the knowledge base.",
                chunks=[]
            )

        # ── Build context ─────────────────────────────────────────
        context_parts = []
        for i, c in enumerate(chunks, 1):
            context_parts.append(
                f"[Chunk {i} | source: {c.source} | page: {c.page} | score: {c.score:.3f}]\n"
                f"{c.text}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # ── Generate ──────────────────────────────────────────────
        llm = ChatGroq(
            api_key    = os.getenv("GROQ_API_KEY"),
            model      = "llama-3.3-70b-versatile",
            temperature= 0.2,
        )

        messages = [
            SystemMessage(content=(
                "You are an educational institution RAG assistant."

                "Answer ONLY using the retrieved context."
                "If the information is unavailable in the retrieved context,"
                "clearly say that."

                "Always mention the source document and page number if available."
            )),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {req.query}"),
        ]

        response = llm.invoke(messages)
        return QueryResponse(answer=response.content, chunks=chunks)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
