
import os
import uuid
import shutil
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
    Form,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone, ServerlessSpec
from google.genai import types

load_dotenv()


# ============================================================
# IN-MEMORY INGESTION JOB STORE
# ============================================================

ingestion_jobs: dict[str, dict] = {}


# ============================================================
# FASTAPI APPLICATION SETUP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("tmp_uploads", exist_ok=True)
    yield
    shutil.rmtree("tmp_uploads", ignore_errors=True)


app = FastAPI(
    title="MediRAG API",
    description="RAG backend with Adaptive Retrieval Intelligence",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class QueryRequest(BaseModel):
    query: str
    namespace: str
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
    retrieval: dict = {}


class IngestStatusResponse(BaseModel):
    job_id: str
    status: str
    message: str
    total: int
    processed: int
    error: Optional[str] = None


# ============================================================
# PDF INGESTION PIPELINE
# ============================================================

def _run_ingestion(
    job_id: str,
    file_path: str,
    namespace: str,
    use_image_captions: bool,
):
    """
    Blocking ingestion function executed in a background thread.
    """

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from pinecone import Pinecone

        job = ingestion_jobs[job_id]
        job["status"] = "running"

        # ----------------------------------------------------
        # STEP 1: LOAD PDF TEXT
        # ----------------------------------------------------

        job["message"] = "Extracting text from PDF..."

        from load import load_pdf

        text_docs = load_pdf(file_path)

        job["message"] = (
            f"Extracted {len(text_docs)} text pages."
        )

        # ----------------------------------------------------
        # STEP 2: OPTIONAL IMAGE CAPTION EXTRACTION
        # ----------------------------------------------------

        caption_docs = []

        if use_image_captions:

            job["message"] = "Extracting images..."

            from multimodal_info import extract_images_from_pdf

            image_docs = extract_images_from_pdf(file_path)

            job["message"] = (
                f"Captioning {len(image_docs)} images with Gemini..."
            )

            from image_to_text import images_to_text_gemini

            caption_docs = images_to_text_gemini(image_docs)

            job["message"] = (
                f"Generated {len(caption_docs)} image captions."
            )

        all_docs = text_docs + caption_docs

        # ----------------------------------------------------
        # STEP 3: CHUNK DOCUMENTS
        # ----------------------------------------------------

        job["message"] = "Chunking documents..."

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

        chunked_docs = []

        for doc in all_docs:

            for chunk in splitter.split_text(doc["text"]):

                chunked_docs.append({
                    "text": chunk,
                    "metadata": doc["metadata"],
                })

        job["total"] = len(chunked_docs)

        job["message"] = (
            f"Created {len(chunked_docs)} chunks."
        )

        # ----------------------------------------------------
        # STEP 4: CONNECT TO PINECONE AND GEMINI
        # ----------------------------------------------------

        job["message"] = "Connecting to Pinecone..."

        pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY")
        )

        client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY")
        )

        index_name = "workflow-testing"

        if not pc.has_index(index_name):

            job["message"] = "Creating Pinecone index..."

            pc.create_index(
                name=index_name,
                dimension=3072,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                ),
            )

        index = pc.Index(index_name)

        # ----------------------------------------------------
        # STEP 5: GENERATE DOCUMENT EMBEDDINGS
        # ----------------------------------------------------

        records = []

        for doc in chunked_docs:

            embedding = client.models.embed_content(
                model="gemini-embedding-001",
                contents=doc["text"],
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"
                ),
            )

            vector = embedding.embeddings[0].values

            meta = doc["metadata"]

            records.append({
                "id": str(uuid.uuid4()),
                "values": vector,
                "metadata": {
                    "text": doc["text"],
                    "doc_id": str(meta.get("doc_id", "")),
                    "page": int(meta.get("page", 0)),
                    "section": str(meta.get("section", "")),
                    "source": str(meta.get("source", "")),
                    "type": str(meta.get("type", "")),
                },
            })

        # ----------------------------------------------------
        # STEP 6: UPSERT TO PINECONE
        # ----------------------------------------------------

        batch_size = 50

        for i in range(0, len(records), batch_size):

            index.upsert(
                vectors=records[i:i + batch_size],
                namespace=namespace,
            )

            job["processed"] = min(
                i + batch_size,
                len(records),
            )

            job["message"] = (
                f"Upserted {job['processed']}/{len(records)} chunks..."
            )

        job["status"] = "done"

        job["message"] = (
            f"Ingestion complete. {len(records)} chunks indexed."
        )

        job["processed"] = len(records)

    except Exception as e:

        ingestion_jobs[job_id]["status"] = "failed"
        ingestion_jobs[job_id]["error"] = str(e)
        ingestion_jobs[job_id]["message"] = "Ingestion failed."

    finally:

        if os.path.exists(file_path):
            os.remove(file_path)


# ============================================================
# API ROUTES
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "MediRAG API",
    }


# ============================================================
# CLEAR PINECONE NAMESPACE
# ============================================================

@app.delete("/clear")
def clear_index(namespace: str = Query(..., min_length=1)):

    try:

        pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY")
        )

        index = pc.Index("workflow-testing")

        index.delete(
            delete_all=True,
            namespace=namespace,
        )

        return {
            "status": "ok",
            "message": f"Namespace '{namespace}' cleared.",
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================
# PDF UPLOAD ENDPOINT
# ============================================================

@app.post("/ingest", status_code=202)
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    namespace: str = Form(...),
    use_image_captions: bool = Form(False),
):
    """
    Upload a PDF and start the ingestion pipeline.
    """

    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted.",
        )

    job_id = str(uuid.uuid4())

    os.makedirs("tmp_uploads", exist_ok=True)

    safe_filename = os.path.basename(file.filename)

    tmp_path = os.path.join(
        "tmp_uploads",
        f"{job_id}_{safe_filename}",
    )

    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ingestion_jobs[job_id] = {
        "status": "pending",
        "message": "Job queued.",
        "total": 0,
        "processed": 0,
        "error": None,
    }

    background_tasks.add_task(
        asyncio.to_thread,
        _run_ingestion,
        job_id,
        tmp_path,
        namespace,
        use_image_captions,
    )

    return {
        "job_id": job_id,
        "message": "Ingestion started.",
    }


# ============================================================
# INGESTION STATUS ENDPOINT
# ============================================================

@app.get(
    "/ingest/status/{job_id}",
    response_model=IngestStatusResponse,
)
def ingest_status(job_id: str):

    job = ingestion_jobs.get(job_id)

    if not job:

        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    return IngestStatusResponse(
        job_id=job_id,
        **job,
    )


# ============================================================
# ADAPTIVE RETRIEVAL + RAG QUERY ENDPOINT
# ============================================================

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Adaptive Retrieval Intelligence pipeline:

    1. Initial retrieval
    2. Retrieval quality assessment
    3. Conditional query rewriting
    4. Expanded retrieval
    5. Chunk deduplication
    6. Hybrid reranking
    7. Evidence-grounded answer generation
    """

    if not req.query.strip():

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    if not req.namespace.strip():

        raise HTTPException(
            status_code=400,
            detail="Namespace cannot be empty.",
        )

    try:

        from pinecone import Pinecone
        from langchain_groq import ChatGroq
        from langchain_core.messages import (
            SystemMessage,
            HumanMessage,
        )

        from adaptive_retrieval import (
            AdaptiveRetriever,
            RetrievalConfig,
        )

        # ----------------------------------------------------
        # STEP 1: INITIALIZE CLIENTS
        # ----------------------------------------------------

        pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY")
        )

        index = pc.Index("workflow-testing")

        client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY")
        )

        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="qwen/qwen3.8-27b",
            temperature=0.2,
        )

        # ----------------------------------------------------
        # STEP 2: GEMINI QUERY EMBEDDING FUNCTION
        # ----------------------------------------------------

        def embed_query(query_text: str):

            embedding = client.models.embed_content(
                model="gemini-embedding-001",
                contents=query_text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY"
                ),
            )

            return embedding.embeddings[0].values

        # ----------------------------------------------------
        # STEP 3: GROQ QUERY REWRITING FUNCTION
        # ----------------------------------------------------

        def rewrite_query(original_query: str):

            messages = [
                SystemMessage(
                    content=(
                        "You are a search query optimization assistant. "
                        "Rewrite the user's question into a clearer, "
                        "more specific search query for retrieving "
                        "relevant institutional documents. "
                        "Preserve the original intent and important "
                        "entities. Return only the rewritten query. "
                        "Do not answer the question."
                    )
                ),
                HumanMessage(
                    content=original_query
                ),
            ]

            response = llm.invoke(messages)

            return response.content.strip()

        # ----------------------------------------------------
        # STEP 4: CONFIGURE ADAPTIVE RETRIEVAL
        # ----------------------------------------------------

        config = RetrievalConfig(
            initial_top_k=max(1, min(req.top_k, 20)),
            expanded_top_k=10,
            final_top_k=max(1, min(req.top_k, 10)),
        )

        retriever = AdaptiveRetriever(
            index=index,
            embed_query=embed_query,
            rewrite_query=rewrite_query,
            config=config,
        )

        # ----------------------------------------------------
        # STEP 5: EXECUTE ADAPTIVE RETRIEVAL
        # ----------------------------------------------------

        retrieval_result = retriever.retrieve(
            query=req.query,
            namespace=req.namespace,
        )

        chunks = retrieval_result["chunks"]

        retrieval_info = retrieval_result["retrieval"]

        # ----------------------------------------------------
        # STEP 6: HANDLE EMPTY RETRIEVAL
        # ----------------------------------------------------

        if not chunks:

            return QueryResponse(
                answer=(
                    "No relevant context found in the knowledge base. "
                    "Please try rephrasing your question or check "
                    "whether documents have been uploaded."
                ),
                chunks=[],
                retrieval=retrieval_info,
            )

        # ----------------------------------------------------
        # STEP 7: BUILD EVIDENCE CONTEXT
        # ----------------------------------------------------

        context_parts = []

        for i, chunk in enumerate(chunks, 1):

            context_parts.append(
                f"[Chunk {i} | Source: {chunk['source']} "
                f"| Page: {chunk['page']} "
                f"| Relevance: {chunk['score']:.3f}]\n"
                f"{chunk['text']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # ----------------------------------------------------
        # STEP 8: GENERATE GROUNDED ANSWER
        # ----------------------------------------------------

        messages = [

            SystemMessage(
                content=(
                    "You are an educational institution RAG assistant. "
                    "Answer ONLY using the retrieved context. "
                    "Do not invent facts or make unsupported assumptions. "
                    "If the answer is unavailable in the context, "
                    "clearly state that the information was not found. "
                    "Use clear and concise language."
                )
            ),

            HumanMessage(
                content=(
                    f"Retrieved context:\n{context}\n\n"
                    f"Question: {req.query}"
                )
            ),

        ]

        response = llm.invoke(messages)

        # ----------------------------------------------------
        # STEP 9: RETURN ANSWER, SOURCES AND DIAGNOSTICS
        # ----------------------------------------------------

        return QueryResponse(
            answer=response.content,
            chunks=chunks,
            retrieval=retrieval_info,
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
