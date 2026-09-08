import os
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

from app.indexing.indexer import (
    load_index,
    get_query_engine,
    ROLE_PERMISSIONS,
    get_vector_store,
)
from app.chains.retrieval_chain import build_retrieval_chain

load_dotenv()

# ─────────────────────────────────────────
# Configure LlamaIndex settings
# ─────────────────────────────────────────

Settings.llm = Anthropic(
    model="claude-3-5-haiku-20241022",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1024,
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="all-MiniLM-L6-v2"
)

# ─────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────

app = FastAPI(
    title="Financial Document Intelligence API",
    description="RAG system with RBAC for financial documents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# Load index on startup
# ─────────────────────────────────────────

index = None

@app.on_event("startup")
async def startup():
    global index
    print("Loading index from Neon pgvector...")
    index = load_index()
    print("Index loaded — API ready")


# ─────────────────────────────────────────
# Request/Response models
# ─────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    role: str = "public"
    session_id: Optional[str] = "default"


class QueryResponse(BaseModel):
    answer: str
    role: str
    allowed_classifications: list
    latency_ms: float
    sources: list


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    roles_available: list


# ─────────────────────────────────────────
# Role validation
# ─────────────────────────────────────────

def validate_role(role: str) -> str:
    """Validate the requested role exists."""
    if role not in ROLE_PERMISSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role: {role}. "
                   f"Valid roles: {list(ROLE_PERMISSIONS.keys())}"
        )
    return role


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — confirms API and index are ready."""
    return HealthResponse(
        status="healthy",
        index_loaded=index is not None,
        roles_available=list(ROLE_PERMISSIONS.keys()),
    )


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """
    Ask a question about financial documents.
    Access is controlled by role — each role
    can only retrieve documents it is permitted to see.
    """
    if index is None:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded yet — try again in a moment"
        )

    role = validate_role(request.role)
    start = time.time()

    # Build role-filtered retriever
    allowed = ROLE_PERMISSIONS[role]
    filters = MetadataFilters(filters=[
        MetadataFilter(
            key="classification",
            value=allowed,
            operator=FilterOperator.IN,
        )
    ])

    retriever = index.as_retriever(
        similarity_top_k=3,
        filters=filters,
    )

    # Get query engine with RBAC
    engine = get_query_engine(index, role)

    # Query
    response = engine.query(request.question)

    # Get source documents for transparency
    nodes = retriever.retrieve(request.question)
    sources = [
        {
            "file": n.metadata.get("file_name", "unknown"),
            "classification": n.metadata.get("classification"),
            "score": round(n.score, 3) if n.score else 0,
        }
        for n in nodes
    ]

    latency_ms = round((time.time() - start) * 1000, 2)

    return QueryResponse(
        answer=str(response),
        role=role,
        allowed_classifications=allowed,
        latency_ms=latency_ms,
        sources=sources,
    )


@app.get("/roles")
async def get_roles():
    """List available roles and their permissions."""
    return {
        "roles": {
            role: {
                "classifications": perms,
                "description": {
                    "public": "External users — public documents only",
                    "junior_analyst": "Staff — public and internal documents",
                    "senior_analyst": "Senior staff — including confidential",
                    "executive": "Leadership — all documents",
                }.get(role, "")
            }
            for role, perms in ROLE_PERMISSIONS.items()
        }
    }