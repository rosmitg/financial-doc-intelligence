import os
from urllib.parse import urlparse
from dotenv import load_dotenv
import anthropic

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

load_dotenv()

Settings.llm = Anthropic(
    model="claude-3-5-haiku-20241022",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1024,
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="all-MiniLM-L6-v2"
)

Settings.node_parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
)

DOCUMENT_METADATA = {
    "q3_2024_report.txt": {
        "document_type": "financial_report",
        "classification": "PUBLIC",
        "department": "finance",
        "year": 2024,
        "quarter": "Q3",
    },
    "risk_framework.txt": {
        "document_type": "risk_policy",
        "classification": "INTERNAL",
        "department": "risk_management",
        "year": 2024,
        "quarter": "Q3",
    },
    "digital_strategy.txt": {
        "document_type": "strategy",
        "classification": "CONFIDENTIAL",
        "department": "technology",
        "year": 2024,
        "quarter": "Q3",
    },
    "economic_outlook.txt": {
        "document_type": "research",
        "classification": "PUBLIC",
        "department": "economics",
        "year": 2024,
        "quarter": "Q3",
    },
}

ROLE_PERMISSIONS = {
    "public":          ["PUBLIC"],
    "junior_analyst":  ["PUBLIC", "INTERNAL"],
    "senior_analyst":  ["PUBLIC", "INTERNAL", "CONFIDENTIAL"],
    "executive":       ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"],
}


def get_vector_store() -> PGVectorStore:
    """Connect to pgvector on Neon using DATABASE_URL."""
    url = urlparse(os.getenv("DATABASE_URL"))
    return PGVectorStore.from_params(
        database=url.path.lstrip("/"),
        host=url.hostname,
        password=url.password,
        user=url.username,
        port=str(url.port or 5432),
        table_name="financial_documents",
        embed_dim=384,
        hybrid_search=True,
        text_search_config="english",
        indexed_metadata_keys={
            ("classification", "text"),
            ("department", "text"),
            ("document_type", "text"),
        },
    )


def enrich_metadata(documents):
    """Stamp classification and department onto each document."""
    for doc in documents:
        filename = doc.metadata.get("file_name", "")
        if filename in DOCUMENT_METADATA:
            doc.metadata.update(DOCUMENT_METADATA[filename])
        else:
            doc.metadata.update({
                "document_type": "unknown",
                "classification": "RESTRICTED",
                "department": "unknown",
            })
        print(f"  {filename} → "
              f"classification: {doc.metadata.get('classification')}, "
              f"department: {doc.metadata.get('department')}")
    return documents


def build_index(documents_dir: str = "data/documents") -> VectorStoreIndex:
    """Load, enrich, chunk, embed, and store in pgvector on Neon."""
    print(f"\nLoading documents from {documents_dir}...")
    documents = SimpleDirectoryReader(documents_dir).load_data()
    print(f"Loaded {len(documents)} documents:")

    print("\nEnriching metadata...")
    documents = enrich_metadata(documents)

    print("\nConnecting to Neon pgvector...")
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    print("\nBuilding index (chunking + embedding + storing in Neon)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    print("\nIndex built and stored in Neon pgvector")
    return index


def load_index() -> VectorStoreIndex:
    """Load index from Neon pgvector."""
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )
    print("Index loaded from Neon pgvector")
    return index


def get_query_engine(index: VectorStoreIndex, user_role: str):
    """
    Return a query engine filtered to what this role can access.
    Filter enforced at pgvector SQL level — restricted chunks
    never retrieved, never seen by the LLM.
    Uses direct Anthropic API to avoid LlamaIndex model compatibility issues.
    """
    allowed = ROLE_PERMISSIONS.get(user_role, ["PUBLIC"])

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

    class SimpleQueryEngine:
        def __init__(self, retriever):
            self._retriever = retriever
            self._client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )

        def query(self, question: str):
            nodes = self._retriever.retrieve(question)

            if not nodes:
                return "No relevant documents found for your query."

            context = "\n\n".join([
                f"[Source: {n.metadata.get('file_name', 'unknown')} | "
                f"Classification: {n.metadata.get('classification')}]\n"
                f"{n.text}"
                for n in nodes
            ])

            response = self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": f"""You are a financial analyst assistant.
Answer the question using ONLY the context provided below.
If the context does not contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
                }]
            )

            return response.content[0].text

    return SimpleQueryEngine(retriever)


if __name__ == "__main__":
    index = build_index()

    print("\n" + "=" * 50)
    print("RBAC TEST")
    print("=" * 50)

    tests = [
        (
            "public",
            "What is Westpac net profit and revenue this quarter?"
        ),
        (
            "junior_analyst",
            "What are the credit risk LVR policies?"
        ),
        (
            "senior_analyst",
            "What is the digital transformation investment and AI strategy?"
        ),
    ]

    for role, question in tests:
        print(f"\nRole: {role}")
        print(f"Allowed: {ROLE_PERMISSIONS[role]}")
        print(f"Question: {question}")
        engine = get_query_engine(index, role)
        response = engine.query(question)
        print(f"Response: {response}")
        print("-" * 40)