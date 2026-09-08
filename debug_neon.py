import os
from urllib.parse import urlparse
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

load_dotenv()

Settings.llm = Anthropic(
    model="claude-3-5-haiku-20241022",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1024,
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="all-MiniLM-L6-v2"
)

url = urlparse(os.getenv("DATABASE_URL"))
vector_store = PGVectorStore.from_params(
    database=url.path.lstrip("/"),
    host=url.hostname,
    password=url.password,
    user=url.username,
    port=str(url.port or 5432),
    table_name="financial_documents",
    embed_dim=384,
    hybrid_search=False,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
)

# Test 1 — no filter
print("TEST 1: No filter")
retriever = index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve("digital transformation strategy")
print(f"Found {len(nodes)} nodes")
for n in nodes:
    print(f"  classification: {n.metadata.get('classification')} | score: {n.score:.3f}")

# Test 2 — single EQ filter
print("\nTEST 2: Single EQ filter PUBLIC")
filters = MetadataFilters(filters=[
    MetadataFilter(key="classification", value="PUBLIC", operator=FilterOperator.EQ)
])
retriever = index.as_retriever(similarity_top_k=5, filters=filters)
nodes = retriever.retrieve("financial report revenue")
print(f"Found {len(nodes)} nodes")
for n in nodes:
    print(f"  classification: {n.metadata.get('classification')} | score: {n.score:.3f}")

# Test 3 — IN filter
print("\nTEST 3: IN filter ['PUBLIC', 'INTERNAL']")
filters = MetadataFilters(filters=[
    MetadataFilter(
        key="classification",
        value=["PUBLIC", "INTERNAL"],
        operator=FilterOperator.IN,
    )
])
retriever = index.as_retriever(similarity_top_k=5, filters=filters)
nodes = retriever.retrieve("digital transformation strategy")
print(f"Found {len(nodes)} nodes")
for n in nodes:
    print(f"  classification: {n.metadata.get('classification')} | score: {n.score:.3f}")

