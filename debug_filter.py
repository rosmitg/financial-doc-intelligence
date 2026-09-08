import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from app.indexing.indexer import load_index

load_dotenv()

Settings.llm = Anthropic(
    model="claude-3-5-haiku-20241022",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1024,
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="all-MiniLM-L6-v2"
)

index = load_index()

# Test 1 — no filter at all
print("TEST 1: No filter")
retriever = index.as_retriever(similarity_top_k=3)
nodes = retriever.retrieve("digital transformation strategy")
print(f"Found {len(nodes)} nodes")
for n in nodes:
    print(f"  classification: {n.metadata.get('classification')} | score: {n.score:.3f}")

# Test 2 — single PUBLIC filter
print("\nTEST 2: Single PUBLIC filter")
filters = MetadataFilters(filters=[
    MetadataFilter(
        key="classification",
        value="PUBLIC",
        operator=FilterOperator.EQ,
    )
])
retriever = index.as_retriever(similarity_top_k=3, filters=filters)
nodes = retriever.retrieve("digital transformation strategy")
print(f"Found {len(nodes)} nodes")
for n in nodes:
    print(f"  classification: {n.metadata.get('classification')} | score: {n.score:.3f}")

# Test 3 — IN operator
print("\nTEST 3: IN operator with ['PUBLIC', 'INTERNAL']")
try:
    filters = MetadataFilters(filters=[
        MetadataFilter(
            key="classification",
            value=["PUBLIC", "INTERNAL"],
            operator=FilterOperator.IN,
        )
    ])
    retriever = index.as_retriever(similarity_top_k=3, filters=filters)
    nodes = retriever.retrieve("digital transformation strategy")
    print(f"Found {len(nodes)} nodes")
    for n in nodes:
        print(f"  classification: {n.metadata.get('classification')} | score: {n.score:.3f}")
except Exception as e:
    print(f"Error: {e}")

