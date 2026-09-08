import os
from urllib.parse import urlparse
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.core.query_engine import RetrieverQueryEngine
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

url = urlparse(os.getenv("DATABASE_URL"))
vector_store = PGVectorStore.from_params(
    database=url.path.lstrip("/"),
    host=url.hostname,
    password=url.password,
    user=url.username,
    port=str(url.port or 5432),
    table_name="financial_documents",
    embed_dim=384,
    hybrid_search=True,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
)

# Test 1 — retrieve chunks and print them
print("TEST 1: What chunks does PUBLIC role get?")
filters = MetadataFilters(filters=[
    MetadataFilter(
        key="classification",
        value=["PUBLIC"],
        operator=FilterOperator.IN,
    )
])
retriever = index.as_retriever(similarity_top_k=3, filters=filters)
nodes = retriever.retrieve("What is Westpac net profit and revenue?")
print(f"Retrieved {len(nodes)} nodes")
for n in nodes:
    print(f"\nScore: {n.score:.3f}")
    print(f"Text: {n.text[:200]}")

# Test 2 — call LLM directly with retrieved context
print("\n\nTEST 2: Call Claude directly with retrieved chunks")
if nodes:
    context = "\n\n".join([n.text for n in nodes])
    from anthropic import Anthropic as AnthropicClient
    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""Use only the context below to answer the question.
            
Context:
{context}

Question: What is Westpac net profit and revenue?"""
        }]
    )
    print(f"Claude response: {response.content[0].text}")

# Test 3 — use query engine
print("\n\nTEST 3: Query engine response")
engine = RetrieverQueryEngine.from_args(
    retriever,
    response_mode="compact",
    verbose=True,
)
response = engine.query("What is Westpac net profit and revenue?")
print(f"Engine response: {response}")
print(f"Source nodes: {len(response.source_nodes)}")

