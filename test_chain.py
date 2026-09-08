import os
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from app.indexing.indexer import load_index, ROLE_PERMISSIONS
from app.chains.retrieval_chain import build_retrieval_chain, run_conversation

load_dotenv()

Settings.llm = Anthropic(
    model="claude-3-5-haiku-20241022",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1024,
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="all-MiniLM-L6-v2"
)

print("Loading index from Neon...")
index = load_index()

role = "senior_analyst"
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

print(f"Building LangChain chain for role: {role}")
chain, session_id = build_retrieval_chain(retriever, session_id=role)

questions = [
    "What is Westpac's total digital transformation investment?",
    "What are the three pillars of that strategy?",
    "How much is allocated to mortgage processing AI specifically?",
]

run_conversation(chain, session_id, questions)