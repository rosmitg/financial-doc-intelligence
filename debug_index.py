from app.indexing.indexer import load_index

index = load_index()

# Check what metadata the chunks actually have
retriever = index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve("digital transformation strategy")

print(f"Found {len(nodes)} nodes")
for node in nodes:
    print(f"\nScore: {node.score}")
    print(f"Metadata: {node.metadata}")
    print(f"Text preview: {node.text[:100]}")
