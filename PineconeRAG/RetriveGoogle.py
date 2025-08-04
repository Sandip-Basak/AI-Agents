from pinecone import Pinecone
import google.generativeai as genai

pc = Pinecone(api_key="API_KEY")

# To get the unique host for an index, 
# see https://docs.pinecone.io/guides/manage-data/target-an-index


# index = pc.Index(host="https://my-rag-index-ks1v9lh.svc.aped-4627-b74a.pinecone.io")

index_name = "my-rag-index" # Use the name of your pre-built index
genai.configure(api_key="API_KEY")

# Note: 'models/embedding-001' or 'models/text-embedding-004' are common.
# Check current Gemini embedding model dimensions.
gemini_embedding_model = "models/embedding-001"
# 'models/embedding-001' typically has 768 dimensions.

if index_name in pc.list_indexes().names():
    index = pc.Index(index_name)
    print(f"Successfully connected to index: {index_name}")
    print(f"Index description: {index.describe_index_stats()}")
else:
    print(f"Error: Index '{index_name}' does not exist. Please check the index name.")

query = "Tell me about Marcus King"
query_response = genai.embed_content(model=gemini_embedding_model, content=query)
query_embedding = query_response['embedding']

# Perform similarity search
results = index.query(
    vector=query_embedding,
    top_k=3,
    namespace="Naamspace",
    include_metadata=True
)

print("\nRetrieval Results (Gemini):")
for match in results.matches:
    print(f"Score: {match.score:.2f}")
    print(f"Text: {match.metadata['text']}\n")