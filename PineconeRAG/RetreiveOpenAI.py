# Retrieve using OpenAI Embedding

from pinecone import Pinecone
from openai import OpenAI

pc = Pinecone(api_key="API_KEY")

# To get the unique host for an index, 
# see https://docs.pinecone.io/guides/manage-data/target-an-index


# index = pc.Index(host="https://demo-ks1v9lh.svc.aped-4627-b74a.pinecone.io")
index_name = "demo" # Use the name of your pre-built index

if index_name in pc.list_indexes().names():
    index = pc.Index(index_name)
    print(f"Successfully connected to index: {index_name}")
    print(f"Index description: {index.describe_index_stats()}")
else:
    print(f"Error: Index '{index_name}' does not exist. Please check the index name.")

# OpenAI Embeddings
openai_client = OpenAI(api_key="API_KEY")
# IMPORTANT: This must be the same model used when upserting data
openai_embedding_model = "text-embedding-ada-002"

query_text = "Tell me about the refund policy"
query_response = openai_client.embeddings.create(input=query_text, model=openai_embedding_model)
query_embedding = query_response.data[0].embedding
print(f"Generated OpenAI embedding for query: '{query_text}'")



# Query
try:
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True, # Essential to get the original text content back
        namespace="FAQ" # Uncomment if you used a namespace
    )

    print(f"\nRetrieved {len(results.matches)} results for query: '{query_text}'")
    print("---")
    for i, match in enumerate(results.matches):
        print(f"Result {i+1}:")
        print(f"  ID: {match.id}")
        print(f"  Score: {match.score:.4f}") # Similarity score
        if 'text' in match.metadata:
            print(f"  Text: {match.metadata['text']}\n")
        else:
            print(f"  Metadata: {match.metadata}\n")
    print("---")

except Exception as e:
    print(f"An error occurred during query: {e}")