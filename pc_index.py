from pinecone import Pinecone, ServerlessSpec

# Create a Pinecone object
pc = Pinecone(api_key="pcsk_7A9iSP_CBAVcTci8VLZfhMeyaWCHYyrMw8fWBSA3CVSJC9Gzefbjf2Xtd6vXqzYZqMyax5")
# Create new index
pc.create_index(
    name="choreo-chatbot",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)