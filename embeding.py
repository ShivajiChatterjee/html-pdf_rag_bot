from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import os

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url="https://055211ef-ed58-4ea2-8c81-2398365ff2f3.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="HeVNW3UVWuReNg30uD7GMxH-0CXNHyvuwxC2Hjjsn7Ph4ERjP1d37A",
)

# Define the path to the extracted text file
extracted_text_file_path = 'extracted_text.txt'

def store_text_in_qdrant(file_path, collection_name="text_chunks"):
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Initialize your text embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Read text from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Generate vector embeddings
    embeddings = model.encode([text])

    # Create collection if not exists
    if collection_name not in [col.name for col in qdrant_client.get_collections().collections]:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
        )

    # Store embeddings in Qdrant
    points = [PointStruct(id=i, vector=vector.tolist(), payload={"text": text}) for i, vector in enumerate(embeddings)]
    qdrant_client.upsert(collection_name=collection_name, points=points)

# Example usage
store_text_in_qdrant(extracted_text_file_path)
