import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize Qdrant client with an increased timeout
qdrant_client = QdrantClient(
    url="https://055211ef-ed58-4ea2-8c81-2398365ff2f3.europe-west3-0.gcp.cloud.qdrant.io",
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=120  # Increased timeout (in seconds)
)

# Define the path to the extracted text file
extracted_text_file_path = 'extracted_text.txt'

# Function to split text into smaller chunks (approximately one paragraph or less, with character length limit)
def split_text_into_smaller_chunks(text, max_chunk_length=500):
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]  # Split text into paragraphs and remove empty ones
    chunks = []

    # Split each paragraph into smaller chunks if it exceeds max_chunk_length
    for paragraph in paragraphs:
        while len(paragraph) > max_chunk_length:
            split_point = paragraph[:max_chunk_length].rfind(' ')  # Find the last space within the max_chunk_length
            if split_point == -1:
                split_point = max_chunk_length  # If no space is found, split at the max_chunk_length
            chunks.append(paragraph[:split_point])
            paragraph = paragraph[split_point:].strip()
        if paragraph:
            chunks.append(paragraph)

    return chunks

def store_text_in_qdrant(file_path, collection_name="text_chunks2", batch_size=10):
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Initialize the improved text embedding model for better QA tasks
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    # Read text from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split text into smaller chunks with a maximum length
    chunks = split_text_into_smaller_chunks(text, max_chunk_length=500)
    logging.info(f"Text split into {len(chunks)} smaller semantic chunks")

    # Generate vector embeddings for each chunk
    embeddings = model.encode(chunks)
    logging.info(f"Generated {len(embeddings)} embeddings")

    # Create collection if it doesn't exist
    if collection_name not in [col.name for col in qdrant_client.get_collections().collections]:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
        )
        logging.info(f"Collection '{collection_name}' created")

    # Prepare points for upsert
    points = [
        PointStruct(id=i, vector=vector.tolist(), payload={"text": chunks[i]})
        for i, vector in enumerate(embeddings)
    ]
    logging.info(f"Created {len(points)} points for upsert")

    # Store embeddings in Qdrant in batches
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        qdrant_client.upsert(collection_name=collection_name, points=batch)
        logging.info(f"Upserted batch {i // batch_size + 1}")

# Example usage
store_text_in_qdrant(extracted_text_file_path)
