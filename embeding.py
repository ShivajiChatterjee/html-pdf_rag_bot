import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize Qdrant client with an increased timeout
qdrant_client = QdrantClient(
    url="https://055211ef-ed58-4ea2-8c81-2398365ff2f3.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=120  # Increased timeout (in seconds)
)

# Define the path to the extracted text file
extracted_text_file_path = 'extracted_text.txt'

# Function to split text into semantic chunks (by sentences)
def split_text_by_sentences(text, max_chunk_size=256):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def store_text_in_qdrant(file_path, collection_name="text_chunks", batch_size=10):
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Initialize the text embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Read text from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split text into semantic chunks
    chunks = split_text_by_sentences(text)
    logging.info(f"Text split into {len(chunks)} semantic chunks")

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
