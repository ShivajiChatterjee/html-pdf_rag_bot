import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import requests
import json
import time

# Load environment variables
load_dotenv()

# Load the improved embedding model for question-answering
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# Initialize Qdrant client
qdrant_api_key = os.getenv("QDRANT_API_KEY")
llama_api_key = os.getenv("LLAMA_API_KEY")  # Get Llama API key from environment
qdrant_url = "https://055211ef-ed58-4ea2-8c81-2398365ff2f3.europe-west3-0.gcp.cloud.qdrant.io"  # Hardcoded Qdrant URL
llama_api_url = os.getenv("LLAMA_API_URL")  # Get Llama API URL from environment

def get_qdrant_client():
    try:
        client = QdrantClient(api_key=qdrant_api_key, url=qdrant_url)
        client.get_collections()  # Test the connection without printing collections
        return client
    except Exception as e:
        st.write(f"Error connecting to Qdrant: {repr(e)}")
        return None

qdrant_client = get_qdrant_client()

def count_tokens(text):
    # Simple approximation of token count
    return len(text.split())

# Function to retrieve top chunks from Qdrant while considering token limits
def retrieve_top_chunks_from_qdrant(query, collection_name="text_chunks2", top_k=10, max_context_tokens=8192,
                                    max_response_tokens=300):
    if qdrant_client is None:
        st.write("Qdrant client is not initialized.")
        return None
    try:
        query_vector = model.encode([query])[0].tolist()  # Convert numpy array to list
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        if search_result:
            chunks = [res.payload['text'] for res in search_result]
            return chunks
        else:
            return None
    except Exception as e:
        st.write(f"Error retrieving from Qdrant: {e}")
        return None

# Function to call the Llama model API with streaming enabled
def get_llama_response(query, context):
    headers = {
        'api-key': llama_api_key,  # Use the API key from environment variable
        'Content-Type': 'application/json'
    }
    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "max_tokens": "300",
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "You are an intelligent AI assistant. Use ONLY the provided document context to accurately answer the user's question. If the context doesn't contain the answer, reply with 'The answer is not found in the provided context.' Do not provide any information that isn't explicitly stated in the context."
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nDocument Context: {context}"
            }
        ]
    }

    # Start timing the API call
    start_time = time.time()

    response = requests.post(llama_api_url, headers=headers, data=json.dumps(payload), stream=True)

    if response.status_code == 200:
        result = []
        # Stream the response in chunks
        for chunk in response.iter_lines():
            if chunk:
                result.append(json.loads(chunk.decode('utf-8'))['choices'][0]['message']['content'])
                # Continuously update the result as chunks arrive
                st.write(''.join(result))

        # End timing the API call
        end_time = time.time()
        return ''.join(result)
    else:
        st.write(f"Error calling the Llama API: {response.status_code} - {response.text}")
        return None

# Streamlit app layout
st.title("Enhanced Qdrant Retrieval with Llama Model (Streaming Enabled)")

user_query = st.text_input("Enter your question:")

if user_query:
    top_chunks = retrieve_top_chunks_from_qdrant(user_query, top_k=10)

    if top_chunks:
        combined_context = "\n\n".join(top_chunks)

        # Step 2: Pass the retrieved chunks to the Llama model for answer generation
        llama_response = get_llama_response(user_query, combined_context)

        st.subheader("Llama Model Response:")
        if llama_response:
            st.write(llama_response)
        else:
            st.write("No response generated.")
    else:
        st.write("No relevant chunks found in Qdrant.")
