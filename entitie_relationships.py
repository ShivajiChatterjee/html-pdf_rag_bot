import os
import json
import logging
import time
import requests
import re
from qdrant_client import QdrantClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Qdrant client with custom timeout (e.g., 90 seconds)
qdrant_client = QdrantClient(
    url="https://055211ef-ed58-4ea2-8c81-2398365ff2f3.europe-west3-0.gcp.cloud.qdrant.io",  # Direct Qdrant URL
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=90  # Set custom timeout to 90 seconds
)

# Define your collection name in Qdrant
collection_name = "text_chunks2"

# Prompt template for extracting entities and relationships without descriptions
KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
Format each entity as ("entity"$$$$<entity_name>$$$$<entity_type>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

Format each relationship as ("relationship"$$$$<source_entity>$$$$<target_entity>$$$$<relation>$$$$<relationship_description>)

3. When finished, output.

-Real Data-
######################
text: {text}
######################
output:
"""

# Function to retrieve chunks from Qdrant using offset-based pagination
def retrieve_chunks_from_qdrant(offset=0, limit=5):
    try:
        # Retrieve points from the collection using pagination (offset and limit)
        response = qdrant_client.scroll(
            collection_name=collection_name,
            offset=offset,
            limit=limit
        )

        chunks = [point.payload['text'] for point in response[0]]
        logging.info(f"Retrieved {len(chunks)} chunk(s) from Qdrant. Offset: {offset}")

        return chunks, len(chunks)

    except Exception as e:
        logging.error(f"Error while retrieving chunks from Qdrant: {str(e)}")
        return [], 0

# Function to send a text chunk to LLaMA API with retry logic and exponential backoff
def extract_entities_llm(text_chunk, max_knowledge_triplets=10, max_retries=3):
    api_url = 'https://rag-llm-api.accubits.cloud/v1/chat/completions'
    headers = {
        'api-key': os.getenv("LLAMA_API_KEY"),  # Ensure your LLaMA API key is in the .env file
        'Content-Type': 'application/json'
    }

    # Build the structured prompt
    prompt = KG_TRIPLET_EXTRACT_TMPL.format(max_knowledge_triplets=max_knowledge_triplets, text=text_chunk[:300])

    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are an information extraction assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
        "temperature": 0
    }

    attempt = 0
    backoff = 5  # Start with a 5-second backoff
    while attempt < max_retries:
        try:
            logging.info(f"Sending payload to LLaMA API for chunk: {text_chunk[:60]}...")  # Log a snippet of the chunk
            response = requests.post(api_url, headers=headers, json=payload, timeout=90)

            if response.status_code == 200:
                return response.json().get('choices')[0]['message']['content'].strip()
            else:
                logging.error(f"Error {response.status_code}: {response.text}")
                return None
        except requests.Timeout:
            attempt += 1
            logging.error(f"Timeout occurred while processing chunk (Attempt {attempt}/{max_retries}). Retrying after {backoff} seconds...")
            time.sleep(backoff)
            backoff *= 2  # Exponential backoff: double the wait time
        except Exception as e:
            logging.error(f"Error while calling the LLaMA API: {str(e)}")
            return None

    logging.error(f"Failed to process chunk after {max_retries} attempts.")
    return None

# Function to parse the LLM response, focusing on entities and relationships
def parse_llm_response(response):
    entity_pattern = r'\("entity"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'
    relationship_pattern = r'\("relationship"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'

    entities = re.findall(entity_pattern, response)
    relationships = re.findall(relationship_pattern, response)

    return {
        "entities": [{"entity_name": e[0], "entity_type": e[1]} for e in entities],  # No descriptions
        "relationships": [{"source_entity": r[0], "target_entity": r[1], "relation": r[2], "relationship_description": r[3]} for r in relationships]
    }

# Function to process a single chunk and save the result
def process_single_chunk(chunk):
    llm_response = extract_entities_llm(chunk)
    if llm_response:
        parsed_content = parse_llm_response(llm_response)
        result = {
            "text_chunk": chunk,
            "entities_and_relationships": parsed_content
        }
        return result
    return None

# Function to save the processed result to a single JSON file
def save_result_to_single_json(results, output_file='entities_and_relationships.json'):
    try:
        # If the file exists, load its contents and append the new result
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []

        data.extend(results)

        # Save the updated data back to the file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        logging.info(f"Saved results to {output_file}")

    except Exception as e:
        logging.error(f"Error while saving results to {output_file}: {str(e)}")

# Main function to process chunks using offset-based pagination and concurrent processing
def process_chunks_concurrently():
    offset = 0
    max_workers = 5  # Number of workers for concurrent processing
    while True:
        # Retrieve the next set of chunks using the offset
        chunks, num_chunks = retrieve_chunks_from_qdrant(offset=offset, limit=5)
        if chunks:
            # Use ThreadPoolExecutor to process chunks concurrently
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_chunk = {executor.submit(process_single_chunk, chunk): chunk for chunk in chunks}

                for future in as_completed(future_to_chunk):
                    result = future.result()
                    if result:
                        results.append(result)

            # Save the processed results to a single JSON file
            if results:
                save_result_to_single_json(results)

            # Increment offset based on number of chunks retrieved
            offset += num_chunks

        if num_chunks == 0:
            logging.info("Finished processing all chunks.")
            break  # Exit the loop if there are no more chunks to process

        logging.info("Waiting for 2 seconds before processing the next batch of chunks...")
        time.sleep(2)

# Run the process
if __name__ == "__main__":
    process_chunks_concurrently()
