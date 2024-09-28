import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
import networkx as nx
from pyvis.network import Network
import requests
import json
import tempfile
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load environment variables
load_dotenv()

# Initialize models and clients
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
qdrant_api_key = os.getenv("QDRANT_API_KEY")
llama_api_key = os.getenv("LLAMA_API_KEY")
qdrant_url = "https://055211ef-ed58-4ea2-8c81-2398365ff2f3.europe-west3-0.gcp.cloud.qdrant.io"


# Initialize Qdrant client
def get_qdrant_client():
    try:
        client = QdrantClient(api_key=qdrant_api_key, url=qdrant_url)
        client.get_collections()  # Test the connection
        return client
    except Exception as e:
        st.write(f"Error connecting to Qdrant: {repr(e)}")
        return None


qdrant_client = get_qdrant_client()


# LLaMA-based function to extract continuous keywords/entities
def get_keywords_with_llama(query):
    api_url = 'https://rag-llm-api.accubits.cloud/v1/chat/completions'
    headers = {
        'api-key': llama_api_key,
        'Content-Type': 'application/json'
    }
    # Update the prompt to make it generic and extract continuous entities
    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "max_tokens": 100,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant. Extract and return only a list of entities or important keywords from the user query. Do not provide any additional information or summaries."
            },
            {
                "role": "user",
                "content": f"Query: {query}"
            }
        ]
    }

    # Make the request to the LLaMA API
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        response_json = response.json()
        generated_keywords = response_json['choices'][0]['message']['content']
        return generated_keywords.split(", ")  # Assuming LLaMA returns a comma-separated list of phrases
    else:
        st.write(f"Error: {response.status_code} - {response.text}")
        return []


# LLaMA-based function to extract continuous keywords/entities
def get_keywords_with_llama(query):
    api_url = 'https://rag-llm-api.accubits.cloud/v1/chat/completions'
    headers = {
        'api-key': llama_api_key,
        'Content-Type': 'application/json'
    }
    # Refine the prompt to extract only the keywords/entities
    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "max_tokens": 100,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant. Extract and return only a list of entities or important keywords from the user query. Do not provide any additional information or summaries."
            },
            {
                "role": "user",
                "content": f"Query: {query}"
            }
        ]
    }

    # Make the request to the LLaMA API
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        response_json = response.json()
        generated_keywords = response_json['choices'][0]['message']['content']

        # Clean up the output, remove extra labels, and only return the entity names
        cleaned_keywords = []
        for kw in generated_keywords.split('\n'):
            if 'Entities:' in kw or '*' in kw:
                kw = kw.replace('Entities:', '').replace('*', '').strip()
            if kw:
                cleaned_keywords.append(kw)

        return cleaned_keywords
    else:
        st.write(f"Error: {response.status_code} - {response.text}")
        return []


# Neo4j Client Class
class Neo4jClient:
    def __init__(self, uri, user, password, database=None):
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))

    def close(self):
        if self._driver:
            self._driver.close()

    def fetch_related_data(self, query):
        # Use LLaMA to extract keywords and entities
        keywords = get_keywords_with_llama(query)

        st.write(f"Extracted keywords by LLaMA: {keywords}")  # Debugging step to check extracted keywords

        if not keywords:
            st.write("No valid keywords extracted from the query.")
            return {}, [], None

        # Modified Neo4j query to retrieve nodes and relationships dynamically using exact matching
        nodes_query = """
        MATCH (n)-[r]->(m)
        WHERE toLower(n.name) IN $keywords OR toLower(m.name) IN $keywords
        RETURN n, r, m
        """
        with self._driver.session(database=self._database) as session:
            results = session.run(nodes_query, parameters={"keywords": [kw.lower() for kw in keywords]})
            nodes = {}
            edges = set()  # Use a set to prevent duplicate edges
            highlighted_node_id = None

            # Process results
            for record in results:
                node1 = record['n']
                node2 = record['m']
                relationship = record['r']

                node1_name = node1.get('name', 'Unknown').upper()
                node2_name = node2.get('name', 'Unknown').upper()

                # Add nodes if they are not already present
                if node1_name not in nodes:
                    nodes[node1_name] = {'id': node1.element_id, 'label': node1_name, 'title': node1_name}

                if node2_name not in nodes:
                    nodes[node2_name] = {'id': node2.element_id, 'label': node2_name, 'title': node2_name}

                # Add relationships, but avoid duplicates
                edge = (node1_name, node2_name, relationship.type)
                reverse_edge = (node2_name, node1_name, relationship.type)
                if edge not in edges and reverse_edge not in edges:
                    edges.add(edge)

                # Highlight the node matching the query
                if query.lower() in node1_name.lower():
                    highlighted_node_id = nodes[node1_name]['id']
                elif query.lower() in node2_name.lower():
                    highlighted_node_id = nodes[node2_name]['id']

        return nodes, list(edges), highlighted_node_id

def display_graph(nodes, edges, highlighted_node_id):
    try:
        G = nx.DiGraph()

        # Add nodes to the graph
        for node_id, node_data in nodes.items():
            G.add_node(node_id, label=node_data['label'], title=node_data['title'])

        # Add edges (relationships) to the graph
        for source_id, target_id, relationship in edges:
            G.add_edge(source_id, target_id, label=relationship)

        # Create a Pyvis network with a larger canvas
        net = Network(notebook=False, directed=True, height="800px", width="1000px")
        net.from_nx(G)

        # Customize node and edge visuals
        for node in net.nodes:
            if node['id'] == highlighted_node_id:
                # Highlight the queried node with a larger size and unique color
                node['color'] = 'green'  # Unique color instead of red
                node['size'] = 30  # Make the node larger
                node['borderWidth'] = 3  # Make the border thicker for emphasis
            else:
                node['font'] = {'color': 'black'}  # Default font color

        for edge in net.edges:
            edge['title'] = edge['label']  # Show relationship type on hover
            edge['color'] = 'gray'  # Set color for edges

        # Set physics to false for a more fixed positioning of nodes
        net.set_options("""
            var options = {
              "nodes": {
                "shape": "dot",
                "size": 15,
                "font": {
                  "size": 14,
                  "color": "#000000"
                }
              },
              "edges": {
                "arrows": {
                  "to": { "enabled": true }
                },
                "color": {
                  "color": "#A9A9A9"
                },
                "font": {
                  "align": "middle"
                }
              },
              "interaction": {
                "hover": true,
                "navigationButtons": true
              },
              "physics": {
                "enabled": false
              }
            }
        """)

        # Save the graph as an HTML file
        html_path = tempfile.mktemp(suffix=".html")
        net.save_graph(html_path)

        # Load and return HTML content
        with open(html_path, 'r') as file:
            html_content = file.read()

        # Clean up temporary file
        os.remove(html_path)

        return html_content

    except Exception as e:
        return f"An error occurred: {str(e)}"


def retrieve_top_chunks_from_qdrant(query, collection_name="text_chunks2", top_k=10, max_context_tokens=8192,
                                    max_response_tokens=500):
    if qdrant_client is None:
        st.write("Qdrant client is not initialized.")
        return None
    try:
        query_vector = model.encode([query])[0].tolist()
        search_result = qdrant_client.search(collection_name=collection_name, query_vector=query_vector, limit=top_k)
        if search_result:
            chunks = [res.payload['text'] for res in search_result]
            current_token_count = 0
            combined_chunks = []
            for chunk in chunks:
                chunk_tokens = len(chunk.split())
                if current_token_count + chunk_tokens + max_response_tokens > max_context_tokens:
                    break
                combined_chunks.append(chunk)
                current_token_count += chunk_tokens
            return combined_chunks
        else:
            return None
    except Exception as e:
        st.write(f"Error retrieving from Qdrant: {e}")
        return None


def stream_llama_response(query, context):
    api_url = 'https://rag-llm-api.accubits.cloud/v1/chat/completions'
    headers = {
        'api-key': llama_api_key,
        'Content-Type': 'application/json'
    }
    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "max_tokens": 500,  # Limit the max tokens generated
        "stream": True,  # Enable streaming
        "messages": [
            {
                "role": "system",
                "content": "You are an intelligent AI assistant. Provide a comprehensive and detailed answer using both the provided document context and related graph data."
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nDocument Context: {context}"
            }
        ]
    }

    # Make the request to the API
    response = requests.post(api_url, headers=headers, data=json.dumps(payload), stream=True)

    if response.status_code == 200:
        text_container = st.empty()  # Initialize an empty container to hold the streamed response
        response_text = ""  # Initialize empty string to accumulate streamed text

        for chunk in response.iter_lines():
            if chunk:
                try:
                    # Decode the chunk to get the generated content
                    chunk_data = json.loads(chunk.decode('utf-8').replace('data: ', ''))
                    content = chunk_data.get('choices', [{}])[0].get('delta', {}).get('content', '')

                    if content:
                        response_text += content  # Accumulate the generated text
                        text_container.write(response_text)  # Update the response in real-time
                except json.JSONDecodeError:
                    # Handle the case where the chunk is not valid JSON (sometimes happens during streaming)
                    continue

        return response_text  # Return the final accumulated text after streaming is done

    else:
        # Handle errors from the API
        st.write(f"Error calling the Llama API: {response.status_code} - {response.text}")
        return None


st.title("Full-Scale Retriever with Qdrant and Neo4j")

user_query = st.text_input("Enter your question:")

if user_query:
    st.write("Retrieving data from Qdrant...")
    top_chunks = retrieve_top_chunks_from_qdrant(user_query, top_k=10)

    if top_chunks:
        st.write("Querying related data from Neo4j...")
        neo4j_client = Neo4jClient("bolt://localhost:7687", "neo4j", "shivaji123", "pdf2")
        nodes, edges, highlighted_node_id = neo4j_client.fetch_related_data(user_query)
        neo4j_client.close()

        if nodes or edges:
            html_content = display_graph(nodes, edges, highlighted_node_id)
            st.subheader("Graphical Relationship from Neo4j:")
            st.components.v1.html(html_content, height=600)
        else:
            st.write("No related graph data found in Neo4j. Aborting LLaMA generation...")  # No graph data
        combined_context = "\n\n".join(top_chunks)
        if nodes or edges:
            combined_context += "\n\nRelated Graph Data:\n"
            combined_context += "\n".join([f"Node: {node['label']}" for node in nodes.values()])
            combined_context += "\n".join(
                [f"Relationship: {source_id} -[{rel}]-> {target_id}" for source_id, target_id, rel in edges])

        if nodes or edges:  # Only generate LLaMA response if relevant data was found
            st.write("Generating response with Llama model...")
            # Stream the response in real-time
            stream_llama_response(user_query, combined_context)
        else:
            st.write("Skipping LLaMA response generation as no graph data was found.")
    else:
        st.write("No relevant chunks found in Qdrant.")
