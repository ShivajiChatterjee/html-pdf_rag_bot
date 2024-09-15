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
import time
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

# Initialize Neo4j client
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
        # Extract keywords from the user's query
        keywords = extract_keywords(query)

        # Refined Neo4j query to only retrieve relevant nodes and relationships
        nodes_query = """
        MATCH (n)-[r]-(m)
        WHERE any(keyword IN $keywords WHERE n.name CONTAINS keyword OR m.name CONTAINS keyword)
        RETURN n, r, m
        """
        with self._driver.session(database=self._database) as session:
            results = session.run(nodes_query, parameters={"keywords": keywords})
            nodes = {}
            edges = []
            for record in results:
                node1 = record['n']
                node2 = record['m']
                relationship = record['r']
                # Only include nodes that match the query keywords
                if any(keyword.lower() in node1.get('name', '').lower() for keyword in keywords) or \
                   any(keyword.lower() in node2.get('name', '').lower() for keyword in keywords):
                    for node in [node1, node2]:
                        node_id = node.id
                        name = node.get('name', 'Unknown')
                        if node_id not in nodes:
                            nodes[node_id] = {'label': name, 'title': name}
                    edges.append((node1.id, node2.id, relationship.type))
        return nodes, edges

def extract_keywords(text):
    # Basic keyword extraction using NLTK
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    keywords = [word for word in word_tokens if word.isalnum() and word.lower() not in stop_words]
    return keywords

def display_graph(nodes, edges):
    try:
        G = nx.DiGraph()
        for node_id, node_data in nodes.items():
            G.add_node(node_id, label=node_data['label'], title=node_data['title'])
        for source_id, target_id, relationship in edges:
            G.add_edge(source_id, target_id, label=relationship)
        net = Network(notebook=False, directed=True)
        net.from_nx(G)
        for node in net.nodes:
            node['font'] = {'color': 'black'}
        for edge in net.edges:
            edge['title'] = edge['label']
            edge['color'] = 'gray'
        net.set_options("""
            var options = {
              "nodes": { "shape": "dot", "size": 15, "font": { "size": 14, "color": "#000000" }},
              "edges": { "arrows": { "to": { "enabled": true }}, "color": { "color": "#A9A9A9" }, "font": { "align": "middle" }},
              "physics": { "forceAtlas2Based": { "gravitationalConstant": -50, "centralGravity": 0.01, "springLength": 100, "springConstant": 0.08, "damping": 0.4, "avoidOverlap": 1 }, "minVelocity": 0.75, "solver": "forceAtlas2Based" },
              "interaction": { "hover": true, "navigationButtons": true }
            }
        """)
        html_path = tempfile.mktemp(suffix=".html")
        net.save_graph(html_path)
        with open(html_path, 'r') as file:
            html_content = file.read()
        os.remove(html_path)
        return html_content
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to retrieve top chunks from Qdrant
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

# Function to call the Llama model API
def get_llama_response(query, context):
    api_url = 'https://rag-llm-api.accubits.cloud/v1/chat/completions'
    headers = {
        'api-key': llama_api_key,
        'Content-Type': 'application/json'
    }
    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "max_tokens": "500",  # Increase max tokens for a longer response
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "You are an intelligent AI assistant. Provide a comprehensive and detailed answer using the provided document context. If the context does not contain the answer, reply with 'The answer is not found in the provided context.'"
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nDocument Context: {context}"
            }
        ]
    }
    start_time = time.time()
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))
    end_time = time.time()
    st.write(f"Llama API call duration: {end_time - start_time} seconds")
    if response.status_code == 200:
        try:
            response_data = response.json()
            result = response_data['choices'][0]['message']['content']
            return result
        except (KeyError, json.JSONDecodeError):
            st.write("Error parsing the Llama API response.")
            return None
    else:
        st.write(f"Error calling the Llama API: {response.status_code} - {response.text}")
        return None

# Streamlit app layout
st.title("Full-Scale Retriever with Qdrant and Neo4j")

user_query = st.text_input("Enter your question:")

if user_query:
    st.write("Retrieving data from Qdrant...")
    top_chunks = retrieve_top_chunks_from_qdrant(user_query, top_k=10)

    if top_chunks:
        combined_context = "\n\n".join(top_chunks)
        st.write("Generating response with Llama model...")
        llama_response = get_llama_response(user_query, combined_context)

        st.subheader("Textual Answer from Llama Model:")
        if llama_response:
            st.write(llama_response)
        else:
            st.write("No response generated.")
    else:
        st.write("No relevant chunks found in Qdrant.")

    st.write("Querying related data from Neo4j...")
    neo4j_client = Neo4jClient("bolt://localhost:7687", "neo4j", "shivaji123", "pdf")
    nodes, edges = neo4j_client.fetch_related_data(user_query)
    neo4j_client.close()

    if nodes or edges:
        html_content = display_graph(nodes, edges)
        st.subheader("Graphical Relationship from Neo4j:")
        st.components.v1.html(html_content, height=600)
    else:
        st.write("No related graph data found in Neo4j.")
