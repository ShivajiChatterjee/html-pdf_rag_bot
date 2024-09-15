import streamlit as st
from neo4j import GraphDatabase
import networkx as nx
from pyvis.network import Network
import tempfile
import os

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
        # Fetch nodes and relationships related to the query
        nodes_query = """
        MATCH (n)-[r]-(m)
        WHERE n.name CONTAINS $query OR m.name CONTAINS $query
        RETURN n, r, m
        """

        with self._driver.session(database=self._database) as session:
            results = session.run(nodes_query, parameters={"query": query})

            nodes = {}
            edges = []

            # Process results
            for record in results:
                node1 = record['n']
                node2 = record['m']
                relationship = record['r']

                # Add nodes with their names or labels
                for node in [node1, node2]:
                    node_id = node.id
                    name = node.get('name', 'Unknown')
                    if node_id not in nodes:
                        nodes[node_id] = {'label': name, 'title': name}

                # Add relationships
                edges.append((node1.id, node2.id, relationship.type))

        return nodes, edges


def display_graph(nodes, edges):
    try:
        G = nx.DiGraph()  # Directed graph for relationships

        # Add nodes with labels and properties
        for node_id, node_data in nodes.items():
            G.add_node(node_id, label=node_data['label'], title=node_data['title'])

        # Add edges with relationships
        for source_id, target_id, relationship in edges:
            G.add_edge(source_id, target_id, label=relationship)

        # Create a Pyvis network
        net = Network(notebook=False, directed=True)
        net.from_nx(G)

        # Style the graph and make the relationships visible
        for node in net.nodes:
            node['font'] = {'color': 'black'}  # Change font color to black for visibility

        for edge in net.edges:
            edge['title'] = edge['label']  # Show relationship type on hover
            edge['color'] = 'gray'  # Set a color for edges

        # Adjust layout settings for better visualization
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
              "physics": {
                "forceAtlas2Based": {
                  "gravitationalConstant": -50,
                  "centralGravity": 0.01,
                  "springLength": 100,
                  "springConstant": 0.08,
                  "damping": 0.4,
                  "avoidOverlap": 1
                },
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based"
              },
              "interaction": {
                "hover": true,
                "navigationButtons": true
              }
            }
        """)

        # Save the graph as an HTML file
        html_path = tempfile.mktemp(suffix=".html")
        net.save_graph(html_path)

        # Load the HTML content
        with open(html_path, 'r') as file:
            html_content = file.read()

        # Clean up the temporary file
        os.remove(html_path)

        return html_content

    except Exception as e:
        return f"An error occurred: {str(e)}"


# Streamlit App
st.title("Neo4j Graph Query App")

query = st.text_input("Enter your query:")

if st.button("Submit Query"):
    if query:
        client = Neo4jClient("bolt://localhost:7687", "neo4j", "shivaji123", "pdf")
        nodes, edges = client.fetch_related_data(query)
        client.close()

        if not nodes and not edges:
            st.write("No data found for the provided query.")
        else:
            html_content = display_graph(nodes, edges)
            st.components.v1.html(html_content, height=600)
    else:
        st.write("Please enter a query.")
