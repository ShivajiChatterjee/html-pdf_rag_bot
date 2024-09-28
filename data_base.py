import json
from neo4j import GraphDatabase
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Neo4j connection
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

    # Function to create nodes and relationships in Neo4j
    def create_nodes_and_relationships(self, entities, relationships):
        with self._driver.session(database=self._database) as session:
            # Create nodes (entities)
            for entity in entities:
                entity_type = re.sub(r'[^A-Za-z0-9_]', '_', entity["entity_type"]).upper()  # Sanitize and uppercase entity type
                entity_name = entity.get("entity_name", "").upper()  # Uppercase entity name
                if entity_name:  # Ensure entity has a name
                    logging.info(f"Creating entity: {entity_name} of type {entity_type}")
                    session.run(f"""
                        MERGE (e:{entity_type} {{name: $name}})
                    """, name=entity_name)
                else:
                    logging.warning("Skipped creating entity due to missing name.")

            # Create relationships with dynamic relationship types and description (without merging relationships)
            for relationship in relationships:
                source_name = relationship.get("source_entity", "").upper()
                target_name = relationship.get("target_entity", "").upper()
                relationship_type = re.sub(r'[^A-Za-z0-9_]', '_', relationship["relation"])
                relationship_description = relationship.get("relationship_description", "")

                # Ensure relationship type starts with a letter or underscore
                if not re.match(r'^[A-Za-z_]', relationship_type):
                    relationship_type = f"R_{relationship_type}"

                if source_name and target_name:
                    logging.info(f"Creating relationship: {source_name} -[{relationship_type}]-> {target_name}")
                    # Use CREATE to ensure multiple relationships can be created even between the same nodes
                    session.run(f"""
                        MATCH (source {{name: $source_name}}), (target {{name: $target_name}})
                        CREATE (source)-[r:{relationship_type} {{description: $description}}]->(target)
                    """, source_name=source_name,
                           target_name=target_name,
                           description=relationship_description)
                else:
                    logging.warning("Skipped creating relationship due to missing source or target.")

# Load the entities and relationships from JSON and store them in Neo4j
def load_entities_and_relationships(json_file, neo4j_client):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for entry in data:
            entities = entry.get("entities_and_relationships", {}).get("entities", [])
            relationships = entry.get("entities_and_relationships", {}).get("relationships", [])

            if entities or relationships:  # Process even if only entities or relationships exist
                neo4j_client.create_nodes_and_relationships(entities, relationships)
            else:
                logging.warning("No entities or relationships found in the JSON entry.")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {str(e)}")
    except Exception as e:
        logging.error(f"An error occurred while loading data: {str(e)}")

# Main function to load the data into Neo4j
if __name__ == "__main__":
    # Connect to Neo4j
    neo4j_client = Neo4jClient("bolt://localhost:7687", "neo4j", "shivaji123", "pdf2")

    # Load and store the entities and relationships
    json_file = "cleaned_entities_and_relationships.json"  # Adjust the path to your file
    load_entities_and_relationships(json_file, neo4j_client)

    # Close Neo4j connection
    neo4j_client.close()

    print("Entities and relationships have been successfully stored in Neo4j.")
