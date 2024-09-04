# import time
# import spacy
# from neo4j import GraphDatabase

# class Neo4jHandler:
#     def __init__(self, uri, user, password):
#         self.driver = GraphDatabase.driver(uri, auth=(user, password))

#     def close(self):
#         self.driver.close()

#     def create_node(self, label, properties, database_name="pdf1"):
#         with self.driver.session(database=database_name) as session:
#             session.run(
#                 f"MERGE (n:{label} {{name: $name}})",
#                 name=properties['name']
#             )
#             print(f"Created node: {label} with name: {properties['name']}")

#     def create_relationship(self, node1, node2, relationship, database_name="pdf1"):
#         with self.driver.session(database=database_name) as session:
#             session.run(
#                 f"MATCH (a {{name: $node1}}), (b {{name: $node2}}) "
#                 f"MERGE (a)-[r:{relationship}]->(b)",
#                 node1=node1,
#                 node2=node2
#             )
#             print(f"Created relationship: {relationship} between {node1} and {node2}")

# def extract_text(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         text = file.read()
#     return text

# def extract_and_store_entities(file_path):
#     # Create the Neo4jHandler instance
#     neo4j_handler = Neo4jHandler("bolt://localhost:7687", "neo4j", "Pavan1234")

#     # Load the English NLP model
#     nlp = spacy.load("en_core_web_sm")
#     nlp.max_length = 2000000  # Increase max length for larger documents

#     # Extract text from the text file
#     content = extract_text(file_path)

#     # Start the timer for execution time
#     start_time = time.time()

#     # Split content into manageable chunks
#     chunk_size = 500000
#     for i in range(0, len(content), chunk_size):
#         chunk = content[i:i + chunk_size]
#         doc = nlp(chunk)

#         # Extract named entities and create nodes in Neo4j
#         for ent in doc.ents:
#             print(f"Detected entity: {ent.text} of type {ent.label_}")
#             if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'CARDINAL', 'MONEY', 'PERCENT']:
#                 neo4j_handler.create_node(ent.label_, {'name': ent.text})

#         # Create relationships between entities found in the same sentence
#         for sent in doc.sents:
#             # Collect entities in the sentence
#             sentence_entities = [ent.text for ent in sent.ents if ent.label_ in ['PERSON', 'ORG', 'GPE']]
#             for i in range(len(sentence_entities)):
#                 for j in range(i + 1, len(sentence_entities)):
#                     # Create a relationship between each pair of entities
#                     neo4j_handler.create_relationship(sentence_entities[i], sentence_entities[j], 'RELATED_TO')

#     # Close the Neo4j connection
#     neo4j_handler.close()

#     # Calculate and print execution time
#     execution_time = time.time() - start_time
#     print(f"Execution time: {execution_time:.2f} seconds")

# if __name__ == "__main__":
#     # Replace 'path_to_your_text_file.txt' with the actual file path
#     file_path = r"D:\VS_Code\text_extract\new_neo\extracted_text.txt"
#     extract_and_store_entities(file_path)


import time
import spacy
from neo4j import GraphDatabase

class Neo4jHandler:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_node(self, label, properties, database_name="pdf2"):
        try:
            with self.driver.session(database=database_name) as session:
                session.run(
                    f"MERGE (n:{label} {{name: $name}})",
                    name=properties['name']
                )
                print(f"Created node: {label} with name: {properties['name']}")
        except Exception as e:
            print(f"Error creating node: {e}")

    def create_relationship(self, node1, node2, relationship, database_name="pdf2"):
        try:
            with self.driver.session(database=database_name) as session:
                session.run(
                    f"MATCH (a {{name: $node1}}), (b {{name: $node2}}) "
                    f"MERGE (a)-[r:{relationship}]->(b)",
                    node1=node1,
                    node2=node2
                )
                print(f"Created relationship: {relationship} between {node1} and {node2}")
        except Exception as e:
            print(f"Error creating relationship: {e}")

def extract_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""

def extract_and_store_entities(file_path):
    # Create the Neo4jHandler instance
    neo4j_handler = Neo4jHandler("bolt://localhost:7687", "neo4j", "Pavan1234")

    # Load the English NLP model
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2000000  # Increase max length for larger documents

    # Extract text from the text file
    content = extract_text(file_path)
    if not content:
        return

    # Start the timer for execution time
    start_time = time.time()

    # Split content into manageable chunks
    chunk_size = 500000
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i + chunk_size]
        doc = nlp(chunk)

        # Extract named entities and create nodes in Neo4j
        for ent in doc.ents:
            print(f"Detected entity: {ent.text} of type {ent.label_}")
            neo4j_handler.create_node(ent.label_, {'name': ent.text})

        # Create relationships between entities found in the same sentence
        for sent in doc.sents:
            # Collect entities in the sentence
            sentence_entities = [ent.text for ent in sent.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'CARDINAL', 'MONEY', 'PERCENT','price', 'dividend', 'P/E ratio', 'EPS', 'marketCap']]
            for i in range(len(sentence_entities)):
                for j in range(i + 1, len(sentence_entities)):
                    # Create a relationship between each pair of entities
                    neo4j_handler.create_relationship(sentence_entities[i], sentence_entities[j], 'RELATED_TO')

    # Close the Neo4j connection
    neo4j_handler.close()

    # Calculate and print execution time
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    # Replace 'path_to_your_text_file.txt' with the actual file path
    file_path = r"D:\VS_Code\text_extract\new_neo\extracted_text.txt"
    extract_and_store_entities(file_path)
