
# Standard RAG Quadrant Project


## Project Overview
The Financial RAG Quadrant Project is designed to handle complex, unstructured financial data by implementing a RAG framework. This project includes a semantic chunking mechanism that can process and analyze both HTML and PDF files, including those with embedded images. The project is tailored for the financial domain, using real financial data for testing.

## Features
Unstructured PDF Text Extraction: Extracts text from PDF files, including embedded images, enabling detailed analysis.
HTML Text Extraction with Trafilatura: Efficiently processes HTML files to extract text content.
Graph RAG: Implements a graph-based RAG system using Neo4j for efficient data retrieval.
Qdrant Vector Database: Manages and stores vector embeddings for semantic chunking of text data, enabling advanced retrieval capabilities.
Neo4j & Cypher: Utilizes Neo4j as the graph database and Cypher as the query language for data storage and retrieval.
Financial Domain Focus: Tailored for financial data, making it suitable for financial analysis and decision-making.
Self-RAG Mechanism: Enhances the system's capabilities post initial implementation.
Streamlit Integration: Provides a user-friendly interface for interacting with the RAG system, making it accessible for users without deep technical expertise.

## Technologies Used
Python: Core programming language for development.
Trafilatura: Used for extracting text from HTML files.
Qdrant: Vector database for storing and managing text embeddings.
Neo4j: Graph database for storing and querying data.
Cypher: Query language for interacting with the Neo4j graph database.
Streamlit: Framework for building the interactive web application.
## Usage
Upload Files: Use the Streamlit interface to upload HTML and PDF files.
Text Extraction: The system will automatically extract and semantically chunk the text from the uploaded files.
Vector Storage: Text embeddings are stored in the Qdrant vector database for efficient retrieval.
Querying: Use the built-in interface to query the graph database using Cypher.
Results: View the results of your queries, including any relevant financial data retrieved by the system.
Future Work
Enhanced Self-RAG Mechanism: Further development to improve system accuracy and efficiency.
Support for Additional File Formats: Expand the system to support more file types.
Advanced Querying Features: Implement more complex querying capabilities to handle sophisticated financial queries.
