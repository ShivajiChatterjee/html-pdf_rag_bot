
# Standard RAG Quadrant Project


## Project Overview
The Financial RAG Quadrant Project is designed to handle complex, unstructured financial data by implementing a RAG framework. This project includes a semantic chunking mechanism that can process and analyze both HTML and PDF files, including those with embedded images. The project is set up to work within the financial domain, using real financial data for testing.

## Features
Unstructured PDF Text Extraction: Extracts text from PDF files, including images, enabling detailed analysis.
HTML Text Extraction with Trafilatura: Efficiently processes HTML files to extract text content.
* Graph RAG: Implements a graph-based RAG system using Neo4j for efficient data retrieval.
* Neo4j & Cypher: Utilizes Neo4j as the graph database and Cypher as the query language for data storage and retrieval.
* Financial Domain Focus: Tailored to work with financial data, making it suitable for use in financial analysis and decision-making.
* Self-RAG Mechanism: After initial implementation, a self-RAG process enhances the system's capabilities.
* Streamlit Integration: A user-friendly interface for interacting with the RAG system, making it accessible for users without deep technical expertise.

## Technologies Used
* Python: Core programming language used for development.
* Trafilatura: Used for extracting text from HTML files.
* Neo4j: Graph database used for storing and querying data.
* Cypher: Query language for interacting with the Neo4j graph database.
* Streamlit: Framework for building the interactive web application.

## Usage
* Upload Files: Use the Streamlit interface to upload HTML and PDF files.
* Text Extraction: The system will automatically extract and semantically chunk the text from the uploaded files.
* Querying: Use the built-in interface to query the graph database using Cypher.
* Results: View the results of your queries, including any relevant financial data retrieved by the system.
Future Work
* Enhanced Self-RAG Mechanism: Further development of the self-RAG process to improve system accuracy and efficiency.
* Support for Additional File Formats: Expand the system to support more file types.
* Advanced Querying Features: Implement more complex querying capabilities to handle sophisticated financial queries.
