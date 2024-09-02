# Standard RAG Quadrant Project

## Project Overview
The **Financial RAG Quadrant Project** is designed to handle complex, unstructured financial data by implementing a Retrieval-Augmented Generation (RAG) framework. This project incorporates a semantic chunking mechanism capable of processing and analyzing both HTML and PDF files, including those with embedded images. It is tailored specifically for the financial domain, utilizing real financial data for testing and validation.

## Features
- **Unstructured PDF Text Extraction**: Extracts text from PDF files, including images, to facilitate detailed analysis.
- **HTML Text Extraction with Trafilatura**: Efficiently processes HTML files to extract text content.
- **Graph RAG**: Implements a graph-based RAG system using Neo4j for efficient data retrieval.
- **Qdrant Vector Database**: Manages and stores vector embeddings for semantic chunking of text data, enabling advanced retrieval capabilities.
- **Neo4j & Cypher**: Utilizes Neo4j as the graph database and Cypher as the query language for data storage and retrieval.
- **Financial Domain Focus**: Tailored for financial data, making it highly suitable for financial analysis and decision-making.
- **Self-RAG Mechanism**: Enhances the system’s capabilities post initial implementation, allowing for adaptive learning.
- **Streamlit Integration**: Provides a user-friendly interface for interacting with the RAG system, making it accessible for users without deep technical expertise.

## Technologies Used
- **Python**: Core programming language for development.
- **Trafilatura**: Tool used for extracting text from HTML files.
- **Qdrant**: Vector database for storing and managing text embeddings.
- **Neo4j**: Graph database used for storing and querying data.
- **Cypher**: Query language for interacting with the Neo4j graph database.
- **Streamlit**: Framework used for building the interactive web application.

## Usage
1. **Upload Files**: Use the Streamlit interface to upload HTML and PDF files.
2. **Text Extraction**: The system will automatically extract and semantically chunk the text from the uploaded files.
3. **Vector Storage**: Text embeddings are stored in the Qdrant vector database for efficient retrieval.
4. **Querying**: Utilize the built-in interface to query the graph database using Cypher.
5. **Results**: View the results of your queries, including any relevant financial data retrieved by the system.

## Future Work
- **Enhanced Self-RAG Mechanism**: Further development to improve system accuracy and efficiency.
- **Support for Additional File Formats**: Expand the system to support a wider variety of file types.
- **Advanced Querying Features**: Implement more complex querying capabilities to handle sophisticated financial queries.
