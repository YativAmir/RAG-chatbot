# Civics Learning 2040

Dive into a project that blends education and technology to transform how civics content is delivered. Originally developed as a freelance initiative, this repository leverages state-of-the-art NLP techniques—like document chunking, vector embeddings, and retrieval-augmented generation—to build an intelligent system for processing and retrieving educational material. It’s a showcase of modern approaches in AI, offering insights and tools that can inspire innovation for NLP engineers, AI researchers, and data scientists alike.\

## Project Overview

The main goal of this project was to provide an engaging, educational experience in civics using advanced NLP techniques. The code demonstrates how to:
- Process and split large documents (both PDFs and DOCX files) into manageable chunks.
- Generate vector embeddings using both OpenAI and BAAI models.
- Store and index these embeddings using Pinecone.
- Perform semantic search and retrieval to answer civics-related queries.
- Build conversational retrieval chains for question answering.

## Core Functionality

The repository contains multiple approaches to document indexing and retrieval:

- **CreateIndexWithSplitting.py:**  
  A script that combines text from PDF documents, splits it into chunks using a recursive text splitter, generates embeddings, and indexes them in Pinecone for efficient retrieval.

- **CreatingIndexByDocWithBAAI.py:**  
  An enhanced version where each document is processed separately using the BAAI/bge-m3 model for improved results.

- **RAGWithPinecone&BAAI.py:**  
  Implements a Retrieval-Augmented Generation (RAG) approach that leverages conversational context to rephrase queries and retrieve answers from the indexed documents.

- **TextRetrievalCheck.py:**  
  A helper script for testing and validating the text retrieval process from the Pinecone index.

- **oldCodeVersion.py:**  
  An earlier version of the code that performed similar tasks but has been superseded by the improved versions above.

## Technology Stack & Dependencies

The project uses the following technologies and libraries:
- **Programming Language:** Python
- **Document Processing:**  
  - [PyPDFLoader](https://github.com/langchain-ai/langchain) for PDFs  
  - [docx2txt](https://github.com/ankushshah89/python-docx2txt) for DOCX files
- **Text Splitting:** RecursiveCharacterTextSplitter from LangChain
- **Embeddings:**  
  - OpenAI Embeddings (via LangChain's OpenAI integration)  
  - BAAI/bge-m3 model using SentenceTransformer
- **Vector Database:** [Pinecone](https://www.pinecone.io/)
- **Conversational Chain & RAG:** LangChain framework for prompt templates and retrieval-augmented generation

## Installation & Setup

To get started, you will need:
- **API Keys:**  
  - Pinecone API key  
  - OpenAI API key (if using OpenAI embeddings)
- **Documents:** Your own PDFs or DOCX files that you wish to index and retrieve.
- **Dependencies:**  
  Install the required Python packages (a `requirements.txt` file should be included or generated) using:
  ```bash
  pip install -r requirements.txt
  ```
- **Configuration:**  
  Update the API keys and file paths in the scripts as needed.

## Repository Structure

```
├── CreateIndexWithSplitting.py
│   - Splits and indexes text from PDF documents using recursive splitting.
├── CreatingIndexByDocWithBAAI.py
│   - Processes DOCX files using manual chunking with the BAAI/bge-m3 model for improved results.
├── oldCodeVersion.py
│   - An older iteration of the indexing and retrieval code (retained for reference).
├── RAGWithPinecone&BAAI.py
│   - Implements a conversational retrieval chain leveraging Pinecone and BAAI embeddings.
├── TextRetrievalCheck.py
│   - A script to test and verify the retrieval accuracy from the Pinecone index.
└── requirements.txt
    - List of Python packages required to run the project.
```

## Usage

1. **Set Up API Keys and File Paths:**  
   Edit the scripts to include your Pinecone and OpenAI API keys. Also, update the file paths to point to your own documents.

2. **Index Your Documents:**  
   Run either `CreateIndexWithSplitting.py` or `CreatingIndexByDocWithBAAI.py` depending on your preferred method of chunking and indexing.

3. **Test Retrieval:**  
   Use `TextRetrievalCheck.py` to run sample queries and verify that the documents are correctly indexed and can be retrieved based on semantic similarity.

4. **Conversational Querying:**  
   Use `RAGWithPinecone&BAAI.py` to interact with the system in a multi-turn conversational manner for retrieval-augmented generation.

## Target Audience & Use Cases

This project is ideal for:
- **NLP Engineers & AI Engineers:**  
  Interested in document indexing, semantic search, and conversational retrieval systems.
- **Data Scientists & Machine Learning Professionals:**  
  Who want to explore vector embeddings and advanced retrieval techniques in a real-world context.
- **Educational Technologists:**  
  Looking at innovative approaches to teach civics through interactive, AI-driven content.

## Future Enhancements

While there are no immediate plans for further enhancements due to the project's freelance nature, the repository serves as a robust foundation that can be extended with:
- Improved user interfaces.
- More dynamic document handling.
- Enhanced conversational AI capabilities.

## Additional Notes

- **CreatingIndexByDocWithBAAI.py:**  
  This script required a significant amount of manual tuning to generate the best chunks for embedding. It demonstrates a high level of customization and might be particularly interesting for those looking to optimize document processing for improved retrieval accuracy.

- **oldCodeVersion.py:**  
  This file is retained as a reference to earlier development stages. While it does not perform as well as the newer implementations, it provides valuable insights into the project’s evolution.

---

Feel free to adjust or add any details to better reflect your vision and technical expertise.
