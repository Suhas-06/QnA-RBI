# Q&A AI Assistant

## Overview

This project utilizes advanced NLP and AI techniques to provide accurate answers from RBI documents. It combines embeddings and the Retrieval-Augmented Generation (RAG) approach to enhance user interaction with document content. Key components include:

- **Embeddings.py**: Generates embeddings from RBI documents (or any PDF files) and stores them in a Milvus vector store.
- **QnA.py**: A Flask web app that utilizes the embeddings from the Milvus vector store to answer user queries using the RAG approach.
- **.env**: Contains sensitive user data, such as API keys and server addresses.
- **cert.pem**: SSL certificate for secure communication.
- **index.html**: Frontend for user interaction with the application.

## Use Cases

- **Policy Monitoring and Analysis**: Financial institutions need to stay updated with changes in RBI policies and guidelines.
- **Semantic Document Retrieval**: Researchers require specific information across a vast archive of RBI documents.
- **Historical Trend Analysis**: Analysts aim to understand long-term trends in India's monetary policy and economic landscape.

## Prerequisites

Before running the project, ensure you have the following:

- Milvus server set up and running.
- Required API keys (if applicable).
- File paths for any local files you need to reference.

## Setup

### Replace API Keys and URLs:

1. Open `.env` and update the following placeholders:
   - `API_KEY=YOUR_API_KEY`
   - `SERVER_ADDRESS=YOUR_SERVER_ADDRESS`
   - Add any other required configurations.

### Update File Paths:

1. Replace all placeholder file paths in the scripts with your actual file paths.

### Install Dependencies:

Run the following command to install the necessary packages:

```bash
pip install -r requirements.txt
