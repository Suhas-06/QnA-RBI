# Question & Answering chatbot for specific documents

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

- An IBM API key for using IBM models.
- A Milvus server set up and running.
- File paths for any local files you need to reference.

## Setup

- Replace API Keys and URLs: Open `.env` and update the placeholders
- Update File Paths: Replace all placeholder file paths in the scripts with your actual file paths.
- Install Dependencies: Install the necessary Python packages using pip. You may need packages like `Flask`, `pymilvus`, `transformers`, and others.

## Usage

### Run the Flask Application:
- To start the application, run:
```bash
python QnA.py
```
### Access the Application:
- Open a web browser and navigate to http://127.0.0.1:5000 to use the application locally or open the `index.html` on any browser after you run the flask app
