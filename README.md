# CHATBOT USING LANGCHAIN

A powerful AI Chatbot built using LangChain, LLMs, and Retrieval-Augmented Generation (RAG).  
This chatbot supports custom context ingestion, embeddings generation, and intelligent response generation using your own data.

 Features

RAG-based Question Answering
-  Embeddings generation and content generation using web scrapping and combining data from various sources 
-  Fast and optimized vector search (Chroma / FAISS supported)
-  Custom prompt templates for accurate responses
-  Modular code structure (easy to extend)
- Multiple Models created

** Project Structure
project/
├── BENNY.py # Main chatbot script
├── BENNY_PRO.py # Extended/custom version
├── common.py # Utility functions
├── templates.py # Prompt templates
├── store_model_embeddings.py # Script to generate embeddings and stores model information
├── requirements.txt # Project dependencies
├── vector_created_2/ # Embeddings folder (ignored)
├── context/ # Your private documents (ignored)
└── .env # API keys (ignored)

Some files such as **embeddings**, **context documents**, and the **.env file** are intentionally not included in this repository for privacy and future options.

These files contain:

- private project data  
- API keys / tokens  
- some really good content generated and then converted into embeddings

To run the pro model chatbot, users should add their own context files however base model embeddings are added in vector_created_2
