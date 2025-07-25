# RAG-for-Machon-Lev
This repository contains the code for a Retrieval-Augmented Generation (RAG) system designed for Machon Lev.

## Project Description
- process.py: This is a python code to process a given JSON format file and split it into chunks of a chosen size. In our project we chose 150.
- EmbeddingFaiss.py: This is a python code that uses the chunks from the proceess codes and creates embeddings with a selected model and creates a Faiss index from those embeddings. We chose 'all-mpnet-base-v2' as an embedding model.
- LLM.py: This is a python code used to initialize the tokenizer and large language model to use to generate responses. We chose a modified Mistral-7B from hugging face to fit our google colab: 'MaziyarPanahi/Mistral-7B-Instruct-v0.2-GPTQ'.
- Generator.py: This is the python code to generate the response. It first builds a prompt, extracts the sources of the selected chunks, and runs the pipeline connecting the prompt with the LLM.
- 
