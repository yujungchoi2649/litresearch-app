# AI-Powered Literature Research Assistant

An AI tool that automates academic literature research and generates novel research hypotheses using Retrieval-Augmented Generation (RAG) + LLM reasoning.

## What it does
1. Fetches papers from arXiv, Semantic Scholar and Sciencedirect on any research topic
2. Embeds and indexes them locally using FAISS + sentence-transformers
3. Uses Llama 3.3 70B (via Groq) to synthesize the field, identify gaps, and generate novel hypotheses
4. Saves a structured Markdown report with references

## Setup

### 1. Clone the repo and create conda environment


conda create -n litresearch python=3.11

conda activate litresearch



### 2. Install dependencies


pip install requests sentence-transformers faiss-cpu langchain langchain-community groq python-dotenv



### 3. Add API keys to `.env`


SEMANTIC_SCHOLAR_API_KEY=your_key_here

GROQ_API_KEY=your_key_here



### 4. Run the notebooks in order
- `week1_data_pipeline.ipynb` — fetch papers, build FAISS index
- `week2_llm_synthesis.ipynb` — synthesize, find gaps, generate hypotheses

## Tech stack
- Retrieval: arXiv API, Semantic Scholar API
- Embeddings: `all-MiniLM-L6-v2` (sentence-transformers)
- Vector store: FAISS (local)
- LLM: Llama 3.3 70B via Groq (free tier)
- Environment: Python 3.11, conda, VS Code
