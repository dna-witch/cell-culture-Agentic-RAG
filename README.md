# Cell Culture RAG

## Intro
This is a RAG-enabled LLM application that acts as an expert on cell culture techniques and knowledge.

## Environment Setup

```powershell
conda 

```

### Important Packages

* crawl4ai
* pypdf2
* supabase
* langchain
* huggingface_hub
* streamlit
* gradio
* sentence-transformers

## Data Collection

* Web crawling and scraping through Crawl4AI
* PDF processing scripts

## Data Processing

* chunking and embedding generation pipeline

## Database Setup

* configure Supabase and store embeddings

## Retrieval Mechanism

* test different retrieval approaches

## LLM Integration

* connect the retrieval system with the language model

## UI Development

* use Streamlit or Gradio to build an interface for using the model

## Further Testing and Optimization

* hybrid search
* agentic RAG
* query rewriting
* relevance feedback
* contextual compression