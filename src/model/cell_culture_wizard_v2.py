"""
cell_culture_wizard.py

This module hosts the RAG (Retrieval-Augmented Generation) model 
for the Cell Culture Wizard application using direct Hugging Face Inference API.
"""
from __future__ import annotations

import os
import asyncio
from typing import List, Dict, Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain.schema import Document
from supabase import Client, create_client
import torch

load_dotenv()  # Load environment variables from .env file

# Environment variables
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Jaume/gemma-2b-embeddings")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemma-2-2b-it")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
PROVIDER = os.getenv("PROVIDER", "nebius")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Initialize HF Inference Client once\
hf_client = InferenceClient(
    api_key=HUGGINGFACE_API_KEY,
    provider=PROVIDER
)

SYSTEM_PROMPT = """You are an expert biologist specializing in cell culture techniques and protocols.
You are tasked with providing accurate and detailed information about cell culture practices, including but not limited to:
- Cell line maintenance and storage
- Media preparation and supplementation
- Sterilization techniques
- Contamination prevention and control
- Cell passage and subculturing methods
- Troubleshooting common cell culture issues
- Best practices for handling and storing cell cultures

You should provide clear, concise, and scientifically accurate responses to user queries based on provided context.
If you do not know the answer, respond with "I don't know" or "I'm not sure." Do not hallucinate."""

# ---------- Embedding & Retrieval Utilities ----------
from langchain_community.embeddings import HuggingFaceEmbeddings

async def create_vector_embedding(text: str) -> List[float]:
    model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return model.embed_query(text)

async def retrieve_documents_from_supabase(query_embedding: List[float], top_k: int = 5) -> List[Document]:
    supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    result = supabase_client.rpc(
        'match_documents',
        {'query_embedding': query_embedding, 'match_count': top_k}
    ).execute()
    documents: List[Document] = []
    for item in result.data:
        doc = Document(
            page_content=item['content'],
            metadata={
                'id': item['id'],
                'url': item['url'],
                'chunk_id': item['chunk_id'],
                'source': item.get('metadata', {}).get('source', 'Unknown'),
                'similarity': item['similarity']
            }
        )
        documents.append(doc)
    return documents

async def retrieve_relevant_docs(question: str, top_k: int = 5) -> str:
    try:
        embedding = await create_vector_embedding(question)
        docs = await retrieve_documents_from_supabase(embedding, top_k)
        if not docs:
            return "No relevant documents found."
        chunks = []
        for doc in docs:
            chunks.append(f"{doc.metadata['url']}\n\n{doc.page_content}")
        return "\n\n---\n\n".join(chunks)
    except Exception as e:
        return f"Error retrieving documents: {e}"

# ---------- Main Query Function ----------
async def query_cell_culture_expert(
    question: str,
    stream_handler=None,
    message_history: List[str]|None = None,
    top_k: int = 5
) -> Dict[str, Any]:
    # 1. Retrieve context documents
    embedding = await create_vector_embedding(question)
    source_docs = await retrieve_documents_from_supabase(embedding, top_k)
    context = "\n\n---\n\n".join(
        [f"{doc.metadata['url']}\n\n{doc.page_content}" for doc in source_docs]
    )

    # 2. Build the prompt string
    prompt = f"{SYSTEM_PROMPT}\n\nContext information is below.\n---------------------\n{context}\n---------------------\nQuestion: {question}\nAnswer:"  

    # 3. Call Hugging Face Inference API
    params = {"max_new_tokens": 1024, "temperature": 0.7, "top_p": 0.95, "do_sample": True}

    if stream_handler:
        # Fetch full response then stream token-by-token
        resp = await asyncio.to_thread(
            hf_client.text_generation,
            model=LLM_MODEL,
            inputs=prompt,
            parameters=params,
            stream=False
        )
        full_text = resp.generated_text
        for char in full_text:
            await stream_handler(char)
        answer = full_text
    else:
        resp = await asyncio.to_thread(
            hf_client.text_generation,
            model=LLM_MODEL,
            inputs=prompt,
            parameters=params,
            stream=False
        )
        answer = resp.generated_text

    return {"answer": answer, "source_documents": source_docs}
