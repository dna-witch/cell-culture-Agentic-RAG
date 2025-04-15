"""
`chunker.py`

This module contains functions to intelligently split text into chunks,
process those chunks, create vector embeddings, 
and store everything in a vector database for efficient retrieval.
"""

import os
import sys
import json
import requests
import asyncio

from dotenv import load_dotenv
load_dotenv()

__location__ = os.path.dirname(os.path.abspath(__file__))
__base_dir__ = os.path.dirname(os.path.dirname(__location__))
__output__ = os.path.join(__base_dir__, 'data', 'raw', 'web_crawled')

from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer 
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from xml.etree import ElementTree as ET  

from supabase import create_client, Client

# Initialize global variables for models and tokenizers
# These will be loaded only once to save memory and time
model = None
tokenizer = None
embedding_model = SentenceTransformer('intfloat/e5-base')  # Embedding dimension is 768

# Initialize the Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
    )

@dataclass
class ProcessedChunk:
    url: str
    chunk_id: int
    # title: str
    # summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def split_text_into_chunks(text: str, chunk_size: int = 1280, chunk_overlap: int = 100) -> List[str]:  # Default chunk size is 1280 characters
    """
    Splits the given text into chunks of specified size.

    Args:
        text (str): The text to be split.
        chunk_size (int): The size of each chunk. Default is 1280 characters, because the embedding model can only handle up to 256 words.
        chunk_overlap (int): The number of overlapping characters between chunks. Default is 100 characters.

    Returns:
        List[str]: A list of text chunks.
    """
    text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, 
                                         chunk_overlap=chunk_overlap, # 100 characters overlap
                                        )
    chunks = text_splitter.split_text(text)
    return chunks
    
# async def get_title_and_summary(chunk: str) -> Dict[str, str]:
#     """
#     Extracts the title and summary from the chunk.

#     Args:
#         chunk (str): The text chunk.
        
#     Returns:
#         Dict[str, str]: A dictionary containing the title and summary.
#     """
#     global model, tokenizer
#     # Only load the model and tokenizer once
#     if model is None or tokenizer is None:
#         model_name = "google/flan-t5-small"  # ~300MB model. If it doesn't work, try "facebook/bart-large-cnn" (~400MB)
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#     # Create the pipeline
#     pipe = pipeline(
#         "summarization",  # Task: text2text-generation or summarization
#         model=model,
#         tokenizer=tokenizer,
#         max_length=250,
#         device=0 if torch.cuda.is_available() else -1
#     )

#     # Title: prompt template and chain
#     title_prompt = PromptTemplate(input_variables=["text"],
#                                   template="If this seems like the start of a document or a chapter, extract the title. Otherwise, derive a short descriptive title. \n\n{text} \n\nTitle:")
#     title_chain = LLMChain(llm=HuggingFacePipeline(pipeline=pipe), prompt=title_prompt)

#     # Summary: prompt template and chain
#     summary_prompt = PromptTemplate(input_variables=["text"],
#                                     template="Summarize the main points of the following text in 3-5 sentences. \n\n{text} \n\nSummary:")
#     summary_chain = LLMChain(llm=HuggingFacePipeline(pipeline=pipe), prompt=summary_prompt)

#     # Run the chains
#     title = await title_chain.arun(chunk)
#     summary = await summary_chain.arun(chunk)
    
#     return {"title": title, "summary": summary}

async def create_vector_embedding(text: str) -> List[float]:
    """
    Creates a vector embedding for the given text using
    a model from Langchain.

    Args:
        text (str): The text to be embedded.

    Returns:
        List[float]: A list representing the vector embedding.
    """
    # Only load the model once
    global embedding_model
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(model_name="Jaume/gemma-2b-embeddings",
                                                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                                                encode_kwargs={"normalize_embeddings": True})  # Cosine similarity works better with normalized embeddings
    
    # embedding = embedding_model.embed_documents([text])[0]
    embedding = embedding_model.encode(text, normalize_embeddings=True).tolist()
    
    return embedding

async def process_chunk(chunk: str, chunk_id: int, url: str) -> ProcessedChunk:
    """
    Processes a single chunk of text to extract title, summary,
    metadata, and create a vector embedding.

    Args:
        chunk (str): The text chunk.
        chunk_id (int): The ID of the chunk.
        url (str): The URL from which the chunk was extracted.
    
    Returns:
        ProcessedChunk: An object containing the processed chunk data.
    """
    # # Extract title and summary
    # title_summary = await get_title_and_summary(chunk)
    
    # Create vector embedding
    embedding = await create_vector_embedding(chunk)

    # Create metadata
    metadata = {
        "source": "ATCC",
        "chunk_size": len(chunk),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }

    # Create the processed chunk object
    return ProcessedChunk(
        url=url,
        chunk_id=chunk_id,
        # title=title_summary["title"],
        # summary=title_summary["summary"],
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )

async def store_chunk_in_supabase(chunk: ProcessedChunk) -> None:
    """
    Stores the processed chunk in Supabase.
    """

    try:
        data = {
            "url": chunk.url,
            "chunk_id": chunk.chunk_id,
            # "title": chunk.title,
            # "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }

        result = supabase.table("documents").insert(data).execute()
        print(f"Stored chunk {chunk.chunk_id} from {chunk.url} in Supabase.")
        return result
    except Exception as e:
        print(f"Error storing chunk in Supabase: {e}")
        return None
    
async def process_and_store_document(url: str, markdown: str):
    """
    Processes the entire document by splitting it into chunks,
    and stores each chunk in Supabase.

    Args:
        url (str): The URL of the document.
        markdown (str): The raw markdown content of the document.
    """
    # Split the markdown into chunks
    chunks = split_text_into_chunks(markdown)

    # Process chunks in parallel
    process_tasks = [process_chunk(chunk, i, url)
             for i, chunk in enumerate(chunks)]
    processed_chunks = await asyncio.gather(*process_tasks)

    # Store chunks in Supabase in parallel
    store_tasks = [store_chunk_in_supabase(chunk)
                   for chunk in processed_chunks]
    await asyncio.gather(*store_tasks)