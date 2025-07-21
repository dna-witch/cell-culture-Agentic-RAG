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

from langchain.text_splitter import MarkdownTextSplitter

from openai import AsyncOpenAI

from xml.etree import ElementTree as ET  

from supabase import create_client, Client

# Initialize the OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Initialize the Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
    )

@dataclass
class ProcessedChunk:
    url: str
    chunk_id: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def split_text_into_chunks(text: str, chunk_size: int = 5000, chunk_overlap: int = 250) -> List[str]:  # Default chunk size is 1280 characters
    """
    Splits the given text into chunks of specified size.

    Args:
        text (str): The text to be split.
        chunk_size (int): The size of each chunk. Default is 5000 characters.
        chunk_overlap (int): The number of overlapping characters between chunks. Default is 250 characters.

    Returns:
        List[str]: A list of text chunks.
    """
    text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, 
                                         chunk_overlap=chunk_overlap, # 100 characters overlap
                                        )
    chunks = text_splitter.split_text(text)
    return chunks
    
async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """
    Extracts the title and summary from the chunk.

    Args:
        chunk (str): The text chunk.
        url (str): The URL from which the chunk was extracted.
        
    Returns:
        Dict[str, str]: A dictionary containing the title and summary.
    """
    system_prompt = """You are an AI assistant that extracts title and summaries from document chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a chunk from the middle of a document, derive a short and descriptive title.
    For the summary: Summarize the main points of the text in 3-5 sentences.
    Keep both title and summary concise yet informative.
    """
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Limit to first 1000 characters for context
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error extracting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def create_vector_embedding(text: str) -> List[float]:
    """
    Creates a vector embedding for the given text using OpenAI's embeddings API.

    Args:
        text (str): The text to be embedded.

    Returns:
        List[float]: A list representing the vector embedding.
    """
    try:
        response = await openai_client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL"),
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating vector embedding: {e}")
        return []*1536  # Return an empty list of the expected dimension (1536 for most OpenAI models)

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
    # Extract title and summary
    title_summary = await get_title_and_summary(chunk, url)
    
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
        title=title_summary["title"],
        summary=title_summary["summary"],
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
            "title": chunk.title,
            "summary": chunk.summary,
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
