"""
Cell Culture Expert Module (cell_culture_expert.py)
"""

from __future__ import annotations as _annotations

import os
import asyncio
import httpx
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
import logfire  # Remove if not needed

from supabase import Client

# Load environment variables
load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

SYSTEM_PROMPT="""You are an expert biologist specializing in cell culture techniques and protocols.
You are tasked with providing accurate and detailed information about cell culture practices, including but not limited to:
- Cell line maintenance and storage
- Media preparation and supplementation
- Sterilization techniques
- Contamination prevention and control
- Cell passage and subculturing methods
- Troubleshooting common cell culture issues
- Best practices for handling and storing cell cultures

You should provide clear, concise, and scientifically accurate responses to user queries based on provided context.
If you do not know the answer, respond with "I don't know" or "I'm not sure." Do not hallucinate.
You should not provide personal opinions or unverified information.
Do not answer questions that are not related to cell culture techniques and protocols.
"""

model = OpenAIModel(LLM_MODEL)

# ---------- DEFINE AGENT DEPENDENCIES ----------
@dataclass
class CellCultureAIDeps:
    """Dependencies for Cell Culture AI Agent."""
    supabase: Client
    openai_client: AsyncOpenAI

# ---------- CREATE THE AGENT ----------
cell_culture_agent = Agent(
    model,
    system_prompt=SYSTEM_PROMPT,
    deps_type=CellCultureAIDeps,
    retries=2
)

# ---------- VECTOR EMBEDDING FUNCTIONALITY ----------
async def create_vector_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """
    Creates a vector embedding for the given text using OpenAI's embeddings API.

    Args:
        text (str): The text to be embedded.
        openai_client (AsyncOpenAI): The OpenAI client instance to use for API calls.
    Returns:
        List[float]: A list representing the vector embedding.
    """
    global EMBEDDING_MODEL
    try:
        response = await openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating vector embedding: {e}")
        return []*1536  # Return an empty list of the expected dimension (1536 for most OpenAI models)

# ---------- TOOLS FOR THE AGENT ----------
@cell_culture_agent.tool
async def retrieve_relevant_documents(ctx: RunContext[CellCultureAIDeps], user_query: str) -> str:
    """
    Retrieve relevant document chunks based on the user's query.
    
    Args:
        ctx (RunContext[CellCultureAIDeps]): The run context containing dependencies.
        user_query (str): The user's query to search for relevant documents.
    
    Returns:
        str: A string containing the relevant document chunks.
    """
    try:
        # Get embedding for the user query
        query_embedding = await create_vector_embedding(user_query, ctx.deps.openai_client)
        
        result = ctx.deps.supabase.rpc(
            'match_docs',
            {
                'query_embedding': query_embedding,
                'match_count': 5,  # Limit to top 5 results
                # 'filter': {'source': 'ATCC'}
            }
        ).execute()

        if not result.data:
            return "No relevant documents found."
        
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)

        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
    except Exception as e:
        print(f"Error retrieving relevant documents: {e}")
        return f"An error occurred while retrieving documents: {str(e)}"
    
@cell_culture_agent.tool
async def list_documents(ctx: RunContext[CellCultureAIDeps]) -> List[str]:
    """
    List all available documents in the database.
    
    Args:
        ctx (RunContext[CellCultureAIDeps]): The run context containing dependencies.
    
    Returns:
        List[str]: List of unique URLs for all documents.
    """
    try:
        # Query Supabase for unique URLS
        result = ctx.deps.supabase.from_('documents') \
            .select('url') \
            .execute()
        
        if not result.data:
            return []
        
        # Extract unique URLs
        unique_urls = sorted(set(doc['url'] for doc in result.data))
        return unique_urls
    except Exception as e:
        print(f"Error listing documents: {e}")
        return []

@cell_culture_agent.tool
async def get_page_content(ctx: RunContext[CellCultureAIDeps], url: str) -> str:
    """
    Get the full content of a specific page by combining all its chunks.
    
    Args:
        ctx (RunContext[CellCultureAIDeps]): The run context containing dependencies.
        url (str): The URL of the page to retrieve content from.
    
    Returns:
        str: The content of the page with all chunks combined in order, or an error message if not found.
    """
    try:
        result = ctx.deps.supabase.from_('documents') \
            .select('title, content, chunk_id') \
            .eq('url', url) \
            .order('chunk_id') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
        
        # Format the page with its title and combined content
        page_title = result.data[0]['title'].split(' - ')[0]
        formatted_content = [f"# {page_title}\n"]

        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
        
        # Join everything together
        return "\n\n".join(formatted_content)

    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"An error occurred while retrieving content for {url}: {str(e)}"