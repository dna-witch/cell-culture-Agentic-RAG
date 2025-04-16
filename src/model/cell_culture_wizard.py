"""
cell_culture_wizard.py

This module hosts the agentic RAG (Retrieval-Augmented Generation) model 
for the Cell Culture Wizard application.
"""
from __future__ import annotations as _annotations

from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List

import os
import asyncio
import httpx  # Remove if not used
import logfire

from pydantic_ai import Agent, ModelRetry, RunContext, AIEngine
from huggingface_hub import InferenceClient
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch  # Ensure PyTorch is installed for model inference
from transformers import pipeline
# from pydantic_ai.models.openai import OpenAIModel  # Free alternative to OpenAI??
# from openai import AsyncOpenAI
from supabase import Client, create_client

load_dotenv()  # Load environment variables from .env file

embedding_model = os.getenv("EMBEDDING_MODEL", "Jaume/gemma-2b-embeddings")

@dataclass
class AgentDeps:
    """
    Dependencies for the Cell Culture Wizard agent.
    Contains the Supabase client and HuggingFace inference client.
    This allows the agent to access the database and perform inference tasks.
    """
    supabase: Client
    hf_client: InferenceClient

system_prompt = """You are an expert biologist specializing in cell culture techniques and protocols.
You are tasked with providing accurate and detailed information about cell culture practices, including but not limited to:
- Cell line maintenance and storage
- Media preparation and supplementation
- Sterilization techniques
- Contamination prevention and control
- Cell passage and subculturing methods
- Troubleshooting common cell culture issues
- Best practices for handling and storing cell cultures

You should provide clear, concise, and scientifically accurate responses to user queries. 
You will be provided with a context that includes relevant information from the Cell Culture Wizard database (knowledge database). 
Your responses should be based on the provided context and your expertise in cell culture. 
You should also be able to handle follow-up questions and provide additional information as needed, but always grounded in the context provided or 
additional information from documents retrieved from the knowledge database.
You should always strive to provide the user with the most accurate and helpful information possible based on the context provided.

You are not allowed to make up information or provide answers that are not supported by the context or your expertise.
If you do not know the answer to a question, you should respond with "I don't know" or "I'm not sure" rather than guessing.
You should not provide personal opinions or unverified information.
Do not answer questions that are not related to cell culture techniques and protocols.
"""

# Define a custom model class for integration with the agent
class HuggingFaceModel(AIEngine):
    """
    A custom model class that wraps the HuggingFace Inference API for integration with the Pydantic AI Agent.

    Parameters:
    - model (str): The model identifier for the HuggingFace model to use.
    - provider (str): The provider for the HuggingFace model, e.g., "hf-inference".
    - api_key (str): The API key for accessing the HuggingFace Inference API.
    - **kwargs: Additional keyword arguments for the model configuration, to pass to the inference call.
    """
    def __init__(self, model_name: str = os.getenv("LLM_MODEL", "google/gemma-2-2b-it"), provider: str = "hf-inference", api_key: str = os.getenv("HUGGINGFACE_API_KEY"), **kwargs):
        super().__init__()
        self.client = InferenceClient(provider=provider, api_key=api_key)
        self.model_name = model_name
        self.default_kwargs = kwargs  # Store default kwargs for the model inference
    
    async def chat(self, messages: List[dict], **kwargs) -> str:  # List[dict] or List[str] ?
        all_kwargs = {**self.default_kwargs, **kwargs}
        prompt = messages[-1]  # Assuming the last message is the user query
        
        # Reference chat completion task documentation:
        # https://huggingface.co/docs/inference-providers/tasks/chat-completion
        def call_inference():
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **all_kwargs
            )
        
        response = await asyncio.to_thread(call_inference)
        message = response.choices[0].message
        if isinstance(message, dict):
            return message.get("content", "")
        return message

# Define the agent with the HuggingFace model, system prompt, and dependencies
cell_culture_expert = Agent(
    HuggingFaceModel(model_name=os.getenv("LLM_MODEL"), provider="nebius"),
    system_prompt=system_prompt,
    deps_type=AgentDeps,
    retries=2
)

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
    embedding_model = HuggingFaceEmbeddings(model_name="Jaume/gemma-2b-embeddings",
                                            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                                            encode_kwargs={"normalize_embeddings": True})  # Cosine similarity works better with normalized embeddings
    try:
        embedding = embedding_model.encode(text, normalize_embeddings=True).tolist()
        return embedding
    except Exception as e:
        print(f"Error creating vector embedding: {e}")
        return [0] * 768  # Return a zero vector if embedding fails
    

# Define the tools that the Agent can use
@cell_culture_expert.tool
async def retrieve_relevant_docs(ctx: RunContext[AgentDeps], user_query: str) -> str:
    """
    Retrieve relevant documents from the knowledge database based on the user's query.

    Args:
        ctx (RunContext[AgentDeps]): The context containing dependencies like Supabase client.
        user_query (str): The user's query to search for relevant documents.

    Returns:
        str: A string containing the retrieved document chunks (Top 5 Most Relevant), or an error message.
    """
    try:
        # Get the embedding for the user query
        query_embedding = await create_vector_embedding(user_query)
        
        # Query the knowledge database (Supabase) for relevant documents
        result = ctx.deps.supabase.rpc('match_documents', 
                                       {
                                           'query_embedding': query_embedding,
                                           'match_count': 5,  # Limit to top 5 most relevant documents
                                       }  # Can add more advanced filters here if needed
                                           ).execute()
        if not result.data:
            return "No relevant documents found."
        
        # Format the retrieved documents into a string
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
{doc['url']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
    except Exception as e:
        print(f"Error retrieving relevant information: {e}")
        return f"An error occurred while retrieving relevant information: {str(e)}"
