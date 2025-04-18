"""
cell_culture_wizard.py

This module hosts the RAG (Retrieval-Augmented Generation) model 
for the Cell Culture Wizard application using Langchain.
"""
from __future__ import annotations

from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

import os
import asyncio
# import logfire

# Langchain imports
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import HuggingFaceEndpoint
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import Document

import torch
from huggingface_hub import InferenceClient  # Use InferenceClient for streaming
from supabase import Client, create_client

load_dotenv()  # Load environment variables from .env file

# Environment variables
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Jaume/gemma-2b-embeddings")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemma-2-2b-it")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
PROVIDER = os.getenv("PROVIDER", "nebius")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")

# ---------- VECTOR EMBEDDING FUNCTIONALITY ----------
def initialize_embedding_model():
    """
    Initializes the HuggingFace embedding model for vector embeddings.
    This function is called only once to load the model into memory.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}  # Normalize embeddings for better cosine similarity
    )

async def create_vector_embedding(text: str) -> List[float]:
    """
    Asynchronously creates a vector embedding for the given text.
    
    Args:
        text (str): The text to be embedded.

    Returns:
        List[float]: The vector embedding of the text.
    """
    embedding_model = initialize_embedding_model()
    try:
        embedding = embedding_model.embed_query(text)
        return embedding
    except Exception as e:
        # logfire.error(f"Error creating vector embedding: {e}")
        print(f"Error creating vector embedding: {e}")
        return []*768  # Return a zero vector if embedding fails

# ---------- CREATE DATABASE (Supabase) AND INFERENCE (HuggingFace) CLIENTS ----------
def initialize_hf_client() -> InferenceClient:
    """
    Initializes the HuggingFace client so that it can be used to query the LLM model.
    """

    return InferenceClient(
        provider=PROVIDER,
        api_key=HUGGINGFACE_API_KEY,
    )

# Function to create a Supabase client
def initialize_supabase_client() -> Client:
    """
    Initializes and returns a Supabase client for database operations.
    
    Returns:
        Client: The initialized Supabase client.
    """
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Define the Retrieval function
async def retrieve_documents_from_supabase(query_embedding: List[float], top_k: int = 5) -> List[Document]:
    """
    Retrieves the top _k_ most relevant documents from Supabase based on the provided query embedding.
    
    Args:
        query_embedding (List[float]): The vector embedding of the query.
        top_k (int): The number of top documents to retrieve.

    Returns:
        List[Document]: A list of retrieved documents.
    """
    supabase_client = initialize_supabase_client()
    
    result = supabase_client.rpc(
        'match_documents',
        {
            'query_embedding': query_embedding,
            'match_count': top_k
        }
    ).execute()

    # Convert the Supabase result to a list of Document objects
    documents = []
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
    """
    Retrieve relevant documents from the knowledge database based on the user's query.

    Args:
        question (str): The user's query to search for relevant documents.
        top_k (int): Number of documents to retrieve.

    Returns:
        str: A string containing the retrieved document chunks, or an error message.
    """
    try:
        query_embedding = await create_vector_embedding(question)
        documents = await retrieve_documents_from_supabase(query_embedding, top_k)

        if not documents:
            return "No relevant documents found for the given query."

        formatted_chunks = []
        for doc in documents:
            chunk_text = f"""
{doc.metadata['url']}

{doc.page_content}
"""
            formatted_chunks.append(chunk_text)
        
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
    except Exception as e:
        # logfire.error(f"Error retrieving relevant documents: {e}")
        return f"An error occurred while retrieving relevant documents: {str(e)}"

def create_prompt_template():
    """
    Create and return the prompt template for the RAG chain.
    """
    system_template = SYSTEM_PROMPT
    human_template = """
    Context information is below.
    ---------------------
    {context}
    ---------------------
    
    Given the context information and not prior knowledge, answer the following question:
    {question}
    """
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    
    return chat_prompt

class CustomRetriever:
    """
    Custom retriever class that uses Supabase to retrieve documents based on embeddings.
    This class mimics the interface of a Langchain retriever while using direct Supabase queries.
    """
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.embedding_model = initialize_embedding_model()
    async def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents for the given query.

        Args:
            query (str): The query string to search for.
        Returns:
            List[Document]: A list of relevant documents.
        """
        query_embedding = self.embedding_model.embed_query(query)
        documents = await retrieve_documents_from_supabase(query_embedding, top_k=self.top_k)
        return documents

async def query_cell_culture_expert(
        question: str,
        stream_handler=None,
        message_history=None,
        top_k: int = 5,
        ) -> Dict[str, Any]:
    """
    Query the Cell Culture Expert with a question and return the response.
    
    Args:
        question (str): The question to ask the expert.
        stream_handler: Optional callback handler for streaming responses.
        message_history: Optional history of messages for context.
        top_k (int): Number of top documents to retrieve.

    Returns:
        Dict[str, Any]: The response from the expert, including the answer and relevant documents.
    """
    callbacks = []
    if stream_handler:
        streaming_callback = StreamingCallbackHandler()
        streaming_callback.streaming_callback = stream_handler
        callbacks.append(streaming_callback)
    
    # Initialize the LLM
    llm = initialize_llm(streaming=bool(stream_handler), callbacks=callbacks)

    # Initialize the retriever
    retriever = CustomRetriever(top_k=top_k)

    # Set up memory for the conversation
    memory = None
    if message_history:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer"
        )

        for i in range(0, len(message_history), 2):
            if i + 1 < len(message_history):
                memory.chat_memory.add_user_message(message_history[i])
                memory.chat_memory.add_ai_message(message_history[i + 1])
    
    # Retrieve relevant documents
    documents = await retriever.get_relevant_documents(question)

    # Create the prompt template
    prompt = create_prompt_template()

    # Create the RetrievalQA chain
    chain_type_kwargs = {
        "prompt": prompt,
        "verbose": True
        }
    
    if memory:
        chain_type_kwargs["memory"] = memory
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # Use "stuff" chain type for simplicity
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

    # Run the chain with the question
    result = await asyncio.to_thread(
        qa_chain.run,
        {"question": question}
    )

    # Prepare the response
    return {
        "answer": result["result"],
        "source_documents": result["source_documents"]
    }
