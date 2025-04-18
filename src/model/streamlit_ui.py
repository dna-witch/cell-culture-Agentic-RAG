from __future__ import annotations
from typing import List, Dict, Any, Literal, TypedDict
import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime

import streamlit as st
# import logfire

from cell_culture_wizard_v2 import query_cell_culture_expert, retrieve_relevant_docs

load_dotenv()  # Load environment variables from .env file

# hf_client = InferenceClient(provider=os.getenv("PROVIDER"),
#                             api_key=os.getenv("HUGGINGFACE_API_KEY"))
# supabase: Client = Client(os.getenv("SUPABASE_URL"), 
#                           os.getenv("SUPABASE_SERVICE_KEY"))

class ChatMessage(TypedDict):
    """Format for chat messages sent to the browser/API."""
    role: Literal['user', 'assistant', 'system']
    content: str
    timestamp: str

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def add_message(role: str, content: str):
    """Add a message to the session state and log it."""
    timestamp = datetime.now().isoformat()
    message = ChatMessage(role=role, content=content, timestamp=timestamp)
    st.session_state.messages.append(message)
    # logfire.log(f"{role.capitalize()} message added", message=message)

    # Update chat history for RAG model
    if role in ['user', 'assistant']:
        st.session_state.chat_history.append(content)

async def stream_response(query: str):
    """Stream the response from the RAG model."""
    message_placeholder = st.empty()
    full_response = ""

    # Streaming callback
    async def stream_handler(token: str):
        nonlocal full_response
        full_response += token
        message_placeholder.markdown(full_response)
    
    # Call the RAG model with streaming
    response = await query_cell_culture_expert(
        question=query,
        stream_handler=stream_handler,
        message_history=st.session_state.chat_history
    )

    # Add the final response to the chat
    add_message('assistant', full_response)

    # Return source documents for optional display
    return response.get('source_documents', [])

def display_chat_history():
    """Display the chat history in the Streamlit app."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            st.caption(message["timestamp"])

async def main():
    """Main function to run the Streamlit chat application using the RAG model."""
    st.title("Cell Culture Wizard")
    st.subheader("Ask the wizard your cell culture questions and get expert advice!")
    st.markdown("This application uses a Retrieval-Augmented Generation (RAG) model to provide answers based on a knowledge base of cell culture documents.")

    initialize_session_state()
    display_chat_history()

    # User input for the chat
    user_input = st.chat_input("What cell culture knowledge do you seek today...")

    # Handle user input
    if user_input:
        add_message("user", user_input)
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        # Stream the response from the RAG model
        with st.chat_message("assistant"):
            source_docs = await stream_response(user_input)
        # Optionally display source documents if available
        with st.expander("View Source Documents", expanded=False):
            for i, doc in enumerate(source_docs):
                st.markdown(f"### Source {i+1}")
                st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                st.markdown(f"**URL:** {doc.metadata.get('url', 'Unknown')}")
                st.markdown(f"**Content:** {doc.page_content}")
                st.markdown("---")

if __name__ == "__main__":
    asyncio.run(main())