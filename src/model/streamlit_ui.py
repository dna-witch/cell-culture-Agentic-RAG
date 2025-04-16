from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import json
import logfire
from supabase import Client
from huggingface_hub import InferenceClient

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)

from cell_culture_wizard import cell_culture_expert, AgentDeps

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

hf_client = InferenceClient(provider=os.getenv("PROVIDER"),
                            api_key=os.getenv("HUGGINGFACE_API_KEY"))
supabase: Client = Client(os.getenv("SUPABASE_URL"), 
                          os.getenv("SUPABASE_SERVICE_KEY"))

class ChatMessage(TypedDict):
    """Format for chat messages sent to the browser/API."""
    role: Literal['user', 'model']
    timestamp: str
    content: str

def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user", avatar=":person:"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant", avatar=":robot:"):
            st.markdown(part.content)

async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation history in the UI as
    `st.session_state.messages`.
    """
    # Prepare dependencies for the agent
    deps = AgentDeps(supabase=supabase, hf_client=hf_client)

    # Run the agent in a stream
    async with cell_culture_expert.run_stream(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[:-1],  # Exclude the last message (user input)
    ) as result:
        # Gather partial text to show incrementally
        partial_text = ""
        message_placeholder = st.empty()

        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            # Update the message placeholder with the current partial text
            message_placeholder.markdown(partial_text)
        
        filtered_messages = [msg for msg in result.new_messages()
                             if not (hasattr(msg, 'parts') and
                                     any(part.part_kind == 'user-prompt'
                                         for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)

        # Update the session state with the final message
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )

async def main():
    st.title("Cell Culture Wizard")
    st.subtitle("Ask the wizard your cell culture questions and get expert advice!")
    st.markdown("This app uses an Agentic RAG-enabled Large Language Model to answer your cell culture questions.")
    
    # Initialize session state for messages if not already done
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display existing messages in the chat history
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)
    
    # Chat input for the user
    user_input = st.chat_input("What cell culture knowledge do you seek today...")
    
    if user_input:
        # Explicitly append a new request to the conversation
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )

        # Display the user's message immediately
        with st.chat_message("user", avatar=":person:"):
            st.markdown(user_input)
        
        # Run the agent with streaming response
        with st.chat_message("assistant", avatar=":robot:"):
            await run_agent_with_streaming(user_input)

if __name__ == "__main__":
    asyncio.run(main())