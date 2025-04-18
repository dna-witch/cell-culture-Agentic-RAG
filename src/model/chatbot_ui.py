from __future__ import annotations
import os
import asyncio
import json
import streamlit as st

from supabase import Client
from openai import AsyncOpenAI

from typing import Literal, TypedDict
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# ---------- IMPORT THE AI EXPERT MODULE ----------
from cell_culture_expert import cell_culture_agent, CellCultureAIDeps

# ---------- ALL MESSAGE PART CLASSES ----------
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

# ---------- OPENAI AND SUPABASE CLIENTS ----------
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase_client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

class ChatMessage(TypedDict):
    """Formats a chat message for the UI."""
    role: Literal['user', 'model']
    timestamp: str
    content: str

def display_message_part(part):
    """
    Display a single message part in the Streamlit UI.
    Customize how you want to display each part type.

    Args:
        part: The message part to display.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System Prompt:** {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)

async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.

    Args:
        user_input (str): The input from the user.
    """
    deps = CellCultureAIDeps(
        supabase=supabase_client,
        openai_client=openai_client
    )

    async with cell_culture_agent.run_stream(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[:-1]
    ) as result:
        partial_text = ""
        message_placeholder = st.empty()

        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)
        
        filtered_messages = [msg for msg in result.new_messages()
                             if not (hasattr(msg, 'parts') and
                                     any(part.part_kind == 'user-prompt' for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)

        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )

# ---------- MAIN FUNCTION ----------
async def main():
    """Main function to run the Streamlit chat application using the RAG model."""
    st.title("Ask Agar!")
    st.subheader("Ask Agar your cell culture questions and get expert advice!")
    st.markdown("This application uses an Agentic Retrieval-Augmented Generation (RAG) model to provide answers based on a knowledge base of cell culture documents.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("What cell culture knowledge do you seek today...")

    if user_input:
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )

        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            await run_agent_with_streaming(user_input)

if __name__ == "__main__":
    asyncio.run(main())