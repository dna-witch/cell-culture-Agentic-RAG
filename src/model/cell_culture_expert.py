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

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Jaume/gemma-2b-embeddings")
LLM_MODEL = os.getenv("LLM_MODEL")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
PROVIDER = os.getenv("PROVIDER", "nebius")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")