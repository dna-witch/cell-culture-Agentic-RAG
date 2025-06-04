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

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI

from supabase import Client
from multi_agent_system import MultiAgentDeps, run_multi_agent

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

# ---------- MAIN RUN FUNCTION ----------
@cell_culture_agent.run
async def run_cell_culture_agent(
    ctx: RunContext[CellCultureAIDeps],
    user_query: str,
    expression: str | None = None,
) -> str:
    """Entry point that delegates to the multi-agent system."""
    deps = MultiAgentDeps(
        supabase_clients=[ctx.deps.supabase],
        openai_client=ctx.deps.openai_client,
    )
    return await run_multi_agent(deps, user_query, expression)