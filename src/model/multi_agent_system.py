"""
`multi_agent_system.py`
==================
Multi-Agent System module.

This module outlines four cooperative agents used in the retrieval-augmented
pipeline:

* **Retrieval Agent** - Queries the vector database and returns chunks of
  documents relevant to the user's question.
* **Reasoning Agent** - Consumes retrieved passages and synthesizes them with the
  question to generate intermediate explanations.
* **Calculation Agent** - Handles quantitative tasks such as unit conversions or
  statistical operations that support the reasoning process.
* **Planning Agent** - Orchestrates the overall workflow by deciding when to call
  each agent and composing their outputs into the final answer.

Together these agents provide a structured approach to searching, analyzing and
responding to cell culture questions.
"""

from __future__ import annotations
import os
import asyncio
from dataclasses import dataclass
from typing import List, Sequence

from supabase import Client, create_client
from openai import AsyncOpenAI

from .cell_culture_expert import create_vector_embedding


@dataclass
class MultiAgentDeps:
    """Shared dependencies for the multi agent RAG system."""
    supabase_clients: Sequence[Client]
    openai_client: AsyncOpenAI


async def retrieve_from_db(client: Client, query_embedding: List[float], match_count: int = 5) -> List[dict]:
    """Query a single Supabase database for relevant documents."""
    result = client.rpc(
        'match_docs',
        {
            'query_embedding': query_embedding,
            'match_count': match_count
        }
    ).execute()
    return result.data or []


async def retrieval_agent(deps: MultiAgentDeps, query: str) -> List[dict]:
    """Retrieve relevant documents from all configured databases."""
    embedding = await create_vector_embedding(query, deps.openai_client)
    tasks = [retrieve_from_db(c, embedding) for c in deps.supabase_clients]
    results = await asyncio.gather(*tasks)
    combined: List[dict] = []
    for res in results:
        combined.extend(res)
    return combined


async def reasoning_agent(deps: MultiAgentDeps, query: str, docs: List[dict]) -> str:
    """Use the language model to analyse the documents and reason about the answer."""
    context = "\n\n".join(doc.get('content', '') for doc in docs)
    messages = [
        {"role": "system", "content": "You are a scientific reasoning assistant."},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}\n\nProvide your reasoning."}
    ]
    response = await deps.openai_client.chat.completions.create(model=os.getenv("LLM_MODEL"), messages=messages)
    return response.choices[0].message.content


async def calculation_agent(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    import sympy as sp
    try:
        result = sp.simplify(expression)
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


async def planning_agent(deps: MultiAgentDeps, query: str, reasoning: str, calc_result: str, docs: List[dict]) -> str:
    """Generate the final answer using the language model."""
    context = "\n\n".join(doc.get('content', '') for doc in docs)
    messages = [
        {"role": "system", "content": "You are an expert cell culture assistant."},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}\n\nReasoning:\n{reasoning}\n\nCalculation result: {calc_result}\n\nProvide the final answer."}
    ]
    response = await deps.openai_client.chat.completions.create(model=os.getenv("LLM_MODEL"), messages=messages)
    return response.choices[0].message.content


async def multi_agent_query(query: str, expression: str | None = None) -> str:
    """High level helper to run the multi agent pipeline."""
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    # Example uses the same database twice, but this structure allows multiples
    supabase_clients = [create_client(supabase_url, supabase_key)]

    deps = MultiAgentDeps(supabase_clients=supabase_clients, openai_client=openai_client)

    docs = await retrieval_agent(deps, query)
    reasoning = await reasoning_agent(deps, query, docs)
    calc_result = ""
    if expression:
        calc_result = await calculation_agent(expression)
    final = await planning_agent(deps, query, reasoning, calc_result, docs)
    await openai_client.close()
    return final


if __name__ == "__main__":
    user_q = input("Enter your question: ")
    expr = input("Optional calculation (press enter to skip): ") or None
    answer = asyncio.run(multi_agent_query(user_q, expr))
    print("\nAnswer:\n", answer)
