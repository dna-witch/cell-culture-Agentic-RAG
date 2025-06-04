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

from dotenv import load_dotenv
load_dotenv()

from supabase import Client, create_client
from openai import AsyncOpenAI

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel


# Shared model used for all delegate agents
model = OpenAIModel(os.getenv("LLM_MODEL"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

@dataclass
class MultiAgentDeps:
    """Shared dependencies for the multi agent RAG system."""
    supabase_clients: Sequence[Client]
    openai_client: AsyncOpenAI

# ---------- Helper Functions ----------
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

# <----- Agent Definitions and Tools ----->


# -------- RETRIEVAL AGENT --------
retrieval_agent = Agent(
    model,
    system_prompt="You retrieve relevant document chunks for a given query.",
    deps_type=MultiAgentDeps,
    retries=2,
)

@retrieval_agent.tool
async def list_documents(ctx: RunContext[MultiAgentDeps]) -> List[str]:
    """
    Return the unique URLs for all documents across databases.
    
    Args:
        ctx (RunContext[MultiAgentDeps]): The run context containing dependencies.
    
    Returns:
        List[str]: List of unique URLs for all documents.
    """
    try:
        urls: set[str] = set()
        for client in ctx.deps.supabase_clients:
            result = client.from_("documents").select("url").execute()
            if result.data:
                urls.update(doc["url"] for doc in result.data)
        return sorted(urls)
    except Exception as e:
        print(f"Error listing documents: {e}")
        return []

@retrieval_agent.tool
async def get_page_content(ctx: RunContext[MultiAgentDeps], url: str) -> str:
    """Return the full content of a page by combining its chunks."""
    try:
        client = ctx.deps.supabase_clients[0]
        result = (
            client.from_("documents")
            .select("title, content, chunk_id")
            .eq("url", url)
            .order("chunk_id")
            .execute()
        )
        if not result.data:
            return f"No content found for URL: {url}"
        page_title = result.data[0]["title"].split(" - ")[0]
        formatted_content = [f"# {page_title}\n"]
        for chunk in result.data:
            formatted_content.append(chunk["content"])
        return "\n\n---\n\n".join(formatted_content)
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"An error occurred while retrieving content for {url}: {str(e)}"


@retrieval_agent.tool
async def retrieve_relevant_documents(
    ctx: RunContext[MultiAgentDeps], user_query: str, match_count: int = 5
) -> List[dict]:
    """Fetch relevant document chunks for a query."""
    embedding = await create_vector_embedding(user_query, ctx.deps.openai_client)
    tasks = [
        retrieve_from_db(client, embedding, match_count)
        for client in ctx.deps.supabase_clients
    ]
    results = await asyncio.gather(*tasks)
    combined: List[dict] = []
    for res in results:
        combined.extend(res)
    return combined


@retrieval_agent.run
async def run_retrieval(ctx: RunContext[MultiAgentDeps], query: str) -> List[dict]:
    """Retrieve relevant documents using the retrieval tools."""
    return await retrieve_relevant_documents(ctx, query)



# -------- REASONING AGENT --------
reasoning_agent = Agent(
    model,
    system_prompt="You analyze retrieved passages and produce detailed reasoning.",
    deps_type=MultiAgentDeps,
    retries=2,
)

@reasoning_agent.run
async def run_reasoning(
    ctx: RunContext[MultiAgentDeps], query: str, docs: List[dict]
) -> str:
    """Use the language model to analyse the documents and reason about the answer."""
    context = "\n\n".join(doc.get("content", "") for doc in docs)
    messages = [
        {"role": "system", "content": "You are a scientific reasoning assistant."},
        {
            "role": "user",
            "content": f"Question: {query}\n\nContext:\n{context}\n\nProvide your reasoning.",
        },
    ]
    response = await ctx.deps.openai_client.chat.completions.create(
        model=os.getenv("LLM_MODEL"), messages=messages
    )
    return response.choices[0].message.content


# -------- CALCULATION AGENT --------
calculation_agent = Agent(
    model,
    system_prompt="You evaluate mathematical expressions to help with reasoning.",
    deps_type=MultiAgentDeps,
)

@calculation_agent.run
async def run_calculation(ctx: RunContext[MultiAgentDeps], expression: str | None = None) -> str:
    """Evaluate a mathematical expression safely.

    The function attempts to parse the expression from plain text or LaTeX
    and returns the simplified numerical result.  If parsing fails, the
    original error message is returned so the planning agent can handle it.
    """
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr
    expr = expression.strip()

    # Remove any assignment portion like "Concentration ="
    if "=" in expr:
        expr = expr.split("=", 1)[1].strip()
    try:
        # Attempt to parse LaTeX if expression contains LaTeX markers
        if "\\" in expr:
            try:
                from sympy.parsing.latex import parse_latex
                sym_expr = parse_latex(expr)
            except Exception:
                sym_expr = parse_expr(expr, evaluate=True)
        else:
            sym_expr = parse_expr(expr, evaluate=True)

        result = sp.N(sp.simplify(sym_expr))
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


# -------- PLANNING AGENT --------
planning_agent = Agent(
    model,
    system_prompt="You plan and synthesize the final answer using reasoning, calculations and context.",
    deps_type=MultiAgentDeps,
    retries=2,
)

@planning_agent.run
async def run_planning(
    ctx: RunContext[MultiAgentDeps],
    query: str,
    reasoning: str,
    calc_result: str,
    docs: List[dict],
) -> str:
    """Generate the final answer using the language model."""
    context = "\n\n".join(doc.get("content", "") for doc in docs)
    messages = [
        {"role": "system", "content": "You are an expert cell culture assistant."},
        {
            "role": "user",
            "content": f"Question: {query}\n\nContext:\n{context}\n\nReasoning:\n{reasoning}\n\nCalculation result: {calc_result}\n\nProvide the final answer.",
        },
    ]
    response = await ctx.deps.openai_client.chat.completions.create(
        model=os.getenv("LLM_MODEL"), messages=messages
    )
    return response.choices[0].message.content


async def multi_agent_query(query: str, expression: str | None = None) -> str:
    """High level helper to run the multi agent pipeline."""
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    # Example uses the same database twice, but this structure allows multiples
    supabase_clients = [create_client(supabase_url, supabase_key)]

    deps = MultiAgentDeps(
        supabase_clients=supabase_clients, openai_client=openai_client
    )

    docs = await retrieval_agent.run(query, deps=deps)
    reasoning = await reasoning_agent.run(query, docs=docs, deps=deps)
    calc_result = ""
    if expression:
        calc_result = await calculation_agent.run(expression, deps=deps)
    final = await planning_agent.run(
        query, reasoning=reasoning, calc_result=calc_result, docs=docs, deps=deps
    )
    await openai_client.close()
    return final

# ---------- MAIN ENTRY POINT ----------
async def run_multi_agent(
    deps: MultiAgentDeps, query: str, expression: str | None = None
) -> str:
    """Run the multi-agent pipeline using provided dependencies."""
    docs = await retrieval_agent.run(query, deps=deps)
    reasoning = await reasoning_agent.run(query, docs=docs, deps=deps)
    calc_result = ""
    if expression:
        calc_result = await calculation_agent.run(expression, deps=deps)
    final = await planning_agent.run(
        query, reasoning=reasoning, calc_result=calc_result, docs=docs, deps=deps
    )
    return final


if __name__ == "__main__":
    user_q = input("Enter your question: ")
    expr = input("Optional calculation (press enter to skip): ") or None
    answer = asyncio.run(multi_agent_query(user_q, expr))
    print("\nAnswer:\n", answer)