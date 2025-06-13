
"""
End-to-end LangGraph assignment implementing a Supervisor, Router, LLM, RAG, Web Crawler,
and Validation nodes with conditional control‑flow.

"""

from __future__ import annotations

import operator
from typing import Annotated, Sequence, TypedDict, Literal

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langchain.vectorstores import FAISS
import httpx

# ---------------------------------------------------------------------
# Models & Embeddings
# ---------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

# Minimal corpus for demo
source_docs = [
    "Python is a high‑level, interpreted programming language.",
    "LangGraph lets you compose LLM-centric pipelines with graph semantics.",
    "Gemini‑1.5‑flash is a lightweight Google multi‑modal model ideal for fast inference.",
]
vector_store = FAISS.from_texts(source_docs, embeddings)

# ---------------------------------------------------------------------
# Output schema and parser (Pydantic)
# ---------------------------------------------------------------------
class Answer(BaseModel):
    answer: str = Field(..., description="Concise, factual answer to the user query.")

parser = PydanticOutputParser(pydantic_object=Answer)

# ---------------------------------------------------------------------
# Agent State definition
# ---------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    last_result: str | None
    validated: bool | None

# ---------------------------------------------------------------------
# Helper node implementations
# ---------------------------------------------------------------------

# ----------------------
# Supervisor + Router  
# ----------------------
# Supervisor now only forwards state; routing logic is moved to a separate function.

def supervisor_node(state: AgentState) -> AgentState:
    """Pass the current state unchanged."""
    return state


def router(state: AgentState) -> str:
    """Return which generation path should run next."""
    user_msg = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    ).lower()
    if any(w in user_msg for w in ("internet", "search")):
        return "Web Crawler"
    if any(w in user_msg for w in ("doc", "knowledge")):
        return "RAG Call"
    return "LLM Call"


def llm_node(state: AgentState) -> AgentState:
    user_msg = state["messages"][-1].content
    res = llm.invoke(user_msg + "\n" + parser.get_format_instructions())
    parsed = parser.parse(res.content)
    state["messages"].append(AIMessage(content=parsed.answer))
    state["last_result"] = parsed.answer
    return state


def rag_node(state: AgentState) -> AgentState:
    user_msg = state["messages"][-1].content
    # Retrieve
    docs = vector_store.similarity_search(user_msg, k=3)
    context = "\n".join(d.page_content for d in docs)
    prompt = f"Answer using only the context below.\n\nContext:\n{context}\n\nQuestion: {user_msg}\n{parser.get_format_instructions()}"
    res = llm.invoke(prompt)
    parsed = parser.parse(res.content)
    state["messages"].append(AIMessage(content=parsed.answer))
    state["last_result"] = parsed.answer
    return state


def web_crawler_node(state: AgentState) -> AgentState:
    user_msg = state["messages"][-1].content
    query = user_msg.split("?")[0]  
    url = f"https://duckduckgo.com/?q={query}&format=json&pretty=1"
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(url)
            data = response.json()
            snippet = data.get("Abstract", "") or (data.get("RelatedTopics", [{}])[0].get("Text", ""))
    except Exception:
        snippet = "Couldn't fetch live data."
    prompt = (
        f"Using the information below, answer the user.\n\n"
        f"Snippet:{snippet}\n\nQuestion:{user_msg}\n{parser.get_format_instructions()}"
    )
    res = llm.invoke(prompt)
    parsed = parser.parse(res.content)
    state["messages"].append(AIMessage(content=parsed.answer))
    state["last_result"] = parsed.answer
    return state


# ----------------------
# Validator split into two parts
# ----------------------

def validator_node(state: AgentState) -> AgentState:
    """Sets a flag in state indicating whether the last answer is valid."""
    answer = state.get("last_result") or ""
    state["validated"] = bool(answer.strip()) and len(answer) < 1000
    return state


def validator_check(state: AgentState) -> str:
    """Return routing label based on validation flag."""
    return "pass" if state.get("validated") else "fail"


# ---------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------
workflow = StateGraph(AgentState)

# Register nodes
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("LLM", llm_node)
workflow.add_node("RAG", rag_node)
workflow.add_node("Web Crawler", web_crawler_node)
workflow.add_node("Validator", validator_node)

# Supervisor routes to desired node
workflow.add_conditional_edges(
    "Supervisor",
    router,  
    {
        "RAG Call": "RAG",
        "LLM Call": "LLM",
        "Web Crawler": "Web Crawler",
    },
)

# After any generation, go to Validator
for n in ["LLM", "RAG", "Web Crawler"]:
    workflow.add_edge(n, "Validator")

# Validation routes
workflow.add_conditional_edges(
    "Validator",
    validator_check,
    {
        "pass": END,
        "fail": "Supervisor",
    },
)

# Set entrypoint so the graph has a START → Supervisor edge
workflow.set_entry_point("Supervisor")

graph = workflow.compile()

# ---------------------------------------------------------------------
# Entry point utility
# ---------------------------------------------------------------------

def run_query(question: str) -> AgentState:
    start_state: AgentState = {
        "messages": [HumanMessage(content=question)],
        "last_result": None,
        "validated": None,
    }
    return graph.invoke(start_state)


if __name__ == "__main__":
    out_state = run_query("explain in detail about brahmos missile?")
    print("Final answer:", out_state["last_result"])
