---
title: "Hybrid Search Inside an Agent: One Retriever, Two Frameworks"
description: "Part 3 of the hybrid search series: I hand the Part 2 retriever to an agent that decides when to search, when to refine, and when to answer — and build the exact same thing in both LangChain and Google ADK. Agentic RAG is a pattern, not a framework feature."
---

*Hybrid Search series &mdash; 1. [Why Hybrid](/2026/07/14/why-hybrid-search.html) &middot; 2. [Building the Retriever](/2026/07/16/building-the-hybrid-retriever.html) &middot; 3. Hybrid Search Inside an Agent (you're here)*

**Objective:** The capstone of the series &mdash; organized around one question: *[Part 2](/2026/07/16/building-the-hybrid-retriever.html) built a retriever you call once; what changes when you hand it to an **agent** that decides when to search, whether to refine, and when it has enough to answer &mdash; and does the framework even matter, given I built the exact same thing in [LangChain](https://www.langchain.com) and [Google ADK](https://google.github.io/adk-docs/)?*

The moment it clicked was watching the agent search *twice*. I asked it "How do statins affect mortality?", and in the streamed steps it fired the search tool, read the results, and then &mdash; with no prompting from me &mdash; fired it *again* with a reworded query. That self-directed second search is the whole idea of this post in one screenshot. The retriever from [Part 2](/2026/07/16/building-the-hybrid-retriever.html) didn't change one line; I just handed it to something that decides *when* to use it.

## Plain RAG vs agentic RAG

Ordinary RAG is a straight line: **retrieve &rarr; generate.** One blind retrieval up front, then the model writes an answer from whatever came back &mdash; even if the first query was the wrong one. Agentic RAG wraps that retrieval in a *reasoning loop*:

![Two panels. Top, plain RAG: a straight line from query to retrieve to generate to answer — retrieval happens once, before the model reasons. Bottom, agentic RAG: the model runs a loop of query, analyze, search with the hybrid_search_tool, and evaluate; from evaluate it either loops back to search with a refined query or moves on to synthesize with citations, then answer. The model decides what to search, whether to search again, and when it has enough. The retriever is unchanged — just a tool the agent can call.](/images/hybrid/hybrid-plain-vs-agentic-rag.svg)

The model drives: *analyze &rarr; search &rarr; evaluate &rarr; refine &rarr; synthesize.* It decides **what** to search for, **whether** the results are good enough, and **when** to stop and answer. Crucially, none of this touches the retriever &mdash; the dense + sparse + [RRF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) machine from Part 2 is exactly the same. It just becomes a **tool** the agent may choose to call, once or many times.

## The retriever becomes a tool

Turning the retriever into a tool is almost nothing &mdash; but two small things matter more than they look.

**The tool must *return* a string, not print one.** The agent never sees your stdout; it only sees the tool's **return value**. So the tool runs the full `search` &rarr; `rrf` &rarr; `retrieve` &rarr; `format` and returns the formatted text:

```python
from langchain.tools import tool

@tool
def hybrid_search_tool(query: str, top_k: int = 5) -> str:
    """
    Hybrid search tool that supports both keyword and semantic search.

    This tool can be called multiple times with different queries to refine.
    """
    results = search(query, top_k)                                     # Part 2's [dense, sparse]
    fused   = rrf([rank_list(results[0]), rank_list(results[1])])      # Part 2's RRF
    records = qdrant_client.retrieve("pubmedqa", ids=[i[0] for i in fused[:top_k]])
    rec_by_id = {r.id: r for r in records}
    lines = [f"{rank}. pubid={rec_by_id[id].payload['pubid']}. text={rec_by_id[id].payload['text']}"
             for rank, (id, _) in enumerate(fused[:top_k], start=1)]
    return "\n".join(lines)                                            # ← the model reads THIS
```

**The docstring *is* the spec.** The model doesn't read your code; it reads the tool's name, signature, and **docstring** to decide when and how to call it. Mine literally says *"This tool can be called multiple times with different queries to refine"* &mdash; and that one sentence is an open invitation to loop. The self-refinement I opened with isn't magic; it's the docstring plus a system prompt that tells the model to evaluate and refine. (Both are deliberately minimal here, for teaching &mdash; just enough to make the loop visible. In a real product the docstring and the system prompt would be far more elaborate: worked examples of good and bad queries, an explicit output and citation format, and guardrails on when *not* to call the tool.)

## Build it in LangChain

With the tool defined, the agent is one call. `create_agent` takes a model, the tools, and a system prompt that encodes the loop:

```python
from langchain.agents import create_agent

SYSTEM_PROMPT = """You are a biomedical research assistant.
Your role is to answer the user's questions by following the process below:
analyze question -> search -> evaluate -> refine/decompose -> synthesize.
ALWAYS search. Cite the pubids you use."""

agent = create_agent(model="gpt-4o", tools=[hybrid_search_tool], system_prompt=SYSTEM_PROMPT)
```

Call it with `agent.invoke(...)` and ask about ILC2s in chronic rhinosinusitis, and it returns a *grounded* answer &mdash; every claim traceable to a retrieved abstract, ending in `[pubid: 25429730]`. Nothing invented; the citation is a document it actually pulled.

But the payoff is in the *stream*, where you can watch the loop turn. `agent.stream(..., stream_mode="updates")` emits each step, so I print tool calls with a wrench and messages with a speech bubble:

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "How do statins affect mortality?"}]},
    stream_mode="updates",
):
    for step, data in chunk.items():
        msg = data["messages"][-1]
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                print(f"🔧 {tc['name']}({tc['args']})")
        elif msg.content:
            print(f"💬 {msg.content[:300]}")
```

And here is the moment &mdash; the actual output, untouched:

```
🔧 hybrid_search_tool({'query': 'statins effect on mortality', 'top_k': 5})
💬 1. pubid=25447567. text=[MeSH Terms] Aged, Body Mass Index, ...
🔧 hybrid_search_tool({'query': 'statins impact mortality meta-analysis', 'top_k': 5})
💬 The effects of statins on mortality have been extensively studied ...
```

Look at the two tool calls. The first searches `"statins effect on mortality"`; the agent reads the results, decides they're not enough, and searches **again** with `"statins impact mortality meta-analysis"` &mdash; a query *it* wrote, sharpening "effect" into "impact … meta-analysis." That reformulate-and-retry is agentic RAG. Under the hood LangChain runs this as a [ReAct](https://arxiv.org/abs/2210.03629) loop on [LangGraph](https://langchain-ai.github.io/langgraph/), but from where I sit it's just: the model got the tool, and the docstring told it it could knock twice.

## The same thing in Google ADK

Now the interesting test: build the *identical* agent in a different framework and see what actually differs. In [Google ADK](https://google.github.io/adk-docs/), the tool isn't decorated at all &mdash; it's a **plain function**. ADK reads its type hints and docstring to build the schema, exactly the role `@tool` played in LangChain:

```python
def hybrid_search_tool(query: str, top_k: int = 5) -> str:
    """
    Hybrid search tool that supports both keyword and semantic search.

    This tool can be called multiple times with different queries to refine.
    """
    return _retriever.search_as_text(query, top_k)

root_agent = LlmAgent(
    name="biomedical_rag",
    model="openai/gpt-4o",
    description="Agentic RAG over PubMedQA using hybrid search",
    instruction="""
    You are a biomedical research assistant.
    Your role is to answer the user's questions by following the process below:
    analyze question -> search -> evaluate -> refine/decompose -> synthesize.
    ALWAYS search. ALWAYS Cite the pubids you use.
    """,
    tools=[hybrid_search_tool],
)
```

Same tool body, same instruction, same loop &mdash; three surface differences. The model id gains a provider prefix: `"openai/gpt-4o"`, because ADK routes through [litellm](https://github.com/BerriAI/litellm) (so swapping to `"google/gemini-2.0-flash"` is a one-string change), where LangChain's `langchain-openai` took a bare `"gpt-4o"`. The system prompt is called `instruction`. And you don't write a run loop at all &mdash; `adk web m3_hybrid_search/agentic_rag_adk` discovers the `root_agent` and gives you a chat UI, the same [declarative-agent pattern](/2026/07/08/your-first-google-adk-agent.html) from my ADK series.

If that shape looks familiar, it should: this is the **exact same `LlmAgent` + tool structure** as the [buyer and seller negotiation agents](/2026/07/09/adk-state-and-control.html) from that series &mdash; a search tool where those had a pricing tool. The agent doesn't know or care that its tool happens to run a hybrid retriever.

## A class to hold the shared state

One aside worth its own section. In the notebook, the retriever was a pile of module-level globals &mdash; `dense_model`, `sparse_model`, `qdrant_client`, and a fistful of functions all reaching for them. I refactored the retriever into a class that **owns its shared state**:

```python
class HybridRetriever:
    """Owns the two embedding models + the Qdrant index; knows how to search & fuse."""

    def __init__(self, collection="pubmedqa", cache_dir=_FASTEMBED_CACHE):
        self.collection = collection
        self.dense  = TextEmbedding("BAAI/bge-large-en-v1.5",  cache_dir=cache_dir)
        self.sparse = SparseTextEmbedding("prithivida/Splade_PP_en_v1", cache_dir=cache_dir)
        self.qdrant_client = QdrantClient(":memory:")

    def search(self, query, top_k=10):          # instance method: reads self.dense/sparse/client
        ...

    def search_as_text(self, query, top_k=5) -> str:   # search → rrf → retrieve → format
        ...

    @staticmethod
    def _rrf(rank_lists, k=60, default_rank=1000):     # pure function of its inputs → no self
        ...

# build the retriever ONCE, at import time
_retriever = HybridRetriever()
_retriever.index(load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train"))
```

Quick recap on instance vs static methods: **a method that reads the models or the index is an instance method** (it needs `self`); **a method that's a pure function of its arguments is a `@staticmethod`** &mdash; `_rrf` and `_rank_list` just transform lists, so they take no `self`. Forget the decorator on one of those and Python quietly passes the first argument in as `self`, and the shapes stop lining up. Meanwhile the tool and `root_agent` stay **module-level on purpose**: the class is the reusable retriever; the thin top-level function and agent are the *framework boundary*, the only part that knows it's ADK.

## The framework mapping

Lay the two builds side by side and the differences are all scaffolding &mdash; the retriever and the loop are identical:

![One retriever in the center — the Part 2 hybrid retriever running search then RRF then retrieve then format to a string. On the left the LangChain scaffold wraps it with an @tool decorator and create_agent (model gpt-4o, system prompt). On the right the Google ADK scaffold wraps the identical retriever with a plain function (type hints plus docstring as schema) and an LlmAgent (model openai/gpt-4o, instruction). Both call the same retriever and run the same reason-and-call loop. Agentic RAG is a pattern, not a framework feature.](/images/hybrid/hybrid-two-scaffolds.svg)

| Concern | LangChain | Google ADK |
|---|---|---|
| **define a tool** | `@tool def f(...)` &mdash; docstring = spec | **plain function** in `tools=[...]` &mdash; type hints + docstring = spec |
| **build the agent** | `create_agent(model, tools, system_prompt)` | `LlmAgent(model=, tools=[...], instruction=)` |
| **model id** | `"gpt-4o"` (`langchain-openai`) | `"openai/gpt-4o"` (litellm prefix) |
| **system prompt** | `system_prompt=` | `instruction=` |
| **run it** | `agent.invoke` / `agent.stream({"messages": [...]})` | `adk web` (or a `Runner`) |
| **the reasoning loop** | ReAct via LangGraph | `LlmAgent` tool-calling loop |
| **tool &harr; state** | closure / instance | `tool_context: ToolContext` param |

The punchline is the whole reason I built it twice: **agentic RAG is a *pattern*, not a framework feature** &mdash; a retriever wrapped as a tool, plus an LLM that decides when to call it. LangChain hands you `create_agent`; ADK hands you `LlmAgent`; the tool body and the reason-and-call loop are the same on both sides. Pick the framework you already live in.

## The one-liner that ties it together

Distilled: **the retriever from Part 2 never changed &mdash; I just gave it to a model that decides when to search, when to refine, and when to answer, and that pattern is identical in LangChain and ADK.** Plain RAG retrieves once and hopes; agentic RAG makes retrieval a tool and lets the reasoning loop use it as many times as the question needs.

That closes the arc: [*why* hybrid](/2026/07/14/why-hybrid-search.html) &rarr; [*building* the retriever](/2026/07/16/building-the-hybrid-retriever.html) &rarr; *putting it inside an agent*. But there's an honest gap I've dodged all three parts, and it nags at me: I keep *asserting* hybrid beats either half, and showing anecdotes that it does &mdash; but I never **measured** it. **Coming later:** the evaluation post neither notebook wrote &mdash; precision@k, recall, and MRR over a labeled query set, to find out whether hybrid actually wins, and by how much. Assertions are cheap; numbers aren't. Thanks for reading the series.
