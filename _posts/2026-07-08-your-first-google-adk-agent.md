---
title: "Your First Google ADK Agent: What You Write vs. What the Framework Builds"
description: "Google ADK is declarative — you describe an agent as one object and adk web conjures the runtime, memory, and chat UI. A field guide to what you write, what the framework builds, and when to take back the wheel."
---

*ADK series — 1. Your First ADK Agent (you're here) · 2. [State & Control](/2026/07/09/adk-state-and-control.html) · 3. [Agents Over the Wire](/2026/07/11/agents-over-the-wire.html) · 4. [Two Substrates](/2026/07/12/two-substrates.html)*

**Objective:** A first-person field guide to writing your first agent in [Google's Agent Development Kit (ADK)](https://google.github.io/adk-docs/) &mdash; organized around one question: *what does the framework do for you, and what do you actually write?*

I came to ADK straight from building an [MCP server by hand](/2026/06/30/beyond-tools-mcp-resources-and-prompts.html) &mdash; wiring the client session, driving the tool loop, moving the messages around myself. So when I sat down to write my first ADK agent, I braced for more of the same. Instead I wrote *one object*, typed `adk web`, and a chat window opened with my agent already talking to its tools.

That gap &mdash; between the little I wrote and the lot that appeared &mdash; is the whole story of ADK. Everything below is one question asked twice: **what do I write, and what does the framework build?**

## The whole agent is one object

ADK is *declarative*. You don't write an orchestration loop; you describe an agent as an object and hand it over. Here is the simplest possible one:

```python
from google.adk.agents import LlmAgent

def get_quick_estimate(address: str) -> dict:
    """Return a rough market estimate for a property address."""
    ...

root_agent = LlmAgent(
    name="basic_agent",
    model="openai/gpt-4o",
    description="A simple real estate assistant that estimates property values.",
    instruction=(
        "You are a helpful real estate assistant. When asked about a property, "
        "use the get_quick_estimate tool to look up its value, then give a "
        "brief summary including the estimate and confidence level."
    ),
    tools=[get_quick_estimate],
)
```

Four fields carry the entire identity: `name` (unique, and it shows up in the UI), `model` in `provider/model` format (routed through litellm, so `openai/gpt-4o` today and `google/gemini-2.0-flash` with a one-string change tomorrow), `instruction` (the system prompt), and `tools` &mdash; plain Python functions the model is allowed to call. No class to subclass, no run loop to write. The tool here just looks up a rough price for our running listing, 742 Evergreen Terrace.

One detail that isn't cosmetic: the variable is named `root_agent`. Hold that thought &mdash; it's load-bearing.

## Bringing in MCP tools, no glue code

A plain function works, but the tools I actually wanted were already living in the MCP server from the last series. In ADK I didn't re-implement them &mdash; I dropped an `MCPToolset` straight into the same `tools=[]` list:

```python
import sys
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StdioConnectionParams,
    StdioServerParameters,
)

root_agent = LlmAgent(
    name="mcp_tools_agent",
    model="openai/gpt-4o",
    description="Real estate pricing agent with MCP-discovered tools.",
    instruction=(
        "You are a real estate pricing analyst for Austin, TX. "
        "You have MCP tools that were auto-discovered from a pricing server. "
        "Always call your available tools before giving any estimates."
    ),
    tools=[
        MCPToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=sys.executable,
                    args=["m1_mcp/pricing_server.py"],
                )
            )
        )
    ],
)
```

At startup ADK spawns that MCP server as a subprocess, runs the `tools/list` handshake, and wraps each discovered tool so the model sees it exactly like a local function. When the model picks one, ADK runs the `tools/call` round-trip for me. This is the *list, then fetch* rhythm from the [MCP series](/2026/06/30/beyond-tools-mcp-resources-and-prompts.html) &mdash; except I never write it. The toolset does, and it owns the subprocess lifecycle too.

One sharp edge worth naming up front: `MCPToolset` discovers **tools only**. The resources, templates, and prompts that the MCP series spent so long on have no slot in an ADK agent &mdash; ADK is a tools client. If you leaned on prompts or resources, they don't cross this bridge.

## What `adk web` builds for you

So I have an object. How does it become a running, chatting agent? One command:

```bash
adk web adk_tutorials/
```

`adk web` walks the folder, and every subfolder holding an `__init__.py` and an `agent.py` that defines `root_agent` becomes a selectable agent in a dropdown. *That* is the discovery contract &mdash; and why the variable must be named `root_agent`. From that one object, the framework builds the entire runtime around it:

![On the left, the one thing you write: a single agent.py defining root_agent as an LlmAgent with name, model, instruction, and tools. The adk web command discovers and wires it. On the right, the runtime builds four things for free: the Runner (the model-to-tool loop), the SessionService (conversation memory), a Chat UI web app, and agent discovery that finds root_agent.](/images/adk/adk-what-you-write-vs-runtime.svg)

- **Runner** &mdash; the execution engine that drives the model &harr; tool loop and emits events. The loop I would otherwise hand-write.
- **SessionService** &mdash; conversation memory, storing every turn per session. In-memory by default; point it at a database with `--session_service_uri="sqlite:///sessions.db"` and the interface is identical.
- **A chat UI** &mdash; a local web app to talk to the agent, with no frontend code from me.
- **Discovery** &mdash; it found `root_agent` and dropped it in the menu.

The contrast with doing this over raw MCP is the whole point. There, I wrote the client session and drove the loop myself. Here I write the *what* &mdash; the agent as an object &mdash; and the framework builds the *how*. (There's even a free `--a2a` flag that hands every agent a JSON-RPC endpoint and an Agent Card so agents can call *each other* &mdash; but that's a later part.)

## The twist: who picks the next step?

Here's where it rhymes with the MCP series' *[who's in the driver's seat?](/2026/06/30/beyond-tools-mcp-resources-and-prompts.html)* An `LlmAgent` is **reasoning-driven**: the *model* decides the next action each turn &mdash; call a tool, call another, or just answer. Dynamic, and exactly what you want when the path isn't known ahead of time.

But sometimes you don't want the model improvising. Sometimes the steps are fixed and you want them to run the same way every time. ADK's **workflow agents** put the control flow back into code &mdash; *you* decide the order, not the model:

![Two contrasting styles. On the left, an LlmAgent where the model sits in the middle and dynamically chooses its next step at runtime — call a tool, call another, or answer. On the right, workflow agents whose control flow is fixed in code: Sequential runs sub-agents A then B then C in order, Parallel fans out to concurrent branches and merges them, and Loop repeats a step until a stop condition. LlmAgent thinks; workflow agents march.](/images/adk/adk-who-drives.svg)

Three shapes, one line each to reach for. `SequentialAgent` runs sub-agents in order, threading each one's output into the next. That threading is worth seeing up close: each sub-agent writes its result to an `output_key`, and the next one reads it by name with a `{placeholder}` right inside its own instruction:

```python
market_brief = LlmAgent(
    name="market_brief",
    model="openai/gpt-4o",
    instruction="Write a 2-line market summary for the Austin 78701 ZIP.",
    output_key="market_summary",           # writes its output to state["market_summary"]
)

offer_drafter = LlmAgent(
    name="offer_drafter",
    model="openai/gpt-4o",
    instruction=(
        "Read {market_summary} and draft a one-line opening buyer offer "
        "for 742 Evergreen Terrace. Output ONLY the offer text."
    ),
    output_key="offer_text",               # {market_summary} is filled in from state
)
```

Then you just list them in order &mdash; here a third `polisher` turns the raw offer into a sendable email:

```python
root_agent = SequentialAgent(
    name="negotiation_pipeline",
    sub_agents=[market_brief, offer_drafter, polisher],
)
```

That's the whole hand-off: *research &rarr; draft &rarr; polish*, with `{market_summary}` and `{offer_text}` quietly carrying state forward between steps. (There's a sharp edge when branches run *in parallel* and reuse the same key &mdash; but that's for a future part.)

`ParallelAgent` runs sub-agents concurrently, each in its own branch:

```python
root_agent = ParallelAgent(
    name="market_signals",
    sub_agents=[schools, comps, inventory],
)
```

Fan out to gather independent signals at once &mdash; schools, comps, inventory &mdash; then let a downstream step merge them.

`LoopAgent` repeats sub-agents until a stop condition:

```python
root_agent = LoopAgent(
    name="haggle_loop",
    sub_agents=[haggler],
    max_iterations=5,
)
```

The haggler proposes a price each turn; a callback flips `escalate = True` to break the loop the moment the price lands in range.

And they compose &mdash; a `SequentialAgent` of `LlmAgent`s is the everyday pattern: deterministic scaffolding on the outside, reasoning on the inside. The mnemonic that stuck for me: **`LlmAgent` thinks; workflow agents march.** When you want the model to find the path, reach for `LlmAgent`. When *you* already know the path and want it repeatable, take the wheel back with a workflow agent.

## The one-liner that ties it together

Distilled into one sentence: **you write the agent as one declarative object; `adk web` builds the runtime, memory, and UI around it; and when you need determinism, workflow agents let you take back the wheel.** You describe the *what*, and the framework supplies the *how*.

That framing is the whole reason ADK felt light to pick up &mdash; almost everything on the right side of that first diagram is machinery I'd have written by hand over raw MCP, handed to me for the price of naming a variable `root_agent`.

**Coming in the future:** a look under the hood of what the runtime handed us &mdash; the four state scopes that make memory actually work, the `output_key` clobber that can bite you when you reach for `ParallelAgent`, and how agents start talking to *each other* over A2A.
