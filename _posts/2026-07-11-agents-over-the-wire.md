---
title: "Agents Over the Wire: An A2A Field Guide"
description: "Part 3 of the ADK series: when agents stop sharing one in-process state bag and become separate services — A2A discovery via Agent Cards, the task lifecycle, contextId threading, and Parts vs Artifacts."
---

*ADK series — 1. [Your First ADK Agent](/2026/07/08/your-first-google-adk-agent.html) · 2. [State & Control](/2026/07/09/adk-state-and-control.html) · 3. Agents Over the Wire (you're here) · 4. [Two Substrates](/2026/07/12/two-substrates.html)*

**Objective:** A field guide to [A2A](https://a2a-protocol.org/) &mdash; the protocol for agents that *don't* share memory &mdash; organized around one question: *[Part 2's](/2026/07/09/adk-state-and-control.html) agents all shared one in-process `state` bag; what happens when they don't, and have to find and talk to each other over the network?*

Every agent so far has lived in one Python process, reading and writing one shared `state`. [Part 2's capstone](/2026/07/09/adk-state-and-control.html) leaned on exactly that: the buyer wrote `output_key="buyer_offer"`, the seller read `{buyer_offer}`, and the `LoopAgent` tied it all together *because they were the same program*. Pull them apart into separate services and three things you got for free suddenly need a protocol: **discovery** (how do I find the other agent?), **a conversation unit** (how do we track one exchange?), and **a message shape** (what actually crosses the wire?). That protocol is A2A.

## A2A vs MCP: vertical and horizontal

The cleanest way to place A2A is next to something you already know from the [MCP series](/2026/06/28/why-mcp-scales-as-n-plus-m.html): **MCP is *vertical*, A2A is *horizontal*.**

![Two agent boxes, a buyer and a seller, connected horizontally by an A2A link labeled message/send — they are peers. Each agent also connects vertically downward to its own MCP servers via tools/call. MCP is the vertical axis (agent to tools and data); A2A is the horizontal axis (agent to agent). One agent is often both an MCP client reaching down and an A2A peer reaching across.](/images/adk/adk-mcp-vs-a2a.svg)

- **MCP** runs *down*: an agent reaches to tools and data. One side is smart (the agent); the other serves. Stateless per call. `tools/list`, `tools/call`.
- **A2A** runs *across*: two *autonomous agents* talk as peers. Both sides are smart, both hold goals and state, either can initiate. `Agent Card`, `message/send`.

The detail that makes it click: **the same agent is usually both.** The seller reaches *down* over MCP for its pricing and inventory tools (including the secret floor price), and it answers *across* over A2A when a buyer &mdash; or an orchestrator &mdash; sends it an offer. MCP client below, A2A server above. They're complementary, not competing.

## Finding the other agent: the Agent Card

In-process, "finding" the seller was an import. Over the wire, an agent advertises itself with an **Agent Card** &mdash; a small JSON document at a well-known URL describing who it is and what it can do. Discovery is a two-step: **discover, then send.**

![A client first discovers an agent by fetching its Agent Card from the well-known URL, which returns name, url, skills, and capabilities; then the client sends a message with message/send to the agent server running under adk web --a2a, which creates a Task whose lifecycle runs submitted, working, completed (or failed).](/images/adk/adk-discover-send.svg)

You don't write the card by hand. Run `adk web --a2a` and ADK builds one for each agent from its `agent.json`, served at `/<agent>/.well-known/agent-card.json`. On the client side, `A2ACardResolver` fetches it and hands you a typed client:

```python
from a2a.client import A2ACardResolver, A2AClient

# discovery — the A2A equivalent of MCP's tools/list
resolver = A2ACardResolver(httpx_client=http, base_url=seller_url)
card = await resolver.get_agent_card()          # GET /.well-known/agent-card.json
client = A2AClient(httpx_client=http, agent_card=card)
```

The card tells you the agent's `name`, `url`, `skills`, and `capabilities` (does it stream? push?). Now you can send it a message &mdash; without ever importing its code.

## The task lifecycle

A `message/send` doesn't just get a reply; it creates a **Task** &mdash; a unit of work with a lifecycle, like a support ticket:

```
submitted  →  working  →  completed        (or → failed)
```

The nuance worth internalizing early: **`completed` means the *turn* finished, not that the *goal* was reached.** Send the seller a lowball offer and the Task still comes back `completed` &mdash; the agent processed your message and answered "I counter at $477K." `failed` is reserved for protocol errors and crashes. Whether a *deal* happened is your application's logic to decide, not a task state. (I proved this to myself by sending a deliberately broken, price-less envelope: still `completed`, with the seller politely asking me to resend.)

## contextId: threading a conversation across the wire

Here's the one that maps most directly onto [Part 2](/2026/07/09/adk-state-and-control.html). In-process, continuity came from the shared `state` bag. Over the wire, it comes from a **`contextId`** &mdash; A2A's equivalent of ADK's `session_id`, just one layer down (HTTP instead of in-process). The handshake:

1. **Round 1:** send *without* a `contextId`. The server assigns one and returns it.
2. **Rounds 2+:** echo that same `contextId` back, and the server loads the conversation history for it.

```python
context_id = None                                # round 1: let the server assign it
for round_num, price in enumerate([432_000, 440_000, 446_000], start=1):
    request = SendMessageRequest(
        id=f"req_{uuid.uuid4().hex[:8]}",
        params=MessageSendParams(
            message=Message(
                messageId=f"msg_{uuid.uuid4().hex[:8]}",
                role=Role.user,
                parts=buyer_offer_parts(round_num, price),
                contextId=context_id,            # None → server assigns; then reused
            )
        ),
    )
    task = (await client.send_message(request)).root.result
    if context_id is None:
        context_id = task.context_id             # capture it once, thread it after
```

Why it matters is best seen by *removing* it. Thread the `contextId` and the seller remembers the floor it looked up in round 1: `$432K → counter`, `$440K → counter`, `$446K → ACCEPT` (that last one clears the $445K floor). Now send a *fourth* offer of `$440K` with **no** `contextId` &mdash; the seller has zero memory, re-derives the floor from scratch, and counters at $477K exactly like it's round 1. The `contextId` is the *only* thing threading the conversation; without it, every request is a clean slate.

## Parts and Artifacts

One more shape to know. An A2A `Message` isn't a string &mdash; it's a list of **Parts**, and a Task can also carry **Artifacts**:

- **`TextPart`** &mdash; human-readable text (what the LLM reads).
- **`DataPart`** &mdash; a structured dict (what your code parses &mdash; prices, IDs, no `json.dumps` gymnastics).
- **`FilePart`** &mdash; binary with a MIME type (PDFs, images).

You'll often send the *same* offer as a `TextPart` **and** a `DataPart` in one message &mdash; human and machine copies side by side. **Artifacts** are different: they're the Task's *durable, named outputs*, separate from the message history. Think of it as **messages are the email thread; artifacts are the attached report.** So when you pull a reply out, go **artifacts-first, then fall back to history**:

```python
def extract_agent_text(task: Task) -> str:
    for artifact in task.artifacts or []:        # the deliverable, if any
        for part in artifact.parts:
            if isinstance(part.root, TextPart):
                return part.root.text
    for msg in reversed(task.history or []):     # otherwise, the last agent turn
        if msg.role == Role.agent:
            for part in msg.parts:
                if isinstance(part.root, TextPart):
                    return part.root.text
    return "(no response)"
```

## send vs stream

Finally, two ways to call: **`message/send`** is one request → one response, which is all this series needs. **`message/stream`** rides Server-Sent Events (the same SSE from the [MCP transports post](/2026/07/07/mcp-transports-sse-vs-streamable-http.html)) so the server can push incremental events as it works &mdash; partial text, tool calls, tool results &mdash; instead of making you wait for the final answer. Same discovery, same Task; only the delivery differs.

## The one-liner that ties it together

Distilled: **A2A gives peer agents the three things a shared process used to give them for free** &mdash; *discovery* (the Agent Card), *a conversation thread* (the `contextId`), and *a message shape* (Parts and Artifacts, wrapped in a Task with a lifecycle). MCP was the vertical reach down to tools; A2A is the horizontal reach across to other agents.

**Coming in the future:** I've now built the buyer↔seller negotiation *one* way &mdash; in-process, in [Part 2](/2026/07/09/adk-state-and-control.html). Next I build the *exact same* negotiation the *other* way, as independent services over A2A, and hold the two substrates side by side: what changed, what didn't, and how to decide which one a real system wants. That's the capstone.
