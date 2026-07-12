---
title: "Same Negotiation, Two Substrates: In-Process vs. the Wire"
description: "The ADK series capstone: I built the same buyer↔seller negotiation twice — once in one process, once across the network as A2A services — and held them side by side. What changed, what didn't, and how to choose."
---

*ADK series — 1. [Your First ADK Agent](/2026/07/08/your-first-google-adk-agent.html) · 2. [State & Control](/2026/07/09/adk-state-and-control.html) · 3. [Agents Over the Wire](/2026/07/11/agents-over-the-wire.html) · 4. Two Substrates (you're here)*

**Objective:** The capstone of the ADK series &mdash; organized around one question: *I built the same buyer↔seller negotiation twice, once in one process and once across the network. What actually changed, and what does that teach me about where to draw agent boundaries?*

I built the same negotiation twice. The first time, in [Part 2](/2026/07/09/adk-state-and-control.html), everything lived in one process and shared one `state` bag. The second time, the buyer and seller were separate services that discovered each other over [A2A](/2026/07/11/agents-over-the-wire.html) and never shared memory at all. Here's the thing that surprised me: **the negotiation *logic* is identical.** Buyer proposes, seller counters or accepts, repeat until a deal or a round cap. Only the **substrate** underneath it changed &mdash; and it changed exactly three mechanisms.

## Substrate #1, in one paragraph

The [in-process build](/2026/07/09/adk-state-and-control.html) was a `LoopAgent` wrapping a `SequentialAgent(buyer → seller)`. The buyer wrote `output_key="buyer_offer"`; the seller read `{buyer_offer}` and wrote `output_key="seller_response"`; the loop threaded them through one shared `state`. Termination was a structured signal: the seller called `submit_decision`, and an `after_agent_callback` flipped `escalate=True` on `ACCEPT` (capped by `max_iterations`). Information asymmetry &mdash; only the seller sees the floor price &mdash; came from a `before_tool_callback` allowlist. One program, one deploy, the framework driving the loop.

## Substrate #2: the A2A matchmaker

Now pull them apart. The buyer and seller run as independent A2A services (`adk web --a2a`), and a **matchmaker** &mdash; a plain script that knows *nothing* about how either agent is built &mdash; relays between them. It discovers both by Agent Card, keeps a **separate `contextId` per agent**, and loops:

```python
# Discover both peers — no imports, just their Agent Card URLs
buyer_client  = A2AClient(httpx_client=http,
    agent_card=await A2ACardResolver(httpx_client=http, base_url=buyer_url).get_agent_card())
seller_client = A2AClient(httpx_client=http,
    agent_card=await A2ACardResolver(httpx_client=http, base_url=seller_url).get_agent_card())

buyer_ctx = seller_ctx = None          # SEPARATE threads — the two agents never share one
seller_reply = None

for round_num in range(1, MAX_ROUNDS + 1):
    prompt = (f"The seller responded: {seller_reply}. Make your next offer."
              if seller_reply else "Make your opening offer for 742 Evergreen Terrace.")
    buyer_task, buyer_ctx = await send_a2a_message(buyer_client, prompt, buyer_ctx)
    offer = extract_agent_text(buyer_task)                 # artifacts-first

    seller_task, seller_ctx = await send_a2a_message(
        seller_client, f"The buyer offers: {offer}. Respond with ACCEPT or COUNTER.", seller_ctx)
    seller_reply = extract_agent_text(seller_task)

    if re.search(r"\bACCEPT\b", seller_reply, re.I) and not re.search(r"\bCOUNTER\b", seller_reply, re.I):
        print(f"Deal reached in round {round_num}!"); break
else:
    print("Max rounds reached — no agreement.")
```

Two subtleties I had to get right, both of which the framework had hidden from me in substrate #1. First, **the matchmaker is the *only* thing that knows about both agents** &mdash; each side just sees messages "from a user," so I thread `buyer_ctx` and `seller_ctx` separately and f-string the seller's reply into the buyer's next prompt myself. Second, **termination is now string-matching**, not a callback: `\bACCEPT\b` with a word boundary and *not* `COUNTER` (the `'ACCEPT' in 'acceptable'` trap from [Part 2](/2026/07/09/adk-state-and-control.html) is back, uncaught by any structured `submit_decision`), and the no-deal path rides Python's `for…else`.

## The substrate contrast

Same two agents, same job &mdash; two completely different pipes underneath:

![The same buyer and seller negotiation on two substrates. Left, in-process: a LoopAgent framework box contains the buyer and seller, which talk through one shared session state bag via output_key and placeholders; the framework orchestrates. Right, networked A2A: the buyer and seller are independent services with no shared memory, and a matchmaker script you own relays between them over HTTP with message/send, keeping a separate contextId thread for each.](/images/adk/adk-two-substrates.svg)

Line them up and every difference is one of three swapped mechanisms &mdash; plus who's holding the wheel:

| Concern | In-process `LoopAgent(Sequential)` | Networked A2A matchmaker |
|---|---|---|
| **How the offer reaches the other agent** | shared session state: `output_key` → `{placeholder}` | `message/send`, relayed over HTTP |
| **Conversation threading** | one `state` bag (implicit) | a `contextId` **per agent** (buyer & seller each their own) |
| **Termination** | `submit_decision` → `after_agent_callback` sets `escalate=True` (+ `max_iterations`) | matchmaker matches `\bACCEPT\b ∧ ¬COUNTER` (+ max rounds) |
| **Information asymmetry** | `before_tool` allowlist (seller-only floor tool) | separate `contextId`s keep threads isolated |
| **Coupling / deploy** | one process, one deploy, tightest | independent services, discovered via Agent Card |
| **Who orchestrates** | the `LoopAgent` (the framework) | your matchmaker (you) |

The negotiation *reasoning* &mdash; how the buyer picks a number, how the seller weighs its floor &mdash; sits in the agents and never moved. Everything in that table is *substrate*.

## Same protocol, two dialects

A quick aside, because it surprised me. I wrote the matchmaker two ways. One uses the **typed A2A SDK** &mdash; `SendMessageRequest`, `Message`, `TextPart` objects, with `A2AClient` handling the JSON-RPC framing. The other drops to **raw `httpx`** and hand-builds the envelope:

```python
# same message/send, no SDK — just the JSON-RPC dict
request_body = {
    "jsonrpc": "2.0",
    "id": f"req_{uuid.uuid4().hex[:8]}",
    "method": "message/send",
    "params": {"message": {
        "messageId": f"msg_{uuid.uuid4().hex[:8]}",
        "role": "user",
        "parts": [{"kind": "text", "text": text}],
        **({"contextId": context_id} if context_id else {}),   # thread only when we have one
    }},
}
resp = await http.post(agent_url, json=request_body)
```

Identical protocol, two dialects &mdash; typed objects vs. a plain dict on `POST`. That's the point of A2A being *just HTTP + JSON-RPC*: any language that can POST JSON can play, no shared SDK required. The typed SDK is nicer to live in; the raw version shows there's no magic underneath.

## Promote the matchmaker into its own agent

The matchmaker above is a *script* &mdash; a pure A2A client. But it can become an agent itself. Wrap the orchestration as an ADK agent and serve it with `adk web --a2a`, and it gets its own Agent Card and endpoint. Now it's **both**:

- an **A2A server** to its callers (they send it "negotiate 742 Evergreen Terrace" via `message/send`), and
- an **A2A client** to the buyer and seller (it discovers and calls them, keeping their separate `contextId`s).

That dual role is how A2A networks grow past two agents: the two peers stay hidden behind the matchmaker's card, and *its* callers don't know or care that there's a negotiation happening underneath &mdash; recursion all the way down. One detail ties it back to [Part 1](/2026/07/08/your-first-google-adk-agent.html): the matchmaker's root is a **workflow agent, not a plain `LlmAgent`**. Its job is deterministic relay-and-terminate control flow &mdash; *code drives*, not a model improvising &mdash; exactly the "take back the wheel" distinction from the very first post. (For sub-agents that are genuinely remote, ADK's `RemoteA2aAgent` wraps a card URL so you can drop it into `sub_agents=[...]` as if it were local.)

## When to choose which

So which substrate? The honest answer is that it's a *boundary* decision, not a *better/worse* one:

- **In-process** is tight, fast, and simplest: one deploy, one language, the framework doing the orchestration and termination for you. Reach for it when you own all the agents and they ship together. It's the right default.
- **Networked A2A** buys you *independence*: separate teams, languages, and scaling; agents you can swap as long as they speak A2A and present compatible skills; a single place (the matchmaker) for logging, retries, and human-in-the-loop gates. You pay for it in moving parts &mdash; discovery, wire format, and new failure modes that a single process never had.

The question A2A really answers isn't "how do agents talk?" &mdash; it's "*where do I draw the boundary between them?*" In-process and over-the-wire are the two answers, and now you've seen the same negotiation wear both.

## The one-liner that ties it together

Four posts, one thread: **[a single declarative agent](/2026/07/08/your-first-google-adk-agent.html) → [the state and control the runtime handed you](/2026/07/09/adk-state-and-control.html) → [the wire for when agents don't share memory](/2026/07/11/agents-over-the-wire.html) → the same job on both substrates.** The negotiation logic never changed across any of it. What changed was how much of the machinery I wrote versus how much the framework built &mdash; and, in the end, where I chose to draw the line between one agent and the next. That line *is* the design.

Thanks for reading the series.
