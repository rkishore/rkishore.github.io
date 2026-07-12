---
title: "State & Control: Reaching Into What the ADK Runtime Handed You"
description: "Part 2 of the ADK series: the four state scopes that are ADK's memory model, the output_key clobber that bites parallel branches, and callbacks as deterministic policy the model can't prompt its way past."
---

*ADK series — 1. [Your First ADK Agent](/2026/07/08/your-first-google-adk-agent.html) · 2. State & Control (you're here) · more coming*

**Objective:** A field guide to the machinery the [ADK](https://google.github.io/adk-docs/) runtime built for you in [Part 1](/2026/07/08/your-first-google-adk-agent.html) &mdash; organized around one question: *now that the framework runs the loop, how do I see and shape what flows through it?*

[Part 1](/2026/07/08/your-first-google-adk-agent.html) ended on a comfortable note: workflow agents let you *take back the wheel* and decide the path. But that's the **coarse** wheel &mdash; who picks the next step. The runtime handed me more than a loop. It handed me a *memory* every step reads and writes, and a set of *interception points* around every model and tool call. This post reaches into both. Two axes: **state** is what persists; **callbacks** are the fine wheel.

## The state the runtime kept for you

Part 1 showed `output_key` threading one agent's output into the next. That value lands in one place: `tool_context.state` &mdash; a dict-like bag any tool can read and write. You opt a tool into it just by declaring a `tool_context: ToolContext` parameter; ADK injects it:

```python
def record_offer(price: int, tool_context: ToolContext) -> dict:
    """Record the user's latest offer price in session state."""
    history = tool_context.state.get("offer_history", [])
    history.append(price)
    tool_context.state["offer_history"] = history
    tool_context.state["user:total_offers"] = len(history)
    return {"recorded_price": price, "total_offers": len(history)}
```

Look at those two writes. `offer_history` and `user:total_offers` go into the *same* `state` object, but they don't live the same life. The **prefix on the key** decides that &mdash; and there are four scopes:

![Four concentric boxes: the outermost app scope is shared across every user and session; inside it user scope is bound to one user across their sessions; inside that the default no-prefix session scope lasts one conversation; innermost, temp scope lasts a single turn and is never persisted. A tool selects a scope by prefixing the key.](/images/adk/adk-state-scopes.svg)

- **`(no prefix)`** &mdash; *session* scope. Lives for this one conversation. The default.
- **`user:`** &mdash; bound to the `user_id`; survives *across* that user's sessions. `user:total_offers` keeps counting after the chat window closes and reopens.
- **`app:`** &mdash; bound to the app; shared across *all* users and sessions. A global counter.
- **`temp:`** &mdash; lives only for the current invocation; never persisted. A scratchpad for one turn.

The nesting is the whole mental model: outer scopes are longer-lived and more widely shared. (In `adk web` the `user_id` defaults to `"user"`, so `user:` and session look identical in a workshop &mdash; the distinction bites in production, with real, separate users.)

## When two branches write the same key: the `output_key` clobber

Part 1 flagged a *"sharp edge when branches run in parallel and reuse the same key"* and promised to come back to it. Here it is.

In a `SequentialAgent`, keys thread forward safely: each agent runs in order, so `market_brief` writes `market_summary`, *then* `offer_drafter` reads it. No collision. A `ParallelAgent` breaks that assumption &mdash; its branches run **concurrently, into the same `state` bag**. If two branches write the *same* `output_key`, they clobber each other and you keep whichever finished last:

```python
# BROKEN — three branches, one key. Two results silently vanish.
ParallelAgent(name="appraisals", sub_agents=[
    LlmAgent(name="appraiser_1", ..., output_key="appraisal"),   # clobbered
    LlmAgent(name="appraiser_2", ..., output_key="appraisal"),   # clobbered
    LlmAgent(name="appraiser_3", ..., output_key="appraisal"),   # last one wins
])
```

The fix is a discipline, not a feature: **give every parallel branch its own key**, then gather them in a downstream step. This is exactly how the real fan-out demo is wired &mdash; three market signals, three distinct keys:

```python
root_agent = ParallelAgent(
    name="market_signals",
    sub_agents=[
        LlmAgent(name="schools_signal",   ..., output_key="schools"),
        LlmAgent(name="comps_signal",     ..., output_key="comps"),
        LlmAgent(name="inventory_signal", ..., output_key="inventory"),
    ],
)
```

Then a reconciler downstream reads all three by name in its instruction &mdash; `Reconcile {schools}, {comps}, {inventory}...` &mdash; and nothing is lost. Distinct keys on the way out, gather on the way in.

## Callbacks: the fine wheel

State is what flows through the loop. **Callbacks are how you intercept an individual step.** The mental model that made them click for me: *instructions are suggestive; callbacks are deterministic.* You can tell the LLM "never offer above $460,000" in its instruction, and most of the time it obeys &mdash; but push it ("the seller is playing hardball") and GPT-4o will occasionally generate a $470,000 offer anyway. The instruction is a *nudge*. A callback that inspects the actual argument is a *guarantee* the model physically cannot prompt its way past.

ADK gives you a hook on both sides of every model and tool call. Three carry most of the weight:

![A left-to-right pipeline: a before_model gate redacts or scrubs the prompt, then the LLM runs; a before_tool gate does allowlisting and argument checks and can return a dict to block the tool; the Tool runs; an after_tool gate observes or logs and can return a dict to override the result, which then flows back to the model.](/images/adk/adk-callbacks.svg)

**`before_model_callback(callback_context, llm_request)`** &mdash; fires before each LLM request. Mutate the request in place to scrub what the model should never see:

```python
def redact_pii(callback_context, llm_request):
    """before_model: scrub SSN-shaped strings from every user message."""
    for content in llm_request.contents or []:
        for part in content.parts or []:
            if part.text and SSN_RE.search(part.text):
                part.text = SSN_RE.sub("[REDACTED]", part.text)
    return None
```

**`before_tool_callback(tool, args, tool_context)`** &mdash; fires before each tool call. Return `None` to let it run; return a **`dict` to block it** (that dict becomes the tool's result). This is where the budget cap lives &mdash; and the key move is that it inspects the *argument*, not the instruction text:

```python
BUYER_BUDGET = 460_000

def enforce_budget(tool, args, tool_context):
    # allowlist first (which tools), then argument policy (which values)
    if tool.name not in ALLOWED_TOOLS:
        return {"error": f"tool '{tool.name}' is not permitted"}
    if tool.name == "submit_decision":
        price = args.get("price")
        if isinstance(price, (int, float)) and price > BUYER_BUDGET:
            return {"error": f"price ${price:,} exceeds buyer budget of ${BUYER_BUDGET:,}"}
    return None  # allow
```

**`after_tool_callback(tool, args, tool_context, tool_response)`** &mdash; fires after the tool returns. Log it for observability, or return a `dict` to **override** the result (a clean fallback when a tool errors):

```python
def log_tool_result(tool, args, tool_context, tool_response):
    """after_tool: observability — and a place to substitute a fallback."""
    print(f"[after_tool] {tool.name} -> {tool_response}")
    return None
```

Notice the **symmetry**: `before_tool` returning a `dict` substitutes the result *before* the tool runs (a block); `after_tool` returning a `dict` substitutes it *after* (an override). Same wheel, both sides of the call &mdash; `None` to pass through, a `dict` to intervene. None of it touches the instruction.

## The capstone: it all comes together

The negotiation orchestrator is where state threading, structured signals, and callbacks lock together. It's a `LoopAgent` wrapping a `SequentialAgent` that runs the buyer, then the seller, for up to five rounds:

![An outer LoopAgent of up to five rounds contains a SequentialAgent running buyer then seller. The buyer writes output_key buyer_offer and reads seller_response; an arrow passes buyer_offer to the seller, which reads it, writes seller_response, and calls submit_decision to record a structured seller_decision; a loop-back arrow carries seller_response into the next round. Each agent has a before_tool allowlist, and the seller alone sees the floor price. The loop repeats until the seller's decision is ACCEPT.](/images/adk/adk-capstone-loop.svg)

Four ideas from this post, wired together:

**State hand-off.** The buyer writes `output_key="buyer_offer"`; the seller reads `{buyer_offer}` and writes `output_key="seller_response"`; next round the buyer reads `{seller_response}`. But round 1 has no seller response yet &mdash; so a `before_agent_callback` seeds it, or the buyer's `{seller_response}` reference would dangle:

```python
def _init_round_state(callback_context):
    if "seller_response" not in callback_context.state:
        callback_context.state["seller_response"] = "(No seller response yet — round 1)"
    return None
```

**A structured signal instead of prose.** How do you know the seller *accepted*? The tempting answer &mdash; `if "ACCEPT" in seller_text` &mdash; is a real bug I want to spare you: the seller's `get_minimum_acceptable_price` tool returns *"minimum **acceptable** price is $445,000,"* and the substring check false-fires on **ACCEPTable** every single round. The fix is to never read prose. The seller *must* call a typed tool that records a structured decision in state:

```python
def submit_decision(action: str, price: int, tool_context: ToolContext) -> dict:
    action_upper = action.strip().upper()
    if action_upper not in ("ACCEPT", "COUNTER"):
        return {"error": f"action must be ACCEPT or COUNTER, got: {action}"}
    tool_context.state["seller_decision"] = {"action": action_upper, "price": price}
    return {"recorded": action_upper, "price": price}
```

**Escalation off that signal.** An `after_agent_callback` reads the *dict field* &mdash; not text &mdash; and breaks the loop deterministically:

```python
def _check_agreement(callback_context):
    decision = callback_context.state.get("seller_decision")
    if isinstance(decision, dict) and decision.get("action") == "ACCEPT":
        callback_context.actions.escalate = True   # stop the LoopAgent
    return None
```

**Information asymmetry via allowlists.** Each agent gets a `before_tool_callback` allowlist. The buyer can call the market-facing pricing tools; the seller's allowlist *additionally* permits `get_minimum_acceptable_price` &mdash; the secret floor. The buyer can never call it, even if the model tries, because the callback blocks it. Policy, not prompt.

It all bolts together in a few lines:

```python
buyer  = LlmAgent(name="buyer",  ..., output_key="buyer_offer",
                  before_agent_callback=_init_round_state,
                  before_tool_callback=_enforce_buyer_allowlist)

seller = LlmAgent(name="seller", ..., output_key="seller_response",
                  tools=[..., submit_decision],
                  before_tool_callback=_enforce_seller_allowlist,
                  after_agent_callback=_check_agreement)

root_agent = LoopAgent(
    name="negotiation",
    sub_agents=[SequentialAgent(name="round", sub_agents=[buyer, seller])],
    max_iterations=5,
)
```

## The one-liner that ties it together

Distilled: **state is what persists** &mdash; a memory whose lifetime you pick with a key prefix; **callbacks are the fine wheel** &mdash; deterministic policy on each step, blocking before a call or overriding after it; and **structured-signal tools** turn an agent's decision into state you can test instead of prose you have to parse. Part 1 picked the *path*; Part 2 controls the *data and the individual steps*.

**Coming in the future:** so far every agent has lived in one process, sharing one `state` bag. The next question is what happens when they *don't* &mdash; when the buyer and seller are separate services that have to discover and call each other over the network. That's A2A: Agent Cards, the task lifecycle, and threading a conversation across the wire &mdash; its own post.
