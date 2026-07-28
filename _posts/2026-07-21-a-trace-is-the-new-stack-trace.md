---
date: 2026-07-21 09:00:00 -0400
title: "A Trace Is the New Stack Trace: Debugging Silent Agent Failures"
description: "Part 1 of a four-part series on shipping LLM systems you can trust: logs are flat, traces are trees, and that structural difference is the only thing that lets you blame a step instead of a request. The failure taxonomy, the single highest-value field in the UI, and four engineered traps from a real multi-agent run."
---

*Observe, Evaluate, Guard, Optimize series &mdash; 1. Observability (you're here) &middot; 2. [Evaluation](/2026/07/22/scoring-the-middle.html) &middot; 3. [Guardrails](/2026/07/23/prompts-suggest-guardrails-enforce.html) &middot; 4. [Cost](/2026/07/24/the-output-cost-floor.html)*

**Objective:** The opening of a four-part series on what "production" actually means for an LLM system, organized around one question: *a multi-agent system gave a confident, fluent, wrong answer and nothing errored &mdash; so how do you find out **which step** produced it?*

**Series thesis, stated once:** you can't manage what you can't see &mdash; and in a multi-agent system you have to see it *per component*, not just at the edges. Part 1 is about seeing.

A customer asks SecureBank's support agent: *"What's the overdraft fee?"* It answers **"$25"**, confidently and politely. The policy says **$35**. Nothing errored. No exception, no stack trace, no log line, no alert &mdash; a plausible, fluent, wrong answer, indistinguishable at the edges from a correct one. This is the **silent failure**, and it is the *default* failure mode of agents rather than an exotic one. The whole post is one question: where did "$25" come from?

## The system under the microscope

Everything below comes from one deliberately small system, built and then broken on purpose: a **multi-agent FinTech support agent** for a fictional SecureBank, wired up in [LangGraph](https://langchain-ai.github.io/langgraph/) and traced to [LangSmith](https://smith.langchain.com).

```
query → supervisor (intent classifier)
      ├─ "policy"          → policy agent      (RAG over 4 markdown policy docs in Chroma)
      ├─ "account_status"  → account agent     (mock account database lookup)
      └─ "escalation"      → escalation agent  (empathetic handoff, no retrieval)
```

Three specialists, one supervisor that makes a single classification call and routes to exactly one of them. That's it. It is small enough to hold in your head and still large enough that a wrong answer has **four** plausible origins &mdash; which is precisely the point.

## Three words that get used interchangeably, and shouldn't

| | What it is | What it answers |
|---|---|---|
| **Logging** | flat events | "did X happen?" |
| **Monitoring** | aggregates over time | "is the system healthy?" |
| **Observability** | structured per-request trees | "why did *this* request fail?" |

You need all three, but they are not three peers on a shelf. **Monitoring is a *view built on top of* observability**, not a sibling of it &mdash; the dashboard is computed *from* the trace data. Observability is the foundation layer, and the other two are projections of it.

## The one idea: logs are flat, traces are trees

A log is a list of events. A **trace** is a *parent-child run tree* for one request &mdash; and that structure is the entire difference:

![Two panels. Left, logging: a flat list of timestamped events — query received, LLM call ok, response sent, no errors. Nothing in the list says which agent ran or what the model actually saw, so it can only answer "did X happen?". Right, observability: the same request as one parent-child run tree — a chain run for the support graph containing an llm run for classify_intent with 87 to 3 tokens, then a chain run for escalation_agent containing an llm run with 235 to 103 tokens, and a dashed empty slot where a retriever run would be. The absent retriever run is the evidence that this request never did retrieval, so the tree can answer "why did this request fail?". The takeaway: the parent-child pointer is what lets you attribute a failure to a step rather than to a request.](/images/agentops/agentops-flat-vs-tree.svg)

The vocabulary is worth being precise about, because these three terms get muddled constantly:

- **One trace = one request.** It contains many **runs**.
- **One run = one step.** Runs form a parent-child tree. Four types: `chain` &middot; `llm` &middot; `retriever` &middot; `tool`.
- **"Span" is just the [OpenTelemetry](https://opentelemetry.io) word for a run.** Same concept, different vocabulary.
- **Only `llm` runs carry tokens.** Retrievers and tools cost **zero** tokens themselves &mdash; a fact that turns out to be the entire economics of Part 4.

And a `chain` run's latency and cost are the **roll-up of its subtree**, not work it did itself. That's why the top-level number is never the interesting one.

## The failure taxonomy: route → retrieve → assemble → generate

Back to "$25 when the policy says $35." A trace lets you enumerate the suspects exhaustively:

1. **Route** &mdash; the supervisor sent it to the wrong specialist.
2. **Retrieve** &mdash; three distinct sub-modes: (a) wrong document, (b) right document but wrong chunk, (c) right chunk, but **buried at rank 4** while `k=3`.
3. **Assemble** &mdash; `format_docs` truncated, reordered, or dropped a chunk *between retrieval and the prompt*. **This is the one everybody forgets**, and it's the reason "the retriever returned the right doc" is not the same claim as "the right doc reached the model."
4. **Generate** &mdash; two sub-modes: hallucination, or **distractor confusion**, where the model grabbed a different *real* number out of another retrieved chunk. That second one is nasty precisely because it's arguably still "grounded."

The fixes are different for every branch, which is the whole reason the taxonomy earns its keep: wrong doc → embeddings or query rewriting; wrong chunk → `chunk_size` and `overlap`; buried → `k` or reranking; distractor → *fewer, tighter* chunks. That last one is worth flagging early, because it inverts an intuition: less context is sometimes both **cheaper and more accurate**. Part 4 comes back to it.

## The single highest-value click

Here's the move that collapses that whole taxonomy into one binary. In the trace tree, open the **final `llm` run** and read its **`input`** &mdash; the fully rendered prompt, exactly as the model received it:

![Across the top, the four-stage pipeline: route, then retrieve, then assemble, then generate, then answer. A dashed callout drops from the boundary between assemble and generate to a highlighted focal box: read the final llm run's input, the fully rendered prompt. From that box the diagnosis splits two ways. If the correct fact is absent from the prompt, the bug is upstream — retrieval, chunking or assembly — and nothing downstream could have saved it. If the correct fact is present in the prompt, the bug is downstream in generation — either hallucination or distractor confusion, where the model grabbed a different real number from another retrieved chunk. The whole pipeline bisects at that single field.](/images/agentops/agentops-bisect-prompt.svg)

- Correct fact **absent** from the prompt → it's a **retrieval / chunking / assembly** bug. Nothing downstream could have saved it; no amount of prompt engineering fixes a fact the model never saw.
- Correct fact **present** in the prompt → it's a **generation** bug.

The entire pipeline bisects at that one field. Logs hand you the query and the answer, which are the two ends; the interesting failures all live in the middle, and only the trace shows you what the model actually saw.

## Four traps, four different lessons

The demo is built with `chunk_size=200, chunk_overlap=20` **deliberately** &mdash; tiny fragments, so facts like *"$35 per transaction, maximum 3 per day ($105)"* split across chunk boundaries. The failures below are engineered, not accidents:

```python
agent = build_support_agent(collection_name="observability_demo",
                           chunk_size=200, chunk_overlap=20)
```

Here is what actually happened when I ran the four traps and read the trees:

| Query | What it did | Stage | The lesson |
|---|---|---|---|
| "How much does overdraft protection cost?" | **passed** &mdash; got $12 right | &mdash; | the trap is *probabilistic*; n=1 proves nothing |
| "I'm really upset about being charged $105! What is your overdraft policy?" | misrouted to `escalation`, `Sources: []` | **route** | structural absence as evidence |
| "Does ACC-12345 qualify for the monthly fee waiver?" | right answer, zero reasoning | **route** (multi-hop) | a correct output is not a correct process |
| "How much does a replacement debit card cost?" | said "free" &mdash; truth is $5 | **generate** | conditions deleted, not facts invented |

Three of them are worth telling in full.

### Structural absence is evidence

The emotional phrasing in *"I'm really upset about being charged $105! What is your overdraft policy?"* tipped the supervisor into classifying it as `escalation`. The customer got a warm apology and no fee breakdown.

The tell in the trace isn't an error. It's a **hole**: there is **no `retriever` run anywhere in the tree.** The escalation agent doesn't retrieve, so the absence of that run *is* the proof that this request never touched the policy documents. You cannot see that in a log, because nothing failed &mdash; you can only see it in a structure where the missing child is visible against its siblings.

### Cross-source condition-stripping &mdash; not hallucination

Asked what a replacement debit card costs, the agent answered **"free."** The truth is **$5** standard, $25 expedited.

Read the rendered prompt and the mechanism is right there. It took *"free"* from `fraud_policy.md`, where replacement is genuinely free &mdash; **but only for fraud cases** &mdash; and *"$25 expedited"* from `account_fees.md`, and then merged two *conditional* policies into one *unconditional* claim by **deleting the conditions**.

This is not a hallucination. Every clause traces to a real retrieved chunk. Nothing was invented. And that is exactly why it's the error class that **survives human review**: a reviewer scanning for made-up facts finds none. It also names the real defect precisely &mdash; too *many* retrieved chunks gave the model two conditional sources to blend.

### A correct output is not evidence of a correct process

*"Does my account ACC-12345 qualify for the monthly fee waiver?"* got the **right answer**. Ship it?

No. The trace shows the account agent read a denormalized `monthly_fee_waived: True` boolean off the account record and reported it. It **never consulted the $1,500 balance rule** in the policy documents. The flag duplicates a policy rule, so:

- policy threshold changes → docs get updated, the flag goes stale → the agent is confidently wrong for every affected customer (in fintech, that's a compliance incident, not a bug ticket);
- the balance changes without a flag recompute → same thing;
- it can't answer *"why?"*;
- it doesn't generalize to any policy that lacks a precomputed flag.

And the reason this one is genuinely dangerous: it is **invisible**. It's right today, so nobody looks. It detonates months later. This trap, more than any other, is why Part 2 scores routing, retrieval and grounding **separately** rather than just grading the final answer.

Two of the four failures share one root cause &mdash; **single-label routing on a multi-intent query** &mdash; and neither is fixable by any amount of `chunk_size` or `k` tuning. But note the honest ordering: the *architectural* fixes (parallel fan-out for the independent case, a planner or sequential edge for the dependent multi-hop case) are not where I'd start. Letting the escalation agent *also* retrieve, or adding a tone instruction to the policy agent, collapses the routing trap without touching the graph at all. **Architecture changes are the last resort, not the first.**

## Averages lie, and they lie in a specific way

Once the tree is in place, monitoring is free &mdash; and immediately misleading if you read it naively. The agent's average latency is around **900ms**. That number describes **no request that has ever happened**:

```
escalation  ~400ms    1 cheap LLM call, no retrieval
account     ~550ms    supervisor + 1 LLM call over small JSON
policy     ~8000ms    supervisor + retriever + ~1,240-token prompt
```

This isn't "usually fast, sometimes slow." It's **two populations**. `p50=400ms / p95=8000ms` is a *mixture distribution*, and everyone in that tail is a policy user &mdash; which is to say, your highest-value traffic. The average sits in a valley where nothing lives.

So: **segment by tag before believing any percentile.** Tagging costs one argument:

```python
result = app.invoke(
    {"query": query, "intent": "", "response": "", "context": "", "retrieved_sources": []},
    config={"tags": [f"agent-type:{tag}", "demo-monitoring"]},
)
```

Then read p95 *per route*, and click into the single slowest trace. The waterfall then distinguishes two things a percentile never can: a slow policy trace with **small retriever latency and huge LLM latency** means prompt bloat &mdash; cut `k`. A slow **`retriever` run itself** means a vector-store problem (index size, HNSW `ef_search`), and cutting `k` won't help at all. Same p95, opposite fixes.

**Monitoring detects; tracing diagnoses.** They're a pipeline, not a choice.

## Where the retriever's cost actually lands

One number from the trace tree reframes the whole cost conversation, so it's worth planting here:

```
[LLM #1]  "should I retrieve?"           →    87 prompt +  3 completion
   └── [Retriever] vector search         →     0 tokens   (it's a database query)
[LLM #2]  "here's the context, answer"   → 1,240 prompt + 85 completion
                                             ↑ the retriever's real cost lands HERE
```

The retriever run costs **nothing**. Its real cost is the **prompt inflation on the next LLM call** &mdash; 87 tokens becomes 1,240. Which means `k=5` versus `k=2` is completely invisible on the retriever's own latency and cost, and enormous one run downstream. You can only see that relationship in a tree that puts the two runs side by side. Part 4 is entirely about the consequences.

## You don't need LangChain for any of this

The most common misconception about LangSmith is that it only works with LangChain. It doesn't &mdash; and the two mechanisms that make it framework-agnostic **compose** rather than compete:

```python
from langsmith import traceable
from langsmith.wrappers import wrap_openai

client = wrap_openai(OpenAI())      # → llm child runs, with token counts

@traceable(run_type="chain")        # → establishes the PARENT run
def answer(q):
    docs = search(q)                # @traceable(run_type="retriever") → child run, doc-list UI
    return client.chat.completions.create(...)   # → child llm run
```

People treat `wrap_openai()` and `@traceable` as alternatives. They're **layers**: `@traceable` supplies the parent context and the `run_type`, `wrap_openai` supplies token-accounted LLM children. Together they reconstruct exactly the same tree LangChain gives you for free, over plain SDK calls.

Worth knowing the neighbors, too: [Langfuse](https://langfuse.com) (open-source, self-hostable &mdash; the answer when the data can't leave your VPC, which is a live concern in fintech), [Arize Phoenix](https://phoenix.arize.com) (open-source, eval-leaning), [OpenTelemetry](https://opentelemetry.io) (the vendor-neutral standard), and [W&B](https://wandb.ai) for broader MLOps. And in production you sample **10&ndash;20%** of traffic, not 100%.

## Why this had to come first

There's a structural reason observability is Part 1 rather than a nice-to-have chapter somewhere later. Look at the signature of any evaluator worth writing:

```python
faithfulness(answer, retrieved_context)
```

That second argument is an **intermediate value that never leaves the pipeline.** It isn't in the request and it isn't in the response. Routing accuracy needs the supervisor's decision; MRR needs the ranked document list; assembly bugs need the rendered prompt. **Logs record endpoints; evaluators score the middle.** Of the five evaluators in Part 2, exactly one &mdash; end-to-end correctness &mdash; can be computed from what logging alone gives you.

Observability isn't a sibling of evaluation. It's a **precondition** for it.

## The one-liner that ties it together

Distilled: **a trace is the new stack trace &mdash; you don't step through code, you step through the run tree, and the parent-child pointer is what lets you blame a *step* instead of a request.** Read the final LLM run's input, and the whole pipeline bisects into "the fact never arrived" versus "the fact arrived and the model fumbled it."

Two lines from these runs I expect to keep repeating: *a correct output is not evidence of a correct process*, and *logs record endpoints &mdash; the interesting failures live in the middle*.

**Coming next:** seeing a failure isn't measuring it. Part 2 turns "this looks wrong" into a **score** &mdash; and shows why a single "is it correct?" number can tell you that something broke but never *where*. It starts from a 16-point gap between two metrics on the same 15 questions, and reads that gap as a diagnosis.
