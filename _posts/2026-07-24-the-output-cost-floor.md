---
title: "The Output Cost Floor: Why Halving the Context Didn't Halve the Bill"
description: "Part 4 and the close of the Observe, Evaluate, Guard, Optimize series: I cut prompt tokens by 52.9% and the bill fell 48.5%. The gap between those two numbers is the whole economics of an LLM system — output bills 4-5x input, so it's a floor your input savings can't cross."
---

*Observe, Evaluate, Guard, Optimize series &mdash; 1. [Observability](/2026/07/21/a-trace-is-the-new-stack-trace.html) &middot; 2. [Evaluation](/2026/07/22/scoring-the-middle.html) &middot; 3. [Guardrails](/2026/07/23/prompts-suggest-guardrails-enforce.html) &middot; 4. Cost (you're here)*

**Objective:** The close of the series, organized around one question: *I halved the context I feed the model and the bill didn't halve &mdash; **why**, and what does that tell me about which lever to pull next?*

**Recap of the thesis:** [Part 1](/2026/07/21/a-trace-is-the-new-stack-trace.html) gave us sight, [Part 2](/2026/07/22/scoring-the-middle.html) measurement, [Part 3](/2026/07/23/prompts-suggest-guardrails-enforce.html) enforcement. Part 4 makes it affordable &mdash; and shows that cost optimization is inseparable from the evaluation we built in Part 2.

I ran the same eight queries through two configurations of the SecureBank agent. Baseline: `chunk_size=1000, top_k=5`. Optimized: `chunk_size=400, top_k=3`. Prompt tokens dropped **52.9%**. The bill dropped **48.5%**. Completion tokens barely moved.

Those two percentages should be the same number, and they aren't. The gap is the entire economics of this system.

## The one idea: input and output aren't priced the same

Input and output tokens are billed at different rates, and not slightly: **output bills roughly 4&ndash;5&times; input.** On `gpt-4o-mini` that's $0.15 per million in against $0.60 per million out; on `gpt-4o`, $2.50 against $10.00. The list prices move, the ratio is what to memorize.

So your bill is two piles of very different character: a **big, cheap** input pile and a **small, expensive** output pile. Trimming retrieved context attacks the big cheap pile &mdash; which is the right first move, because it's where the volume is &mdash; but it leaves the small expensive pile completely untouched:

![Two stacked bars, one above the other. The baseline bar, built with chunk size 1000 and top k of 5, is mostly a long pale block of prompt tokens — the big, cheap pile — ending in a short dark block of completion tokens, the small pile that is priced roughly four to five times higher per token. An arrow labeled "trim the retrieved context: chunk 1000 to 400, k 5 to 3" leads down to the optimized bar, where the prompt block has shrunk dramatically while the completion block is exactly the same width as before. A bracket under the unchanged completion block marks it as the floor that input savings cannot cross. Two number chips report the result: prompt tokens fell 52.9 percent, but cost fell only 48.5 percent. The caption explains that the gap between those two percentages is the output floor made visible — percentage saved on tokens is not percentage saved on dollars.](/images/agentops/agentops-cost-floor.svg)

That untouched pile is a **floor**. No amount of context trimming crosses it, and as you trim harder it becomes a larger and larger share of what's left &mdash; so each additional token you cut buys less than the one before. **Percentage saved on tokens is not percentage saved on dollars**, and the two diverge in a direction you can predict. Which is also the practical instruction: match the lever to whichever side of *your* bill dominates. If output is your bulk, no context trimming will save you and you should be shortening responses instead.

## Two tools, two jobs: predict, then measure

```python
import tiktoken
from langchain_community.callbacks.manager import get_openai_callback
```

These get muddled constantly, and the distinction is simply **before versus after**.

**`tiktoken` counts locally, for free, *before* the call.** Nothing leaves your machine. It's how you price a system prompt, or check whether `chunk_size` &times; `k` will blow the context window, without spending anything to find out:

```python
encoder = tiktoken.encoding_for_model("gpt-4o-mini")
count   = len(encoder.encode(supervisor_prompt))
print(f"Hidden cost: {count} tokens × every call = {count * 1000:,} tokens/day at 1K queries")
```

**`get_openai_callback()` measures actual tokens and actual dollars from a real run.** Note what it aggregates: **every** call inside the `with` block &mdash; supervisor *and* specialist together, which for this agent is exactly the unit you care about, one customer query:

```python
with get_openai_callback() as cb:
    result = ask(app, query)
print(f"{cb.prompt_tokens} prompt | {cb.completion_tokens} completion | ${cb.total_cost:.6f}")
```

And the obvious objection, since Part 1 already gave us LangSmith traces with token counts on every `llm` run: why bother? Because they do different jobs. **LangSmith is visibility** &mdash; per-run breakdown, the waterfall, full inputs and outputs, best consumed by a human debugging one trace. **The callback is action** &mdash; in-process, so it can drive a cost threshold, a per-intent budget, or an automated before/after table in CI. Predict with `tiktoken`, measure with the callback, *understand* in LangSmith.

While we're on `tiktoken`, the misconception worth killing: **a token is not a word.** Roughly 1 token ≈ 0.75 words ≈ 4 characters, and the split is subword &mdash; `overdraft` tokenizes as `over` + `draft`. A 1,000-word policy document is about 1,333 tokens, and tokenizers differ per model, so count with the model's own.

## Where the money actually goes

Two findings from measuring, both of which are Part 1 findings wearing a price tag.

**The system prompt tax.** The supervisor's classification prompt is ~90 tokens. The policy agent's is ~120. That's **210 tokens billed on every single policy query, carrying zero new information** &mdash; 210,000 tokens a day at a thousand queries a day. Because that prefix is **byte-identical on every call**, it's also the single clearest candidate for provider-level prompt caching.

**Multi-agent cost is deeply asymmetric.** Not all queries are the same query:

| Path | Input tokens | Output tokens | What drives it |
|---|---|---|---|
| **Policy (RAG)** | ~830&ndash;1,430 | ~53&ndash;153 | **retrieved context** (600&ndash;1,200 of it) |
| Account | ~400 | ~73 | the account JSON |
| Escalation | ~220 | ~103 | prompt only &mdash; no retrieval |

A policy query costs several times what an escalation query costs, and the driver isn't the model or the system prompt &mdash; it's the **retrieved documents**. Which is precisely the Part 1 observation restated: the retriever run itself costs *zero* tokens, and its real cost is the prompt inflation on the **next** LLM call, 87 tokens becoming 1,240. This table is what that sentence looks like on an invoice.

It's also the same finding as the latency mixture from Part 1. Policy is both the ~8s p95 tail *and* the token bill, for one shared reason. When your slowest path and your most expensive path are the same path, you have one problem, not two.

## Four levers, in the order to reach for them

![Four numbered cards in the order to reach for them. One, trim the retrieved context: cut top k and chunk size, the highest impact and lowest effort, with the risk that if the relevant chunk sat at rank four you just lost it, so check MRR after every cut. Two, caching: provider-side prompt caching bills a repeated static prefix at a fraction of the input price, while semantic caching in your own vector store eliminates the call entirely on a hit — completely different mechanisms, not two names for one thing. Three, the Batch API: about fifty percent off for anything not real-time and tolerant of up to twenty-four hours, which is exactly what an offline evaluation run is. Four, model routing, deliberately last, because the complexity classifier itself costs tokens and latency and is a new component that can fail or misroute. Below the cards, a strip on picking by bottleneck: input-bound workloads want trimming and caching, easy-query-heavy traffic wants routing, and latency-tolerant bulk work wants the Batch API.](/images/agentops/agentops-four-levers.svg)

Two of these are routinely confused, and one is routinely reached for first when it should be last.

**Prompt caching and semantic caching are completely different mechanisms.** Prompt caching is provider-side and automatic: a repeated static prefix bills at a fraction of the normal input rate. It reduces **input cost** and nothing else. Semantic caching is *your* infrastructure &mdash; embed the query, look it up in your vector store, and on a hit **skip the LLM call entirely**. Reported hit rates run 15&ndash;60% depending on traffic shape. One shaves the price of a call; the other deletes the call. Conflate them and you'll reach for the wrong one when the bill spikes.

**Model routing goes last, not first.** Sending easy queries to a cheaper model is the most *intuitive* optimization, which is exactly why it's over-reached-for. The complexity classifier itself costs tokens and latency, and it is a **new component that can fail or misroute** &mdash; and Part 2 already showed what misrouting does to every downstream metric. Start with the free things (prompt caching) and the simple things (trim `k`), and build routing only once the quality difference justifies a whole new failure mode.

## MRR as a pre-flight check for a cost cut

Cutting `top_k` from 5 to 3 saves real money. It also risks deleting the chunk that contains the answer, and "risks" is doing a lot of work in that sentence &mdash; normally you find out by shipping it.

But Part 2 already produced the instrument. The MRR table doesn't just report a score, it reports **the rank of the relevant document for every individual query**. Read it as a cost instrument and it tells you, in advance and per query, exactly which cuts are safe:

| Query | Rank | Headroom at `k=3` |
|---|---|---|
| seven queries with a clean single-document mapping | 1 | plenty &mdash; two positions of slack |
| "What are the wire transfer fees?" | 2 | one position of slack |
| "How much does a replacement debit card cost?" | 3 | **none &mdash; one cut from breaking** |

The replacement-card query sits *exactly* at the `k=3` boundary. Trim to `k=2` and that answer disappears &mdash; and the failure would arrive as a confidently wrong response, the Part 1 silent-failure shape, not as an error anyone would page on. That single row turns cost optimization from a guess into something surgical. **MRR is the pre-flight check for a `k` cut: trim only as far as your retrieval headroom allows.**

## The step everyone skips

The before/after protocol is five steps, and the fifth is the one that gets dropped:

1. Baseline on a **fixed** set of queries.
2. Change **one** variable.
3. Re-run the **same** queries.
4. Compare tokens and dollars.
5. **Re-run the Part 2 evaluators.**

Cost savings that come with quality degradation are **false savings**. The demo run included a quality smoke test at step 4½, which is genuinely better than nothing and genuinely not enough:

```python
QUALITY_CHECKS = {
    "What is the overdraft fee?": ["overdraft", "fee"],
    "What credit score do I need for a personal loan?": ["credit", "loan"],
    "What is the balance on ACC-12345?": ["balance", "12450", "12,450"],
}
```

That's a substring check &mdash; Part 2's `keyword_correctness` in miniature, with all of its blind spots intact. It would happily pass an answer quoting the right number under the wrong label. The real gate is the LLM-judge correctness evaluator and the MRR table, re-run on the optimized configuration and compared against the baseline experiment. Cost and quality are **one problem measured two ways**, not two dashboards owned by two people.

The target is the **Pareto frontier**: maximum quality per unit cost. Off it in one direction you're wasteful, paying more for nothing; off it in the other you're cheap and wrong &mdash; and cheap-and-wrong is not a saving, it's a different bill arriving later.

## The trade-off isn't always a trade-off

Here's my favorite result in the whole series, and it only exists because all four modules ran over the same system.

Cutting `k` is *framed* as a cost-versus-quality trade. But Part 1's worst failure &mdash; the **cross-source condition-stripping** that answered "free" for a $5 replacement card &mdash; happened because the model had **too many** retrieved chunks: two conditional policies from two different documents, which it merged into one unconditional claim by deleting the conditions. More context handed it more distractors.

So **fewer, tighter chunks can be simultaneously cheaper *and* more accurate.** The cost lever and the quality lever pointed the same direction on that query. It doesn't always happen &mdash; but knowing it *can* is the difference between treating every optimization as a sacrifice and actually looking at the number.

## The one-liner that ties it together

Distilled: **output tokens bill 4&ndash;5&times; input, so the small output pile is a floor your context trimming can never cross &mdash; which is why cutting prompt tokens 52.9% only cut the bill 48.5%.** Match the lever to whichever side of the bill dominates, and verify every cut with the evaluators from Part 2.

Three lines to keep: *percentage saved on tokens is not percentage saved on dollars* &middot; *MRR is the pre-flight check for a `k` cut* &middot; *cost and quality aren't separate dashboards &mdash; every optimization is a bet you verify with your evaluators.*

## Closing the series

Four posts, one system, one thesis: **observe, evaluate, guard, optimize &mdash; you can't manage what you can't see, per component.** Here is the whole thing on one page &mdash; a single request from the input guards, through the agent, to the output guards, with each part's instrument annotated where it attaches:

![A left-to-right request-response spine across the top. A request enters an INPUT GUARDS gate that runs, cheapest-first, moderation, regex, an LLM injection classifier, and Presidio PII redaction. It then reaches the agent — Part 1's graph — where a supervisor routes to a specialist; the policy RAG path is shown as supervisor to retrieve to generate. The generated answer passes through an OUTPUT GUARDS gate that validates with Guardrails AI (SSN patterns, toxicity, competitor names) and redacts PII with Presidio, then returns as the response. Below the spine, four bands show where each part's instrument attaches: Part 1 Observability traces every box as one parent-child run tree per request; Part 2 Evaluation scores each hop — routing, retrieval via precision, recall and MRR, faithfulness, and correctness; Part 3 Guardrails is the two gates, cheapest-first and fail-open on the way in, validate-and-redact and fail-closed on the way out; Part 4 Cost meters every LLM run, where the retriever's cost lands on the generate step and output tokens bill four to five times input. A caveat band notes that in this series the guards ran only in Part 3's pipeline while Parts 1, 2 and 4 exercised the bare agent, so this diagram is the composed production system — how the four pieces fit into one request — not a single run that used all four.](/images/agentops/agentops-whole-system.svg)

One honest note on that diagram: the guards ran only in [Part 3](/2026/07/23/prompts-suggest-guardrails-enforce.html)'s guarded pipeline &mdash; Parts 1, 2 and 4 all exercised the *bare* agent, because each was isolating one concern. So this is the **composed** system, the thing you'd actually ship: how the four pieces fit into a single production request, not a single run that happened to use all four at once. Building them separately and composing them at the end is the normal shape of this work.

What I didn't expect going in was how much mileage came from **the same four demo traps**, run through all four lenses. The replacement-card query alone showed up as a *generation* blend in Part 1's trace tree, as a **rank-3 retrieval** miss in Part 2's MRR table, and as the query with **zero headroom** in the cost cut above &mdash; three views of one defect, each visible only through its own instrument. The fee-waiver trap that returned the right answer off a stale boolean is the argument for scoring routing, retrieval and grounding *separately*. And the input filter that ran clean while blocking nothing is Part 1's lesson &mdash; *a passing run proves the code executed, not that it worked* &mdash; wearing a security hat.

That's the discipline, and it's the thing I'd want a reader to take away over any individual metric: **production isn't a feature you add at the end. It's what you commit to seeing, scoring, guarding, and pricing &mdash; per component, not just at the edges.**
