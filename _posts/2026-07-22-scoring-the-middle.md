---
title: "Scoring the Middle: Why One Correctness Number Can't Find Your Bug"
description: "Part 2 of the Observe, Evaluate, Guard, Optimize series: the agent scored 0.93 faithfulness and 0.77 correctness on the same 15 questions, and that 16-point gap is a diagnosis. Five eval layers, MRR with its misses named, DeepEval metric directions (one of them inverted), and why temperature=0 still needs three repetitions."
---

*Observe, Evaluate, Guard, Optimize series &mdash; 1. [Observability](/2026/07/21/a-trace-is-the-new-stack-trace.html) &middot; 2. Evaluation (you're here) &middot; 3. [Guardrails](/2026/07/23/prompts-suggest-guardrails-enforce.html) &middot; 4. [Cost](/2026/07/24/the-output-cost-floor.html)*

**Objective:** Part 2 of the series, organized around one question: *[Part 1](/2026/07/21/a-trace-is-the-new-stack-trace.html) let me **see** each component &mdash; how do I turn "this looks wrong" into a **number**, and why does one number never tell me **where** it went wrong?*

**Recap of the thesis:** Part 1 gave us traces, the ability to see per component. Part 2 turns seeing into measuring. The key move is the same move, one layer up: **don't score the edges, score the middle.**

The agent scored **0.77 on correctness** and **0.93 on faithfulness**. Same 15 questions, same run, same model. If those two were the same metric they'd be equal &mdash; they aren't, and that **16-point gap is a diagnosis**. It says the bug is in *retrieval*, not generation, and it says so before I've opened a single trace. This post is about why a single "is it correct?" score can never say that.

## The one idea: a single score detects, per-layer evaluators localize

A multi-agent RAG system has **five** places to be wrong, and each of them deserves its own evaluator:

![Five stacked rows, one per evaluation layer. Row one, route: the routing_accuracy evaluator, free and deterministic, asks did the supervisor pick the right agent. Row two, retrieve: precision at k and recall at k ask how much of what came back was relevant, and how much of what was relevant came back. Row three, rank: mean reciprocal rank asks how high the right document was ranked. Row four, faithfulness: an LLM judge over the answer and the retrieved context asks whether every claim is grounded. Row five, correctness: an LLM judge over the answer and the ground truth asks whether the facts match. A bracket on the right shows that a single end-to-end "is it correct?" score collapses all five rows into one number. The takeaway bar: end-to-end metrics detect that something is wrong; per-layer evaluators localize which layer is wrong.](/images/agentops/agentops-five-layers.svg)

Collapse those five rows into one end-to-end number and you learn *that* the system is broken, never *where*. It's the same logic as Part 1's monitoring-versus-tracing distinction, restated at the eval layer: **end-to-end metrics detect; per-layer evaluators localize.**

An evaluator in LangSmith is just a function of a run and a labeled example, returning a keyed score:

```python
def routing_evaluator(run, example):
    predicted = run.outputs.get("intent", "")
    expected  = example.outputs.get("intent", "")
    return {"key": "routing_accuracy", "score": 1.0 if predicted == expected else 0.0}
```

That one is free and deterministic. Note what it reads: `run.outputs["intent"]` &mdash; the supervisor's internal decision, which never appears in the customer-facing response. This is Part 1's point made concrete. **You cannot write this evaluator without traces.**

## Faithful ≠ correct, and the distinction is the whole game

The two LLM-judge evaluators look superficially similar and measure orthogonal things:

- **Faithfulness** asks *"is the answer grounded in the context we actually retrieved?"* &mdash; signature `f(answer, retrieved_context)`. Ground truth is not involved.
- **Correctness** asks *"does it match the labeled ground truth?"* &mdash; signature `f(answer, expected_answer)`. The context is not involved.

Because they take different inputs, they can disagree in both directions:

![A two-by-two grid with correctness increasing along the horizontal axis and faithfulness increasing along the vertical axis. Bottom left, low faithfulness and low correctness: hallucinating, ungrounded and wrong. Bottom right, low faithfulness and high correctness: right answer with zero grounding, a shortcut that is fragile the day it breaks. Top left, high faithfulness and low correctness: faithful to the wrong or incomplete document — the RAG signature, and where the run landed with faithfulness 0.93 and correctness 0.77. Top right, high on both: grounded and right, the only quadrant you can ship. A marked point sits in the top-left quadrant at correctness 0.77 and faithfulness 0.93. The caption notes that a single score cannot distinguish these four quadrants.](/images/agentops/agentops-faithful-vs-correct.svg)

Retrieve the *wrong* document and summarize it perfectly and you are **faithful and wrong** &mdash; top-left. Answer correctly from the model's prior knowledge while ignoring the retrieved context entirely and you are **correct and ungrounded** &mdash; bottom-right, and fragile in exactly the way Part 1's fee-waiver trap was fragile: right today, silently wrong the day the shortcut breaks.

So the run's **0.93 / 0.77** lands squarely in the top-left quadrant, and the shape of the gap is the message: *faithfulness above correctness* means **"the model honestly reports what it retrieved, but what it retrieved was incomplete."** That is the RAG signature, and it points the fix at a retrieval knob &mdash; `chunk_size`, `k`, the embeddings &mdash; not at the generation prompt. One number could never have said that. Two numbers said it for free.

The judge itself is unremarkable, which is rather the point &mdash; the discipline is in *what you ask it to compare*, not in the plumbing:

{% raw %}
```python
FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Assess whether the answer is faithful to the provided context.\n"
     "Score 1.0 = fully faithful: every claim is supported by the context\n"
     "Score 0.5 = partially faithful: some claims are unsupported\n"
     "Score 0.0 = not faithful: claims contradicting or absent from context\n"
     'Respond ONLY with JSON: {{"score": <float>, "reason": "<one sentence>"}}'),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:\n{answer}"),
])
```
{% endraw %}

Two details that matter more than they look. **Ask for a `reason` alongside the score** &mdash; a bare float is unactionable, and the one-sentence reason is what turns a bad score into a bug report. And **have a fallback for unparseable JSON**, because a judge that occasionally returns prose will otherwise take your whole eval run down with it.

## MRR measures retrieval &mdash; and a bad score accuses two suspects

Faithfulness and correctness both sit downstream of retrieval. To measure retrieval *itself* you need metrics over the returned documents, and the place to start is the classic pair &mdash; both computed **at `k`**, since `k` is how many documents your retriever hands back:

- **Precision@k &mdash; of what we retrieved, how much is relevant?** This is the **noise** measure. Return 5 chunks of which 2 actually bear on the question and precision is 0.4; the other three are distractors sitting in the prompt, costing tokens and &mdash; as Part 1's condition-stripping trap showed &mdash; actively inviting the model to blend sources.
- **Recall@k &mdash; of everything relevant, how much did we retrieve?** This is the **completeness** measure. If the answer spans 3 chunks and only 2 came back, recall is 0.67, and nothing downstream can recover the third. It's the metric that catches the "the fact never arrived" half of Part 1's bisect.

The two pull against each other on a single knob: **raise `k` and recall goes up while precision goes down.** That sentence is the whole `top_k` tuning problem, and it's why the cost cut in Part 4 is a real trade rather than free money.

What neither of them can tell you is *where* in the list the relevant document landed &mdash; retrieve the right chunk at position 1 or at position 5 and precision@5 and recall@5 are identical. Since a RAG prompt is read top-down and the tail gets truncated first, position matters. That's the gap **Mean Reciprocal Rank** fills: for each query, find the position of the first relevant document, take `1/rank`, average across queries.

```python
reciprocal_ranks = []
for item in mrr_queries:
    docs = retriever.invoke(item["query"])
    rank = next((i for i, d in enumerate(docs, 1)
                 if d.metadata.get("source") == item["relevant_source"]), 0)
    reciprocal_ranks.append(1.0 / rank if rank else 0.0)

mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
```

*Reciprocal* rank, not mean rank, and the choice is deliberate: moving a document from position 2 to position 1 matters enormously (1.0 vs 0.5), while moving it from 10 to 9 barely registers (0.111 vs 0.100). That curve matches how RAG actually consumes results &mdash; you usually need *one* good chunk, near the top.

I ran it over ten queries, chosen so that several are genuinely ambiguous, against four policy documents:

| Query | Rank | RR |
|---|---|---|
| seven queries with a clean single-document mapping | 1 | 1.00 |
| "What are the **wire transfer fees**?" | 2 | 0.50 |
| "How much does a **replacement debit card** cost?" | 3 | 0.33 |
| "What **interest rate** will I get?" | &mdash; | 0.00 |
| | | **MRR = 0.783** |

Naming the misses is the whole value of the exercise, because each one has a *different* cause:

- **"Wire transfer fees" ranked #2** &mdash; vocabulary overlap. The term genuinely lives in two documents (`account_fees.md` and `transfer_policy.md`); the retriever picked the other one first. That's a real retrieval weakness.
- **"Replacement debit card" ranked #3** &mdash; and here's the satisfying bit. This is the *same query* that Part 1 caught as a generation-layer blend, where the agent merged "free" from the fraud policy with "$25 expedited" from the fee schedule. Part 1 named the symptom; the MRR table names its **retrieval-layer root cause**. Same trap, two layers, two views.
- **"What interest rate will I get?" scored a reciprocal rank of zero** &mdash; the sentinel for *not retrieved at all*. But before blaming the retriever: this query is *genuinely ambiguous*. "Interest rate" matches savings APY **and** loan APR, and the label says `account_fees.md`. The retriever wasn't obviously wrong; **the label was.**

That last one is the lesson worth carrying: **a bad eval score is an accusation against either your system or your ground truth, and the number alone cannot tell you which.** Only inspection can. A metric you never drill into is a metric you're trusting blind.

## Match the tool to the check

Not everything needs an LLM. The cheap deterministic evaluator in the hill-climb suite extracts figures from the expected answer and checks whether they appear in the actual one:

```python
def keyword_correctness(run, example):
    key_terms = re.findall(r"\$[\d,.]+|\d+(?:\.\d+)?%?|acc-\d+",
                           example.outputs.get("answer", "").lower())
    if not key_terms:
        return {"key": "keyword_correctness", "score": 0.5}
    actual = run.outputs.get("answer", "").lower()
    matches = sum(1 for term in key_terms if term in actual)
    return {"key": "keyword_correctness", "score": round(matches / len(key_terms), 4)}
```

Free, instant, reproducible &mdash; and blind in two specific directions. It **false-positives** on right-numbers-wrong-assignment: an answer saying *"$105 per transaction, maximum $35"* contains both figures and scores 1.0, while being exactly backwards. It **false-negatives** on formatting: `$1,500` and `$1500` are different strings. So use a substring rule for *mechanical* checks and an LLM judge for *semantic* ones &mdash; spending a nondeterministic LLM call to paper over a comma is the wrong trade in both directions.

And that `0.5` in the early-return deserves a hard look. It fires when the expected answer contains no numbers at all, and it means **"no signal"** &mdash; but it enters the mean exactly like a real score. Every example that hits that branch drags both arms of an experiment toward 0.5 and *compresses the very delta you're trying to read*. Before trusting a small improvement, count how many of your examples took that path.

## DeepEval: 50+ metrics, and one of them runs backwards

[LangSmith](https://docs.smith.langchain.com/evaluation) and [DeepEval](https://deepeval.com) are complements, not rivals. LangSmith gives you tracing, a comparison UI, and custom `f(run, example)` evaluators for interactive iteration; DeepEval gives you 50+ pre-built metrics and a pytest-native runner that exits non-zero, which is what you want gating a pull request.

The table worth memorizing is the one about **direction**, because the metrics do not all point the same way:

| Metric | Requires on the test case | Direction | Success condition |
|---|---|---|---|
| `FaithfulnessMetric` | `input`, `actual_output`, **`retrieval_context`** | higher = better | `score >= threshold` |
| `AnswerRelevancyMetric` | `input`, `actual_output` | higher = better | `score >= threshold` |
| `HallucinationMetric` | `input`, `actual_output`, **`context`** | **lower = better** | `score <= threshold` |
| `GEval` | whatever is in `evaluation_params` | higher = better | `score >= threshold` |

The course notes I was working from stated that `HallucinationMetric` returning 1.0 means *no* hallucination. That is inverted, and I only caught it by reading the installed package: in `deepeval/metrics/hallucination/hallucination.py`, the score is `hallucination_count / number_of_verdicts` &mdash; the **fraction of supplied contexts the answer contradicts** &mdash; and success is `self.score <= self.threshold`. **1.0 means it contradicted every context.** Which has a follow-on the docs don't shout about: a `threshold=0.7` copied from the faithfulness example is wildly permissive here, passing an answer that contradicts 70% of its contexts. Hallucination wants a threshold near **0&ndash;0.2**.

Second gotcha, same family, differently annoying: **faithfulness reads `retrieval_context`; hallucination reads `context`.** Different required fields for what feels like the same thing. Set only one and your loop dies on a missing-parameter error halfway through a paid run, which is why the test cases set both to the same list:

```python
ctx = [result["context"]] if result["context"] else ["No context retrieved."]
LLMTestCase(input=query, actual_output=result["response"],
            retrieval_context=ctx, context=ctx)
```

## G-Eval, and the negative control that makes a suite trustworthy

Some qualities have no reference answer at all. Empathy in an escalation response is one: there's no ground truth string to compare against, only a rubric. That's what **G-Eval** is for &mdash; you write criteria in prose, and the judge applies them:

```python
empathy_metric = GEval(
    name="Empathy",
    criteria=("Evaluate whether the response shows genuine empathy... A highly empathetic "
              "response should: 1) Acknowledge the customer's frustration or distress, "
              "2) Validate their feelings without being dismissive, 3) Offer a clear next "
              "step (escalation, contact info), 4) Use warm, professional language."),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)
```

Two things are load-bearing here. **`evaluation_params` is a visibility list** &mdash; it declares which test-case fields the judge is allowed to see. Empathy is a property of the *response alone*, so the judge gets `ACTUAL_OUTPUT` and nothing else; include the input and the tone of the *question* starts leaking into a score that's supposed to be about the answer. And **the criteria are the metric.** "Be empathetic" is a bad criterion because the judge then invents its own rubric, differently each run. Numbered, specific, actionable criteria are what make a G-Eval score reproducible &mdash; criteria quality *is* metric quality.

Now the part that separates a real suite from a decorative one. The three empathy test queries are two furious customers **and one dry factual question**: *"What is the wire transfer fee?"* That one **should score low**. It's a **negative control**. Same reasoning put the deliberately vague *"What interest rate will I get?"* in the DeepEval set. A suite where everything passes has told you nothing about your metric's *sensitivity* &mdash; it's equally consistent with "the system is great" and "the metric is broken."

## `temperature=0` is not determinism

I ran the whole evaluation with `num_repetitions=3`:

```python
evaluate(run_agent, data=EXERCISE_DATASET_NAME,
         evaluators=[routing_evaluator, faithfulness_evaluator, correctness_evaluator],
         experiment_prefix="exercise-eval-student",
         num_repetitions=3,
         metadata={"model": "gpt-4o-mini", "version": "route-faith-correct"})
```

Why, when every LLM call in the system is `temperature=0`? Because **`temperature=0` is greedy decoding, not a determinism guarantee.** Floating-point addition isn't associative, and the batch your request lands in on the server changes the reduction order; mixture-of-experts routing adds its own drift. Low variance is not zero variance.

So a single run is a sample from a distribution, and two means are only trustworthy-different if their **ranges don't overlap**. `num_repetitions` isn't there to make the mean prettier &mdash; it's there to **measure the noise floor so you can tell signal from it.** One repetition is fast and noisy, three is the usual balance, five or more is robust and expensive.

There's a companion fact from Part 1 that's easy to get backwards: given a fixed index, **retrieval is deterministic**. Same query → same embedding → same top-k, every time. So re-running an *identical* query tests **generation**, not retrieval. To exercise retrieval you must **vary the phrasing** &mdash; which is exactly where vocabulary mismatch bites, and exactly what the MRR query set was built to do.

## Hill climbing, and designing an experiment that *can* move

With evaluators in place, improvement becomes a loop: fix the **worst** metric first, change **one** variable, re-evaluate, then confirm the others didn't regress. Not globally optimal &mdash; attributable and debuggable, which is the point.

The hill-climb run moved `top_k` from 1 to 5 while **pinning `chunk_size=200`**, and that pinning is not incidental. With large chunks, `top_k=1` already carries the whole fact, so both arms score identically and you'd conclude *"top_k doesn't matter"* from an experiment **structurally incapable of detecting that it does**. Designing an experiment that *can* move the needle is part of the method, not a detail of the setup.

The suite also carries `routing_accuracy` as a **control**. Changing `top_k` must not move it. If it does, something else changed and the comparison is contaminated &mdash; which is a much better thing to learn from a control than from a confusing result three experiments later.

Two more disciplines that cost nothing and save arguments:

- **Curation is a separate step from evaluation.** The three edge-case examples appended to the dataset &mdash; a multi-part question (single-label routing against two intents), a nonexistent `ACC-00000` (the refusal path, not the happy path), and an exact-threshold boundary ("exactly $1,500", where the policy text and the model's arithmetic disagree) &mdash; are **not** scored by the `evaluate()` that already ran. They land on the *next* one. That's not just bookkeeping: grow the dataset mid-experiment and your two experiments have different denominators, so the comparison is dishonest. Coverage beats volume anyway &mdash; 15 examples across every path beat 200 happy-path policy questions.
- **Offline eval is not an A/B test.** No traffic split, no significance testing, no weeks of data. What it buys instead is *minutes-not-weeks* feedback on a fixed, reproducible dataset. It's the gate that decides what earns an A/B test. A pipeline, not rivals.

## The one-liner that ties it together

Distilled: **one "is it correct?" score tells you the system is broken; five per-layer evaluators tell you which layer broke &mdash; and the *gap between two* of them is often the diagnosis by itself.** Faithfulness catches hallucination, correctness catches retrieval failure; ship both or you're blind on one axis.

Three lines I expect to reuse: *a bad eval score is an accusation against either your system or your ground truth, and you can't tell which from the number* &middot; *`num_repetitions` isn't there to make the mean prettier, it's there to measure the noise so you can tell signal from it* &middot; *a suite where everything passes has told you nothing.*

**Coming next:** we can now see the system and score it. But evaluation measures quality *on average*, and an average is exactly the wrong instrument for the one response that leaks a customer's SSN. Part 3 is the first **enforcement** layer &mdash; four guardrail strategies, why input and output guards defend against completely different threats, and a security control of mine that ran clean and guarded nothing at all.
