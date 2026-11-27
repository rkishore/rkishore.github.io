---
date: 2026-08-25 18:30:00 -0400
title: "Don't Let the Model Do the Math"
description: "Part 5 of the callsense series: how a call gets scored. The model judges five dimensions; code applies the weights and throws the model's own total away. In the demo it said 3.0 and the code said 3.5. Then one critical compliance flag sends the call to a person, whatever the average."
---

{% include info.html text="This content was written with assistance from Claude AI." %}

*callsense series &mdash; 1. [The Problem](/2026/08/05/callsense-the-call-nobody-reviews.html) &middot; 2. [Six Slots, One Pipeline](/2026/08/10/callsense-six-slots-one-pipeline.html) &middot; 3. [Measuring the Ear](/2026/08/15/callsense-measuring-the-ear.html) &middot; 4. [When the Input Is the Prompt](/2026/08/20/callsense-when-the-input-is-the-prompt.html) &middot; 5. Don't Let the Model Do the Math (you're here) &middot; 6. [From Clean Clone to Deployed](/2026/08/30/callsense-from-clean-clone-to-deployed.html) &middot; 7. [The Guard as a Tool](/2026/10/30/callsense-the-guard-as-a-tool.html) &middot; 8. [Measuring the Guard](/2026/11/04/callsense-measuring-the-guard.html)*

**Objective:** [Part 1](/2026/08/05/callsense-the-call-nobody-reviews.html) promised a score that doesn't depend on how a model felt that run. This part is how callsense splits the work between model and code, organized around one question: *which parts of a quality score need judgement, and which must be arithmetic you can check?*

Here is the demo. The language model (LLM) looked at a call and proposed an overall score of **3.0**. The screen shows **3.5**. And the call ended with status `flagged_for_review`, not `completed`, despite that 3.5.

Both of those are on purpose. The rest of this post explains why.

## 1. Five dimensions, five weights

callsense gives every call a quality-assurance (QA) score on five dimensions. Each one gets an integer from 1 to 5 and a written justification. The overall score is a **weighted score**: each dimension's score is multiplied by its weight, and the results are added up.

| Dimension | Weight |
|---|---|
| Professionalism | 15% |
| Empathy | 20% |
| Problem Resolution | 30% |
| Compliance | 20% |
| Communication Clarity | 15% |

The weights live in one table in `src/graph/state.py`. Resolving the customer's problem counts for most. Tone and clarity count for least.

There is one small trap. Computers can't store 0.15 or 0.3 exactly, so some score combinations come out as values like 3.0500000000000003. So the result is rounded to two places before anyone sees it.

## 2. What the model is told

The model gets a long system prompt. Four things in it matter most.

- **3 is competent, 5 is rare.** A call where nothing went wrong is a 3, not a 4. The prompt says that a model scoring every dimension 4 or 5 is "describing your relief that the call was fine rather than assessing it". That is an anti-inflation rule.
- **Anchors at 1, 3 and 5.** Each dimension says what a 1, a 3 and a 5 look like. Compliance is described as a floor: score 3 unless procedure was handled with unusual rigour.
- **Justifications are written to the agent.** "You reversed the charge before verifying identity", not "the agent failed to verify identity". Anything specific must cite a time as MM:SS, so a coach can act on it.
- **Short calls are not penalised.** A three-minute call that resolved the issue is better handling than a twelve-minute one that resolved the same issue.

What the prompt does not contain is the weights. The model never learns that Problem Resolution counts for 30%. It judges five things and does not know how they will be combined.

It also receives the call summary as context, after the transcript. [Part 2](/2026/08/10/callsense-six-slots-one-pipeline.html) covers why the summary stage comes first.

## 3. The model proposes; code decides

The model still has to fill in an `overall_score` field, because the output schema requires one. This is **structured output**: the model must return data in a fixed shape, which the code validates, instead of free text. So the model proposes a total. That is deliberate: the UI shows its figure beside the real one.

Then code ignores it. After the call returns, `run_qa_scoring` multiplies each dimension's score by its weight, sums them, rounds to two places, and overwrites the field. The model's figure is discarded, not averaged and not reconciled.

![On the left, the model's output in a unit test: five dimension scores of 1, 2, 3, 4 and 5 for professionalism, empathy, problem resolution, compliance and communication clarity, and below them the model's own overall score of 1.0, struck through and labelled discarded. An arrow leads to a green box where code multiplies each score by its weight: 1 times 0.15, plus 2 times 0.20, plus 3 times 0.30, plus 4 times 0.20, plus 5 times 0.15. That adds to 0.15, 0.40, 0.90, 0.80 and 0.75, and is rounded to two places. An arrow leads to the result on the right, 3.0. A strip along the bottom restates the demo: the model said 3.0, the code says 3.5, and the status is flagged for review because of one critical flag.](/images/callsense/callsense-recompute.svg)

Two unit tests hold this in place.

**The first checks the model is ignored.** The test makes the model say 3.0 while scoring every dimension 5. The result must be 5.0, and it must also differ from the model's figure. The second condition matters. A test that only checked for 5.0 would pass on code that trusted the model, any time the model happened to be right.

**The second checks each weight.** All fives can't catch a transposed weight, because any weights that sum to 1 turn five 5s into a 5. So the second test uses five different scores, 1 through 5. The answer must be exactly 3.0, which is what the figure above shows. Swap any two different weights and it changes.

In the demo, the scorecard has Professionalism at 4 and Compliance at 1. The model said 3.0. The code said 3.5. The UI shows each dimension's weight beside its score, so anyone can redo the sum.

**Verdict:** judgement goes to the model, arithmetic goes to code. Two runs over the same five scores always give the same total. A language model can't promise that.

## 4. Don't trust its ID either

The same schema requires a `call_id`, a unique identifier in UUID format, which the model has to make up. When I first ran the schema against three providers, OpenAI and Gemini both invented one.

OpenAI returned `12345678-abcd-ef01-2345-6789abcdef01`, an obvious placeholder. It is still a valid UUID, so validation passes. Gemini invented a plausible random one, which is worse, because it looks real.

So the code overwrites it with the call's real ID, in the same step that overwrites the score. A test checks that the result carries the transcript's ID and not the model's. Without it, a report could quietly be filed under an identifier that belongs to no call.

## 5. Structured output, checked across three providers first

Before writing the scorer I ran a spike: a short piece of throwaway code that answers one question. The question was whether the nested schema, with five dimensions and a list of flags, would work on OpenAI, Gemini and Groq.

*Would nesting break structured output?* It didn't. All three returned a valid result and all three caught the identity-verification violation I had planted in the test transcript.

The risk was real, though, just in a different place. It was per model, not per provider. Of the Groq models I tried, only `openai/gpt-oss-120b` worked. One other model failed with "Failed to call a function", and a third did not support tool calling at all. Structured output on Groq depends on tool calling. The default Groq model is now `openai/gpt-oss-120b`.

The same spike gave me the first latency numbers for one scoring call:

| Provider | Latency |
|---|---|
| Groq | 1.84 s |
| OpenAI | 5.32 s |
| Gemini | 6.86 s |

The pipeline makes two LLM calls, so those add up. About 4 s on Groq, about 14 s on Gemini.

Each LLM stage makes up to three attempts, the default for `MAX_RETRIES_PER_NODE`. It waits 1 second after the first failure and 2 seconds after the second. If all three fail, the stage raises an error. It does not return a made-up default.

## 6. Compliance flags: severity is not averaged

The five scores say how well the call was handled. Compliance flags say whether a rule was broken. They are separate on purpose.

The prompt says to raise a flag only for a genuine procedural violation: identity not verified before an account action, information disclosed to an unverified caller, a required disclosure omitted, or a commitment the agent had no authority to make. Style and warmth belong in the justifications. The prompt also says an empty list is a normal outcome, and that inventing a flag is worse than leaving it empty. Every flag raised will be reviewed by someone.

Each flag has a severity, and the prompt defines all four:

- **critical:** a customer or the business was materially exposed.
- **high:** a required control was skipped.
- **medium:** procedure was followed late or incompletely.
- **low:** the deviation was technical and caused no exposure.

Then comes the decision the pipeline exists for. `route_after_qa` checks whether any flag is critical. If so, the call goes to supervisor review. If not, it goes to the report.

![The QA result flows into a diamond that asks whether any compliance flag is critical. If yes, the call goes to supervisor review, with status FLAGGED_FOR_REVIEW and its analysis kept for the human. If no, it goes to the report with status COMPLETED. A note beside the boxes says that however well the call scored, one critical flag is enough. Along the bottom are the four severities the model is told about: critical means materially exposed, high means a required control skipped, medium means followed late or incompletely, and low means technical with no exposure.](/images/callsense/callsense-severity-routing.svg)

Severity is never turned into a number and mixed into the average. A call can score well overall and still go to a person.

That is what happened in the demo. A red flag at 00:25 read "account action taken without identity verification". The overall was 3.5. The status was `flagged_for_review`.

One more detail. On this path the UI still shows the full summary and scorecard, under a "Held for supervisor review" heading. The person needs the scorecard to see why the call was flagged. [Part 2](/2026/08/10/callsense-six-slots-one-pipeline.html) describes the bug that came from two different paths reaching supervisor review.

## 7. What isn't checked yet

Three gaps, stated plainly.

- **Nothing verifies the justifications.** The prompt says to ground every one in the transcript. No code checks that it did. This is the missing critic from Part 2.
- **There is no human-scored golden set.** A golden set is a small collection of calls whose scores people have agreed on, used to test changes against. Without one, I can't say how closely the model's judgement matches a human reviewer's. The score is stable, but whether it is *right* is unmeasured. [Safe, Structured, and Wrong](/2026/07/31/safe-structured-and-wrong.html) is about that gap: output can be well-formed and still wrong.
- **Output guardrails are skipped.** I considered a guardrails framework for the model's output and left it out on purpose. Both calls already use structured output, so types, score ranges and enums are validated at the boundary. The overall score is recomputed in code. PII can't leak through the output, because redaction happens before the model sees the transcript. What remains is content that is grounded in nothing. I treat that as a prompting and evaluation problem, not a validation one. See [Prompts Suggest, Guardrails Enforce](/2026/07/23/prompts-suggest-guardrails-enforce.html) for more on guarding inputs and outputs.

**Verdict:** the arithmetic is reproducible. Whether the judgement is right is still open.

## The one-liner that ties it together

Distilled: **let the model judge and let code count, then let one critical flag outrank every average.**

**Coming next:** Part 6, taking it from "works on my machine" to a clean clone, a container, and an attempt at a public host.
