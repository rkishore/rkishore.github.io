---
title: "Safe, Structured, and Wrong"
description: "Part 4 of the fine-tuning series: my fine-tuned medical model was asked about asthma and answered about COPD — 0.2 accuracy, 0.2 helpfulness, and 0.8 safety. A safety gate would have shipped it. One number, especially a safety number, will lie to you."
---

*Fine-Tuning in Practice &mdash; 1. [The Mechanics](/2026/07/28/how-lora-and-qlora-work.html) &middot; 2. [When Not to Fine-Tune](/2026/07/29/when-not-to-fine-tune.html) &middot; 3. [The Dataset Audit](/2026/07/30/the-dataset-audit-that-predicted-the-eval.html) &middot; 4. Safe, Structured, and Wrong (you're here) &middot; 5. [It Took All Day](/2026/08/01/it-took-all-day-to-fine-tune-a-small-model.html)*

**Objective:** [Part 3](/2026/07/30/the-dataset-audit-that-predicted-the-eval.html) got the safety score to go *up*. This post asks whether that score can be trusted — organized around one question: *if a single evaluation number says "safe," is it?* The answer, from one particular row of my eval, is a flat no.

My fine-tuned medical model was asked about **asthma** and answered about **COPD** — the wrong disease. It scored **0.2 on accuracy, 0.2 on helpfulness — and 0.8 on safety.** A safety gate would have shipped it. That single row is the whole argument.

## The one idea: a single number, especially a safety number, lies

I scored every answer with an LLM-as-judge ([GPT-4o-mini](https://platform.openai.com/docs/models), temperature 0) across **three independent evaluators — helpfulness, accuracy, safety** — deliberately using a different model family from the one under test (OpenAI grading Qwen) to dodge self-evaluation bias. Three axes, not one, and here is why one would have failed me:

![Safe, structured, and wrong: one eval row. The model was asked about asthma and answered about COPD, the wrong disease — well-formatted, politely hedged, with a disclaimer. Three independent axes: accuracy scores 0.2 and helpfulness scores 0.2, both correctly cratering because the answer is about the wrong disease; but safety scores 0.8 because the answer hedged and disclaimed politely. A well-mannered wrong answer passes a safety check — a safety gate alone would have shipped it. Two of the three evaluators caught it — accuracy and helpfulness both scored 0.2 — while the safety score, the one a gate would trust, did not.](/images/finetuning/finetuning-safe-structured-wrong.svg)

The COPD answer is the argument in one row. Because it was about the wrong disease, accuracy and helpfulness correctly crater to **0.2**. But safety stays at **0.8** — because the answer *hedged, disclaimed, and was politely formatted.* Everything a safety rubric rewards, it did. It just did it about the wrong illness.

**A well-mannered wrong answer passes a safety check.** The failure that hurts someone doesn't look unsafe; it looks *polished*. Which is exactly the failure mode fine-tuning is especially prone to produce, because — per [Part 2](/2026/07/29/when-not-to-fine-tune.html) — fine-tuning installs *form*, and a safety score is heavily a measure of form.

So: **measure safety, but never *only* it.** Correctness has to be its own axis, or a confidently-wrong-but-polite model slips straight through. Safety scores give false comfort precisely when you most need real information.

## Averages hide the miss that matters

There's a second way the numbers lie, and it's about aggregation. That model's *aggregate* accuracy was **0.72** — a perfectly respectable-looking mean. The 0.2 on a treatment question was invisible inside it:

![Averages hide the miss that matters. Per-example accuracy scores plotted as dots from 0 to 1. Most cluster high between 0.7 and 0.9. One sits far out at 0.2 — a wrong first-line treatment answer. A dashed line marks the mean at 0.72, sitting comfortably inside the healthy cluster, giving no hint of the outlier. The mean says "usually fine"; the single 0.2 is the shape of a system that hurts someone occasionally. For anything safety-critical, inspect the tail per example, don't trust the average.](/images/finetuning/finetuning-averages-hide-tail.svg)

A mean of **0.72 with a 0.2 hiding inside it** is exactly the shape of a system that hurts someone *occasionally*. In most software, the average case is the product. In safety-critical software, **the tail is the product** — the average is a comfortable fiction that averages a good answer and a dangerous one into a B-minus. You cannot find that 0.2 by reading the mean. A box plot or five-number summary would at least flag that it *exists* — the right first move, and more honest than a `mean ± SD` band, whose lower edge can still sit above a lone catastrophic outlier. But no summary statistic can tell you that the 0.2 means *wrong disease*; for that, you read the *row*.

So the method has two non-negotiable halves: **multiple independent axes** (so correctness can dissent from safety) *and* **per-example inspection** (so the tail can dissent from the mean). Drop either and the COPD answer ships.

## A corollary against a bad habit

One habit worth naming because it's seductive: **helpfulness ≠ readability.** A longer, better-formatted, nicely-bulleted answer is *not* more helpful if it's about the wrong thing. A decent LLM judge catches that — it reads the content. A length or formatting heuristic does not; it would have *rewarded* the COPD answer for being well-structured. If your "helpfulness" metric is secretly a readability metric, you've built a machine that prefers confident wrong answers.

## The one-liner that ties it together

Distilled: **a single evaluation number — especially a safety number — will lie to you, because the failure that hurts someone hides inside a healthy average and behind polite formatting.** You need independent axes so correctness can contradict safety, and per-example inspection so the tail can contradict the mean.

It was safe, structured, and wrong — and the one evaluator a safety gate would trust was the only one it fooled.

**Coming next:** all of this assumed the training even *ran*. Getting a 1.5B QLoRA job to complete on free hardware was the single hardest part of the week — an infrastructure gauntlet nobody screenshots. [Part 5](/2026/08/01/it-took-all-day-to-fine-tune-a-small-model.html) is that day.
