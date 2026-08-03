---
title: "The Dataset Audit That Predicted the Eval"
description: "Part 3 of the fine-tuning series: with model and hyperparameters held fixed, the dataset is the only thing that moves the outcome — and you can measure its quality upfront. A one-hour audit (3.2% vs 99.4% disclaimers) forecast which way the post-training safety scores would move."
---

*Fine-Tuning in Practice &mdash; 1. [The Mechanics](/2026/07/28/how-lora-and-qlora-work.html) &middot; 2. [When Not to Fine-Tune](/2026/07/29/when-not-to-fine-tune.html) &middot; 3. The Dataset Audit (you're here) &middot; 4. [Safe, Structured, and Wrong](/2026/07/31/safe-structured-and-wrong.html) &middot; 5. [It Took All Day](/2026/08/01/it-took-all-day-to-fine-tune-a-small-model.html)*

**Objective:** [Part 2](/2026/07/29/when-not-to-fine-tune.html) said *if* you fine-tune, the dataset decides everything. This post makes that measurable — organized around one question: *can I tell whether a dataset will help me **before** I train on it?* I spent an hour auditing two datasets before running a single training step, and those numbers forecast which way the eval scores would move.

## The one idea: the dataset is the only variable, and it's measurable upfront

I held everything else fixed — same 1.5B model, same LoRA config (`r=16`, `α=32`, all-linear, 4-bit NF4), same everything. **The dataset was the only thing I deliberately changed.** So whatever moved between runs, the data moved it. And the useful discovery is that you can measure the relevant property of a dataset *statically*, before training — data quality isn't a soft virtue you assess in hindsight, it's the load-bearing variable, and it has a number.

Two datasets:

- **ChatDoctor** (112k rows, a raw scrape): **63% persona contamination** — nearly two-thirds of answers opened with something like *"Welcome to Chat Doctor."* And just **3.2% carried a safety disclaimer.**
- **WikiDoc, reformatted by a stronger model** (2.1k rows): **0% persona**, **99.4% disclaimers.**

Those four numbers took an hour of grepping and sampling to produce. Here's what they bought.

## The audit predicted the eval

I ran both fine-tunes and scored them with an LLM-as-judge (base → fine-tuned deltas). The safety axis is the headline:

![The audit predicted the eval. Two datasets, each showing a before-training audit number predicting an after-training eval delta. ChatDoctor, 112k rows raw scrape: 63 percent persona contamination and only 3.2 percent of answers carrying a safety disclaimer, which predicted a post-training safety change of minus 0.04, a regression. WikiDoc, 2.1k rows reformatted by a stronger model: 0 percent persona and 99.4 percent disclaimers, which predicted a post-training safety change of plus 0.14. A static property of the data, measured before a single training step, forecast the model's behavior after — same model and LoRA config both times, the dataset the only variable.](/images/finetuning/finetuning-audit-predicts-eval.svg)

Read the loop: **3.2% disclaimers → safety *fell* (−0.04). 99.4% disclaimers → safety *rose* (+0.14).** A static property of the data, measured beforehand, forecast a behavioral property of the model afterward. This is the strongest single claim in the series — the closed loop from audit to eval. The model learns the distribution you show it; if 96.8% of your answers skip the disclaimer, the model learns that skipping it is normal.

The persona number tells the same story in a different register. **63% of ChatDoctor answers said "Welcome to Chat Doctor" — so the model learned to say that too.** It couldn't *not*: that phrase was the single most reliable pattern in the data.

## Quantity was never the constraint

Here's the reflex the audit kills: *"112k rows must beat 2k rows."* It didn't, and it was never going to. Filtering ChatDoctor for contamination still left **~24k clean rows** — and I only trained on **2,000**. The winning run used a *tiny* clean dataset. **1k clean beats 100k noisy**, and it isn't close. Once you've seen the audit→eval loop, "collect more data" is revealed as the wrong instinct when the data you have is dirty; you want *cleaner*, not *more*.

## The nuance that makes "clean good, noisy bad" too coarse

I want to be honest about what the data quality did and didn't control, because the clean summary is wrong in an instructive way. Look at all three axes, not just safety:

![The full delta table, base to fine-tuned, on three axes. The noisy ChatDoctor fine-tune: accuracy minus 0.10, helpfulness minus 0.18, safety minus 0.04 — a regression on all three. The clean WikiDoc fine-tune: accuracy plus 0.06, helpfulness minus 0.14, safety plus 0.14. The shared column is helpfulness: both fine-tunes lost it, because both trained on shorter examples — an artifact independent of data quality. Data quality controlled the accuracy and safety direction, not the length.](/images/finetuning/finetuning-delta-table.svg)

**Both** fine-tunes lost helpfulness (−0.18 and −0.14). Both made answers shorter, because both trained on short examples — and shorter answers score lower on helpfulness regardless of how clean they are. That's an artifact of *example length*, not of *quality*. So the precise statement is:

> Data quality controlled the **accuracy and safety** direction. It did not control the **length**, and length dragged helpfulness down in both arms.

"Clean good, noisy bad" is too coarse. The true shape: **clean data buys you accuracy and safety, and still costs you completeness** if your examples are terse. Two independent levers — quality and length — and the audit only forecasts the one it measures. (If I re-ran this, I'd audit *answer length distribution* too, and reformat toward longer clean examples.)

## The one-liner that ties it together

Distilled: **with the model and config fixed, the dataset is the only variable that moves the outcome — and its quality is a number you can read before training, not a verdict you get after.** You can measure whether a dataset will help you before you train on it.

**Coming next:** so I trained on clean data and the safety score went *up*. Am I sure the model is actually *safe*? In [Part 4](/2026/07/31/safe-structured-and-wrong.html), the fine-tuned model is asked about asthma, answers about COPD, and scores **0.8 on safety** — a well-mannered wrong answer that a safety gate would have shipped.
