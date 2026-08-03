---
title: "When Not to Fine-Tune"
description: "Part 2 of the fine-tuning series: I fine-tuned a capable 1.5B model on 112k real medical Q&A pairs and it got worse on every axis. Fine-tuning is the third thing you try — the ladder is prompt engineering, then RAG, then fine-tuning — because it changes form, not facts."
---

*Fine-Tuning in Practice &mdash; 1. [The Mechanics](/2026/07/28/how-lora-and-qlora-work.html) &middot; 2. When Not to Fine-Tune (you're here) &middot; 3. [The Dataset Audit](/2026/07/30/the-dataset-audit-that-predicted-the-eval.html) &middot; 4. [Safe, Structured, and Wrong](/2026/07/31/safe-structured-and-wrong.html) &middot; 5. [It Took All Day](/2026/08/01/it-took-all-day-to-fine-tune-a-small-model.html)*

**Objective:** [Part 1](/2026/07/28/how-lora-and-qlora-work.html) explained the *machine*. This post is about the decision to switch it on &mdash; organized around one question: *I had a capable model and a real dataset; should I have fine-tuned at all?* The honest answer, from my own runs, is usually **no** &mdash; and understanding *why* is the most useful thing fine-tuning taught me.

I fine-tuned a perfectly capable 1.5B model ([Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)) on **112,000 real medical Q&A pairs**. It got **worse** on every axis I measured — accuracy down, helpfulness down, even safety down. That failure is the spine of this post, because the reason it failed is a rule you can apply *before* you spend a GPU-hour.

## The one idea: fine-tuning is the third thing you try

There's a ladder, and the order is not negotiable:

![The fine-tuning ladder as three ascending steps. Step one, try first: prompt engineering — cheap, instant, fully reversible, and it got the base model about 80 percent of the way there. Step two, then: RAG — adds a retrieval system, and it is what installs facts. Step three, last resort: fine-tuning — a GPU job that is costly, hard to reverse, and locks you to open-weight models; it changes form, not facts. A rising arrow shows cost and irreversibility increasing from prompt engineering up to fine-tuning, with the note that each step fixes a different problem.](/images/finetuning/finetuning-the-ladder.svg)

**Prompt engineering → RAG → fine-tuning**, in that order, because each step costs more and is less reversible than the one below it — *and none of them fix the same problem.* That last clause is the part people skip. They reach for fine-tuning to fix something that lives two rungs down.

The evidence for rung one is embarrassingly cheap to gather. A good system prompt got the base model **~80% of the way** to where I wanted it. Before you spend anything, test whether prompting alone clears your bar — often it does, and you find out in minutes instead of a day (Part 5 is that day).

## Fine-tuning changes *form*, not *facts*

Here is the mental model that would have saved me the run. Fine-tuning shifts the model's *output distribution* — its tone, its format, its register, its default verbosity. It does **not** reliably install *knowledge*. It's a style transfer, not a fact transplant.

The proof from my own runs is almost too on-the-nose. There was a question with a wrong first-line asthma treatment in the base model's answer. No system prompt fixed it. And **neither did fine-tuning** — one fine-tune left the answer wrong, and the *other* made the model answer about a **different disease entirely** (that COPD answer is the whole of [Part 4](/2026/07/31/safe-structured-and-wrong.html)). Two fine-tunes, two ways of being wrong, zero facts installed. A factual gap is a *retrieval* problem, and no amount of gradient descent on the wrong tool closes it.

The corollary is the clean division of labor worth memorizing:

> **RAG for the facts, LoRA for the form.**

RAG installs *what is true* (house policies, current dosages, the document you must cite). LoRA installs *how it should sound* (register, structure, citation conventions). Cranking rank to 256 does **not** rescue knowledge injection — Part 1's honest caveat, now with a scar. If your problem is "the model doesn't know X," raising `r` is you turning the wrong knob harder.

## Why the 112k-row fine-tune actively hurt

This is the counterintuitive bit: it didn't just fail to help, it *regressed*. The base model wrote a complete, on-topic **1,228-character** answer. The fine-tuned model wrote **239 characters**, cut off mid-sentence, with the training set's persona leaking in:

![What "fine-tuning hurt" looks like: two length bars. The base model answer is 1,228 characters, a long complete bar. The fine-tuned v1 answer on the noisy dataset is 239 characters, less than a fifth as long, cut off mid-sentence, with the dataset persona ("Welcome to Chat Doctor") leaking in. The model changed the form — chopping the answer and importing the training set's habits — without adding any correctness.](/images/finetuning/finetuning-v1-collapse.svg)

The model faithfully learned the *form* of its training data: short, abrupt, persona-stamped answers. Since the data was noisy, "learning the form" meant importing its worst habits. It changed exactly what fine-tuning changes — and that was the problem. (The *why* behind "the data was noisy" is [Part 3](/2026/07/30/the-dataset-audit-that-predicted-the-eval.html): I could have predicted this from the dataset alone, before training.)

## The cost nobody puts on the slide

Even when fine-tuning *works*, there's a strategic bill that the compute cost hides: **fine-tuning locks you to open-weight models.** The moment you invest in a fine-tune of an open model, you've opted out of the frontier — and frontier models improve *monthly*. Your painstakingly fine-tuned 1.5B is competing against next quarter's base model that a prompt change would have gotten you for free. That's not a GPU-hour cost; it's an opportunity cost that compounds.

Weigh a fine-tune against three budgets, not one: **cost, latency, and accuracy** — and against the reversibility you're giving up. Prompt changes are free and a lot easier to undo. A fine-tune is a fork you maintain.

## The one-liner that ties it together

Distilled: **fine-tuning is a style transfer, not a fact transplant — so it's the third rung (prompt → RAG → fine-tune), and reaching for it to install knowledge is turning the wrong knob harder.** If prompting gets you 80% of the way, fine-tuning is buying the last 20% at 100× the cost and none of the reversibility.

**Coming next:** suppose you *do* decide to fine-tune. The entire outcome then rides on your dataset — and *not* on the size of it. In [Part 3](/2026/07/30/the-dataset-audit-that-predicted-the-eval.html) I spend an hour auditing two datasets *before* training anything, and those numbers forecast the post-training eval scores almost exactly.
