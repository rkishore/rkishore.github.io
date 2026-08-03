---
title: "The 20-Minute Fine-Tune That Took All Day"
description: "Part 5 and the close of the fine-tuning series: 'QLoRA fine-tunes a model on a free GPU in 20 minutes' is true in principle and took me a full day. OOM on a T4, a lost run at 157/250 steps, a P100 that breaks 4-bit, and two GPUs that made it fail instead of faster."
---

*Fine-Tuning in Practice &mdash; 1. [The Mechanics](/2026/07/28/how-lora-and-qlora-work.html) &middot; 2. [When Not to Fine-Tune](/2026/07/29/when-not-to-fine-tune.html) &middot; 3. [The Dataset Audit](/2026/07/30/the-dataset-audit-that-predicted-the-eval.html) &middot; 4. [Safe, Structured, and Wrong](/2026/07/31/safe-structured-and-wrong.html) &middot; 5. It Took All Day (you're here)*

**Objective:** The close of the series — organized around one question: *the tutorials say "QLoRA fine-tunes a model on a free GPU in 20 minutes," so why did it take me a full day?* The gap between those two sentences is the whole post. Knowing the failure modes in advance turns out to be worth more than knowing the config.

"QLoRA fine-tunes a model on a free GPU in 20 minutes." True — *in principle*. It took me a full day, and none of the day was the 20 minutes. It was the infrastructure gauntlet the screenshots leave out.

## The one idea: the tutorials show you the happy path

Nearly every fine-tuning tutorial is a screenshot of the one run that worked. The real cost of "accessible" fine-tuning is everything *around* that screenshot — a sequence of failures each of which is obvious in hindsight and invisible in advance. Here is the actual gauntlet, in order, with what each fix cost:

![The infrastructure gauntlet as five rows, each a trap then its fix then the trade it cost. One: OOM on a T4 at batch-2 and 512 tokens, fixed by gradient_checkpointing=True, trading memory for about 2 times slower steps. Two: Colab free disconnects long sessions, losing a run at step 157 of 250, with no clean fix. Three: a P100 breaks bitsandbytes 4-bit because NF4 kernels need compute capability 7.5 or higher and the P100 is 6.0, fixed by switching to the T4 — the better-sounding card doesn't work. Four: two T4s make it fail not faster, because the trainer auto-wraps in DataParallel and crashes, fixed by hiding the second GPU with CUDA_VISIBLE_DEVICES set to 0. Five: what finally worked was Kaggle on a single T4, about 11 seconds per step, about 45 minutes, stable. Every fix was a trade.](/images/finetuning/finetuning-infra-gauntlet.svg)

Four of those deserve the story, because each one teaches something the config file can't.

## OOM on a T4 — memory for time

First wall: out-of-memory on a T4 at batch-2, 512-token sequences. The fix is one line — `gradient_checkpointing=True` — and it works by *recomputing* activations on the backward pass instead of storing them. But recomputation isn't free: it runs the forward math twice, so steps get **~2× slower.** That's the first trade of the day, and it's the mechanism [Part 1](/2026/07/28/how-lora-and-qlora-work.html) flagged — *activations still must be stored under LoRA*, so memory still scales with batch and sequence length. The "0.4% of parameters are trainable" headline does nothing for your activation memory.

## The P100 that "should" have worked

I switched GPUs to dodge a different bug and hit `named symbol not found in ops.cu`. The cause is a hardware floor most tutorials never mention: **4-bit NF4 kernels need compute capability ≥ 7.5 (Turing).** The T4 is 7.5 and works. The P100 is 6.0 (Pascal) — an older, in some ways *beefier* card — and it simply can't run the kernels. **The "better" card doesn't work; the humble T4 does.** "Runs on a GPU" is not a property of your model; it's a property of the exact silicon you were handed.

## Two GPUs made it fail, not faster

The one I'd never have predicted. Given 2× T4, the trainer helpfully auto-wraps the model in `DataParallel` — and promptly crashes with `chunk expects at least a 1-dimensional tensor`. A notebook won't do real multi-GPU training out of the box anyway, so the second card is pure liability. The fix is to *hide* it: `CUDA_VISIBLE_DEVICES=0`. **The second GPU didn't speed anything up. It just gave the job a new way to crash.**

## And Colab quietly disconnects

Somewhere in the middle, Colab's free tier disconnected a long session and I **lost a run at 157/250 steps.** Free-tier idle timeouts don't care that you're mid-train. There's no clever fix here, only a lesson: checkpoint often, and don't run anything on free Colab you're not prepared to lose.

## What finally worked

Kaggle, a single T4, `CUDA_VISIBLE_DEVICES=0`, gradient checkpointing on: **~11 s/step, ~45 minutes, stable.** And the loss did exactly its job and nothing more — train loss fell **2.27 → 2.04** while validation stayed **~2.16.** That gap is the whole series in one number: the model was **learning the *style*** of the training data (train loss dropping) **without getting more correct** (val loss flat). The [eval in Part 4](/2026/07/31/safe-structured-and-wrong.html) later confirmed exactly that — form moved, facts didn't.

## The one-liner that ties it together

Distilled: **"runs on a free T4" is true, and it is not the same sentence as "runs on the free T4 you were assigned today."** Every fix was a trade — memory for speed, speed for stability, a faster card for a compatible one — and knowing those trades in advance is worth more than knowing the config.

## Closing the series

Five posts, one small model, one thesis: **fine-tuning changes form, not facts — so it's the last rung of the ladder, its outcome is set by the dataset you can audit in advance, its safety score lies, and getting it to run at all is most of the work.**

The tie-back to [Part 1](/2026/07/28/how-lora-and-qlora-work.html) is literal. The adapter this day produced is **74 MB** — because, as the mechanics post showed, you only ever trained `B·A`, not `W`. Deploying it is the merge-first pattern from Part 1 and a cost-vs-toggle decision: bake it into a BF16 checkpoint for zero inference latency, or keep it as a hot-swappable 74 MB file beside a shared base. That 74 MB is the entire point of LoRA made concrete — a full day of gauntlet, and what you carry out is small enough to drop in a chat.

The honest summary of the week: I spent a day fine-tuning a model that got *worse*, learned to predict that from the dataset in an hour I didn't spend first, caught the dangerous answer only because I refused to trust one number, and shipped 74 MB. Every one of those is a lesson I'd rather have than the 20-minute screenshot. Thanks for reading the series.
