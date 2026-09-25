---
title: "Lessons from building a .NET Embedding Server"
description: "A retrospective on Quilha, a .NET ONNX embedding server built to test whether a .NET-native model call beats a Python sidecar. Called in-process, it did: 1.5–1.7× the throughput on short text, level on long, and 1.4–6.2× less memory per cell. Dynamic batching lost 61% on mixed CPU traffic, while dropping fixed padding was worth about 7×."
date: 2026-09-24 21:00:00 -0400
---

**Objective:** A retrospective on Quilha, a .NET server that turns text into embeddings: what I set out to measure, and what the measurements taught. Quilha was a learning experiment, and it's now concluded. This is the post I promised in [the parity post](/2026/09/23/the-divergence-before-the-tensor.html), which covered the correctness side of the same work; that post's probes and results are public at [rkishore/dotnet-embedding-parity](https://github.com/rkishore/dotnet-embedding-parity).

Quilha started from one hypothesis: **a .NET-native model call, made either in-process or through a .NET sidecar, is cheaper to run than the usual Python sidecar.**

## Motivation

Microsoft's .NET AI tooling had matured quickly: [`Microsoft.Extensions.AI`](https://learn.microsoft.com/en-us/dotnet/ai/microsoft-extensions-ai) (MEAI) gave a common abstraction over providers, [Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/what-is-foundry-local) ran ONNX models on-device, and [Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/overview/) handled orchestration. The serving layer in the middle looked missing. Foundry Local is built for one user on one machine, and [its documentation](https://learn.microsoft.com/en-us/azure/foundry-local/what-is-foundry-local#can-foundry-local-run-on-a-server) points multi-user serving to a dedicated server framework. MEAI gives you interfaces, not an engine.

I built an **embedding** server on **ONNX** models for practical reasons. Embedding models are small (the one here is about 90 MB), so they load in seconds, run on a CPU, and let me change one thing and re-measure the same afternoon. ONNX Runtime ships the same core in its .NET and Python packages, so comparing the two isolates the serving layer from the kernels underneath. And the questions a small model raises, about batching, padding, concurrency and where a serving framework helps at all, are the same ones you'd ask of bigger models, even where the answers differ.

**The hypothesis held, most clearly in-process.** Measured side by side from the same caller, in-process .NET served short text **1.5–1.7×** faster than the better Python sidecar and matched it on long text. Per cell, it used **1.4–2.0×** less memory than a single-worker Python sidecar and **4.5–6.2×** less than a four-worker one, and under load its tail latency was up to **2.6×** better. A .NET sidecar kept part of that lead: **1.24–1.41×** on short text over HTTP, and well under half the memory of four Python workers, though about the same as one. Two results pointed the other way. **Dynamic batching**, a key feature I'd hoped to benefit from, was **61% slower** than no batching on realistic mixed traffic, and the biggest win of the project, about **7×**, came from something simpler: dropping fixed padding.

I concluded Quilha at this point, because the advantages belong to .NET, not to Quilha. Libraries such as **[Microsoft's Semantic Kernel ONNX connector](https://github.com/microsoft/semantic-kernel)** already make in-process model calls, and the separate-server case is taken by **Hugging Face's [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)** (TEI). 

## Background: four ideas you need first

**An embedding** is a list of numbers (here, 384 of them) that stands for the meaning of a piece of text, so that similar texts get similar lists. A semantic search system embeds its documents once, embeds each query as it arrives, and returns the documents whose embeddings are closest. Retrieval-augmented generation (RAG) apps commonly use this step to find the text an LLM answers from.

**The model runs in ONNX Runtime.** [ONNX](https://onnx.ai/) is a file format for trained models, and [ONNX Runtime](https://onnxruntime.ai/) is Microsoft's engine for running them. Running the model once is an **inference**. The model is [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (L6), a small, widely used English embedding model. Its sibling all-MiniLM-L12-v2 (L12) has twice the layers and serves below as the bigger-model check.

**Inputs are measured in tokens, and a batch has to be a rectangular grid of numbers (a tensor).** The model reads **tokens**, word pieces from a fixed vocabulary. A search query might be 6–10 tokens and a document passage 100–400. To run several texts in one inference, they go in together as one **tensor**, with one row per text. Every row must be the same length, so shorter texts are filled out with **padding** to match the longest. The model computes over the padding like any other token, and the result is thrown away.

![Three texts put into one batch. The query "cheap flights to paris" is 4 tokens, the query "is it raining" is 3, and the passage "the museum is open every day except monday and closes at six" is 12. The model expects one rectangular grid of numbers, a tensor, so the longest text sets the width at 12 columns and the two queries are filled out with padding. Of the 36 cells, 19 are real tokens and 17, or 47%, are padding; the model computes over all of them and throws the padding results away. Simplified: roughly one token per word, with the special start and end tokens left out.](/images/quilha-retrospective/batch-is-a-rectangle.svg)

**One inference can already use every CPU core.** ONNX Runtime splits the matrix multiplications inside a single inference across all its cores ([intra-op parallelism](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)). Whether one inference has *enough* work to keep them all busy is something only measurement can answer. If it does, there's no idle capacity for batching to fill; if it doesn't, there is. That answer turns out to decide nearly everything below.

![Two cases, each with four CPU cores. Left, if one inference leaves cores idle: one request on its own keeps one core busy and three idle, while four requests batched into one inference keep all four busy, so batching wins. Right, if one inference already fills the cores: one request keeps all four busy, and a batch of four is four times the work on the same four cores, so there is nothing to fill and batching gains nothing.](/images/quilha-retrospective/why-batching-should-help.svg)

Three load-testing terms:

1. **Concurrency** is how many requests are in flight at once. At 100 concurrent requests, the load tester runs 100 simulated users, each sending a request, waiting for the answer, and sending the next straight away.
2. **req/s** is requests completed per second, the **throughput**. By [Little's law](https://en.wikipedia.org/wiki/Little%27s_law) it is concurrency divided by the time per request, so 100 requests in flight that each take 0.1 s complete about 1,000 per second.
3. **p95** is the **95th-percentile latency**: 95% of requests finished at least this fast. It describes the slow tail a user occasionally feels.

## Experimental Methodology

Every number here comes from one machine: an **Azure `Standard_D4ads_v7` VM, 4 vCPUs, x86-64, 16 GB, Ubuntu 24.04**, CPU only. Text came from three corpora built from [MS MARCO](https://microsoft.github.io/msmarco/), a standard search dataset:

| corpus | content | typical length |
|---|---|---|
| `query` | search queries | ~6–10 tokens |
| `passage` | document passages | ~100–400 tokens |
| `mixed` | an even shuffle of both | either |

The load generator ran on the same machine and used about 2.5–3% of it. There were four experiments, each run in its own session, so absolute numbers can differ slightly between them; ratios are only taken within one experiment.

| experiment | what it compares | model | runs per cell |
|---|---|---|---|
| **In-process vs. sidecar** | one .NET caller calling the model directly vs. calling .NET and Python sidecars over HTTP, 1–64 concurrent calls | L6 in full, L12 on three cells | 5 × 30 s |
| **.NET vs. Python over HTTP** | Quilha vs. two Python servers, driven by [k6](https://k6.io/) at 1–100 concurrent requests | L6 | 5 × 30 s |
| **Batching over HTTP** | Quilha with batching on vs. off, driven by k6 at 1–100 concurrent requests | L6 | 5 × 30 s |
| **Batching at the engine** | one inference over a batch of N vs. N separate inferences at once, no server involved | L6 and L12 | 3–4 × 30 s |

Four practices ran through all of it:

- **Predictions were written down before each run**, and the ones that failed were kept and scored.
- **Arms were interleaved A-B-B-A**, so slow drift in the machine can't masquerade as a difference. Ordering alone had shifted an earlier comparison by 0.6–10.2%.
- **An output check came before any timing.** The same texts went to every server, and their embeddings had to agree to a cosine similarity of at least 0.9999.
- **A drift check** compared the start of each sweep with the end. My first version fired by chance about one sweep in seven; I recorded that failure and redesigned it.

**The harness and raw data are in a private repository**, so each table states its own method, and I've left out any number whose method I couldn't state in a sentence.

## Results

### Lesson 1: in-process .NET beat a Python sidecar on short text and on memory

The choice a .NET team weighs is where the model runs: **in-process**, inside the app with no network, or in a **[sidecar](https://learn.microsoft.com/en-us/azure/architecture/patterns/sidecar)**, a separate server on the same machine that the app calls over HTTP. So one .NET caller program ran on the machine and changed only what each call did:

- **`inproc`:** call the model directly, through .NET's standard [`IEmbeddingGenerator`](https://learn.microsoft.com/en-us/dotnet/ai/microsoft-extensions-ai) interface.
- **`quilha-http`:** call Quilha over HTTP, the same .NET engine behind a network hop.
- **Two Python sidecars**, because a Python team faces a real choice. `python-baseline` is one [uvicorn](https://www.uvicorn.org/) worker with ONNX Runtime free to use all four cores. `python-prod` is four workers with one core's worth of ONNX Runtime threads each, the usual way around Python's [global interpreter lock](https://docs.python.org/3/glossary.html#term-global-interpreter-lock).

Everything, the caller included, shared the same four cores, because that's what a sidecar means. L6 on all three corpora at 1, 4, 16 and 64 concurrent calls, five 30-second runs per cell, A-B-B-A order, zero errors; then L12 on three of those cells. Each cell compares against **whichever Python sidecar was faster there**: the single-worker one at one call at a time, the four-worker one above that. That was the conservative choice, fixed before the run.

**Throughput.** In-process .NET against the better Python sidecar, L6:

| corpus | 1 concurrent | 4 | 16 | 64 |
|---|--:|--:|--:|--:|
| `query`: short | 1.52× | **1.72×** | 1.63× | 1.54× |
| `mixed` | 0.99× | 1.06× | 1.08× | 1.07× |
| `passage`: long | 0.97× | 1.02× | 1.03× | 1.02× |

On short text, in-process was **1.52–1.72×** faster. On long text it was level: within 3% in every cell, slightly behind at one call at a time. Mixed traffic sat in between.

**Memory.** Peak memory for the whole deployment under load, meaning the caller plus every server process it needs, on L6:

| corpus, concurrency | `inproc` | `quilha-http` | `python-prod` (4 workers) |
|---|--:|--:|--:|
| `query`, 1 | **204 MiB** | 305 MiB | 1,019 MiB |
| `query`, 64 | **232 MiB** | 336 MiB | 1,096 MiB |
| `mixed`, 64 | **250 MiB** | 397 MiB | 1,393 MiB |
| `passage`, 64 | **251 MiB** | 392 MiB | 1,314 MiB |

The single-worker Python sidecar peaked at 301–456 MiB across all cells. Compared per cell, in-process used **1.4–2.0× less** memory than single-worker Python and **4.5–6.2× less** than four workers across all cells; the four cells in the table give 4.7–5.6×. These are summed resident memory ([RSS](https://en.wikipedia.org/wiki/Resident_set_size)) sampled throughout each run, which overcounts the four Python workers because they share some pages, but their total stays well over 800 MiB in every row. The gap is structural: **each Python worker loads its own copy of the model**, while one .NET process shares one copy. On L12 the gap grew, as four copies of a bigger model predict. Quilha's own sidecar, at 305–397 MiB, was about level with single-worker Python; its memory edge is over the four-worker deployment.

**Tail latency.** At 16 and 64 concurrent calls, **in-process had the lowest p95 of any arm in every cell, on every corpus.** On short queries at 64 concurrent calls it was 63.7 ms against 163.9 ms for the better Python sidecar, **2.6× better**; on long passages, 419 against 500 ms. At one call at a time it can lose: on mixed traffic, 13.1 ms against Python's 10.2.

**On L12 the lead shrinks.** On short queries in-process was 1.14× faster than the better Python sidecar at one call at a time and 1.16× at 64; on passages, 0.99×. The VM ran about 15% slower in that session, probably after being moved to different hardware, so the bigger model can't be cleanly separated from the machine change, though ratios within the session are unaffected. The tail advantage under load held on L12 too.

The trade-off: in-process, **the model's memory counts against the app's own limit, and the model can't be scaled separately from the app.** If you need that, you want a sidecar.

### Lesson 2: the gain is part network hop, part .NET's lower overhead

The in-process lead has two parts, and each was measured.

**The network hop: 1.12–1.31×.** In the same experiment, in-process was 1.12–1.31× faster than Quilha's own sidecar on short text, 1.31× at 4 concurrent calls falling to 1.12× at 64, with no difference on mixed or long text, and its p95 was 15–29% lower under load. I'd predicted at least 1.5×; same-machine HTTP cost less than I assumed. On L12 the hop cost nothing visible: in-process and sidecar were within 1% on median latency at one call at a time, and at 64 concurrent short calls, throughput was roughly the same. Because both ratios come from one experiment, they divide cleanly: the rest of the L6 lead, about 1.3–1.4× at 4 and 64 concurrent calls, is Quilha's sidecar outrunning Python's.

**The framework: a .NET sidecar against Python.** A separate experiment compared the servers directly, all over HTTP, driven by k6 rather than the .NET caller, with batching off (lesson 3 explains why). L6, five 30-second runs per cell, zero errors, run-to-run variation under 4.3%. Quilha against the faster Python arm in each cell:

| corpus | 1 concurrent | 10 | 50 | 100 |
|---|--:|--:|--:|--:|
| `query`: short | 1.24× | 1.34× | 1.32× | **1.41×** |
| `mixed` | 1.06× | 1.01× | 1.02× | 1.01× |
| `passage`: long | 1.00× | 1.01× | 1.02× | 1.03× |

Quilha was never slower. On short inputs, at 100 concurrent requests, it held 925 req/s against 656 and 601. On long inputs, all three servers pin at **150–156 req/s from 10 concurrent requests onward**: ONNX Runtime saturates the CPU, and the calling language stops mattering. **When inference is cheap, the framework's overhead is the bottleneck, and .NET's was smaller.** That 1.24–1.41× is close to the 1.3–1.4× worked out from the in-process experiment above, so the two experiments tell the same story. They used different load generators and concurrency levels, though, so their ratios shouldn't be multiplied together.

One cost: **where throughput ties, Quilha's sidecar had the worse tail.** At 100 concurrent requests its p95 was 886 ms on `passage` against single-worker Python's 753 ms (18% worse), and 547 against 487 ms on `mixed` (12% worse). Throughput tied, so average latency did too (Little's law again); the worse p95 means Quilha's latencies were more spread out, most likely because it ran every in-flight request at once on four cores while Python's thread pool queued them, though I didn't test that. On `query`, where it won throughput, it also won the tail, 142 ms against 194.

These numbers come after fixing five flaws in the comparison, four of which had flattered Quilha; a sixth, found later, is too small to change them. All six are in [the appendix](#appendix-six-flaws-in-the-net-vs-python-comparison).

### Lesson 3: dynamic batching doesn't pay on CPU

A key aspect I was exploring, and hoping to benefit from, was **dynamic batching**: collect requests as they arrive, run them through the model together, and hand each caller its answer. I built it with a queue, a configurable batch size and wait window, and graceful shutdown.

Quilha with batching on (dispatching immediately with no wait window, the most favourable setting) against batching off, over HTTP. L6, five 30-second runs per cell, zero errors, shown at 100 concurrent requests:

| traffic | batching off | batching on | change |
|---|--:|--:|--:|
| `query`: short | 1006 req/s | 1053 req/s | **+4.6%** |
| `passage`: long, 100–400 tokens | 163 req/s | 114 req/s | **−30%** |
| `mixed`: short and long | 278 req/s | 110 req/s | **−61%** |

On long and mixed traffic, batching made the server slower. Two costs explain why.

**Cost 1: every request in a batch pays for the longest one.** With batching off, each request runs alone at its own length. With batching on, a 7-token query that shares a batch with a 300-token passage is padded to 300 and costs as much as the passage. The mixed row shows it: with batching off, mixed traffic beats pure passages (278 req/s against 163), because half of it is cheap queries. With batching on it drops to 110, level with passages at 114. Pure passages pay too, since a 120-token passage batched with a 300-token one is padded to 300.

![Three panels. Top: with batching off, each request runs alone at its own length: a 7-token query, a 120-token passage and a 300-token passage. Middle: with batching on, all three share one tensor padded to the longest member, so the query gets 293 tokens of padding and the shorter passage 180, and both cost a 300-token pass; because one inference already used all four cores, nothing is won back. Bottom: measured at 100 concurrent requests on all-MiniLM-L6-v2, five runs per bar on a 4-vCPU Azure VM: mixed traffic 278 req/s with batching off and 110 with it on, a 61% loss; pure passages 163 off and 114 on, a 30% loss. Batched mixed traffic runs like all passages.](/images/quilha-retrospective/batch-costs-longest-member.svg)

**Cost 2: with no idle cores, a batch runs worse than separate inferences.** To rule out bad batching code, the next test skipped the server and asked ONNX Runtime directly: is one inference over a batch of N cheaper per item than N separate inferences at once? The input was one 11-token sentence padded to 128 tokens, as Quilha worked at the time. A-B-B-A, four 30-second repetitions per cell on L6 and three on L12. Cost per item in milliseconds, batched arm:

| batch size N | 1 | 2 | 4 | 8 | 16 | 32 |
|---|--:|--:|--:|--:|--:|--:|
| **L6** (6 layers) | 8.71 | 8.07 | 8.11 | 7.18 | 8.32 | 8.00 |
| **L12** (12 layers) | 16.50 | 15.50 | 13.98 | 15.35 | 15.93 | 15.11 |

**No trend.** Per-item cost moves around by up to 15% but doesn't fall as the batch grows 32 times bigger, at either model size. This is ONNX Runtime on a CPU, not the server code. Batching exists to fill idle hardware: a small inference leaves most of a GPU unused, but on a CPU, ONNX Runtime already spreads one 128-token inference across all four cores, leaving nothing for a batch to fill.

It's slightly worse than that. On L6, separate inferences side by side topped out at about 148 items/s, against about 125 for batched: several independent inferences share busy cores better than one large one. So long traffic pays cost 1 and a little of cost 2; I didn't measure the split.

Uniform short traffic gained a little, **+4.6% to +13.7%** on `query` across concurrency levels. I turned batching off by default. **The GPU case is untested.**

### Lesson 4: dropping fixed padding was worth about 7×

That test sentence was 11 real tokens padded to 128, so **91% of every inference was arithmetic on padding**, and Quilha padded everything to 128 at the time, batched or not. I switched to padding each request only to its own length and re-ran the engine test, three repetitions per cell:

- **L6:** single-inference cost fell from 8.61 ms to 1.33 ms, **6.5–7.7× faster** across batch sizes.
- **L12:** from 17.42 ms to 2.82 ms, **6.2–6.8× faster**.

That was the biggest improvement in the project by a wide margin.

It also reversed the batching result, which is what makes the mechanism convincing. At 11 tokens, one inference is too small to keep four cores busy, so a batch has idle capacity to fill: at a batch of 16, batching was **1.47× faster** than separate inferences on L6 and **1.58×** on L12. **The parallelism has to come from somewhere.** Long inputs supply it themselves; short inputs need batching to supply it.

That doesn't rescue batching in the server, for two reasons:

- **The engine test used one sentence, repeated**, so nothing needed padding. Real traffic mixes lengths, and every short request gets padded back up into the regime where batching adds nothing, as the `mixed` row showed.
- **The wait got relatively worse.** Quilha's default wait window was 20 ms. Against a 1.33 ms inference, that's **15 times the work it's waiting to speed up**.

### Lesson 5: the advantages belong to the category, not the product

**Every in-process advantage comes with any .NET library that runs an ONNX model inside the app**: no second process, no network hop, one copy of the weights. Semantic Kernel's connector gets them too; I didn't benchmark it, but nothing in the mechanism is specific to Quilha. On the server side, TEI serves many model families and ships GPU images, while Quilha had one verified model, and "it's written in .NET" isn't a reason to pick a server you only call over HTTP.

So the gap I'd found was real for .NET as a platform, and already closed for a new product. **An empty niche is evidence too**: here, that what .NET teams need is a library, not another server. What Quilha produced was the measurements in this post and the parity work.

### What this doesn't show

- **One machine shape**: four vCPUs, x86-64, CPU only. Saturation may behave differently on 32 cores.
- **One full model sweep**: L6. L12 appears only on three in-process cells and in the engine-level batching test.
- **No GPU**, where batching should win because a small inference really does leave the chip idle.
- **No Triton or vLLM.** [vLLM](https://github.com/vllm-project/vllm) is built for large generative models. [Triton Inference Server](https://github.com/triton-inference-server/server) is the strong baseline I didn't run: its ONNX backend takes tensors, not text, so it would need a Python tokenizer in front, bringing back the dependency under test.
- **Python's own batching wasn't re-measured** after its padding was fixed. The mechanism predicts it hurts there too, but that's a prediction.

## Conclusion

**A .NET-native model call is cheaper than a Python sidecar.** In-process, it served short text 1.5–1.7× faster and used 1.4–6.2× less memory per cell, depending on how Python is deployed; as a sidecar, it kept a 1.24–1.41× short-text lead and well under half the memory of multi-worker Python. Beyond that, **how much work the model does per request decides everything**: whether the framework matters (only while inference is cheap), whether in-process is faster (clearly on short text, not on long), and whether batching helps (only when inputs are short and alike). Measure what the model is doing before building anything around it.

The correctness half of this work, with public probes, predictions and results, is at [github.com/rkishore/dotnet-embedding-parity](https://github.com/rkishore/dotnet-embedding-parity).

## Appendix: six flaws in the .NET-vs-Python comparison

The benchmark collected **six asymmetries, five of them in my favour**. Flaws 1–5 were fixed before the HTTP comparison in lesson 2; flaw 6 was found the day after it.

1. **Quilha computed the wrong embedding.** It read one special token where the model expects an average over all token vectors. Its vectors scored a cosine of 0.62 against the correct ones and skipped work Python was doing. *In my favour.*
2. **The Python servers still padded to 128 tokens after Quilha had stopped.** Worth about 7× on short inputs. *In my favour.*
3. **Only Quilha was instrumented.** A timing header on one code path cost about 9.4% of a request. *Against me.*
4. **The Python server handled one request at a time.** A blocking model call inside an `async def` handler stalls FastAPI's event loop; [FastAPI's docs](https://fastapi.tiangolo.com/async/) say to use a plain `def`. On long passages Python gained 7% from 1 to 100 concurrent requests, where Quilha gained 53%. *Hugely in my favour.*
5. **Python ran only as a single worker at first**, a deployment few teams would ship. The four-worker arm fixed that. *In my favour.*
6. **Quilha's tokenizer dropped some text**: accented words became unknown tokens and some symbols were deleted. That meant about 0.4% fewer tokens on the passage corpus and none on queries, so I didn't re-run the comparison. The in-process experiment in lesson 1 ran after the fix, and the parity post has the full story. *In my favour.*

**The two biggest, flaws 1 and 4, weren't found by reading code.** Flaw 1 was caught by the output check. For flaw 4, I'd predicted Quilha's lead would *narrow* under load, and written down beforehand that a widening lead should be read as a fault in the Python arm. It widened. The prediction failed, and the failure was the useful part.
