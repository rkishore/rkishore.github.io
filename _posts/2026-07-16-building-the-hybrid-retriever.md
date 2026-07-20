---
title: "Building the Hybrid Retriever: PubMedQA, Qdrant, and Three Surprises"
description: "Part 2 of the hybrid search series: a build log. I build the dense + sparse + RRF retriever from Part 1 on biomedical abstracts with Qdrant and FastEmbed — and hit three results I didn't fully expect, including a SPLADE subword trap on a rare acronym."
---

*Hybrid Search series &mdash; 1. [Why Hybrid](/2026/07/14/why-hybrid-search.html) &middot; 2. Building the Retriever (you're here) &middot; 3. [Hybrid Search Inside an Agent](/2026/07/20/hybrid-search-inside-an-agent.html)*

**Objective:** A first-person build log &mdash; organized around one question: *[Part 1](/2026/07/14/why-hybrid-search.html) argued hybrid retrieval should beat either method alone; does it actually, when I build it on a domain full of rare, specialized terms? Here's the build, and three results I didn't fully expect.*

[Part 1](/2026/07/14/why-hybrid-search.html) was the *why* &mdash; two retrievers, two blind spots, one rank-based fusion. This is the *how*. And I deliberately changed domains to make it a harder test: Part 1 used e-commerce examples, but here I build on **[PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA)** &mdash; 100 biomedical research abstracts. Biomedical text is **specialized-terminology-heavy with scarce vocabulary overlap**: exactly the niche [SPLADE](https://arxiv.org/abs/2107.05720) is pitched for, and a much harder test of hybrid than product search. If hybrid earns its keep anywhere, it should be here. Five beats to build it, then three surprises from the actual run.

## First, the data

Before any of the beats, the raw material. PubMedQA's `pqa_artificial` split has 211,269 instances; I pull the first 100 into a [pandas](https://pandas.pydata.org/) dataframe with HuggingFace [`datasets`](https://huggingface.co/docs/datasets):

```python
from datasets import load_dataset

dataset   = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")  # 211,269 rows
dataset   = dataset.select(range(100))     # keep the first 100 for this build
source_df = dataset.to_pandas()            # → (100, 5): pubid, question, context, long_answer, final_decision
```

A quick sanity check before trusting it: no duplicate `pubid`s, no null or empty `question`/`long_answer`, and one label skew worth noting &mdash; `final_decision` runs 89 `yes` to 11 `no`. That `source_df` is the thing every beat below operates on.

## Beat 1: document construction is a design decision, not boilerplate

Before any embedding, you have to decide *what text represents a document* &mdash; and this is the single most important modeling choice in the whole build. Each PubMedQA row has structured fields: labeled context paragraphs, a `long_answer`, and a `question`. I join the context paragraphs (prefixed with their section label) and the conclusion &mdash; and I **deliberately leave the `question` out**:

```python
def build_document(row):
    parts = []
    for label, para in zip(row["context"]["labels"], row["context"]["contexts"]):
        parts.append(f"[{label}] {para}")
    parts.append(f"[CONCLUSION] {row['long_answer']}")
    return "\n".join(parts)

source_df["combined_text"] = source_df.apply(build_document, axis=1)
```

Why exclude the `question`? Because *the question is what the user searches with.* Bake it into the document and every doc trivially self-matches its own query &mdash; your retrieval metrics look fantastic and mean nothing. The document should be the *answer-bearing* text, not the query. (The first document, `id 0`, is a chronic-rhinosinusitis abstract about "Group 2 innate lymphoid cells (ILC2s)" &mdash; remember it; it comes back to bite me in surprise 3.)

## Beat 2: two embedders, one dataframe

Now embed that text *twice* &mdash; once dense, once sparse &mdash; with [FastEmbed](https://github.com/qdrant/fastembed):

```python
from fastembed import TextEmbedding, SparseTextEmbedding

dense_model  = TextEmbedding("BAAI/bge-large-en-v1.5")            # 1024-d
sparse_model = SparseTextEmbedding("prithivida/Splade_PP_en_v1")

# .embed() returns a *generator* — wrap it in list() or you get a lazy object, not vectors
source_df["dense_embeddings"]  = list(dense_model.embed(source_texts))
source_df["sparse_embeddings"] = list(sparse_model.embed(source_texts))
```

One gotcha to flag: `.embed()` is lazy &mdash; it hands back a *generator*, so you `list()` it or you end up storing that generator instead of the vectors.

The two representations really are opposite shapes. The dense vector is `(1024,)`, all non-zero. The sparse vector for that same first document has just **120 non-zeros out of a 30,522-word vocabulary** &mdash; over 99% zeros, and every non-zero index decodes back to an actual token. Part 1's dense-vs-sparse table, now in real numbers.

## Beat 3: one collection, two named vectors

A word on the tool first, since the next two beats lean on it. **[Qdrant](https://qdrant.tech) is an open-source vector database** &mdash; it stores your vectors next to their metadata (the *payload*) and runs the approximate-nearest-neighbor search over them, so you don't hand-roll any of that. [Part 1](/2026/07/14/why-hybrid-search.html) surveyed several ways to do this ([Pinecone](https://www.pinecone.io), [`pgvector`](https://github.com/pgvector/pgvector), and others); I build on Qdrant here purely to make the concepts concrete, **not as an endorsement** &mdash; the same dense + sparse + RRF design ports to any of them. From here on, most of the code is really about speaking Qdrant's SDK correctly.

Both vectors live on *one* Qdrant point, in *one* collection. That's what makes a single hybrid query possible. Creating the collection is where the API's asymmetry first bites:

```python
client = QdrantClient(":memory:")   # no server needed for a 100-doc build

client.create_collection(
    "pubmedqa",
    vectors_config={"text-dense": VectorParams(size=1024, distance=Distance.COSINE)},
    sparse_vectors_config={"text-sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))},
)
```

Notice there's a `vectors_config` and a `sparse_vectors_config` &mdash; but **no `dense_vectors_config`.** Reach for that symmetric-sounding name (I did) and Qdrant throws `AssertionError: Unknown arguments`. Dense is just "the vectors"; sparse is the special-cased one. The dense config needs `size=1024` to match [BGE](https://huggingface.co/BAAI/bge-large-en-v1.5); the sparse config needs no size at all (the vocabulary is its dimension).

Then pack each row into a point. The next surprise is that `PointStruct` takes a **singular `vector=`** even though it holds *both* named vectors (pass `vectors=` and it errors):

```python
def make_points(df):
    points = []
    for idx, row in df.iterrows():
        sp = row["sparse_embeddings"]
        points.append(PointStruct(
            id=idx,
            payload={"text": row["combined_text"], "pubid": int(row["pubid"]),
                     "question": row["question"], "final_decision": row["final_decision"]},
            vector={                                             # singular — holds a dict of BOTH
                "text-dense":  row["dense_embeddings"].tolist(),  # numpy → list, or Qdrant rejects it
                "text-sparse": SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist()),
            },
        ))
    return points

client.upsert("pubmedqa", make_points(df))
```

Two more sharp edges baked in: the dense vector has to be `.tolist()`-ed (Qdrant won't take a raw [numpy](https://numpy.org) array), and the sparse vector gets rebuilt as a `SparseVector` from its `.indices` and `.values`. Here's the whole indexing pipeline in one picture:

![The indexing build, left to right. 100 PubMedQA abstracts of structured fields go through build_document, which joins the context paragraphs from [BACKGROUND] to [CONCLUSION] and excludes the question field, producing combined_text. That text is embedded two ways — a dense BGE vector of 1024 dimensions, all non-zero, and a sparse SPLADE vector with about 120 non-zeros out of 30,522. make_points packs each document into one PointStruct with both named vectors, which is upserted into a single Qdrant collection holding text-dense and text-sparse. At query time both vectors fan out in one batch and fuse with RRF, the same pipeline as Part 1.](/images/hybrid/hybrid-build-pipeline.svg)

## Beat 4: the hybrid query is one batch, two requests

With both vectors indexed, a hybrid query is just two `QueryRequest`s fired in a single `query_batch_points` call &mdash; one against each named vector:

```python
def search(query_text, top_k=10):
    dv = list(dense_model.embed([query_text]))[0]
    sv = list(sparse_model.embed([query_text]))[0]
    results = client.query_batch_points("pubmedqa", requests=[
        QueryRequest(query=dv.tolist(), using="text-dense",  limit=top_k, with_payload=True),
        QueryRequest(query=SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist()),
                     using="text-sparse", limit=top_k, with_payload=True),
    ])
    return [results[0].points, results[1].points]   # [dense_results, sparse_results]
```

The `using=` argument is what routes each request to the right index &mdash; `"text-dense"` hits the [HNSW](https://arxiv.org/abs/1603.09320) graph, `"text-sparse"` hits the inverted index. Out comes two ranked lists, ready to fuse.

## Beat 5: fuse with RRF, then read the tea leaves

The fusion is [RRF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) &mdash; the `rrf()` from Part 1 made concrete, ranks only, no scores:

```python
def rank_list(points):                                   # ScoredPoints → [(id, rank)], 1-indexed
    return [(p.id, rank) for rank, p in enumerate(points, start=1)]

def rrf(rank_lists, k=60, default_rank=1000):
    all_ids = {id for rl in rank_lists for id, _ in rl}
    scores = {id: sum(1.0 / (k + dict(rl).get(id, default_rank)) for rl in rank_lists)
              for id in all_ids}
    return sorted(scores.items(), key=lambda kv: -kv[1])  # [(id, rrf_score)] best-first
```

That's the whole retriever. Now the interesting part &mdash; what it actually did.

## Surprise 1: the score-scale gap is real, and it *moves*

Part 1 argued in the abstract that you can't average dense and sparse scores. Here it is on real data. Query: `"immune cells driving nasal inflammation"`:

```
DENSE · cosine                     SPARSE · dot-product
1. id=0   0.729  CRS / ILC2s       1. id=52  12.198  nasal lymphoma
2. id=7   0.652  sPLA2             2. id=0   11.592  CRS / ILC2s
3. id=4   0.631  tumor immunity    3. id=4    7.578  tumor immunity
4. id=52  0.620  nasal lymphoma    4. id=65   6.980  SNPs
5. id=79  0.612  bipolar           5. id=13   5.317  lupus B cells
```

Dense scores sit in a tight `0.61–0.73` band (bounded cosine). Sparse scores run `5.3–12.2` (unbounded dot-product). Averaging `0.729` and `12.198` is meaningless &mdash; the sparse number would swamp the dense one every time.

The subtlety I didn't expect: the gap isn't even a *fixed* ratio you could normalize away. The **same** sparse retriever topped out at `12.198` on this query and `15.770` on the next one (the `ILC2` query below), while dense sat at `~0.7` both times. The sparse scale drifts with the query; there's no single constant that reconciles it. Rank-based fusion sidesteps the whole problem &mdash; and *that* instability, not "sparse is always bigger," is the real argument for RRF.

## Surprise 2: RRF genuinely promoted consensus

Look again at that query. Dense's #1 is `id 0` (the chronic-rhinosinusitis abstract); sparse's #1 is `id 52` (a completely different doc &mdash; nasal-type NK/T-cell lymphoma). The two retrievers *disagree on the top result.* But three documents &mdash; `id 0`, `id 4`, `id 52` &mdash; appear in **both** top lists. Fuse with RRF, and those three consensus docs take the fused top three, in a new order:

![Real results for the nasal-inflammation query. The dense list ranks documents 0, 7, 4, 52, 79; the sparse list ranks 52, 0, 4, 65, 13. The dense top is document 0 and the sparse top is document 52 — different. Three documents appear in both lists: 0, 4, and 52. RRF with k of 60 floats exactly those three to the fused top three: document 0 at 0.0325, document 52 at 0.0320, document 4 at 0.0317. Hand calculation for document 0: dense rank 1 and sparse rank 2 gives 1/61 plus 1/62, equal to 0.0325.](/images/hybrid/hybrid-rrf-consensus.svg)

The hand-math is worth doing once, because it's the entire mechanism. `id 0` is dense rank 1 and sparse rank 2, so its RRF score is `1/(60+1) + 1/(60+2)` = `1/61 + 1/62` = `0.0325` &mdash; the highest of any doc, because it scored well in *both* lists. A document that only one retriever found contributes just one reciprocal, so it can't out-rank a doc both agreed on. Consensus promotion, working on real data, with no score normalization anywhere. This is RRF earning its keep.

## Surprise 3: the SPLADE subword trap

Here's the one that genuinely surprised me &mdash; the non-textbook finding. The textbook says: *sparse retrieval wins on exact, rare terms.* So I queried the rare acronym `"ILC2"` directly, expecting the sparse side to nail it cold:

```
DENSE · cosine                     SPARSE · dot-product
1. id=0   0.717  CRS / ILC2s       1. id=0   15.770  CRS / ILC2s
2. id=7   0.599  sPLA2             2. id=7   12.371  sPLA2
3. id=46  0.557  Lp-PLA2           3. id=85   9.412  RCC
4. id=29  0.549  vein bypass       4. id=31   6.118  Barrett's esophagus
5. id=85  0.548  RCC               5. id=46   5.779  Lp-PLA2
```

The good news: the *real* ILC2 abstract (`id 0`, which literally contains "ILC2s") ranks **#1** on *both* sides &mdash; the exact term still won the top slot. The strange part is everything below it. Positions 2–5 are the **same cast on both sides**: `sPLA2`, `Lp-PLA2`, `RCC` &mdash; papers on phospholipases and renal cell carcinoma that have **nothing** to do with innate lymphoid cells. Why would *either* retriever rank them so high for `ILC2`?

Because to the model, `ILC2` isn't a word &mdash; it's a handful of fragments. Language models don't index whole words; they break text into **subword tokens** from a fixed vocabulary, so frequent words stay intact while rare ones get chopped into smaller pieces (a leading `##` marks a piece that continues the one before it). The scheme BERT uses &mdash; and that SPLADE inherits &mdash; is [WordPiece](https://huggingface.co/docs/transformers/en/tokenizer_summary), and it shatters the rare acronym `ILC2` into exactly those common fragments. I checked the actual split:

![The rare acronym ILC2 is tokenized by WordPiece into three subwords: il, then ##c, then ##2. Two are promiscuous. The ##c subword matches document 85, RCC, which tokenizes to rc and ##c. The ##2 subword matches document 7, sPLA2, which tokenizes to sp, ##la, ##2, and document 46, Lp-PLA2, which tokenizes to lp, dash, pl, ##a, ##2. These three share only a subword with ILC2, nothing conceptual, yet the sparse retriever ranks them in positions 2 through 5. The lesson: exact-term retrieval is only as exact as the tokenizer.](/images/hybrid/hybrid-subword-trap.svg)

For the **sparse** side the cause is mechanical and provable. `ILC2` &rarr; `il ##c ##2`. The `##2` subword lives in `sPLA2` (`sp ##la ##2`) and `Lp-PLA2` (`lp - pl ##a ##2`); the `##c` subword lives in `RCC` (`rc ##c`). Every false friend shares a *subword* with `ILC2` and nothing else. The sparse retriever is doing exactly what it's built to do &mdash; match lexical pieces &mdash; but a rare acronym's pieces are promiscuous, so it drags in chemistry and oncology papers on the strength of a stray `c` and a stray `2`.

But the tidy "sparse stumbles, dense saves it" story doesn't hold here &mdash; because **dense surfaces the same false friends.** `sPLA2`, `Lp-PLA2`, and `RCC` fill dense's positions 2–5 too. I can't dissect a 1024-d embedding as cleanly as a token match, but the reason is almost certainly shared plumbing: [BGE](https://huggingface.co/BAAI/bge-large-en-v1.5) is itself a BERT model over the *same* WordPiece vocabulary, fed the same fragmented `il ##c ##2`, and a bare acronym is a thin semantic signal to begin with. Both retrievers are BERT-family, so they share a tokenizer &mdash; and therefore share this blind spot.

That's what quietly breaks the happy ending. Surprise 2's whole point was that RRF rewards agreement &mdash; but here the two retrievers *agree on the junk.* `sPLA2` is rank 2 in **both** lists, so its fused score (`1/62 + 1/62`) lands it at **#2 overall**, RRF *promoting* a false friend instead of filtering it. Fusion only rescues you when your retrievers fail *differently*; a shared tokenizer is a shared failure mode that no amount of rank fusion can undo. So the real lesson isn't "sparse is fragile" &mdash; it's that **exact-term retrieval is only as exact as the tokenizer, and hybrid can't cover a blind spot both halves have in common.**

## The one-liner that ties it together

Distilled: **the concepts survived contact with a hard domain &mdash; hybrid on biomedical abstracts usually beats either half &mdash; but the interesting lessons were in the failure modes:** the sparse score scale drifts with the query (so fuse on rank), RRF promotes whatever both retrievers agree on (a superpower when they're right, a trap when they share a blind spot), and SPLADE's "exact" matching is only as exact as WordPiece lets it be. Build it, then *read the results* &mdash; that's where the understanding is.

**Coming next:** the retriever works, but so far *I'm* the one deciding when to call it and with what query. In Part 3 I hand it to an agent &mdash; wrap `search` + `rrf` as a [LangChain](https://www.langchain.com) tool and let a `create_agent` ReAct loop decide when to search, reformulate, and search again before it answers. It's the [ADK agent-plus-tool pattern](/2026/07/08/your-first-google-adk-agent.html) from the last series, with hybrid retrieval as the tool.
