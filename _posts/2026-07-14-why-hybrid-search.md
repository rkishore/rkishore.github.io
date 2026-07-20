---
title: "Why Hybrid Search: Two Blind Spots, One Fusion"
description: "Part 1 of a hybrid retrieval series: keyword search and semantic search each have a blind spot. What exactly are they — dense vs sparse vectors, HNSW vs inverted index, BM25 vs SPLADE — and how does fusing both with RRF cover for each, no score normalization required."
---

*Hybrid Search series &mdash; 1. Why Hybrid (you're here) &middot; 2. [Building the Retriever](/2026/07/16/building-the-hybrid-retriever.html) &middot; 3. [Hybrid Search Inside an Agent](/2026/07/20/hybrid-search-inside-an-agent.html)*

**Objective:** A first-person field guide to *hybrid retrieval* &mdash; organized around one question: *keyword search and semantic search each have a blind spot; what exactly are they, and how does running both and fusing the results cover for each?*

The thing that finally made hybrid search click for me wasn't a benchmark. It was two queries that each break a different retriever. Search a product catalog for **"quiet bathroom exhaust fan"** and a keyword engine sails right past the perfect product &mdash; because its description says *"whisper-silent operation"* and never once says "quiet." Now search for **"revent 80 cfm"** &mdash; an exact brand and spec &mdash; and a semantic engine hands you a grab-bag of *similar-feeling* fans while blurring the one exact model you asked for. Two queries, two failures, two *different* retrievers to blame. That's the whole reason hybrid exists, and everything below unpacks it.

## The two blind spots

Each retriever fails in a way the other doesn't:

- **Lexical search** (keyword / [BM25](https://en.wikipedia.org/wiki/Okapi_BM25)) matches *surface tokens*. Its blind spot is **vocabulary mismatch** &mdash; synonyms. "quiet" &ne; "silent" &ne; "low-noise," so a genuinely relevant document that happens to use a different word is *invisible* to it. It can only match words that are literally there.
- **Dense search** (semantic) compresses text to one meaning vector. Its blind spot is **exact rare tokens** &mdash; model numbers, SKUs, specs like "80 cfm." Those get smeared into the general "vibe" of the query, so its nearest neighbors are similar-feeling, not exact.

![Two panels. Left, the lexical blind spot: the query "quiet bathroom exhaust fan" reaches a keyword retriever that matches surface tokens only, so a relevant product described as "whisper-silent operation" is missed because it never contains the word "quiet" — a vocabulary mismatch. Right, the dense blind spot: the query "revent 80 cfm" reaches a dense semantic retriever that compresses to one meaning vector and returns a similar-feeling "quiet 90 cfm fan," blurring the exact model and spec. A bar across the bottom: hybrid runs both, so each retriever covers the other's blind spot — semantic recall plus lexical precision.](/images/hybrid/hybrid-two-blind-spots.svg)

So the pitch writes itself: **run both, then fuse.** Dense brings *semantic recall* (it finds "silent" for "quiet"); sparse brings *lexical precision* (it nails "revent 80 cfm"). Each covers the other's blind spot. The rest of this post is the machinery that makes "run both, then fuse" actually work &mdash; and one subtlety in the fusion step that's easy to get wrong.

## Dense vs sparse: two ways to be a vector

Both retrievers turn text into a vector, but they are *opposite kinds* of vector:

| | **Dense** | **Sparse** |
|---|---|---|
| Made by | `BAAI/bge-large-en-v1.5` | `prithivida/Splade_PP_en_v1` |
| Dimensions | **1024**, fixed | **30,522** &mdash; the BERT WordPiece vocab size |
| Non-zeros | ~all of them | ~**30&ndash;80** (>99% zero) |
| Good at | **semantic / conceptual** match | **exact / lexical** match |
| Similarity | **cosine** | dot product |
| Interpretable? | **No** | **Yes** |

That last row is the one I keep coming back to. A dense vector's 1024 dimensions are *entangled latent features* &mdash; no single dimension "is" a word; you cannot point at index 442 and say what it means. A sparse vector is the opposite: **each non-zero dimension *is* a vocabulary token.** Decode the indices back to tokens and you can read, in plain language, *why* a document matched &mdash; `fan: 2.1`, `exhaust: 1.8`, `airflow: 0.9`. Sparse retrieval is auditable in a way dense retrieval simply isn't.

The move that makes hybrid *possible*, though, is that both vectors live on **one document, in one collection**. A single point carries a `text-dense` vector *and* a `text-sparse` vector side by side &mdash; so you can query the same corpus two ways without duplicating it. (That storage layout is a Part 2 concern; here it's enough to know the two representations coexist.)

## From distance to fast search

A vector is only useful if you can find its neighbors *quickly*. Two more choices hide in here.

First, **what does "close" mean?** For text, the default is **cosine similarity** &mdash; the angle between vectors, ignoring their magnitude. That's deliberate: magnitude tends to track document length and raw term counts, which is *noise*; direction is what carries meaning. (Euclidean distance, where magnitude matters, is the right call for images or spatial data &mdash; not text.)

Second, **how do you search without checking every document?** Exact k-nearest-neighbors compares the query against *every* vector &mdash; **O(N)**, which collapses at scale. The escape is **ANN (Approximate Nearest Neighbor)**: trade a sliver of accuracy for an enormous speedup &mdash; a "good-enough" guess. For dense vectors, the workhorse is **HNSW** (Hierarchical Navigable Small World): a multi-layer proximity *graph* with sparse long-range links up top for coarse jumps and dense short links at the bottom for precision. You **greedily descend from the top layer**, hopping toward the query until you converge on its neighborhood &mdash; a few hundred comparisons instead of millions.

![A layered proximity graph with three levels stacked top to bottom. The top level, Layer 2, holds only a few nodes joined by long-range links, with an entry point on the left. The middle level, Layer 1, has more nodes and medium links. The bottom level, Layer 0, contains every node joined by short links. A highlighted search path starts at the top entry point, greedily hops along each layer toward the node nearest the query, and drops down a layer via dashed arrows, converging in Layer 0 on the nearest-neighbor node beside the query marker. Sparse long-range links up top jump close fast; dense short links at the bottom refine to the true nearest neighbor.](/images/hybrid/hybrid-hnsw.svg)

Sparse vectors don't use a graph at all. They use an **inverted index** &mdash; the classic `term → postings list of document IDs` structure that Elasticsearch and BM25 have run on for decades. Look up each query term, walk its (short) postings list, score. So a hybrid system is really **one collection with two index types**: an HNSW graph for the dense side, an inverted index for the sparse side.

## Classic lexical scoring: TF-IDF and BM25

Before the learned stuff, it's worth being precise about how sparse scoring classically works, because it explains *exactly* where it breaks. **TF-IDF** weighs a term by two factors:

- **TF (term frequency)** &mdash; how often the term appears *in this one document*. More mentions here &rarr; more about this doc.
- **IDF (inverse document frequency)** &mdash; how *rare* the term is *across the whole corpus*. And mind the direction: it's **inverse**. A term in *few* documents gets a **high** weight; a term in *many* documents ("the," "and") gets a **low** one. (This is the one I personally flipped the first time &mdash; appearing in more documents *lowers* the weight.)

Multiply them and the highest weight lands on a term that's **frequent in this document but rare in the corpus** &mdash; a *distinctive* term. And the multiplication is the whole trick: if *either* dial is near zero, the score is near zero no matter how big the other one is.

A tiny collection makes it concrete. Picture three café menus:

| | Menu A | Menu B | Menu C |
|---|---|---|---|
| words | espresso, latte, coffee | coffee, tea, juice | coffee, coffee, bagel |

Two words behave completely differently here. `coffee` appears in **every** menu &mdash; common, so its IDF (rarity) dial sits near zero. `espresso` appears in just **one** &mdash; rare, so its IDF is high. Because TF-IDF *multiplies* the two dials, `coffee`'s score collapses no matter how often it shows up (Menu C says it twice and still loses), while `espresso` passes both tests and becomes the signal that Menu A is *the espresso menu*:

![Three café menus — Menu A: espresso, latte, coffee; Menu B: coffee, tea, juice; Menu C: coffee, coffee, bagel — with "coffee" in all three (common) and "espresso" in one (rare). Below, two scoring rows show TF-IDF as TF times IDF. For coffee the TF bar is high but the IDF rarity bar is near zero (it is in all three menus), so the product is a low score. For espresso, TF is moderate and IDF is high (one of three menus), so the product is a high score. The punchline bar: a search for "espresso latte" ranks Menu A first, because the rare distinctive word carries the weight while ubiquitous coffee barely moves the needle.](/images/hybrid/hybrid-tfidf-cafe.svg)

So a search for **"espresso latte"** ranks Menu A first: the rare, distinctive query words carry almost all the weight, while ubiquitous `coffee` barely moves the needle. That's the lexical half of a hybrid stack in miniature &mdash; a cheap, robust "which document is *distinctively* about these words?" signal.

**BM25** (short for *Best Match 25*, from the Okapi retrieval system) is the industry-standard upgrade, adding two refinements TF-IDF lacks:

1. **TF saturation** &mdash; diminishing returns. The 50th occurrence of a word doesn't make a document 50&times; more relevant; the curve flattens (the `k1` knob).
2. **Document-length normalization** &mdash; penalize long documents so they don't win just by containing more words (the `b` knob).

BM25 is fast, strong, and still a serious baseline. But it has a **fatal limit**: it builds its vector *only from tokens literally present in the text.* It will **never** match a synonym it doesn't contain. That's the lexical blind spot in one sentence &mdash; and exactly the gap the next idea closes.

## SPLADE: sparse that learned to expand

**[SPLADE](https://arxiv.org/abs/2107.05720)** (*Sparse Lexical and Expansion model*) produces the same *shape* of output as BM25 &mdash; a sparse vector over the vocabulary &mdash; but it does two things BM25 cannot: smarter per-term weighting, and, the big one, **term expansion**.

Term expansion means SPLADE **activates related vocabulary terms that are *absent* from the text.** A document about "exhaust fan ventilation" will light up `airflow` &mdash; a word it never uses &mdash; each expansion term carrying its own learned weight (usually lower than the words actually present). *That* is how a sparse retriever starts to cross the vocabulary-mismatch chasm that sank BM25:

![The same document, "exhaust fan ventilation," scored two ways as a sparse vector over the vocabulary. Top, BM25 puts weight only on the tokens literally present — exhaust, fan, ventilation — so a query for "airflow" finds nothing and misses the document. Bottom, SPLADE keeps those present terms at full weight but also activates related vocabulary terms the text never used — airflow and quiet — each at a smaller learned weight, so a query for "airflow" now matches an activated dimension. Learned term expansion lets a still-sparse retriever cross the vocabulary-mismatch gap that sinks BM25.](/images/hybrid/hybrid-bm25-vs-splade.svg)

Where does the expansion come from? It falls out of how BERT was trained. BERT learned language by playing *fill-in-the-blank* &mdash; hide a word in a sentence, guess which word belongs. So at any position in a text, it can rate *every* word in its 30,522-word vocabulary by how well that word would fit there &mdash; including words that aren't actually present. SPLADE puts that skill to work: it runs each word of the document through that fill-in-the-blank scorer, then, for every vocabulary word, keeps the **single highest** score it earned anywhere in the document. (That "keep the max across positions" step is the bit called *max-pooling* &mdash; nothing more mysterious than taking the best score per word.) What's left is a short list of weighted terms: the words the document actually uses, *plus* the words it strongly implies (`ventilation` lighting up `airflow`). Scoring the whole vocabulary by "what fits here?" is *where expansion comes from, for free.*

One honest caveat: SPLADE is still **lexical**, not full semantic search. It bridges the vocabulary gap *partway* &mdash; which is exactly why you still want a dense retriever alongside it.

## RRF: fuse on rank, never on score

Now the payoff. You've run both retrievers; you have two ranked lists. How do you merge them into one?

Here's the trap, straight from one of my own smoke tests. The **same winning document** came back scored **0.835** by the dense retriever (cosine) and **14.742** by the sparse retriever (dot-product):

![A bar chart of the same winning document scored on one shared axis from 0 to 16. The dense cosine score, 0.835, is bounded between 0 and 1, so its bar is a tiny sliver. The sparse dot-product score, 14.742, is unbounded, so its bar is far taller. Both retrievers rank the document number 1. The punchline: averaging 0.835 and 14.742 is meaningless, but both rank #1 — so fuse on rank, not score.](/images/hybrid/hybrid-incompatible-scales.svg)

You cannot meaningfully average 0.835 and 14.742. Cosine is bounded roughly 0&ndash;1; a sparse dot-product runs from 0 to unbounded. They're on **incompatible scales** &mdash; adding the raw numbers is nonsense, and even normalizing them is fragile. But notice what *is* clean: the document is ranked **#1 in both lists**. Ranks are comparable even when scores aren't. That's the insight behind **[Reciprocal Rank Fusion (RRF)](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)** &mdash; it throws the scores away and fuses on **rank alone**:

```
rrf_score(doc) = Σ  1 / (k + rank)      # summed over each list, rank is 1-indexed
```

Three things to know about it:

- **`k` (&#8776;60, from Cormack 2009)** is a smoothing constant so rank-1 doesn't utterly dominate. With `k=0`, a rank-1 hit contributes a full `1.0` and drowns everything else; a larger `k` flattens the curve. Sixty is the well-worn default.
- **Consensus wins.** A document that appears in *both* lists sums *two* reciprocals, so it beats a document that only one retriever found. Agreement between your two retrievers is rewarded automatically &mdash; which is precisely the behavior you want from a hybrid system.
- **Absence is handled gracefully.** A document missing from one list is treated as if it sat at some large default rank &mdash; a small, finite, non-zero contribution rather than a special case.

One more nuance worth naming: there are **two families** of fusion. **Weighted score fusion** (`α·dense + (1−α)·sparse`) *requires* normalizing both scorers to a common 0&ndash;1 range &mdash; because you're literally adding scores. **RRF** needs no normalization at all, because ranks are already on a common scale. Neither is "wrong" &mdash; but when your scorers are on incompatible scales, which dense-cosine and sparse-dot-product *always* are, **RRF is the one that sidesteps the whole problem.**

You can watch both families play out in the commercial vector databases. [Pinecone's hybrid search](https://docs.pinecone.io/guides/search/hybrid-search) picks **weighted score fusion** &mdash; a convex combination `α·dense + (1−α)·sparse` with a single `alpha` knob (`1.0` = pure dense, `0.0` = pure sparse). And because raw sparse scores are unbounded and would otherwise swamp the bounded dense ones, Pinecone makes them comparable by *scaling the query vectors before search* &mdash; the very normalization step family (1) can't avoid. [Qdrant's Query API](https://qdrant.tech/articles/hybrid-search/) picks the other family: **RRF**, which its docs call *"the de facto standard,"* fusing on rank with no normalization (with a distribution-based score-fusion variant on offer for when you *do* want to work in scores). Same two families, two vendors &mdash; and the same trade-off you just reasoned through.

And you don't even need a dedicated vector database. [pgvector](https://github.com/pgvector/pgvector) turns plain PostgreSQL into the dense half &mdash; an `HNSW` index over an embedding column, cosine and friends built in. The sparse half already ships with Postgres: its native full-text search (`tsvector` + `ts_rank`), or [ParadeDB's `pg_search`](https://github.com/paradedb/paradedb) extension when you want *real* BM25 rather than Postgres's tf-idf-flavored ranking. What Postgres *doesn't* give you is a fusion step &mdash; so you [write it yourself in SQL](https://supabase.com/docs/guides/ai/hybrid-search), and the idiomatic way is RRF: rank each result set with a window function and sum `1/(k + rank)` &mdash; the exact `rrf()` from earlier, expressed as a `SELECT`. One database, two indexes, a fusion you own, and no separate search service to run.

That stack is productized, too. [TigerData](https://www.tigerdata.com/search) (the rebranded Timescale) pairs `pgvectorscale` &mdash; a DiskANN index layered on pgvector for the dense side &mdash; with `pg_textsearch` for BM25 on the sparse side, and writes the RRF merge as a single SQL query, exactly the shape above. The same one-database pattern even backs [persistent agent memory](https://www.tigerdata.com/learn/building-ai-agents-with-persistent-memory-a-unified-database-approach) &mdash; a nice bridge toward where this series is headed, when the retriever becomes a tool an agent reaches for.

## The pipeline, end to end

Put it together and the whole system is one query fanning out into two indexes and back into one list:

![A left-to-right pipeline. One query fans out into two paths reading from one collection that stores two named vectors per document. The top path embeds the query as a dense 1024-dimension vector and searches an HNSW graph, producing ranked list D (semantic recall). The bottom path embeds it as a sparse SPLADE vector and searches an inverted index, producing ranked list S (lexical precision). Both lists feed RRF, which merges on ranks only with no normalization and emits one fused list where consensus wins.](/images/hybrid/hybrid-pipeline.svg)

Every piece of this post lives somewhere on that diagram: the two blind spots are why there are two paths; dense/sparse are the two vector types; HNSW and the inverted index are the two ways to search them fast; BM25&rarr;SPLADE is how the sparse path learned to expand; and RRF is the rank-only merge that doesn't care that 0.835 and 14.742 don't compare.

## The one-liner that ties it together

Distilled: **lexical search misses synonyms and dense search blurs exact tokens, so hybrid runs both and fuses their two ranked lists with RRF &mdash; on rank, never score &mdash; because dense cosine and sparse dot-product live on scales you can't average.** Two complementary retrievers, one rank-based merge, no normalization required.

**Coming next:** the concepts become code. In Part 2 I build the retriever for real &mdash; Qdrant named vectors and FastEmbed, one collection carrying both a dense and a sparse vector per document, firing both queries in a single batch, and the `rrf()` function from this post applied to actual results. We'll watch dense-only, sparse-only, and *both* pull up different documents on the same query &mdash; the blind spots, made concrete.
