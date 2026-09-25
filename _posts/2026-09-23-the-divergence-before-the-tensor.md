---
title: "The Divergence Before the Tensor: Embedding Parity in .NET"
description: "I ran three .NET embedding libraries against sentence-transformers on the same model. All three diverge in at least one configuration, and every divergence is in tokenization, not pooling. Unstripped accents cost French recall@10 14.5 points (0.409 → 0.264, −35.5% relative), and under invariant globalization accent stripping silently fails on precomposed text — so a parity check on your laptop says nothing about your container."
date: 2026-09-23 07:30:00 -0400
---

**Objective:** A measured look at embedding parity in .NET, organized around one question: *three .NET libraries run the same model as sentence-transformers, so why do their vectors still differ, and what does that cost retrieval?* Everything below comes from a public evidence repository, [rkishore/dotnet-embedding-parity](https://github.com/rkishore/dotnet-embedding-parity), and each result links to the file it comes from.

***Disclosure***: I built a [.NET embedding server of my own, Quilha, as a learning experiment](/2026/09/25/lessons-from-building-a-dotnet-embedding-server.html), and that work led me here. This post covers only the three external libraries and the tokenizer package underneath one of them.

Here is one sentence: **`Café crème brûlée in São Paulo, naïve résumé`**. Embed it with `sentence-transformers/all-MiniLM-L6-v2` in Python, then embed it with the model's ONNX export in .NET, and compare the two vectors. In [ElBruno.LocalEmbeddings](https://github.com/elbruno/elbruno.localembeddings) 1.6.1 the cosine is **0.333**. Same model, same weights, same pooling math, and a third of a match. [Semantic Kernel](https://github.com/microsoft/semantic-kernel) gets it right with ICU, the system's Unicode library (**1.000000**), but wrong under invariant globalization (**0.333**). Nothing throws and nothing warns. What's going on, and where is the difference coming from?

## The one idea: the divergence lives before the tensor

An embedding pipeline has four steps, and only the last two involve the neural network:

1. **Text:** what the user typed.
2. **Tokenizer:** turns text into token ids by lowercasing, splitting off punctuation, stripping accents, and looking up each piece in a fixed vocabulary. (If subword tokenization is new to you, Hugging Face's [tokenizer summary](https://huggingface.co/docs/transformers/en/tokenizer_summary) covers WordPiece, the scheme BERT uses.)
3. **Model:** turns those ids into one vector per token.
4. **Pooling:** averages them into the single vector you store and search.

![An embedding pipeline in four steps: text, tokenizer, model, pooling. A dashed line between the tokenizer and the model marks the input tensor. Every divergence measured sat in step 2, the tokenizer, and not in the text itself, which is the same input either way; tensor-level parity checks start to the right of the tensor line, and the model and pooling steps were cleared because each library's own token ids reproduce its vectors exactly. Below, the sentence "Café crème brûlée in São Paulo, naïve résumé" through two tokenizers: sentence-transformers gives "cafe cr ##eme br ##ule ##e in sao paulo , naive resume", cosine 1.000; ElBruno 1.6.1 and Semantic Kernel under invariant globalization give "[UNK] [UNK] [UNK] in [UNK] paulo , [UNK] [UNK]", cosine 0.333, because six of the eight words become the unknown token. At the bottom: invariant globalization, used by Alpine and Ubuntu Chiseled .NET images and turned on by the .NET 10 Native AOT templates, makes accent stripping silently fail on precomposed text; Semantic Kernel scores 1.000 with ICU and 0.333 without it.](/images/embedding-parity/parity-before-the-tensor.svg)

**Every divergence I measured happens before the input tensor**, so a tensor-level check can pass while the text-to-vector pipeline is still wrong.

## Three libraries, and all three diverge

To judge a library you need something to judge it against, so I wrote the whole pipeline out in Python and numpy: tokenize, run the model, average, normalize. Each of the ten probe texts is cut at 256 tokens, the default limit Sentence Transformers sets for this model; the underlying transformer accepts up to 512. The score in every table below is **cosine similarity** between two vectors, where 1.000000 means they point in exactly the same direction and anything lower means the .NET library and Python disagree about the same sentence.

But a yardstick is only useful if it's straight. So I checked mine against sentence-transformers 6.0.1 itself — the standard Python implementation, running the original PyTorch weights, no ONNX anywhere. Across all ten texts the two never fall below a cosine of **0.99999996** ([`results/control-sentence-transformers.json`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/results/control-sentence-transformers.json)). When a library diverges below, the yardstick isn't what's wrong.

One more thing had to be ruled out first: a library loading a *different* copy of the model would explain every difference on its own. For SK and ElBruno I hashed the file each one actually loaded, and both matched the reference's sha256 exactly. LMSupply doesn't expose the path it loads from, so I checked its download cache instead, and it holds that same file and nothing else.

With those key aspects out of the way, let's take a closer look at the table of results. 

Quick primer to aid understanding the results: A BERT tokenizer does two jobs in order. First it tidies the text: lowercase it, split punctuation off the words, strip accents. Then it looks up what's left in a fixed vocabulary of about 30,000 pieces. Two pieces of notation show up below: `[UNK]` is the "unknown word" token, used when nothing in the vocabulary matches, and `##` marks a fragment stuck onto the piece before it.

The two SK rows are the same code built two ways: "ICU" is .NET's default, and "invariant" is the globalization-invariant mode those container images need, explained further down. Bold marks a defect ([`RESULTS.md`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/RESULTS.md), [`results/summary.md`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/results/summary.md), [`results/invariant-globalization/summary.md`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/results/invariant-globalization/summary.md)):

| | short ASCII | case + punctuation | realistic passage | accents | newline / tab | CJK + emoji |
|---|--:|--:|--:|--:|--:|--:|
| SK `Connectors.Onnx` 1.80.1-alpha, ICU | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| SK `Connectors.Onnx` 1.80.1-alpha, invariant | 1.000000 | 1.000000 | 1.000000 | **0.333027** | 1.000000 | 1.000000 |
| [LMSupply.Embedder](https://github.com/iyulab/lm-supply) 0.68.0 | 1.000000 | **0.671828** | **0.420103** | **−0.001147** | 1.000000 | **0.517515** |
| LMSupply.Embedder 0.70.0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| ElBruno.LocalEmbeddings 1.6.1 | 1.000000 | 1.000000 | 1.000000 | **0.333027** | **0.936251** | 0.996960 |

The divergence in each row has a different cause:

- **LMSupply 0.68.0 did the second job without the first.** No lowercasing, no punctuation splitting, no accent stripping — so anything beyond plain lowercase words came apart. `The QUICK … didn't … dog?!` turned into `[UNK] [UNK] … didn ##' ##t … dog ##? ##!`: the capitalized words simply weren't in the vocabulary, which only holds lowercase forms. An ordinary English paragraph, no accents or exotic characters anywhere, scored 0.42. I reported it, the maintainer confirmed it, and **0.70.0 came out about ten hours later**, exact on every text up to 256 tokens.
- **ElBruno 1.6.1 has two defects, and both come from the `Microsoft.ML.Tokenizers` package it is built on.** One: that package strips accents only when you set its `RemoveNonSpacingMarks` option, the option is off by default, and ElBruno never turns it on. This model's vocabulary holds `cafe` and not `café`, so every accented word falls through to `[UNK]`. An upstream design review suggests the default itself is deliberate, which makes this a question of which setting a library should choose rather than a bug in the package. Two: a line break or tab with no space beside it is deleted rather than treated as a space, welding two words together — `one\nline` becomes `one ##line`.
- **Semantic Kernel is exact with ICU present** — 1.000000 on every probe. Take ICU away and the accents break. That's the next section, and it's the part that surprised me the most.

**Note that Pooling is correct in all three, and I established that by measurement, not by reading code.** For ElBruno and LMSupply, I captured each library's own token ids and re-embedded them through the reference pooling, which reproduces the library's vectors at **1.000000** ([`reference/verify_tokens.py`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/reference/verify_tokens.py), [`results/token-diff-elbruno.json`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/results/token-diff-elbruno.json), [`results/token-diff-lmsupply.json`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/results/token-diff-lmsupply.json)). SK is exact under ICU. Its invariant-mode vectors are reproduced at ≥ 0.99999 by the reference tokenizer with accent stripping turned off ([`results/attribution.json`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/results/attribution.json)). One more thing, which isn't a defect: all three truncate at 512 tokens, the architecture's limit, rather than the 256 Sentence Transformers uses for this model. It's a policy difference, and the table leaves it out.

I wrote down predictions before any probe ran ([`PREDICTIONS.md`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/PREDICTIONS.md)). They were recorded in a private repository, so their date is stated, not demonstrated by git history. Of 19 predictions, 13 were met, 1 was half refuted, 4 were refuted, and 1 was left open. **Every refutation came from tokenization.** I had read the pooling code, and it was fine. The tokenizer defaults are where I guessed wrong.

## What 0.333 costs retrieval

A cosine of 0.333 on one sentence is a vector-level number. What a developer actually feels is retrieval quality. So the follow-up experiment asks one question: with the default English model, what does a tokenizer that doesn't strip accents cost recall?

The design isolates that one variable ([`recall/PREDICTIONS.md`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/recall/PREDICTIONS.md)). Two arms share the same ONNX session, the same pooling and the same 256-token truncation. They differ only in `strip_accents`. The unstripped arm reproduces ElBruno's and SK-under-invariant's `accents` vector at cosine 1.000000, and every run checks that before scoring. Queries and passages go through the same arm, as they would in an app that uses one library for both. Each corpus is subsampled to **300,000 passages**: every judged passage, plus random distractors. The metric is recall@10 with exact search. **The predictions were committed before any recall code was written**, and the repository history shows the order. Among the six, I predicted a drop of **at least 10%** relative for French and **at least 5%** for Spanish.

Results ([`recall/RESULTS.md`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/recall/RESULTS.md)):

| corpus | recall@10, correct | recall@10, unstripped | absolute change | relative change (95% interval) |
|---|--:|--:|--:|--:|
| MIRACL French | 0.409 | 0.264 | −14.5 points | **−35.5%** (−43.0% to −27.5%) |
| MIRACL Spanish | 0.282 | 0.221 | −6.1 points | **−21.6%** (−27.7% to −15.3%) |
| MIRACL German | 0.279 | 0.276 | −0.3 points | −1.0% (−9.4% to +7.8%): inconclusive |
| MS MARCO (English) | 0.913 | 0.909 | −0.4 points | −0.5% (−0.7% to −0.2%) |

**French loses 14.5 points of recall@10 (0.409 → 0.264), a 35.5% relative drop. Spanish loses 6.1 points (0.282 → 0.221), a 21.6% relative drop.** Both clear their pre-registered thresholds by a wide margin, and both intervals exclude the thresholds.

The absolute and relative numbers need to be read together, because **the baselines are low**. An English model finds only 28–41% of the relevant French, Spanish and German passages in its top 10, even with correct tokenization. The design fixed relative change as the headline in advance, because a given absolute loss is a bigger share of a weak baseline. The absolute column is there so you can check the relative figure against it. A developer who picks the default English model for French content starts from 0.409, and a tokenizer defect takes them to 0.264.

## The failure that depends on the deployment image

This is the finding I didn't go looking for.

Start with how a computer stores `é`. There are two ways. It can be one single character, or it can be a plain `e` followed by a separate accent mark — two characters that display as one. Both look identical on screen, and the first can be turned into the second. That conversion is called **decomposition**.

Accent stripping is built on it. Decompose the word, throw away the accent marks that fall out, and `café` becomes `cafe` — the form the vocabulary actually holds.

Now the .NET part. **Invariant globalization** is a mode that runs an app without ICU, the system library carrying Unicode and language data, in exchange for a smaller install. Without ICU, .NET cannot decompose anything, so `String.Normalize(NormalizationForm.FormD)` hands back whatever you gave it. That is [documented behaviour](https://github.com/dotnet/runtime/blob/6f4751a142ca0e879d60cb4091356bb9d346143e/docs/design/features/globalization-invariant-mode.md#string-normalization), not a bug. But it leaves accent stripping with nothing to strip: `é` stays `é`, the word isn't in the vocabulary, and out comes `[UNK]`. No exception, no warning, no log line.

I measured that on `Microsoft.ML.Tokenizers` 2.0.0's `BertTokenizer` by itself, with no embedding library wrapped around it, comparing its token ids against Hugging Face's tokenizer and scoring the cosine after embedding both ([`results/mltokenizers/accent-summary.md`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/results/mltokenizers/accent-summary.md)). The probe printed the giveaway as it ran: decomposing `é` yields **2 characters under ICU and 1 under invariant mode**.

One thing to be clear about first, because it is the same switch as before. Accent stripping is off unless you ask for it, and the switch that asks — `RemoveNonSpacingMarks` — belongs to `Microsoft.ML.Tokenizers`, not to any of the three libraries. ElBruno's accent defect is simply that it leaves the switch alone. Here I turn it **on**, which is the correct setting for this model, and change nothing else but whether ICU is present.

| probe, accent stripping **on** | with ICU | without ICU |
|---|--:|--:|
| accents, stored as one character | 1.000000 | **0.333027** |
| accents, stored as letter + mark | 1.000000 | 1.000000 |
| a French sentence | 1.000000 | **0.545866** |

Correct with ICU, broken without it — except for text that already arrives as a letter plus a separate mark, which survives because its marks are sitting right there to be thrown away. Nothing but the decomposition step appears to need ICU.

Now hold ICU steady and flip the switch instead:

| probe, **with ICU** | stripping off | stripping on |
|---|--:|--:|
| accents, stored as one character | **0.333027** | 1.000000 |
| accents, stored as letter + mark | **0.333027** | 1.000000 |
| a French sentence | **0.545866** | 1.000000 |

That is the switch working. Now put the two tables next to each other:

- **With ICU, the switch matters.** Off gives 0.333027, on gives 1.000000.
- **Without ICU, the switch changes nothing.** Turn it on and the accented sentence still scores 0.333027, the French one still 0.545866 — the very numbers you get with it off.
- **Text that arrives as a letter plus a mark is the exception.** Its marks need no decomposing, so with the switch on it comes out right with or without ICU.

So getting this right takes both: the switch on **and** ICU present. Miss either one and accented words become `[UNK]`, silently.

That is the whole bug. You asked for accent stripping, the accents survived anyway, and nothing threw or logged. Two things decide whether it works, and your code states neither of them: the image you deploy into, and the form the text arrives in.

I reported this as [dotnet/machinelearning#7728](https://github.com/dotnet/machinelearning/issues/7728). All seven predictions for this probe (MT1–MT7 in [`PREDICTIONS.md`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/PREDICTIONS.md)) were committed before it ran: six were met, and one was partly refuted because two of my own predictions contradicted each other ([`RESULTS.md`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/RESULTS.md)).

Semantic Kernel lands on that same 0.333027 without ICU, which fits the same explanation. I never opened SK's tokenizer code, though, so I'll claim only what I measured: running the text through a tokenizer that skips accent stripping reproduces SK's no-ICU vectors.

Invariant globalization is common where .NET is deployed:

- **Container images.** Microsoft's Alpine and Ubuntu Chiseled .NET images don't include ICU and "only work with apps that are configured for globalization-invariant mode". Their `extra` variants add it ([`dotnet-docker` image variants](https://github.com/dotnet/dotnet-docker/blob/7a1cdd5dd426ae782d7304ab8af476855790186a/documentation/image-variants.md)).
- **Native AOT templates.** `dotnet new console --aot`, `dotnet new worker --aot` and `dotnet new webapiaot` in .NET 10 all set `<InvariantGlobalization>true</InvariantGlobalization>` ([console](https://github.com/dotnet/sdk/blob/fd7d9df34dec5bf71ff2ad9869335644a33b9925/template_feed/Microsoft.DotNet.Common.ProjectTemplates.10.0/content/ConsoleApplication-CSharp/Company.ConsoleApplication1.csproj), [worker](https://github.com/dotnet/aspnetcore/blob/0ef4bbfa3291b306a21e5001cb0491277bdd35bc/src/ProjectTemplates/Web.ProjectTemplates/Worker-CSharp.csproj.in), [webapiaot](https://github.com/dotnet/aspnetcore/blob/0ef4bbfa3291b306a21e5001cb0491277bdd35bc/src/ProjectTemplates/Web.ProjectTemplates/WebApiAot-CSharp.csproj.in)).

So the same code is exact on a developer's machine and wrong in the image that ships. Which leads to the line I'd put on a sticky note: **an oracle runs in the test environment, not the deployment image.** The reference you check yourself against — the oracle — lives where your tests run. Your code doesn't. A parity test that passes on your laptop, where ICU is present, says nothing about a Chiseled container where it isn't.

You can't fix ICU's absence from inside the app, but you can refuse to pretend. The cheapest defence is one check at startup: decompose an `é` and see whether you get two characters back. The repository's [`invariant/Program.cs`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/invariant/Program.cs) demonstrates it (abridged):

```csharp
// Refuse to start rather than embed wrongly in silence.
if ("\u00e9".Normalize(NormalizationForm.FormD).Length != 2)
{
    Console.Error.WriteLine("Refusing to start: Unicode decomposition is unavailable " +
        "(invariant globalization), so accent stripping would silently do nothing.");
    return 1;
}
```

`dotnet run --project invariant` refuses to start. `dotnet run --project invariant -p:InvariantGlobalization=false` does not.

## How I found it: a harness error I kept

I didn't design an invariant-globalization experiment. My first run set `InvariantGlobalization=true` in `Directory.Build.props` out of habit, not as a decision. SK then failed the `accents` probe at **0.333027, exactly ElBruno's value**, to six decimal places. That coincidence was too clean to ignore. I re-ran under .NET's default globalization and SK was exact. The invariant-mode results stay in the repository ([`results/invariant-globalization/`](https://github.com/rkishore/dotnet-embedding-parity/tree/main/results/invariant-globalization)), because the mistake is the configuration many deployments actually run.

## What this doesn't show

**First, who this touches.** It would be easy to read all this as an accents problem. Accents are the clearest case, and the only one I measured for retrieval, but plain English broke too. LMSupply 0.68.0 mangled any word with a capital letter or punctuation stuck to it, until the fix. ElBruno still welds together words split by a line break or tab, which happens in ordinary documents. The `Microsoft.ML.Tokenizers` tokenizer it builds on also drops symbols such as `$ + = <` and `°`, so prices, code and arithmetic are affected. And all three read up to 512 tokens where Sentence Transformers stops this model at 256, so long text diverges no matter which characters are in it.

**Second, what I didn't look at.** Everything here is .NET, one model, and one kind of vocabulary: uncased all-MiniLM-L6-v2, which expects accents to be removed. A cased model, or one using a different tokenizer scheme, would behave differently. The missing-ICU failure is specific to .NET, since it comes from what `String.Normalize` does without ICU. Whether other language ecosystems have tokenizer gaps of their own is a fair question, and one this experiment says nothing about.

- **German is inconclusive.** Recall@10 moved from 0.279 to 0.276, −0.3 points and −1.0% relative. The interval runs from a 9.4% loss to a 7.8% gain. I predicted a drop of at least 5%. That's scored as refuted, but 305 queries can't confirm or rule out an effect that size. It is not evidence that German is unaffected.
- **The English control's small effect comes entirely from mojibake.** MS MARCO lost 0.4 points (0.913 → 0.909, −0.5%). Every one of the 68,981 passages that tokenize differently contains UTF-8-read-as-Latin-1 debris like `Ã©`. With the mojibake removed, none still differ, and no dev query has a relevant passage with a genuinely accented character ([`recall/results/posthoc.json`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/recall/results/posthoc.json)). So the control shows how each arm handles encoding damage. It says nothing about English text with real accents. A pre-registered English subset (−1.76 points, −2.0%) turned out to measure the same mojibake and nothing else.
- **The Spanish interrogative observation is post hoc.** Every Spanish dev query tokenizes differently between the arms, because MIRACL's Spanish queries are questions and the interrogatives carry accents (`¿Cómo`, `¿Cuáles`, `qué`): 1.93 query tokens become `[UNK]` on average, against 0.87 for French ([`recall/results/posthoc.json`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/recall/results/posthoc.json)). Yet Spanish drops less than French. Counting broken query tokens overstates the damage; a plausible reason is that question words carry little retrieval signal. I noticed this after the results were in, and haven't tested it.
- **One model, subsampled corpora.** Everything uses the English all-MiniLM-L6-v2, which is the realistic default but a weak model for these languages. A multilingual model would have different baselines, and I didn't test one. Recall is measured on 300,000-passage subsamples, not the full corpora of 8.8–16 million passages.
- **The recall arms isolate accents.** They reproduce the accent handling of ElBruno and SK-under-invariant, but not ElBruno's newline gluing or other defects.
- **Scorecard:** 4 of the 6 recall predictions were met, and 2 were refuted (German, and the mojibake subset at −1.97% against a −2% threshold).

## Upstream

All five reports include reproductions and token-level evidence:

- [iyulab/lm-supply#12](https://github.com/iyulab/lm-supply/issues/12): embeddings diverge because of WordPiece tokenization. **Fixed in 0.70.0**, published about ten hours after the report, and closed.
- [elbruno/elbruno.localembeddings#56](https://github.com/elbruno/elbruno.localembeddings/issues/56): accented words become `[UNK]` with the default model. Open.
- [dotnet/machinelearning#7724](https://github.com/dotnet/machinelearning/issues/7724): `BertTokenizer` merges words separated only by `\n`, `\t` or `\r`. Open.
- [dotnet/machinelearning#7725](https://github.com/dotnet/machinelearning/issues/7725): `BertTokenizer` silently drops symbol characters, including `$ + = < >` and emoji. Open.
- [dotnet/machinelearning#7728](https://github.com/dotnet/machinelearning/issues/7728): `RemoveNonSpacingMarks = true` fails to strip accents from precomposed characters under invariant globalization. Open.

## The one-liner

**Parity at the tensor isn't parity at the text, and parity on your laptop isn't parity in the image you ship.** If you embed text in .NET, compare token ids against the reference tokenizer, not just output vectors. Include accented, newline and symbol inputs in that check, and run it in the same globalization mode as production. Or refuse to start when decomposition is unavailable.

The probes, runners, predictions, results and reproduction steps are all at [github.com/rkishore/dotnet-embedding-parity](https://github.com/rkishore/dotnet-embedding-parity).
