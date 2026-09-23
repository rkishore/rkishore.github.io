---
title: "The Divergence Before the Tensor: Embedding Parity in .NET"
description: "I ran three .NET embedding libraries against sentence-transformers on the same model. All three diverge in at least one configuration, and every divergence is in tokenization, not pooling. Unstripped accents cost French recall@10 14.5 points (0.409 → 0.264, −35.5% relative), and under invariant globalization accent stripping silently fails on precomposed text — so a parity check on your laptop says nothing about your container."
date: 2026-09-22 20:30:00 -0400
---

**Objective:** A measured look at embedding parity in .NET, organized around one question: *three .NET libraries run the same model as sentence-transformers, so why do their vectors still differ, and what does that cost retrieval?* Everything below comes from a public evidence repository, [rkishore/dotnet-embedding-parity](https://github.com/rkishore/dotnet-embedding-parity), and each result links to the file it comes from.

Here is one sentence: **`Café crème brûlée in São Paulo, naïve résumé`**. Embed it with `sentence-transformers/all-MiniLM-L6-v2` in Python, then embed it with the model's ONNX export in .NET, and compare the two vectors. In [ElBruno.LocalEmbeddings](https://github.com/elbruno/elbruno.localembeddings) 1.6.1 the cosine is **0.333**. Same model, same weights, same pooling math, and a third of a match.

[Semantic Kernel](https://github.com/microsoft/semantic-kernel) gets that same sentence exactly right: **1.000000**, on my machine. Build the identical code the way Microsoft's slimmed-down .NET container images require — they ship without ICU, the system's Unicode library — and it scores **0.333** too. Nothing throws and nothing warns. That one has a section of its own below.

A disclosure first: I wrote .NET embedding code of my own for learning purposes, and that work led me here. I'll write about it separately. This post covers only the three external libraries and the tokenizer package underneath one of them.

## The one idea: the divergence lives before the tensor

An embedding pipeline has four steps, and only the last two involve the neural network:

1. **Text:** what the user typed.
2. **Tokenizer:** turns text into token ids by lowercasing, splitting off punctuation, stripping accents, and looking up each piece in a fixed vocabulary. (If subword tokenization is new to you, Hugging Face's [tokenizer summary](https://huggingface.co/docs/transformers/en/tokenizer_summary) covers WordPiece, the scheme BERT uses. I've seen what it costs retrieval before, when it [shattered a rare acronym](/2026/07/16/building-the-hybrid-retriever.html) into promiscuous fragments.)
3. **Model:** turns those ids into one vector per token.
4. **Pooling:** averages them into the single vector you store and search.

![An embedding pipeline in four steps: text, tokenizer, model, pooling. A dashed line between the tokenizer and the model marks the input tensor. Every divergence measured sat to the left of it, in the tokenizer; tensor-level parity checks start to the right of it, and the model and pooling steps were cleared because each library's own token ids reproduce its vectors exactly. Below, the sentence "Café crème brûlée in São Paulo, naïve résumé" through two tokenizers: sentence-transformers gives "cafe cr ##eme br ##ule ##e in sao paulo , naive resume", cosine 1.000; ElBruno 1.6.1 and Semantic Kernel under invariant globalization give "[UNK] [UNK] [UNK] in [UNK] paulo , [UNK] [UNK]", cosine 0.333, because six of the eight words become the unknown token. At the bottom: invariant globalization, used by Alpine and Ubuntu Chiseled .NET images and turned on by the .NET 10 Native AOT templates, makes accent stripping silently fail on precomposed text; Semantic Kernel scores 1.000 with ICU and 0.333 without it.](/images/embedding-parity/parity-before-the-tensor.svg)

When you port a model, the usual parity check is to feed the same ids to both runtimes and compare the outputs. That check starts at the input tensor, and it takes the token ids as given. **Every divergence I measured happens before that point**, so a tensor-level check can pass while the text-to-vector pipeline is still wrong.

## Three libraries, and all three diverge

I compared each library against a numpy reference on ten probe texts, using the model's own 256-token truncation. The reference is checked against sentence-transformers 6.0.1 running the PyTorch weights, with no ONNX involved. The minimum cosine is **0.99999996** ([`results/control-sentence-transformers.json`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/results/control-sentence-transformers.json)). SK and ElBruno loaded a model file byte-identical to the reference's by sha256; for LMSupply, which doesn't expose its model path, its download cache holds only that same file. The two SK rows are the same code built two ways: "ICU" is .NET's default, and "invariant" is the globalization-invariant mode those container images need, explained further down. Bold marks a defect ([`RESULTS.md`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/RESULTS.md), [`results/summary.md`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/results/summary.md), [`results/invariant-globalization/summary.md`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/results/invariant-globalization/summary.md)):

| | short ASCII | case + punctuation | realistic passage | accents | newline / tab | CJK + emoji |
|---|--:|--:|--:|--:|--:|--:|
| SK `Connectors.Onnx` 1.80.1-alpha, ICU | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| SK `Connectors.Onnx` 1.80.1-alpha, invariant | 1.000000 | 1.000000 | 1.000000 | **0.333027** | 1.000000 | 1.000000 |
| [LMSupply.Embedder](https://github.com/iyulab/lm-supply) 0.68.0 | 1.000000 | **0.671828** | **0.420103** | **−0.001147** | 1.000000 | **0.517515** |
| LMSupply.Embedder 0.70.0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| ElBruno.LocalEmbeddings 1.6.1 | 1.000000 | 1.000000 | 1.000000 | **0.333027** | **0.936251** | 0.996960 |

Each row has a different cause:

- **LMSupply 0.68.0 skipped BERT's basic tokenization entirely.** It had no lowercasing, no punctuation splitting and no accent stripping, so `The QUICK … didn't … dog?!` became `[UNK] [UNK] … didn ##' ##t … dog ##? ##!`. A plain 109-token English paragraph scored 0.42. The maintainer confirmed the report and published **0.70.0 about ten hours after it was filed**, and it's exact on every text up to 256 tokens.
- **ElBruno 1.6.1 has two defects**, both inherited from `Microsoft.ML.Tokenizers`. Accented words become `[UNK]` because `RemoveNonSpacingMarks` defaults to `false`. That's an option default, and upstream design review suggests it's intentional. Separately, a word after a bare `\n` or `\t` gets glued onto the previous one: `one\nline` becomes `one ##line`.
- **Semantic Kernel is exact under .NET's default globalization** and loses accents under the invariant mode. More on that below, because it's the part that surprised me most.

**Pooling is correct in all three, and I established that by measurement, not by reading code.** For ElBruno and LMSupply, I captured each library's own token ids and re-embedded them through the reference pooling, which reproduces the library's vectors at **1.000000** ([`reference/verify_tokens.py`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/reference/verify_tokens.py), [`results/token-diff-elbruno.json`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/results/token-diff-elbruno.json), [`results/token-diff-lmsupply.json`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/results/token-diff-lmsupply.json)). SK is exact under ICU. Its invariant-mode vectors are reproduced at ≥ 0.99999 by the reference tokenizer with accent stripping turned off ([`results/attribution.json`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/results/attribution.json)). One more thing, which isn't a defect: all three truncate at 512 tokens rather than the model's configured 256. It's a policy difference, and the table leaves it out.

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

.NET has a mode called **invariant globalization**. The app runs without ICU, the Unicode and culture library, and gets a smaller footprint in return. Under that mode, `String.Normalize(NormalizationForm.FormD)` returns its input unchanged. That's [documented behaviour](https://github.com/dotnet/runtime/blob/6f4751a142ca0e879d60cb4091356bb9d346143e/docs/design/features/globalization-invariant-mode.md#string-normalization), not a bug. But BERT-style accent stripping works by decomposing `é` into `e` plus a combining accent (Form D) and then dropping the accent. If decomposition does nothing, there's no accent to drop, `é` stays `é`, and the word misses the vocabulary and becomes `[UNK]`. Nothing throws, and nothing warns.

I measured this directly in `Microsoft.ML.Tokenizers` 2.0.0's `BertTokenizer` on its own, with no library around it. I compared token ids against Hugging Face and scored cosine after embedding ([`results/mltokenizers/accent-summary.md`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/results/mltokenizers/accent-summary.md)). The probe process recorded `"\u00e9".Normalize(FormD).Length` as **2 under ICU and 1 under invariant mode**.

| probe | ICU, default | ICU, `RemoveNonSpacingMarks` | invariant, default | invariant, `RemoveNonSpacingMarks` |
|---|--:|--:|--:|--:|
| accents, precomposed | **0.333027** | 1.000000 | **0.333027** | **0.333027** |
| accents, decomposed | **0.333027** | 1.000000 | **0.333027** | 1.000000 |
| a French sentence | **0.545866** | 1.000000 | **0.545866** | **0.545866** |

Read the last column. **Under invariant globalization, `RemoveNonSpacingMarks = true` produces exactly the ids of the option being off**, on every precomposed probe. It still works on text that arrives already decomposed, presumably because the mark test doesn't need ICU; only the decomposition does. So whether accent stripping works depends on two things the code doesn't control: the deployment image and the input's normalization form. I reported it as [dotnet/machinelearning#7728](https://github.com/dotnet/machinelearning/issues/7728), and every one of the seven predictions for this probe (MT1–MT7 in [`PREDICTIONS.md`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/PREDICTIONS.md)) was committed before the probe was run. Six were met. One was refuted in part, because two of my own predictions contradicted each other ([`RESULTS.md`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/RESULTS.md)).

Semantic Kernel shows the same 0.333027 under invariant mode. That's consistent with the same mechanism, but I didn't read or instrument SK's tokenizer, so I'm claiming only the measurement: SK's invariant-mode vectors are reproduced by tokenizing without accent stripping.

Invariant globalization is common where .NET is deployed:

- **Container images.** Microsoft's Alpine and Ubuntu Chiseled .NET images don't include ICU and "only work with apps that are configured for globalization-invariant mode". Their `extra` variants add it ([`dotnet-docker` image variants](https://github.com/dotnet/dotnet-docker/blob/7a1cdd5dd426ae782d7304ab8af476855790186a/documentation/image-variants.md)).
- **Native AOT templates.** `dotnet new console --aot`, `dotnet new worker --aot` and `dotnet new webapiaot` in .NET 10 all set `<InvariantGlobalization>true</InvariantGlobalization>` ([console](https://github.com/dotnet/sdk/blob/fd7d9df34dec5bf71ff2ad9869335644a33b9925/template_feed/Microsoft.DotNet.Common.ProjectTemplates.10.0/content/ConsoleApplication-CSharp/Company.ConsoleApplication1.csproj), [worker](https://github.com/dotnet/aspnetcore/blob/0ef4bbfa3291b306a21e5001cb0491277bdd35bc/src/ProjectTemplates/Web.ProjectTemplates/Worker-CSharp.csproj.in), [webapiaot](https://github.com/dotnet/aspnetcore/blob/0ef4bbfa3291b306a21e5001cb0491277bdd35bc/src/ProjectTemplates/Web.ProjectTemplates/WebApiAot-CSharp.csproj.in)).

So the same code can be exact on a developer's machine and wrong in the image that ships. Which leads to the line I'd put on a sticky note: **an oracle runs in the test environment, not the deployment image.** A parity test that passes on your laptop, under ICU, tells you nothing about a Chiseled container.

The cheapest defence is one line at startup. The repository's [`invariant/Program.cs`](https://github.com/rkishore/dotnet-embedding-parity/blob/main/invariant/Program.cs) demonstrates it (abridged):

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
