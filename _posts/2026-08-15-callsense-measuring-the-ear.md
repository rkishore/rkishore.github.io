---
date: 2026-08-15 18:30:00 -0400
title: "Measuring the Ear"
description: "Part 3 of the callsense series: everything downstream reads Whisper's transcript, so I measured it before trusting it. Why the confidence threshold moved from 0.6 to 0.3 and back, why timing gaps can't tell speakers apart, and how the size of the speech model decides what PII redaction can catch."
---

{% include info.html text="This content was written with assistance from Claude AI." %}

*callsense series &mdash; 1. [The Problem](/2026/08/05/callsense-the-call-nobody-reviews.html) &middot; 2. [Six Slots, One Pipeline](/2026/08/10/callsense-six-slots-one-pipeline.html) &middot; 3. Measuring the Ear (you're here) &middot; 4. [When the Input Is the Prompt](/2026/08/20/callsense-when-the-input-is-the-prompt.html) &middot; 5. [Don't Let the Model Do the Math](/2026/08/25/callsense-dont-let-the-model-do-the-math.html) &middot; 6. [From Clean Clone to Deployed](/2026/08/30/callsense-from-clean-clone-to-deployed.html) &middot; 7. [The Guard as a Tool](/2026/10/30/callsense-the-guard-as-a-tool.html) &middot; 8. [Measuring the Guard](/2026/11/04/callsense-measuring-the-guard.html)*

**Objective:** [Part 2](/2026/08/10/callsense-six-slots-one-pipeline.html) drew seven stages and noted that five never call a language model. One of those five feeds everything after it: transcription. The injection check, the PII redactor and both LLM calls only ever see what Whisper wrote down. This part is the measurements behind that stage, organized around one question: *how far can a speech model's transcript be trusted, and how do you find out before building on it?*

Whisper is an open speech-to-text model. callsense runs it locally through a library called faster-whisper, and it comes in sizes: `tiny`, `base`, `small` and up. Bigger is slower and usually more accurate. Most of what follows happened on `tiny`, the smallest and the default.

## 1. Spikes before code

A *spike* is a throwaway script that answers one question before real code depends on the answer. I kept mine in a `_spikes/` folder, because they are the evidence behind decisions that would otherwise look arbitrary.

The first spike asked: what does faster-whisper actually return? I ran all 10 reference calls through `tiny`: **771 segments, 55 minutes of audio.** A segment is one stretch of speech with a start time, an end time and some confidence numbers.

The first surprise was small. `transcribe()` returns a *generator*, which is a lazy sequence that does its work only when you loop over it. My first call came back in 0.29 seconds having transcribed nothing. The real work happens during iteration, so the code that follows wraps it in `list(...)` and says why in a comment.

The speed was better than I expected: 55 minutes of audio in 45 seconds on a CPU, about **74 times faster than real time**. Re-running the whole corpus costs under a minute, which is what makes measuring cheap.

Whisper also invents things now and then. Across 55 minutes I found one stray "thank you for watching" and three words repeated three times. Rare, but real, so the transcript cleaner is not decoration.

Each segment gets a confidence score. Whisper provides two raw numbers, `avg_logprob` (how likely the model thought its own words were) and `no_speech_prob` (how likely the audio was silence). callsense blends them:

`confidence = 0.7 × clamp(1 + avg_logprob) + 0.3 × (1 − no_speech_prob)`

*Clamp* means "squeeze into the range 0 to 1". Segments scoring below `CONFIDENCE_THRESHOLD` get a `[LOW CONF]` marker in the transcript. The question is where to put that line.

## 2. The threshold that moved twice

*What number separates "unsure" from "fine"?*

I did not know, so the value went through three stages:

![Top: a timeline of three pills joined by arrows. On Aug 5 the confidence threshold was 0.6, a placeholder with a halt ratio of 0.5. On Aug 6 it became 0.3, the spec's value, on the assumption that the tiny model scores low. On Aug 9 it went back to 0.6, set from measurement. Below: a bar chart of segments under each threshold out of 771 from 10 calls. Below 0.3 the bar is empty, 0 or 0.0 percent, labelled never fires, dead code. Below 0.5 it is 12, 1.6 percent, in one file only. Below 0.6 it is 54, 7.0 percent, with 8 of 10 calls at zero, sample_02 at 32 and sample_10 at 22. A footer gives confidence min 0.412, mean 0.750, max 0.924.](/images/callsense/callsense-confidence-threshold.svg)

- **Aug 5: 0.6, a placeholder.** I put a value in `.env.example` so the file was complete. It was a guess.
- **Aug 6: 0.3, the spec's value.** The commit message says `tiny` "routinely scores segments below 0.6", so 0.6 would flag nearly every transcript. The spec used 0.3, and I changed to match. I had not measured anything.
- **Aug 9: back to 0.6, after measuring.** Confidence across the 771 segments ran from a minimum of **0.412** to a maximum of **0.924**, with a mean of **0.750**.

| Threshold | Segments below it |
|---|---|
| 0.3 | 0 of 771 (0.0%) |
| 0.5 | 12 of 771 (1.6%) |
| 0.6 | 54 of 771 (7.0%) |

At 0.3 the flag can never fire. It is dead code. At 0.6 it does what a threshold is for: it separates *calls*. Eight of the ten have zero low segments, while `sample_02` has 32 and `sample_10` has 22. The assumption behind Aug 6 was simply wrong: about 93% of segments sit above 0.6.

Two more things the numbers showed.

**The signal is coarser than "per segment" suggests.** Whisper computes `avg_logprob` and `no_speech_prob` once per 30-second decode window, not per segment. The 771 segments share only **111 distinct values**, about seven segments each. So `[LOW CONF]` markers show up in blocks, not singly.

**The "halt ratio" halts nothing.** A second setting, `LOW_CONFIDENCE_HALT_RATIO` (0.8), is the share of low segments above which the whole transcript counts as unreliable. Despite the name, that only sets a flag for display. No routing step reads it, so the pipeline continues to the summary either way. That is the "flag nothing routes on" from Part 2. I left it as it is and wrote that down in `.env.example`, so the name does not mislead the next reader.

**Verdict:** a threshold has to be set from measurement, and this one tells you *when Whisper is unsure*, not *whether it is wrong*. Those are different claims.

## 3. Who said it: a gap is not a speaker

*Can silence tell you who is talking?*

*Diarization* means labelling each line of a transcript with its speaker, here *agent* or *customer*. The simplest signal is timing. The spec's rule: a silence longer than 1.2 seconds between segments means the speaker changed.

On the whole corpus that rule fires on 16% of segment boundaries. That average describes no actual call. Per call, the rate runs from **3% to 38%**, a 12-fold spread.

To find out why, I hand-labelled two calls. Two terms first. *Recall* is the share of real speaker changes the rule caught. *Precision* is the share of the rule's firings that were real changes.

| Call | Style | Changes caught | Recall | Precision |
|---|---|---|---|---|
| `sample_03` | slow, formal bank call | 8 of 12 | 67% | 80% |
| `sample_01` | fast sales call | 1 of 8 | 12.5% | not recorded |

On the fast call, the gaps at real speaker changes ran 0.26&ndash;1.24 s, and the gaps *within* one speaker's turn ran 0.00&ndash;1.24 s. The two ranges overlap completely. The difference is the pace of the conversation, not noise. The gap signal is not weak. It is *inconsistent*, which is worse, because you can't tell which kind of call you're on.

So the spec ranks content patterns above timing: phrases that mark a speaker, like "thank you for calling". They turned out to need the same discipline.

- **They only anchor.** Patterns matched just 11% of segments. The other 89% get their label by carrying the current speaker forward.
- **The best one is strong.** "thank you for calling" has 15 hits across 10 of 10 calls.
- **Patterns must match what Whisper writes, not what people say.** "assist you" has 14 hits in 6 calls. "how can I help" has 1, because Whisper mangles it into things like "How can you help you today?". Seven of the 30 phrases I imagined had zero hits at all.
- **High frequency is not discrimination.** "my name is" appears 13 times in 8 calls, but both speakers say it. I excluded it.
- **Segment 0 is not always the agent.** Five of the ten calls open with a narrator, because they are training recordings. That is corpus-specific, but it would have mislabelled half this set.

**Verdict:** build the pattern list from the transcripts the model produces, never from your idea of how people talk.

## 4. The redactor reads the transcript, not the audio

*What happens when PII is read out loud?*

Part 4 covers the redactor itself. Here, only what the transcript does to it. A redactor is a set of regular expressions (*regex*: text patterns) that match card numbers, SSNs, emails and phone numbers in the text. It can only match what Whisper wrote.

On Aug 14 I recorded text-to-speech demo clips. I dictated a test card, `4111-1111-1111-1111`. Whisper wrote it back as **25 digits**, and as **19** when read with pauses. Never 16. Repeated digits give the model no acoustic landmarks to count with.

That broke redaction in a quiet way. The card pattern had an upper limit sized to a real card, so it failed on 25 digits. The phone pattern then matched a 10-digit run *inside* the card, redacted that tail, and left the rest of the card in the clear under the wrong label. The fix was to raise the ceiling to 40 digits, so an over-long run is swallowed whole. A noisier transcript beats a leak. The floor of 13 digits stays, because it is what keeps a card apart from a 9-digit SSN or a 10-digit phone number.

Model size matters too. On `tiny`, a different dictated test card, `4539-8712-3456-7890`, came out as `4.539-8712-3456-789-0`. The leading "4." sits outside the matched span and escapes. On `base`, the next size up, it came out clean. So `make demo` pins `base`.

Spoken-word digits are the other hole. If a caller's number is transcribed as "four one one one", a digit pattern matches nothing. In practice, every real spoken number in the ten calls came out as digits, including the spoken phone numbers in three of them. So this is a tail case, but a genuine hole.

**Verdict:** the redactor is only as good as the transcript, so test it on what the ear writes, not on what you typed.

## 5. How far it can be trusted, as of now

Here is the honest ledger.

| Measured | Not measured |
|---|---|
| Confidence distribution over 771 segments | **Word error rate (WER)**: the share of words wrong against a human transcript |
| How often the 1.2 s gap fires, per call | Speaker labels beyond two hand-labelled calls |
| Pattern hit counts across 10 calls | Anything on audio outside these 10 calls |
| Two calls, hand-labelled for speaker changes | Model size on real calls |
| Model size on one dictated card clip | Whether the low-confidence calls are the ones with more wrong words |

WER is the standard measure of a speech model. I have not computed it. The confidence score tells me when Whisper is unsure. It does not tell me when Whisper is wrong, and the two can differ. Everything above measures *behaviour*, not *correctness*.

**Verdict:** trusted enough to build on, with known holes. The holes are written down, which is the point.

## *Update, August 2026: the bigger model made it worse*

On Aug 19 I ran the app in a browser for the first time and found something the tests could not. The spoken phone number in `sample_01` was not redacted. The redactor was fine. The ear had changed what it wrote.

`base` and `small` hear "eight oh one" and write it as a letter-prefixed reference code: `A-01-XXX-XXXX` on `base`, `A01-XXX-XXXX` on `small`. No phone pattern matches that. `tiny` just writes the digits, and they redact.

![A grid of three Whisper sizes against two clips. Dictated card: tiny wrote 4.539-8712-3456-789-0, and the leading 4. escapes, shown in coral; base wrote 4539-8712-3456-7890 and it was redacted, shown in green; small is grey, not measured. Spoken phone in sample_01: tiny wrote 801-XXX-XXXX and it was redacted, green; base wrote A-01-XXX-XXXX and it escapes, coral; small wrote A01-XXX-XXXX and it escapes, coral. A caption reads: no single size redacts both.](/images/callsense/callsense-model-size-redaction.svg)

No size gets both clips right. I chose not to widen the patterns to accept a letter prefix, because that would fit one transcription quirk and cost false positives on ordinary text. `base` wins because the card number is the stronger demo, and the demo runbook no longer promises phone-number redaction from `sample_01`.

The line I keep coming back to: **more capable is not the same as more literal.** The redactor needs the transcript to say what was said. A better model sometimes says what it *thinks was meant*.

## The one-liner that ties it together

Distilled: **the injection check, the redactor and the LLM all trust the transcript, so measure the ear first, and measure what it writes, not what was said.**

**Coming next:** Part 4, the two guards between that transcript and any model: an injection check that fails closed, and PII redaction that leaves no raw copy behind.
