---
date: 2026-08-10 18:30:00 -0400
title: "Six Slots, One Pipeline"
description: "Part 2 of the callsense series: the system design, split into seven stages and then laid out against six slots I believe every well-designed agentic system should have. Control plane, workers, checker, state, eval and audit log. What callsense chose for each, and the two slots where it's still thin."
---

*callsense series &mdash; 1. [The Problem](/2026/08/05/callsense-the-call-nobody-reviews.html) &middot; 2. Six Slots, One Pipeline (you're here) &middot; 3. [Measuring the Ear](/2026/08/15/callsense-measuring-the-ear.html) &middot; 4. [When the Input Is the Prompt](/2026/08/20/callsense-when-the-input-is-the-prompt.html) &middot; 5. [Don't Let the Model Do the Math](/2026/08/25/callsense-dont-let-the-model-do-the-math.html) &middot; 6. [From Clean Clone to Deployed](/2026/08/30/callsense-from-clean-clone-to-deployed.html) &middot; 7. [The Guard as a Tool](/2026/10/30/callsense-the-guard-as-a-tool.html) &middot; 8. [Measuring the Guard](/2026/11/04/callsense-measuring-the-guard.html)*

**Objective:** [Part 1](/2026/08/05/callsense-the-call-nobody-reviews.html) set out the problem: score every call, without letting a caller write to the prompt, without sending PII to a model, and without trusting a model to add up. This part is the design, organized around one question: *what does each part of an agentic system have to decide, and what did callsense decide for each?*

## From the problem to seven stages

Part 1 ended with two lists: five things to produce for every call, and three constraints that stop it from being one prompt. The first design step was to give each of those a home. That split the job into seven stages, run in order:

![The input, a call recording with caller ID and department, flows through seven stages left to right: intake, transcribe, injection check, redact PII, summarize, QA score and report. Under each stage is the model it uses and what it answers from Part 1. Intake uses no model and checks the input. Transcribe uses a speech model and produces output 1, the transcript. Injection check and redact PII use no model and enforce constraints 1 and 2: the input is untrusted, and PII stops at the door. Summarize is LLM call 1 and produces output 2, the summary. QA score is LLM call 2 and produces outputs 3 and 4, the score and compliance flags, and carries constraint 3, that the overall score is computed in code. Report uses no model and produces output 5, the status.](/images/callsense/callsense-seven-stages.svg)

Read left to right:

- **Intake** checks the upload: real audio by its magic bytes, within size and duration limits, and no PII in the caller ID or department fields.
- **Transcribe** turns speech into text and labels each line *agent* or *customer* (output 1).
- **Injection check** scans the transcript for instructions aimed at the reviewer, and halts the call on a match (constraint 1).
- **Redact PII** masks card numbers, SSNs, emails and phone numbers before any language model sees the text (constraint 2).
- **Summarize** writes the structured summary (output 2).
- **QA score** scores the five dimensions and raises compliance flags (outputs 3 and 4). Code, not the model, sets the overall score (constraint 3).
- **Report** assembles the result, sets the status and saves it (output 5).

Now look at the row under the stages. Five of the seven never call a language model: one runs a speech model locally, and the other four are plain code. When I first drew it that way it surprised me, because the brief reads like an LLM project. Most of the pipeline's work is guarding the model, not calling it.

## Six slots, each with a stated choice

Seven stages say what the pipeline does. They don't say how it's routed, checked, remembered, measured or audited. For that I use six slots.

From what I've learned so far, I believe every well-designed agentic system should have at least these six slots. Most systems end up with something in each, whether or not anyone named it. I treat a design as complete when each slot has a stated choice, even if the choice is *"not needed yet, because&hellip;"*.

| Slot | Its job | The question it forces |
|---|---|---|
| **Control plane** | Routes work, caps steps, decides when to stop | What stops this loop, and who picks the next step? |
| **Workers** | The model and tool calls that do the work | Does this step need a model, a tool, or neither? |
| **Checker** | Verifies output before it leaves | What would prove this wrong without asking the same model? |
| **State** | What the system remembers | Who writes each field, and what survives a crash? |
| **Eval** | The bar a change must clear | What number must hold before this ships? |
| **Audit log** | A durable record of every step | Can I replay why this output happened? |

Here is callsense against all six, with the same seven stages now sitting inside them:

![callsense laid out against the six slots. A control-plane band across the top reads: a fixed LangGraph pipeline, routing decided in code, not by a model, no loops, LLM retries capped at 3. Below it, seven stages left to right: intake (no model), transcribe (Whisper, a local speech model), injection check (a checker, 22 regexes), redact PII (no model), summarize (LLM call 1), QA score (LLM call 2, with its overall score set in code), and report (no model). An invalid upload branches down from intake to an error terminal; an injection match and a critical compliance flag both branch down to supervisor review. Three rails run underneath: state, the audit log with dots for STARTED, FAILED, FLAGGED and COMPLETED and a hollow dot where an LLM that gives up leaves no record, and eval, drawn dashed and marked not yet.](/images/callsense/callsense-six-slots.svg)

## 1. Control plane: a fixed pipeline, routed in code

The control plane is a [LangGraph](https://langchain-ai.github.io/langgraph/) state machine with eight nodes: seven stages and three terminals (report, error, supervisor review). Routing happens in four small functions in `edges.py`. Each one reads the state and returns a string, so no model ever picks the next step.

*What stops the loop?* There isn't one. Every path runs forward to a terminal, and every terminal edges to `END`. The only repetition is inside the two LLM stages, which make up to three attempts (by default) with a capped backoff.

Two of the routing choices are worth a sentence each:

- **A detected injection goes to supervisor review, not to the error node.** The brief's wiring said error. But "error" means *we couldn't analyse this call*, while a blocked injection means *we analysed it and stopped on purpose*. Merging them would count every blocked attack as a pipeline failure on the dashboard.
- **The route after transcription has only one outcome.** That's deliberate. It keeps every stage boundary the same shape and leaves a place for a future condition, such as halting on a low-confidence transcript, without rewiring the graph.

**Verdict:** the default I reach for, a fixed pipeline unless the roles truly differ, fits exactly. Nothing here needs a supervisor agent.

## 2. Workers: model, tool, or neither?

Asking that question of each stage is where the surprise came from:

| Stage | Needs | Why |
|---|---|---|
| Intake | neither | Format from magic bytes, size and duration limits, PII in metadata by regex |
| Transcribe | a speech model | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) run locally, plus heuristic speaker labels |
| Injection check | neither | 22 regex patterns |
| Redact PII | neither | Regex for SSNs, cards, emails, phones |
| Summarize | a language model | Structured output, validated by Pydantic |
| QA score | a language model | Five dimensions, each with a justification |
| Report | neither | Assemble, persist, log |

The two LLM calls run in sequence, not in parallel. QA scoring gets the summary as context, because a reviewer who knows the call was about a disputed charge judges the handling differently from one reading the transcript cold. No worker calls a tool. Nothing in this job needs the model to act, only to read and judge.

**Verdict:** one decision per worker, and the model is used only where judgement is the work.

## 3. Checker: deterministic gates first

The checker slot asks what would prove the output wrong *without asking the same model*. callsense answers with code, at two points:

- **Before the model:** the injection check is a fail-closed input guard. A match halts the call before either LLM sees the text. That's the reverse of the usual fail-open posture. Part 4 explains why it's right here; the general trade-off is in [Prompts Suggest, Guardrails Enforce](/2026/07/23/prompts-suggest-guardrails-enforce.html).
- **Before the report:** the model proposes an overall score, and the pipeline throws it away and recomputes it from the weighted dimensions. In the demo call the model said **3.0** and the recomputed score is **3.5**. Then, if any compliance flag is *critical*, the call goes to supervisor review however well it scored.

Supervisor review is the system's way of abstaining. It doesn't guess; it hands the call to a person.

What's missing is a separate critic. The prompt tells the model to ground every justification in the transcript, but nothing checks that it did. And the transcription stage already computes a "this transcript is unreliable" flag that nothing routes on.

**Verdict:** strong on deterministic checks, no critic yet. Part 5 is the full story of the score.

## 4. State: one dict per call, and two exceptions to "one writer per field"

State is a typed dict carried through the graph. Each node returns only the keys it wrote, and almost every key has exactly one writer. The two exceptions are instructive:

- **`transcription` is written twice, on purpose.** PII redaction returns the redacted transcript under the *same key*, replacing the raw one. Downstream stages read that key and there is no raw version left to reach for. Sending unredacted text to an LLM becomes unrepresentable rather than merely discouraged.
- **`status` is written by four nodes, and that one bit me.** Supervisor review is reached from two places: a blocked injection, which arrives with no QA scores, and a critical flag, which arrives with everything. The status is the same on both paths, but the state behind it isn't. Code that read the QA scores unconditionally crashed on the injection path, and 166 tests missed it because none drove an injection through the compiled graph.

*What survives a crash?* By design, nothing mid-call. There's no checkpointer, because each call is one stateless run. A crashed call can simply be run again, and a SHA-256 cache keyed on the audio bytes makes the expensive part, transcription, free the second time. Finished calls land in SQLite.

**Verdict:** right-sized for one machine and one tenant. "Who writes each field?" is worth asking even in a seven-stage pipeline.

## 5. Eval: the slot that isn't filled yet

There are 169 tests, and they need no API keys or real audio. Every one of them mocks the model. They prove the wiring and the security properties. For example, the injection test asserts that neither LLM mock was *called*, which is the guarantee itself rather than a proxy for it. The speech side was measured directly in a set of spikes, which Part 3 covers.

None of that answers the eval slot's question: *what number must hold before a prompt or model change ships?* There's no golden set of calls with human-agreed scores, so a prompt change that makes every score half a point kinder would pass the whole suite. I made the case for [scoring the middle of a pipeline](/2026/07/22/scoring-the-middle.html) in an earlier series; here the middle is still unscored.

**Verdict:** the gap. The next step is a small human-scored golden set, built before any further prompt work.

*Update, November 2026: injection detection now has an eval of its own. A version of the guard that adds an injection classifier to the regexes catches 87.5% of 263 public injections, and wrongly flags 0.35% of 2,025 real call-centre calls; [Part 8](/2026/11/04/callsense-measuring-the-guard.html) covers how that was measured. The pipeline itself still runs the regexes alone, and output quality is still unmeasured: there is no golden set yet, so this verdict stands.*

## 6. Audit log: one verb, and one hole

The audit logger exposes a single method, `log()`. There's no update path and no delete path, and that absence *is* the guarantee: a log you can edit isn't evidence of anything. Each entry is keyed by call ID, and the halted paths are covered:

- every call writes **STARTED** at intake,
- a rejected upload writes **FAILED**,
- a blocked injection and a critical flag each write **FLAGGED**,
- a finished call writes **COMPLETED**.

The hole is the one in the figure. If both LLM stages exhaust their retries, the exception escapes the graph, the UI shows an error, and the log holds a **STARTED** with nothing after it. And the log records outcomes rather than every step. Per-step detail lives in the [traces](/2026/07/21/a-trace-is-the-new-stack-trace.html), which are useful for debugging but aren't a durable audit record.

**Verdict:** right in shape, with one missing terminal: an LLM failure should end in **FAILED** like any other.

## Where it's thin

Three slots are solid: the control plane, the workers and the checker. State is right-sized. Two need work:

1. **Eval:** no golden set, so output quality is unmeasured.
2. **Audit log:** an LLM failure leaves an orphaned **STARTED** row.

Smaller, but worth naming: the low-confidence transcript flag that nothing routes on yet.

## The one-liner that ties it together

Distilled: **put the model only where judgement is the work, and surround it with code that decides, checks and remembers. Then be honest about which slot you haven't filled.**

**Coming next:** Part 3 goes into the stage everything else depends on, the transcript. It covers how far Whisper can be trusted, why the confidence threshold moved twice, and the experiment where a *bigger* speech model made PII redaction *worse*.
