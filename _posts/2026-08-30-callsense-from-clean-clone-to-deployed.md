---
date: 2026-08-30 18:30:00 -0400
title: "From Clean Clone to Deployed"
description: "Part 6 of the callsense series: what it took to make callsense run somewhere other than my laptop. A clean clone that failed tests the developer's machine passed, a container built so it would actually serve, a Hugging Face Space that hit five blockers, and the deployment notes written so a second attempt starts where this one stopped."
---

*callsense series &mdash; 1. [The Problem](/2026/08/05/callsense-the-call-nobody-reviews.html) &middot; 2. [Six Slots, One Pipeline](/2026/08/10/callsense-six-slots-one-pipeline.html) &middot; 3. [Measuring the Ear](/2026/08/15/callsense-measuring-the-ear.html) &middot; 4. [When the Input Is the Prompt](/2026/08/20/callsense-when-the-input-is-the-prompt.html) &middot; 5. [Don't Let the Model Do the Math](/2026/08/25/callsense-dont-let-the-model-do-the-math.html) &middot; 6. From Clean Clone to Deployed (you're here) &middot; 7. [The Guard as a Tool](/2026/10/30/callsense-the-guard-as-a-tool.html) &middot; 8. [Measuring the Guard](/2026/11/04/callsense-measuring-the-guard.html)*

**Objective:** Parts 3&ndash;5 were about what the pipeline does. This part is about where it runs. callsense came out of a capstone brief from an AI engineering course, and a reviewer would run it from a fresh copy, not from my laptop. My laptop was the only place I had ever run it. So I tried a clean clone, then a container, then a public host. Each step exposed something the step before could not show, and the public deployment stopped on cost, not on code. This part is organized around one question: *what does "it works" have to mean before someone else can run it?*

![Four columns, left to right, show where callsense ran and what each step exposed. The laptop column says nothing had been tried anywhere else yet, and make run works. The clean clone column lists what broke: eight tests depended on my local .env file, and the install assumed uv and a git repository. Its fixes are an autouse environment fixture and a search for a suitable Python. The container column, headed avoided, lists three failures caught while building the image and avoided in the Dockerfile, so none reached a host: binding 127.0.0.1, an app directory owned by root, and downloading Whisper weights on the first upload. Its fixes are binding 0.0.0.0, running as UID 1000 and baking the weights in. The Hugging Face Spaces column lists Python 3.10, comments in packages.txt, ZeroGPU-only hardware and an exhausted free quota. It ends in a dashed box reading blocked on cost.](/images/callsense/callsense-four-environments.svg)

## 1. The clean clone

*What breaks when a stranger runs it?*

A **hermetic** test suite is one that carries its own environment. It does not read the developer's `.env` file, their database or their network. It gives the same result on any machine.

Mine was not hermetic. A fresh clone failed eight tests with a `ConfigError` saying `LLM_PROVIDER` was missing. Several code paths read the configuration themselves, by calling `load_config()`, instead of being handed it. That is fine in production, where `.env` exists. It is not fine as a test dependency. Those eight tests passed on my machine only because my own `.env` happened to be there. The suite even behaved differently depending on which provider that file named.

The fix is a pytest fixture, a piece of setup code that runs around a test. This one, `hermetic_environment` in `tests/conftest.py`, is **autouse**: pytest applies it to every test without being asked. It sets the provider variables, and it points `DB_PATH`, the database location, at a temporary directory.

That second part closes a hole I had already fallen into. One integration test had been writing its own failure into my real database on every run. I found it by looking at the dashboard, not by a test: 21 rows of the same failure, and a 4% success rate built entirely from test runs. Nothing asserts on the contents of the developer's own database, so nothing could have caught it.

The install broke too. `make install` ran `uv pip install` into a virtualenv that did not exist yet. It assumed `uv` was installed at all. And `pre-commit install` needs a git repository, which a reviewer working from a zip does not have.

`make install` now creates the virtualenv itself with standard Python tools, and skips the git hooks when there is no repository. The Makefile searches for the first Python 3.10 or newer it can find, because stock macOS still ships 3.9 as `python3`. You can override it with `make install PYTHON=python3.12`. From a fresh copy with no `.git`, no `.env` and no virtualenv, all 169 tests pass, with no API keys and no real audio.

**Verdict:** "passes on my machine" proved the machine, not the code.

## 2. The container

*What does a container assume that a laptop never does?*

A **container** is a packaged copy of the app plus everything it needs, run in isolation. callsense's `Dockerfile` starts from `python:3.11-slim` and installs `ffmpeg`, which faster-whisper uses to decode anything that is not already PCM WAV. Four decisions in it keep the container working once it leaves my laptop. The first three each fix a failure I caught while building and testing the image, before any host saw it:

- **Bind `0.0.0.0`.** A **bind address** is the network interface a server listens on. Inside a container, `127.0.0.1` is the container's own loopback, so a published port would map to nothing and refuse the connection. The first build exposed exactly this: `app.py` bound `127.0.0.1`. Now the Dockerfile sets `GRADIO_SERVER_NAME=0.0.0.0` and `app.py` reads it.
- **Run as UID 1000.** A **UID** is the numeric id of the user a process runs as. Hugging Face Spaces runs containers as UID 1000, and several other platforms do the same. The image ran as root, so `/app` belonged to root. Running it as UID 1000 locally confirmed the problem: writing to `/app/data` was denied. The SQLite database and the model cache would have been unwritable in production only. So the image creates that user and gives it `/app` from the start.
- **Bake the Whisper weights in.** The app warms its speech model at startup, but that is useless if the weights are not on disk. Without them, the first upload pays the download. Baking them in adds about 140 MB to the image and removes a download from every cold start, the first start of a fresh container.
- **Two volumes.** One keeps the SQLite database across restarts. The other keeps the weights, which are otherwise downloaded on every container start.

The README records what I verified: the image builds clean and serves on port 7860 within about two seconds. It is about 2 GB.

Device choice is automatic: an NVIDIA GPU (CUDA) when available, otherwise CPU with `int8`. Apple Silicon falls back to CPU on purpose. CTranslate2, the engine under faster-whisper, has no backend for Apple's GPU, so CPU is the faster choice there.

**Verdict:** a container is a second machine you have not met. Write down what it does not share with yours.

## 3. Hugging Face Spaces: five blockers

*What did a free public host actually require?*

I wanted a public demo on Hugging Face Spaces, which hosts small apps and has a free tier. My deployment notes, `docs/DEPLOYMENT.md`, start with what the app needs from any host:

| Need | Value |
|---|---|
| Python | 3.10+ |
| Memory | about 1 GB |
| System package | `ffmpeg` |
| Port | 7860 by default |
| Bind address | `0.0.0.0` in a container |
| User | UID 1000 |
| Secrets | one provider key |
| Storage | ephemeral is fine: SQLite resets and nothing that matters is lost |

A Space is built from an **SDK**, here a ready-made recipe for a kind of app: Docker for any container, or Gradio for a Gradio app. Spaces needed four code changes, all now in the repo:

- **Python 3.10.** The Gradio SDK's base image is 3.10, and it ignores the `python_version` setting. `StrEnum` and `datetime.UTC` only exist in 3.11, so `state.py` carries a `StrEnum` backport and the code uses `timezone.utc`.
- **`requirements.txt` and `packages.txt`.** The SDK reads neither `pyproject.toml` nor a Dockerfile. `packages.txt` cannot contain comments. It is piped through `xargs` into `apt-get install`, so a comment becomes a list of package names and the build fails.
- **`ssr_mode=False`.** Gradio 5's server-side rendering starts a separate Node.js process next to Python. That is unnecessary behind a platform's proxy.
- **UID 1000**, as above.

Then the blockers, in the order I hit them:

1. **The Docker SDK is paid.** The Gradio SDK is the free route.
2. **Creating a Gradio Space offers only ZeroGPU hardware.** **ZeroGPU** is Hugging Face's shared GPU hardware, and it requires the app to mark at least one function with `@spaces.GPU`. callsense is CPU-only by design, so it fails at startup with `No @spaces.GPU function detected`.
3. **ZeroGPU cannot be downgraded to CPU basic without PRO.**
4. **A workaround that worked.** Create the Space with the static template, which offers CPU basic. Then change `sdk: static` to `sdk: gradio` in the README frontmatter. The Space accepted it.
5. **The free CPU quota is per account, and mine was already exhausted.** With one Space and nothing else running, PRO ($9/month) was the only way forward.

**Verdict:** technically ready, blocked on cost, and written down. The notes open with the status: "attempted, blocked on cost. Everything technical was solved; the free tier was not available."

## 4. Options, costed

*What would I do on the second attempt?*

I did not want to leave a dead end, so the notes carry the next options with a monthly price and an effort estimate:

![Horizontal bars show the monthly cost of four ways to host the callsense demo, with the effort to set each up beside it. Fly.io is about 3 dollars a month and takes about 20 minutes. Hugging Face PRO is 9 dollars and needs no further work because the Space is already built. AWS Lightsail Containers is 10 dollars and takes about an hour. AWS ECS Fargate with a load balancer is 35 to 40 dollars, by far the longest bar, with the range drawn lighter, and takes half a day.](/images/callsense/callsense-hosting-cost.svg)

Fly.io runs the existing Dockerfile unchanged. HF PRO needs nothing more: subscribe and unpause. Lightsail includes HTTPS and a stable URL. ECS with a load balancer is the odd one out. The load balancer costs more than the compute, so it is only worth it if the AWS experience is itself the goal.

Whichever host wins, the notes say to point the demo at Groq. It has a free tier, and the provider spike measured it at 1.84 s against OpenAI's 5.32 s. A public demo should be fast and should not bill a personal OpenAI account.

**Verdict:** a dead end with prices on it is a decision someone can make later, not a restart.

## 5. The demo runbook

*What do I need to say on camera, and what do I need to avoid?*

The same habit applies to the demo runbook, `docs/DEMO.md`. It says it was "written while the reasons were still fresh rather than reconstructed on the day".

**Run `make demo`, not `make run`.** It pins the Whisper model to `base`. That is a trade, not an upgrade. On `tiny`, a dictated card number can come out with a stray `4.` left outside the redaction. `base` renders it cleanly, but it costs the phone number in one sample. [Part 3](/2026/08/15/callsense-measuring-the-ear.html) is the full story.

The audio table says which files do what. `data/samples/` is gitignored, because the reference audio is not ours to republish. `data/demo/` is tracked, so a reviewer who clones the repo can reproduce the security beats.

The walkthrough has seven beats, in order, and each shows something the previous one cannot. It starts with the ordinary path working end to end. Later come PII redacted before the third-party call, and an injection halting the pipeline before any LLM call. It ends with swapping the LLM provider and re-running.

The last section is "Known rough edges, if asked". It points to limits the README already names, among them: spoken-word digits escape a digit regex, only structured PII is redacted, the 50 MB and 60-minute caps pull against each other, and five of the ten reference calls open with a narrator. It also volunteers the strongest one: a larger Whisper model can *reduce* PII detection.

**Verdict:** I would rather say these first than be asked.

*Update, October 2026: a fresh install later broke every transcription. `av`, the audio-decoding library under faster-whisper, released version 19 without an argument that faster-whisper still passes. `av` is now pinned below 19 in `pyproject.toml`, with a comment saying when to lift it. No test noticed, because every test replaces Whisper with a stand-in (a mock). That is the clean-clone lesson one more time. A hermetic suite is not the same as a suite that touches real things. Mine kept the keys, the audio and the database out of the tests. It also kept the real transcriber out.*

## The one-liner that ties it together

Distilled: **"deployed" starts with a stranger's clean clone, and the notes on why you stopped are part of the deployment.**

**Coming next:** Part 7 skips ahead to October: the guards from Part 4, packaged as a tool other programs can call, and a second injection check that reads intent rather than words.
