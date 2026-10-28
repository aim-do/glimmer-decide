# Glimmer Decide

Point your phone camera at a desk, say what you want to do, and get **one object to use**,
or *nothing here fits* / *show me another angle*, with a probability for every option.
**SAM 3.1** then highlights the chosen object on the frame.

Built for the Meta Global AI Developer Hackathon, on Meta models only, no training yet:

| Role | Model | How it decides | Where it runs |
|---|---|---|---|
| **System 1** (default) | [MetaEncoder](https://huggingface.co/facebook/meta-encoder): Muse Glimmer 30B contrastively tuned into a bi-encoder ([paper](https://arxiv.org/abs/2610.11316)) | frame + goal and each option are embedded; cosine similarity -> softmax. One forward pass, no text generated, option embeddings cached | our Modal GPU (`deploy/`) |
| **System 2** (compare / fallback) | [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | reasons, then answers `ANSWER: <letter>` | OpenRouter |
| Highlight | SAM 3.1 | text concept -> boxes | Meta Model API |

```
camera frame ─┐                       ┌─► MetaEncoder: cos(task, option_i) ─► softmax ─► choice + p
goal (text)  ─┼─► task / options ─────┤
options      ─┘   (closed set:        └─► Glimmer: reasoning ─► ANSWER: <letter>        (compare)
                  TAKE_<obj>|NONE|ASK_VIEW)
                                   chosen object ─► SAM 3.1 ─► box overlay
```

The task follows the closed-set format from the MetaEncoder card:

```
User goal: Temporarily hold these sheets together without punching holes or glue

Task: Select the correct option: which object visible in the image should the user use to achieve the goal?
Criteria:
- binder clip: holds a stack of sheets together temporarily, no holes, removable
- stapler: fastens sheets permanently by punching a staple through them
...
- none: no object visible in the frame fits the goal and its constraints
- unclear view: the frame is too blurred, dark, cropped or occluded to decide
```

## Status

| Step | What | State |
|---|---|---|
| 1 | MetaEncoder + Glimmer + SAM 3.1 on camera frames, no training; labeling UI; eval with blind ablation | code done; MetaEncoder needs the Modal deploy below |
| 2 | Collect & label 100-300 frames on real desks; measure MetaEncoder vs Glimmer (accuracy, false pick when absent, latency) | next |
| 3 | Adapt: Glimmer option-logit scoring + LoRA (Fireworks free SFT until 2026-11-30, or Modal), or a small head on frozen MetaEncoder features | |

## Accounts

| Service | For | Status |
|---|---|---|
| Hugging Face | MetaEncoder weights are gated (auto-approve): accept the terms on the model page, then `hf auth login` | needed |
| Modal | GPU for MetaEncoder (no hosted API exists); hackathon credits | needed |
| OpenRouter | Glimmer System 2 | `GLIMMER_API_KEY` |
| Meta Model API | SAM 3.1 | `MODEL_API_KEY` |
| Tailscale | HTTPS to the laptop from a phone (camera needs HTTPS) | optional |

## Run

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Linux/macOS: .venv/bin/pip
cp .env.example .env                               # GLIMMER_API_KEY, MODEL_API_KEY
.venv/Scripts/python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000. The server listens on localhost only, because every request
spends credits. `MOCK=1` runs everything offline with fake decisions.

### MetaEncoder on Modal (one-time)

```bash
.venv/Scripts/pip install -r deploy/requirements.txt
modal token new                                    # browser login
hf auth login                                      # read token of the account that accepted the terms
.venv/Scripts/python deploy/setup_modal.py secrets # HF token + generated ENCODER_TOKEN -> Modal secrets and .env
modal deploy deploy/modal_metaencoder.py
.venv/Scripts/python deploy/setup_modal.py url     # writes ENCODER_URL into .env
```

One H100 (80 GB, bf16 weights ~62 GB). The container stops after 5 idle minutes; the
first request after that waits for a cold start (minutes). The web app pings
`/api/warmup` on load to start it early. Both endpoints require the bearer token.

### From a phone

```bash
tailscale serve --bg 8000
```

then open `https://<laptop>.<tailnet>.ts.net` on the phone (same tailnet).

## Labeling and evaluation

In the app, open *Save sample*, tick every correct action and save. Frames go to
`data/captures/`, labels to `data/captures.jsonl` (both git-ignored).

```bash
python -m eval.run_eval --decider encoder          # MetaEncoder, frame + goal
python -m eval.run_eval --decider glimmer          # Glimmer with reasoning
python -m eval.run_eval --decider encoder --blind  # goal only: how much does the camera matter?
```

Metrics: accuracy, accuracy when the right object is present, **false pick when absent**,
ask-view rate, parse failures, latency; MetaEncoder rows also keep the full distribution
for calibrating a refusal threshold. Hold out whole scenes, not neighbouring frames.

## Layout

```
server/   FastAPI app; encoder.py (MetaEncoder decider), decision.py (Glimmer), sam.py, tasks.py
deploy/   Modal service for MetaEncoder + secrets/URL setup
domains/  action sets (desk.json)
web/      vanilla-JS camera client
eval/     offline evaluation
tests/    pytest (mock mode, no network)
```
