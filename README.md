# Glimmer Decide

Point your phone camera at a desk, say what you want to do, and **Muse Glimmer 30B** picks
the one object to use — or says *nothing here fits* / *show me another angle*.
**SAM 3.1** then highlights the chosen object on the frame.

Built for the Meta Global AI Developer Hackathon.

```
camera frame ─┐
goal (text)  ─┼─► Glimmer 30B: choose 1 of N lettered options ─► hint to the user
options      ─┘          │  (TAKE_<object> … | NONE | ASK_VIEW)
                         └─► chosen object's concept ─► SAM 3.1 ─► box overlay
```

Glimmer works as a **decision model**: a closed action set per domain, a final
`ANSWER: <letter>` line, and explicit refusal options. That gives a measurable task
(accuracy, false picks when the right object is absent, ask-view rate) and a clean
target for the next step, LoRA SFT.

## Status

| Step | What | State |
|---|---|---|
| 1 | Base Glimmer + SAM 3.1, camera web app, sample capture/labeling, eval harness with blind ablation | **done (no training)** |
| 2 | Collect & label 300–1000 frames on real desks, measure base model | next |
| 3 | LoRA SFT on Glimmer (Fireworks free SFT / Modal), compare base vs tuned on held-out scenes | |
| 4 | SAM candidates → numbered marks → Glimmer picks the instance | |

## Run

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Linux/macOS: .venv/bin/pip
cp .env.example .env                               # add GLIMMER_API_KEY (OpenRouter) and MODEL_API_KEY (Meta)
.venv/Scripts/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. Without `GLIMMER_API_KEY` the server runs in **mock mode**
(keyword stub) so the UI works offline.

### From a phone

Browsers only allow the camera on HTTPS (or localhost). Easiest: Tailscale on the
laptop and the phone, then

```bash
tailscale serve --bg 8000
```

and open `https://<laptop>.<tailnet>.ts.net` on the phone. Alternatively use the
**Photo** button, which works over plain HTTP too.

## Where Glimmer runs

Any OpenAI-compatible `/chat/completions` endpoint (`GLIMMER_BASE_URL`, `GLIMMER_MODEL`):

- **OpenRouter** `meta/muse-glimmer-30b` (default; served by DeepInfra / Together, text+image).
- **Fireworks** — serverless Glimmer ended 2026-09-25; on-demand deployment only. Free LoRA SFT until 2026-11-30.
- **Self-hosted vLLM** (e.g. on Modal), needed for a custom LoRA:
  `vllm serve meta-models/Muse-Glimmer-30B --reasoning-parser muse_glimmer --tool-call-parser muse_glimmer`.

Glimmer is not on the Meta Model API; SAM 3.1 is (`sam-3.1` via `POST /v1/responses`).

## Labeling and evaluation

In the app, open *Save sample*, tick every correct action and save. Frames go to
`data/captures/`, labels to `data/captures.jsonl` (both git-ignored).

```bash
python -m eval.run_eval            # frame + goal
python -m eval.run_eval --blind    # goal only: how much does the camera matter?
```

Metrics: accuracy, accuracy when the right object is present, **false pick when absent**,
ask-view rate, parse failures, latency. For a fair test, hold out whole scenes, not
neighbouring frames of one video.

## Layout

```
server/   FastAPI app, Glimmer decision prompt+parser, SAM 3.1 client, domains loader
domains/  action sets (desk.json)
web/      vanilla-JS camera client
eval/     offline evaluation
tests/    pytest (mock mode, no network)
```
