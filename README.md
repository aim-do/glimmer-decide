# Glimmer Decide

Point a camera at a desk (or drop in a photo), say what you want to do, and get **one object
to use** — or *nothing here fits* / *show me another angle* — with a probability for every
option. **SAM 3.1** then highlights the chosen object on the frame.

Built for the Meta Global AI Developer Hackathon, on Meta models only, no training yet:

| Role | Model | How it decides | Where it runs |
|---|---|---|---|
| **System 1** (default) | [MetaEncoder](https://huggingface.co/facebook/meta-encoder): Muse Glimmer 30B contrastively tuned into a bi-encoder ([paper](https://arxiv.org/abs/2610.11316)) | frame + goal and each option are embedded; cosine similarity -> softmax. One forward pass, no text generated, option embeddings cached | your Modal GPU (`deploy/`) |
| **System 2** (compare / fallback) | [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | reasons, then answers `ANSWER: <letter>` | OpenRouter |
| Highlight | SAM 3.1 | text concept -> boxes | Meta Model API |

```
camera frame ─┐                       ┌─► MetaEncoder: cos(task, option_i) ─► softmax ─► choice + p
goal (text)  ─┼─► task / options ─────┤
options      ─┘   (closed set:        └─► Glimmer: reasoning ─► ANSWER: <letter>        (compare)
                  TAKE_<obj>|NONE|ASK_VIEW)
                                   chosen object ─► SAM 3.1 ─► box overlay
```

## Quickstart (5 minutes, no accounts)

Needs Python 3.11+ (tested on 3.13) and git.

**Windows (PowerShell)**

```powershell
git clone https://github.com/aim-do/glimmer-decide
cd glimmer-decide
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m server --mock
```

**macOS / Linux**

```bash
git clone https://github.com/aim-do/glimmer-decide
cd glimmer-decide
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m server --mock
```

Your browser opens http://localhost:8000. `--mock` answers with fake decisions, so you can
try the whole interface — camera, photos, labeling — without keys or a GPU. The first run
creates `.env` from `.env.example`; that is where keys go.

## Turn on the real models

Each key switches on one part; add any subset to `.env` and restart `python -m server`
(without `--mock`). The terminal prints what is configured.

| Part | What you need | `.env` |
|---|---|---|
| Glimmer (reasoning) | an [OpenRouter](https://openrouter.ai/keys) key with a few dollars of credit | `GLIMMER_API_KEY=` |
| SAM 3.1 box highlight | a [Meta Model API](https://dev.meta.ai) key | `MODEL_API_KEY=` |
| MetaEncoder (fast decider) | your own GPU on Modal, next section | `HF_TOKEN=` (+ filled in by setup) |

Glimmer alone is enough for a working demo; MetaEncoder adds the fast one-pass decider.

### MetaEncoder on your own GPU (Modal), one-time setup

MetaEncoder has no hosted API, so it runs on a GPU in your [Modal](https://modal.com) account
(80 GB card; Modal bills per second and the container sleeps when idle).

1. **Hugging Face.** Open [facebook/meta-encoder](https://huggingface.co/facebook/meta-encoder),
   click *Agree and access repository* (approved instantly). Create a token at
   [settings/tokens](https://huggingface.co/settings/tokens): type *Read*, or a fine-grained
   token with *Read access to contents of all public gated repos you can access*.
   Put it in `.env` as `HF_TOKEN=...`.
2. **Modal.** Create an account at [modal.com](https://modal.com) (hackathon credits apply).
3. **Setup** — one command; it opens the browser for the Modal login if needed, stores the
   HF token as a Modal secret, generates the service token, deploys, and writes
   `ENCODER_URL` / `ENCODER_TOKEN` into `.env`:

   ```bash
   python deploy/setup_modal.py all
   ```

4. Run `python -m server`. The **MetaEncoder GPU** panel at the top of the page has
   **Start** (deploy + warm the GPU) and **Stop** (shut the GPU down and undeploy).

The very first Start downloads the 62 GB of weights into a Modal Volume (~14 min); later
starts take ~2 min. While the GPU runs it costs about $4/h on an H100 (`ENCODER_GPU=A100-80GB`
in `.env` is ~$2.5/h); it falls asleep after 5 idle minutes, and **Stop** turns it off
completely so nothing can wake it. From a terminal the same is:

```bash
python deploy/setup_modal.py status   # deployed? GPU running?
python deploy/setup_modal.py start    # deploy (a MetaEncoder request then wakes the GPU)
python deploy/setup_modal.py stop     # GPU off, app undeployed
```

## Ways to give it a picture

| Where | How | Notes |
|---|---|---|
| Laptop / desktop | webcam on http://localhost:8000 | browsers allow the camera on localhost |
| Any computer | **Photo** button, **drag an image** onto the frame, or **paste** one (Ctrl+V) | no camera needed — screenshots and saved photos work |
| Phone on the same Wi-Fi | `python -m server --https`, then open the `https://<computer-ip>:8000` address it prints | the certificate is self-signed, so the phone warns once: *Advanced → Proceed* (iPhone: *Show Details → visit this website*); after that the live camera works. Anyone on that Wi-Fi can open the page and spend your credits |
| Phone anywhere | `tailscale serve --bg 8000` on the computer, then `https://<computer>.<tailnet>.ts.net` on the phone | real certificate, no warning, only your own devices ([Tailscale](https://tailscale.com) on both) |

Phones get the camera only over HTTPS. `--host 0.0.0.0` without `--https` also works, but then
only **Photo** does (it still opens the phone's camera app).

Then type the goal (or tap 🎤 where the browser supports speech), pick the decider —
MetaEncoder, Glimmer, or **Both** to compare — and press **What should I use?**

## Labeling and evaluation

In the app, open *Save sample*, tick every correct action and save. Frames go to
`data/captures/`, labels to `data/captures.jsonl` (both git-ignored).

```bash
python -m eval.run_eval --decider encoder          # MetaEncoder, frame + goal
python -m eval.run_eval --decider glimmer          # Glimmer with reasoning
python -m eval.run_eval --decider both             # one result file per decider
python -m eval.run_eval --decider encoder --blind  # goal only: how much does the camera matter?
```

Metrics: accuracy, accuracy when the right object is present, **false pick when absent**,
ask-view rate, parse failures, latency; MetaEncoder rows also keep the full distribution
for calibrating a refusal threshold. Hold out whole scenes, not neighbouring frames.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Camera does not start | Use http://localhost (not the IP) on the computer, allow the permission, or use Photo / drag / paste. |
| Phone: *This site can't be reached* | Phone and computer on the same Wi-Fi (not a guest or hotspot network, VPN off on the phone); type the exact address the terminal prints. Windows: when the firewall asks, allow Python (tick *Public* too if your Wi-Fi is marked Public). Android: turn mobile data off for the test (some phones route traffic over it when Wi-Fi looks weak) and allow Chrome *Nearby devices* (Settings → Apps → Chrome → Permissions). Chrome or another non-Safari browser on iPhone: Settings → that browser → *Local Network* on. Some routers isolate Wi-Fi devices from each other: then use Tailscale. |
| Phone: *Your connection is not private* | Expected with `--https` (self-signed certificate): *Advanced → Proceed*. It comes back when the computer's address changes and a new certificate is made. |
| Panel says *Not set up yet* | Run `python deploy/setup_modal.py all`; the page picks the result up without a restart. |
| `GatedRepoError` during setup | Accept the terms on the model page and give the HF token gated-repo read access (step 1). |
| Start ends with HTTP 401 | `.env` and Modal disagree on `ENCODER_TOKEN`: rerun `python deploy/setup_modal.py all`. |
| *Starting GPU…* runs much longer than 15 min | `modal app logs glimmer-decide-metaencoder`; press Stop to end the billing. |
| MetaEncoder answers `upstream error (HTTP 404)` | The GPU app is stopped: press Start. |

## How the decision works

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

Probabilities use an uncalibrated tau=0.02. When the top-2 cosine margin is below
`ENCODER_MIN_MARGIN` (default 0.05) the encoder asks for another view instead of guessing
between near-ties; blind mode still reports the model's own pick so the camera ablation
stays honest. Tune the margin on the labeled eval set. `/api/decide` has one shared
deadline (`REQUEST_TIMEOUT_S`, default 240 s) and per-route rate limits; the server listens
on localhost only by default, because every request spends credits.

## Status

| Step | What | State |
|---|---|---|
| 1 | MetaEncoder + Glimmer + SAM 3.1 on camera frames, no training; labeling UI; eval with blind ablation | done, smoke-tested on 2 public photos |
| 2 | Collect & label 100-300 frames on real desks; measure MetaEncoder vs Glimmer (accuracy, false pick when absent, latency) | next |
| 3 | Adapt: Glimmer option-logit scoring + LoRA (Fireworks free SFT until 2026-11-30, or Modal), or a small head on frozen MetaEncoder features | |

### First smoke test (2026-10-28, not an eval)

Two public photos (a stationery desk, a paper shredder), H100 in us-east, client in Europe;
latency is end to end from the laptop, MetaEncoder GPU time per frame is ~0.6 s.

| Frame | Goal | MetaEncoder | Glimmer (reasoning) |
|---|---|---|---|
| stationery | Cut the paper along a straight line | scissors p=0.999, 3.3 s (first call, embeds the options) | scissors, 5.5 s |
| stationery | Fasten these sheets together permanently | stapler p=0.999, 1.6 s | stapler, 2.9 s |
| stationery | Label this box | marker p=0.9997, 1.7 s | marker, 3.6 s |
| shredder | Cut the paper along a straight line | **none p=0.65** vs scissors 0.35, 1.9 s | none, 8.5 s |
| no image | Cut the paper along a straight line | scissors p=0.995 (before the blind-note fix) | ask for view, 2.1 s |

MetaEncoder is 2-4x faster here and agrees with Glimmer when the object is visible, but it
refuses weakly: with the right object absent it said *none* by a thin margin, and with no
image it still picked by the goal text. That is what the labeled eval set has to measure
(false pick when absent) and what a calibrated threshold or a small trained head should fix.

## Layout

```
server/   FastAPI app (python -m server); encoder.py (MetaEncoder decider), encoder_ctl.py (GPU Start/Stop),
          certs.py (self-signed HTTPS for phones), decision.py (Glimmer), sam.py, tasks.py
deploy/   Modal service for MetaEncoder + setup_modal.py (all / status / start / stop)
domains/  action sets (desk.json)
web/      vanilla-JS camera client
eval/     offline evaluation
tests/    pytest (mock mode, no network): python -m pytest
```

## License

The code here is MIT-licensed, see [LICENSE](LICENSE). The models are not part of the repo
and keep their own terms: you accept MetaEncoder's license on its Hugging Face page before
the weights download, and Glimmer and SAM 3.1 are used through OpenRouter and the Meta
Model API under those services' terms.
