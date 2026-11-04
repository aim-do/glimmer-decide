# Glimmer Decide

A camera test bench for open-weight **decision models**, and a way to train your own.

Point a camera at a desk (or drop in a photo), say what you want to do, and get **one object
to use** — or *nothing here fits* / *show me another angle* — with a probability for every
option. **SAM 3.1** then highlights the chosen object on the frame.

## Why

Decision models answer by picking one option from a closed set in a single forward pass,
instead of writing out their reasoning first: the model itself needs ~0.15 s on the GPU, and
the answer arrives in 1-1.5 s end to end instead of several seconds.
The open-weight ones are new, and the released ones were trained mostly on text. This project
has two goals:

1. **A tool to test open-weight decision models on camera (video) frames**, so we can try
   features quickly and keep the ones that work. Point a phone or a webcam at a real scene and
   compare, side by side and with their timings, a bi-encoder (MetaEncoder), a decision head
   (JevAny) and a reasoning model (Muse Glimmer), all built on Meta's open Muse Glimmer 30B.
   Try refusing when nothing fits, asking for a better view, the camera-off ablation,
   highlighting the pick with SAM, and capture from a phone. Then see what a real product needs.
2. **Let users train decision models themselves.** The page that tests the models also
   collects the data: every frame saved with its right answer is a training example. One
   command then fine-tunes JevAny on your frames, or trains a Clef-style decision head from
   scratch, on your own GPU. One line in `.env` puts your fine-tuned model back into the page.

Built for the Meta Global AI Developer Hackathon, on Meta models only.

## What's inside

| Decider | Model | How it decides | Speed: end to end (GPU) | Where it runs |
|---|---|---|---|---|
| **MetaEncoder** (System 1, default) | [facebook/meta-encoder](https://huggingface.co/facebook/meta-encoder): Muse Glimmer 30B contrastively tuned into a bi-encoder ([paper](https://arxiv.org/abs/2610.11316)) | frame + goal and each option are embedded; cosine similarity -> softmax. Option embeddings are cached | ~1.4 s (0.15 s) | your Modal GPU |
| **JevAny** (System 1, **fine-tunable**) | [JevAny-Muse-Glimmer-30B](https://huggingface.co/SimpleJev/JevAny-Muse-Glimmer-30B-LoRA): a LoRA + pointer head on Muse Glimmer 30B ([JevAny](https://github.com/SimpleJev/JevAny)) | frame + goal + labelled options in one prefill; a small trained head scores the options | ~1.4 s (0.17 s) | your Modal GPU |
| **Clef-style Glimmer** (System 1, **trained from scratch**, experiment) | [Clef](https://huggingface.co/Cloudflare/clef)'s joint schema head + a fresh LoRA on Muse Glimmer 30B | one prefill; the head scores all options together | — (training only, not in the page yet) | your Modal GPU |
| **Glimmer** (System 2, compare) | [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | reasons, then answers `ANSWER: <letter>` | 3–9 s | OpenRouter |
| Highlight | SAM 3.1 | text concept -> boxes | | Meta Model API |

Speeds are medians on the same 1024×768 camera frames, from a laptop in Europe to warm H100s in
us-east: the models take ~0.15 s, the rest is the network and Modal's routing (see
[Results so far](#results-so-far)).

```
camera frame ─┐                       ┌─► MetaEncoder: cos(task, option_i) ─► softmax ─► choice + p
goal (text)  ─┼─► task / options ─────┼─► JevAny: one prefill ─► pointer head ─────────► choice + p
options      ─┘   (closed set:        └─► Glimmer: reasoning ─► ANSWER: <letter>        (compare)
                  TAKE_<obj>|NONE|ASK_VIEW)
                                   chosen object ─► SAM 3.1 ─► box overlay
saved frames + right answers ─► fine-tune JevAny / train a Clef-style head ─► back into the page
```

**Contents:** [Run it](#run-it) · [Deploy the GPU services](#deploy-the-gpu-services-modal) ·
[Use the page](#use-the-page) · [Train your own decider](#train-your-own-decider) ·
[How decisions work](#how-decisions-work) · [Results so far](#results-so-far) ·
[Troubleshooting](#troubleshooting) · [Development](#development) · [License](#license)

## Run it

1. **Install** once (Python 3.12+, tested on 3.13, and git):

   ```bash
   git clone https://github.com/aim-do/glimmer-decide
   cd glimmer-decide
   python -m venv .venv            # macOS / Linux: python3 -m venv .venv
   .venv\Scripts\activate          # macOS / Linux: source .venv/bin/activate
   pip install -r requirements.txt -c constraints.txt
   ```

   `constraints.txt` pins every dependency, transitive ones included, to the versions the
   checks ran with; leave it out to take newer ones.

2. **Try it with no keys:** `python -m server --mock` opens http://localhost:8000 with fake
   decisions, so the whole interface works. This first run also creates `.env`.

3. **Add your keys to `.env`.** Each one turns on one part; Glimmer alone is already a demo.

   | Part | In `.env` | Where to get it |
   |---|---|---|
   | Glimmer (reasoning) | `GLIMMER_API_KEY=` | an [OpenRouter key](https://openrouter.ai/keys) with a few dollars of credit |
   | SAM 3.1 box highlight | `MODEL_API_KEY=` | a [Meta Model API](https://dev.meta.ai) key |
   | MetaEncoder | `HF_TOKEN=`, then `python deploy/setup_modal.py all` once | accept the terms on [facebook/meta-encoder](https://huggingface.co/facebook/meta-encoder), then create a *Read* [token](https://huggingface.co/settings/tokens); a [Modal](https://modal.com) account for the GPU |
   | JevAny (and the trainers) | nothing new: `python deploy/setup_modal.py jevany` once | the same Modal account; the weights are open (~60 GB, downloaded once on a cheap CPU container) |

   The setup commands sign you in to Modal in the browser, deploy the GPU service and write
   its URL and token into `.env` themselves (see [Deploy the GPU services](#deploy-the-gpu-services-modal)).

4. **Start the server:** `python -m server --phone` (or just `python -m server` without a
   phone). It prints:

   ```
     This computer:       https://localhost:8000
     Phone (same Wi-Fi):  https://192.168.1.23:8000
   ```

   Open the phone address on your phone; the page on the computer shows it too. The first
   time, each browser warns about the self-signed certificate: *Advanced → Proceed*
   (iPhone: *Show Details → visit this website*). Then the phone camera works.

5. **GPUs:** press **Start** in the *MetaEncoder GPU* (or *JevAny GPU*) panel and wait ~1-2 min;
   press **Stop** when you are done. Each GPU costs ~$4/h while on and sleeps after 5 idle minutes.

6. **Train on your own frames (optional):** save labelled frames in the page, then
   `modal run deploy/modal_jevany_train.py --run desk-v1` and `JEVANY_RUN=desk-v1` in `.env`;
   see [Train your own decider](#train-your-own-decider).

Server options: `--mock` (fake decisions, no keys), `--phone` (HTTPS on the local network),
`--host`, `--port`, `--no-browser`; `python -m server --help` lists them.

## Deploy the GPU services (Modal)

MetaEncoder and JevAny have no hosted API, so the repo deploys them to GPUs in **your** Modal
account; the trainers run there too. Everything is created by two one-time commands:

```bash
python deploy/setup_modal.py all      # MetaEncoder: Modal login, HF token -> Modal secret, deploy, ENCODER_URL/TOKEN -> .env
python deploy/setup_modal.py jevany   # JevAny: deploy, ~60 GB of weights into a Volume on a CPU box, JEVANY_URL -> .env
```

The page picks up the new `.env` values without a restart. What ends up in your Modal account:

| Modal object | What it is | Created by |
|---|---|---|
| app `glimmer-decide-metaencoder` | MetaEncoder service (`deploy/modal_metaencoder.py`, vLLM pooling) | `setup_modal.py all`, the page's Start |
| app `glimmer-decide-jevany` | JevAny service (`deploy/modal_jevany.py`) | `setup_modal.py jevany`, the page's Start |
| app `glimmer-decide-jevany-train` | JevAny fine-tuning job, runs only while training | `modal run deploy/modal_jevany_train.py` |
| app `glimmer-decide-clef-train` | Clef-style training job, runs only while training | `modal run deploy/modal_glimmer_clef.py` |
| Volume `metaencoder-weights` | MetaEncoder weights (62 GB) | the first MetaEncoder Start (~14 min) |
| Volume `jevany-weights` | Muse Glimmer 30B + the JevAny adapter (~60 GB); your runs in `runs/` and `clef-runs/` | `setup_modal.py jevany` |
| Secret `huggingface` | your HF token, for the gated MetaEncoder weights | `setup_modal.py all` |
| Secret `glimmer-decide-auth` | the bearer token both services check (`ENCODER_TOKEN` in `.env`) | `setup_modal.py all` or `jevany` |

Operating the services:

- **Start** in the page = `modal deploy` plus a warm-up call; later starts take ~1-2 min
  (both were ready 105 s after Start). **Stop** shuts the GPU down and undeploys the app, so nothing can wake it
  and bill. Idle GPUs also sleep after 5 minutes. From a terminal:

  ```bash
  python deploy/setup_modal.py status                 # deployed? GPU running? (both)
  python deploy/setup_modal.py start [encoder|jevany] # deploy; a request then wakes the GPU
  python deploy/setup_modal.py stop                   # both GPUs off, apps undeployed
  ```

- Each service needs its own 80 GB card (62 GB and 59 GB of weights): ~$4/h for an H100,
  ~$8/h for both. `ENCODER_GPU` / `JEVANY_GPU=A100-80GB` in `.env` costs ~$2.5/h and is read
  at every Start. Modal bills per second; hackathon credits apply.
- The HF token can also be fine-grained, with *Read access to contents of all public gated
  repos you can access*. JevAny and the trainers need no Hugging Face access at all.

## Use the page

| Where | How | Notes |
|---|---|---|
| Laptop / desktop | webcam on the *This computer* address | browsers allow the camera on localhost |
| Any computer | **Photo** button, **drag an image** onto the frame, or **paste** one (Ctrl+V) | no camera needed — screenshots and saved photos work |
| Phone on the same Wi-Fi | `python -m server --phone`, then open the *Phone* address it prints (the page on the computer shows it too) | the certificate is self-signed, so the phone warns once: *Advanced → Proceed* (iPhone: *Show Details → visit this website*); after that the live camera works. Anyone on that Wi-Fi can open the page and spend your credits |
| Phone anywhere | `tailscale serve --bg 8000` on the computer, then `https://<computer>.<tailnet>.ts.net` on the phone | real certificate, no warning, only your own devices ([Tailscale](https://tailscale.com) on both) |

Phones get the camera only over HTTPS. `--host 0.0.0.0` without `--phone` also works, but then
only **Photo** does (it still opens the phone's camera app).

Type the goal (or tap 🎤 where the browser supports speech), pick the decider — MetaEncoder,
JevAny, Glimmer, or **Compare all** — and press **What should I use?** The button counts the
seconds while the models think; each answer shows how long its model took (⏱) and the total
stays under the button. When you change the goal, the frame or the domain, earlier answers are
dimmed and marked as stale. *blind (no image)* asks the same question without the frame, to see
how much the camera matters.

## Train your own decider

The page collects the data, and two trainers turn it into decision models on your Modal GPU:

| | Fine-tune JevAny | Clef-style Glimmer from scratch (experiment) |
|---|---|---|
| Starts from | the released JevAny adapter (LoRA r8 + pointer head) | empty weights: a fresh LoRA (r 16, B = 0) + a new Clef joint schema head (~140M parameters) |
| Base model | Muse Glimmer 30B, frozen | Muse Glimmer 30B, frozen |
| Loss | JevAny's own trainer (cross-entropy) | cross-entropy + Brier score |
| Frames needed | a few dozen to start | hundreds |
| Back in the page | yes: `JEVANY_RUN=<run>` in `.env` | not yet |

Both need, once: the `modal` CLI (it comes with `requirements.txt`), a Modal login and the
Muse Glimmer weights in your account: `python deploy/setup_modal.py jevany` does both.

### Collect labelled frames

In the page, open *Save sample for the eval set*, tick **every** right object and press
**Save frame + goal + label**. Frames go to `data/captures/`, labels to `data/captures.jsonl`
(both git-ignored); both trainers and the evaluation read that file. Save frames where nothing
fits and frames with a bad view too: they teach refusing. Keep some whole scenes out of
training (move their lines to e.g. `data/heldout.jsonl`) to evaluate on.

### Fine-tune JevAny (warm start)

JevAny's own trainer continues from the released adapter, on the exact text the page sends at
decision time; the 30B base stays frozen in bf16 on one 80 GB GPU.

```bash
modal run deploy/modal_jevany_train.py --smoke        # one generated frame, 20 steps: does training run? ~2.5 min, ~$0.2
modal run deploy/modal_jevany_train.py --run desk-v1  # every sample in data/captures.jsonl, 3 epochs
```

A run prints the loss per record and its decision on the training frames before and after
(also saved to `eval/results/train-<run>.json`), and keeps the new adapter in the
`jevany-weights` Volume under `runs/<run>/`. Flags: `--data`, `--epochs`, `--steps`, `--lr`,
`--head-lr`, `--accum` (`--help` lists them).

**Use it in the page:** `JEVANY_RUN=desk-v1` in `.env`, then **Stop** and **Start** the JevAny
GPU (no server restart needed). Its answers then end with *fine-tuned run: desk-v1*. An empty
`JEVANY_RUN` goes back to the released adapter.

### Clef-style Glimmer from scratch (experiment)

The joint schema head of Cloudflare's [Clef](https://huggingface.co/Cloudflare/clef) on Muse
Glimmer 30B. Nothing is warm-started: the decoder gets a fresh LoRA (B = 0, so it starts out
as the base model) and the head starts from its default initialisation. The head has Clef's
release size (width 1024, 2 evidence-routing + 4 joint layers) and reads the instruction,
every option (its span and its tokens' output embeddings) and the whole sequence, scoring all
options together. The prompt keeps Clef's layout in Muse's chat format
(`train/clef_format.py`); the head is ported from Clef's code (`train/clef_head.py`,
Apache-2.0, see `train/LICENSE-clef`).

```bash
modal run deploy/modal_glimmer_clef.py --selftest     # CPU, ~1 min, cents: real tokenizer + image tokens, spans, head math
modal run deploy/modal_glimmer_clef.py --smoke        # GPU: one generated frame, 20 steps from empty weights, ~1.5 min, ~$0.1
modal run deploy/modal_glimmer_clef.py --run clef-v1  # every saved sample; --epochs --lr --head-lr --accum --lora --width --layers
```

Runs go to the `jevany-weights` Volume under `clef-runs/<run>/` (LoRA adapter, head, settings,
loss, decisions before/after) and to `eval/results/clef-<run>.json`. Each run also reloads its
saved files and checks they decide exactly like the model it just trained. It is not wired
into the page yet.

### Evaluate

```bash
python -m eval.run_eval --decider encoder          # MetaEncoder, frame + goal
python -m eval.run_eval --decider jevany           # JevAny (the run in JEVANY_RUN, once started)
python -m eval.run_eval --decider glimmer          # Glimmer with reasoning
python -m eval.run_eval --decider both             # every configured decider, one result file each
python -m eval.run_eval --decider jevany --data data/heldout.jsonl   # scenes kept out of training
python -m eval.run_eval --decider encoder --blind  # goal only: how much does the camera matter?
```

Metrics: accuracy, accuracy when the right object is present, **false pick when absent**,
ask-view rate, parse failures, latency; MetaEncoder rows also keep the full distribution for
calibrating a refusal threshold. The before/after numbers a training run prints are on its own
training frames; judge a run on scenes it never saw.

**Cost:** training reads ~1 record per second on an H100 (~$4/h), plus 1-2 min to load the
weights: roughly $1 per 1,000 records seen (e.g. 300 frames × 3 epochs).

## How decisions work

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

The options come from `domains/desk.json`; every object becomes a `TAKE_<obj>` action, and
*none* and *unclear view* are always added. MetaEncoder probabilities use an uncalibrated
tau=0.02. When its top-2 cosine margin is below `ENCODER_MIN_MARGIN` (default 0.05) it asks for
another view instead of guessing between near-ties; blind mode still reports the model's own
pick so the camera ablation stays honest. Tune the margin on the labeled eval set. JevAny gets
the same goal, frame and options as a state, an instruction and `{option: meaning}`, and its
head scores them directly.

`/api/decide` has one shared deadline (`REQUEST_TIMEOUT_S`, default 240 s) that also bounds
every upstream call inside it, Modal's poll redirects and the SAM highlight included
(`server/upstream.py`), so a stuck GPU frees its worker at the deadline; plus per-route rate
limits. The server listens on localhost only by default, because every request spends credits.

## Results so far

| Step | What | State |
|---|---|---|
| 1 | MetaEncoder + JevAny + Glimmer + SAM 3.1 on camera frames; labeling UI; eval with blind ablation | done; smoke-tested |
| 2 | Collect & label 100-300 frames on real desks; measure the deciders (accuracy, false pick when absent, latency) | next |
| 3 | Train: fine-tune JevAny on the labeled frames; Clef-style head from scratch as an experiment | code and one-frame GPU tests done (2026-11-04); waits for labeled frames |

**Speed on the same frames (2026-11-04).** Both GPUs warm (ready 105 s after Start), asked
through the local server exactly as the page asks, 3 goals × 2 runs per frame, medians; the
frame is a desk picture sent as the page sends it (JPEG, 1024×768) or a small 320×240 frame.
"GPU" is the time the service itself reports for the model.

| Frame | MetaEncoder: end to end / GPU | JevAny: end to end / GPU |
|---|---|---|
| desk, 1024×768 | 1.43 s / 0.15 s | 1.45 s / 0.17 s |
| small, 320×240 | 1.06 s / 0.06 s | 1.10 s / 0.11 s |

On the same frames the two decision models are equally fast, and the models are the small
part: a round trip from the laptop (Europe) to each service in us-east alone takes ~0.18-0.20 s,
and connection setup, the upload and Modal's routing make up the rest of the ~1.3 s. Both gave
the same answers on the desk frame (scissors, stapler, marker); on the small frame with no desk
object MetaEncoder said *none* for 2 of the 3 goals, JevAny for 1.

**MetaEncoder vs Glimmer (2026-10-28, two public photos, not an eval).** H100 in us-east,
client in Europe; latency end to end from the laptop.

| Frame | Goal | MetaEncoder | Glimmer (reasoning) |
|---|---|---|---|
| stationery | Cut the paper along a straight line | scissors p=0.999, 3.3 s (first call, embeds the options) | scissors, 5.5 s |
| stationery | Fasten these sheets together permanently | stapler p=0.999, 1.6 s | stapler, 2.9 s |
| stationery | Label this box | marker p=0.9997, 1.7 s | marker, 3.6 s |
| shredder | Cut the paper along a straight line | **none p=0.65** vs scissors 0.35, 1.9 s | none, 8.5 s |
| no image | Cut the paper along a straight line | scissors p=0.995 (before the blind-note fix) | ask for view, 2.1 s |

MetaEncoder is 2-4x faster and agrees with Glimmer when the object is visible, but it refuses
weakly: with the right object absent it said *none* by a thin margin, and with no image it still
picked by the goal text.

**JevAny live (2026-11-04).** On a grey frame with no desk object it still picks by the goal
text (scissors 0.88 for *cut the paper*): the released adapter was trained on text decisions,
the reason to fine-tune it on frames.

**Training smoke tests (2026-11-04, H100, one generated frame, answer *none*).** Fine-tuning
JevAny: loss 2.34 → 1.10 → 0.06 → 0.00, scissors 0.88 → none 1.00, peak GPU memory 58 GB,
2.5 min, ~$0.2. Clef-style from empty weights: the untrained head is near uniform (none 0.10
over 10 options); loss 3.16 → 2.63 → 1.55 → 0.00, none 1.00, the saved files reload with
identical probabilities; LoRA on 416 decoder modules (105M parameters) + head (138M), peak
64 GB, 1.4 min, ~$0.1. Both prove that the pipelines train and serve what they trained,
nothing about accuracy: one frame learned by heart says nothing about other frames.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Camera does not start | Use http://localhost (not the IP) on the computer, allow the permission, or use Photo / drag / paste. |
| Phone: *This site can't be reached* | Phone and computer on the same Wi-Fi (not a guest or hotspot network, VPN off on the phone); type the exact address the terminal prints. Windows: when the firewall asks, allow Python (tick *Public* too if your Wi-Fi is marked Public). Android: turn mobile data off for the test (some phones route traffic over it when Wi-Fi looks weak) and allow Chrome *Nearby devices* (Settings → Apps → Chrome → Permissions). Chrome or another non-Safari browser on iPhone: Settings → that browser → *Local Network* on. Some routers isolate Wi-Fi devices from each other: then use Tailscale. |
| Phone: *Your connection is not private* | Expected with `--phone` (self-signed certificate): *Advanced → Proceed*. It comes back when the computer's address changes and a new certificate is made. |
| Panel says *Not set up yet* | Run the command it names (`setup_modal.py all` for MetaEncoder, `setup_modal.py jevany` for JevAny); the page picks the result up without a restart. |
| JevAny Start ends with HTTP 5xx | `modal app logs glimmer-decide-jevany`: usually out of GPU memory (use an 80 GB card), the weights were never downloaded (rerun `setup_modal.py jevany`), or `JEVANY_RUN` names a run that does not exist (fix or empty it). |
| `GatedRepoError` during setup | Accept the terms on the model page and give the HF token gated-repo read access (step 3). |
| Start ends with HTTP 401 | `.env` and Modal disagree on `ENCODER_TOKEN`: rerun `python deploy/setup_modal.py all`. |
| *Starting GPU…* runs much longer than 15 min | `modal app logs glimmer-decide-metaencoder`; press Stop to end the billing. |
| MetaEncoder answers `upstream error (HTTP 404)` | The GPU app is stopped: press Start. |
| An answer says `upstream timed out` | The GPU was cold or busy past `REQUEST_TIMEOUT_S`; it keeps starting, so ask again in a minute. |
| `modal run …` says *not logged in* / *token missing* | `python deploy/setup_modal.py login`, or run `setup_modal.py jevany` once. |
| A training run fails with *run … already exists* | Run names are unique in the Volume: pass another `--run`. |

## Development

```
server/   FastAPI app (python -m server): encoder.py (MetaEncoder), jevany.py (JevAny), decision.py (Glimmer),
          sam.py, tasks.py (domains -> actions), upstream.py (request deadline for upstream calls),
          encoder_ctl.py (GPU Start/Stop), certs.py (self-signed HTTPS for phones)
deploy/   Modal: modal_metaencoder.py and modal_jevany.py (services), modal_jevany_train.py (JevAny fine-tuning),
          modal_glimmer_clef.py (Clef-style training), setup_modal.py (all / jevany / login / status / start / stop)
train/    saved samples -> training rows (jevany_data.py); Clef-style prompt layout and head (clef_format.py, clef_head.py)
domains/  action sets (desk.json)
web/      vanilla-JS camera client, no build step
eval/     offline evaluation (run_eval.py)
tests/    pytest, mock mode: no network, no keys
```

Checks, the same ones CI runs on every push (`.github/workflows/checks.yml`):

```bash
pip install -r requirements-dev.txt
ruff check .
mypy
python -m pytest
```

After changing `requirements*.txt`, refresh the pins with
`uv pip compile requirements.txt requirements-dev.txt --universal --python-version 3.13 -o constraints.txt`.

## License

The code here is MIT-licensed, see [LICENSE](LICENSE), except the joint schema head in
`train/clef_head.py`, ported from Cloudflare's Clef and kept under the Apache License 2.0
(`train/LICENSE-clef`). The models are not part of the repo and keep their own terms: you
accept MetaEncoder's license on its Hugging Face page before the weights download; JevAny
(adapter and code) and Muse Glimmer 30B's open weights are Apache-2.0 (Muse with Meta's usage
policy in its repository); and Glimmer and SAM 3.1 are used through OpenRouter and the Meta
Model API under those services' terms.
