# Glimmer Decide

Point a camera at a desk (or drop in a photo), say what you want to do, and get **one object
to use** — or *nothing here fits* / *show me another angle* — with a probability for every
option. **SAM 3.1** then highlights the chosen object on the frame.

Built for the Meta Global AI Developer Hackathon, on Meta models only, no training yet:

| Role | Model | How it decides | Where it runs |
|---|---|---|---|
| **System 1** (default) | [MetaEncoder](https://huggingface.co/facebook/meta-encoder): Muse Glimmer 30B contrastively tuned into a bi-encoder ([paper](https://arxiv.org/abs/2610.11316)) | frame + goal and each option are embedded; cosine similarity -> softmax. One forward pass, no text generated, option embeddings cached | your Modal GPU (`deploy/`) |
| **System 1**, second opinion | [JevAny-Muse-Glimmer-30B](https://huggingface.co/SimpleJev/JevAny-Muse-Glimmer-30B-LoRA): a LoRA + pointer head on Muse Glimmer 30B ([JevAny](https://github.com/SimpleJev/JevAny)) | frame + goal + labelled options in one prefill; a small trained head scores the options. Trained on text decisions only, so photos are the open question | your Modal GPU (`deploy/`) |
| **System 2** (compare / fallback) | [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | reasons, then answers `ANSWER: <letter>` | OpenRouter |
| Highlight | SAM 3.1 | text concept -> boxes | Meta Model API |

```
camera frame ─┐                       ┌─► MetaEncoder: cos(task, option_i) ─► softmax ─► choice + p
goal (text)  ─┼─► task / options ─────┼─► JevAny: one prefill ─► pointer head ─────────► choice + p
options      ─┘   (closed set:        └─► Glimmer: reasoning ─► ANSWER: <letter>        (compare)
                  TAKE_<obj>|NONE|ASK_VIEW)
                                   chosen object ─► SAM 3.1 ─► box overlay
```

## Run it

1. **Install** once (Python 3.11+ and git):

   ```bash
   git clone https://github.com/aim-do/glimmer-decide
   cd glimmer-decide
   python -m venv .venv            # macOS / Linux: python3 -m venv .venv
   .venv\Scripts\activate          # macOS / Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Try it with no keys:** `python -m server --mock` opens http://localhost:8000 with fake
   decisions, so the whole interface works. This first run also creates `.env`.

3. **Add your keys to `.env`.** Each one turns on one part; Glimmer alone is already a demo.

   | Part | In `.env` | Where to get it |
   |---|---|---|
   | Glimmer (reasoning) | `GLIMMER_API_KEY=` | an [OpenRouter key](https://openrouter.ai/keys) with a few dollars of credit |
   | SAM 3.1 box highlight | `MODEL_API_KEY=` | a [Meta Model API](https://dev.meta.ai) key |
   | MetaEncoder (fast decider) | `HF_TOKEN=`, then run `python deploy/setup_modal.py all` once | accept the terms on [facebook/meta-encoder](https://huggingface.co/facebook/meta-encoder), then create a *Read* [token](https://huggingface.co/settings/tokens); a [Modal](https://modal.com) account for the GPU |
   | JevAny (second fast decider) | nothing new: run `python deploy/setup_modal.py jevany` once | the same Modal account; the weights are open (it downloads ~60 GB once, on a cheap CPU container) |

   MetaEncoder has no hosted API, so the repo deploys it to a GPU in your Modal account
   (`deploy/modal_metaencoder.py`). `setup_modal.py all` signs you in to Modal in the
   browser, stores the HF token there, deploys, and writes `ENCODER_URL` and `ENCODER_TOKEN`
   into `.env` itself.

4. **Start:** `python -m server --phone` (or just `python -m server` without a phone). It prints:

   ```
     This computer:       https://localhost:8000
     Phone (same Wi-Fi):  https://192.168.1.23:8000
   ```

   Open the phone address on your phone; the page on the computer shows it too. The first
   time, each browser warns about the self-signed certificate: *Advanced → Proceed*
   (iPhone: *Show Details → visit this website*). Then the phone camera works.

5. **GPUs:** press **Start** in the *MetaEncoder GPU* (or *JevAny GPU*) panel and wait ~2-3 min;
   press **Stop** when you are done. Each GPU costs ~$4/h while on and sleeps after 5 idle minutes.

### GPU deciders: details

- MetaEncoder and JevAny each need their own 80 GB card (62 GB and 59 GB of weights): running
  both costs ~$8/h. Modal bills per second; hackathon credits apply.
- The HF token can also be fine-grained, with *Read access to contents of all public gated
  repos you can access*. JevAny needs no Hugging Face access at all.
- The very first MetaEncoder Start downloads its 62 GB of weights into a Modal Volume (~14 min);
  JevAny's setup downloads its weights beforehand. Later starts take ~2-3 min. An H100 is
  ~$4/h; `ENCODER_GPU` / `JEVANY_GPU=A100-80GB` in `.env` is ~$2.5/h.
- **Stop** shuts a GPU down and undeploys it, so nothing can wake it. From a terminal:

```bash
python deploy/setup_modal.py status                 # deployed? GPU running? (both)
python deploy/setup_modal.py start [encoder|jevany] # deploy; a request then wakes the GPU
python deploy/setup_modal.py stop                   # both GPUs off, apps undeployed
```

## Ways to give it a picture

| Where | How | Notes |
|---|---|---|
| Laptop / desktop | webcam on the *This computer* address | browsers allow the camera on localhost |
| Any computer | **Photo** button, **drag an image** onto the frame, or **paste** one (Ctrl+V) | no camera needed — screenshots and saved photos work |
| Phone on the same Wi-Fi | `python -m server --phone`, then open the *Phone* address it prints (the page on the computer shows it too) | the certificate is self-signed, so the phone warns once: *Advanced → Proceed* (iPhone: *Show Details → visit this website*); after that the live camera works. Anyone on that Wi-Fi can open the page and spend your credits |
| Phone anywhere | `tailscale serve --bg 8000` on the computer, then `https://<computer>.<tailnet>.ts.net` on the phone | real certificate, no warning, only your own devices ([Tailscale](https://tailscale.com) on both) |

Phones get the camera only over HTTPS. `--host 0.0.0.0` without `--phone` also works, but then
only **Photo** does (it still opens the phone's camera app).

Then type the goal (or tap 🎤 where the browser supports speech), pick the decider —
MetaEncoder, JevAny, Glimmer, or **Compare all** — and press **What should I use?** The button
counts the seconds while the models think; each answer shows how long its model took (⏱)
and the total time stays under the button.

## Labeling and evaluation

In the app, open *Save sample*, tick every correct action and save. Frames go to
`data/captures/`, labels to `data/captures.jsonl` (both git-ignored).

```bash
python -m eval.run_eval --decider encoder          # MetaEncoder, frame + goal
python -m eval.run_eval --decider jevany           # JevAny decision head
python -m eval.run_eval --decider glimmer          # Glimmer with reasoning
python -m eval.run_eval --decider both             # every configured decider, one result file each
python -m eval.run_eval --decider encoder --blind  # goal only: how much does the camera matter?
```

Metrics: accuracy, accuracy when the right object is present, **false pick when absent**,
ask-view rate, parse failures, latency; MetaEncoder rows also keep the full distribution
for calibrating a refusal threshold. Hold out whole scenes, not neighbouring frames.

## Fine-tuning JevAny on your frames

JevAny is the decider you can train, and the samples you save are its training data. Training
runs on a Modal GPU with JevAny's own trainer, continuing from the released adapter (LoRA r8 +
pointer head; the 30B base stays frozen), on the exact text the page sends at decision time.
It needs `python deploy/setup_modal.py jevany` once (the weights it trains on).

```bash
modal run deploy/modal_jevany_train.py --smoke        # one generated frame, 20 steps: does training run? ~2.5 min, ~$0.2
modal run deploy/modal_jevany_train.py --run desk-v1  # every sample in data/captures.jsonl, 3 epochs
```

A run prints the loss per record and its decision on the training frames before and after
(also saved to `eval/results/train-<run>.json`), and keeps the new adapter in the
`jevany-weights` Volume under `runs/<run>/`. To use it in the page: `JEVANY_RUN=desk-v1` in
`.env`, then **Stop** and **Start** the JevAny GPU; its answers then end with *fine-tuned run:
desk-v1*. An empty `JEVANY_RUN` goes back to the released adapter. Flags: `--data`, `--epochs`,
`--steps`, `--lr`, `--head-lr`, `--accum` (`--help` lists them). The before/after numbers are
on the training frames themselves: judge a run with `eval.run_eval` on scenes it never saw.

First smoke test (2026-11-04, H100): a grey frame with no desk object, goal *Cut the paper
along a straight line*, correct answer *none*. The released adapter said scissors 0.88 (none
0.08), the same as the live service on that frame. After 20 steps at lr 1e-4 the loss went
2.34 → 1.10 → 0.06 → 0.00 (from step 4) and the answer became none 1.00; peak GPU memory
58 GB, 2.5 min and ~$0.2 in all. That proves the pipeline trains and serves what it trained,
nothing more: one frame learned by heart says nothing about other frames.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Camera does not start | Use http://localhost (not the IP) on the computer, allow the permission, or use Photo / drag / paste. |
| Phone: *This site can't be reached* | Phone and computer on the same Wi-Fi (not a guest or hotspot network, VPN off on the phone); type the exact address the terminal prints. Windows: when the firewall asks, allow Python (tick *Public* too if your Wi-Fi is marked Public). Android: turn mobile data off for the test (some phones route traffic over it when Wi-Fi looks weak) and allow Chrome *Nearby devices* (Settings → Apps → Chrome → Permissions). Chrome or another non-Safari browser on iPhone: Settings → that browser → *Local Network* on. Some routers isolate Wi-Fi devices from each other: then use Tailscale. |
| Phone: *Your connection is not private* | Expected with `--phone` (self-signed certificate): *Advanced → Proceed*. It comes back when the computer's address changes and a new certificate is made. |
| Panel says *Not set up yet* | Run the command it names (`setup_modal.py all` for MetaEncoder, `setup_modal.py jevany` for JevAny); the page picks the result up without a restart. |
| JevAny Start ends with HTTP 5xx | `modal app logs glimmer-decide-jevany`: usually out of GPU memory (use an 80 GB card) or the weights were never downloaded (rerun `setup_modal.py jevany`). |
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
| 3 | Adapt: fine-tune JevAny on the labeled frames (Modal); later maybe a Clef-style joint head on Glimmer, or a small head on frozen MetaEncoder features | JevAny training code + one-frame GPU smoke test done (2026-11-04); waits for labeled frames |

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
server/   FastAPI app (python -m server); encoder.py (MetaEncoder decider), jevany.py (JevAny decider),
          encoder_ctl.py (GPU Start/Stop), certs.py (self-signed HTTPS for phones), decision.py (Glimmer),
          sam.py, tasks.py
deploy/   Modal services for MetaEncoder and JevAny + setup_modal.py (all / jevany / status / start / stop),
          modal_jevany_train.py (JevAny fine-tuning on a GPU)
domains/  action sets (desk.json)
web/      vanilla-JS camera client
train/    saved samples -> JevAny training rows
eval/     offline evaluation
tests/    pytest (mock mode, no network): python -m pytest
```

## License

The code here is MIT-licensed, see [LICENSE](LICENSE). The models are not part of the repo
and keep their own terms: you accept MetaEncoder's license on its Hugging Face page before
the weights download, and Glimmer and SAM 3.1 are used through OpenRouter and the Meta
Model API under those services' terms.
