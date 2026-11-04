"""JevAny decision service on Modal: Muse Glimmer 30B + the JevAny pointer head on one 80 GB GPU.

JevAny (github.com/SimpleJev/JevAny, Apache-2.0) is a System 1 decision model: the state, the
question and the labelled options go through one backbone prefill and a small learned head
scores the options; no text is generated. The adapter (SimpleJev/JevAny-Muse-Glimmer-30B-LoRA)
holds only the LoRA and that head; the open base weights come from meta-models/Muse-Glimmer-30B.

  python deploy/setup_modal.py jevany          # one time: deploy, download weights, write JEVANY_URL
  modal run deploy/modal_jevany.py::download   # the download step alone, on a cheap CPU container
  modal deploy deploy/modal_jevany.py          # what the page's Start button runs

Secret: glimmer-decide-auth (ENCODER_TOKEN), the same bearer token the MetaEncoder service uses.
JevAny's own HTTP server has no authentication, so this file loads the model in-process
(JevModel) and serves it behind that token:

  POST /decide {"state": {...}, "instructions": "...", "criteria": {label: meaning}, "image": <data URL>?}
    -> {"choice": label, "probabilities": {label: p}, "latency_ms": model time, "run": name | null}

JEVANY_RUN=<name> (in .env when Start deploys) serves a fine-tuned run from the Volume
(runs/<name>/model, made by deploy/modal_jevany_train.py) instead of the released adapter.
"""
import os
import re

import modal
from pydantic import BaseModel, Field

ADAPTER = "SimpleJev/JevAny-Muse-Glimmer-30B-LoRA"
ADAPTER_REV = "8b390d11364e090287645a751658eb086be86fba"
BASE = "meta-models/Muse-Glimmer-30B"
BASE_REV = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"     # pinned by the adapter's release-manifest.json
JEVANY_REV = "75aa500d99878af94beca49bd89382cbee46e424"   # JevAny v0.3.0, the adapter's release
GPU = os.environ.get("JEVANY_GPU") or "H100"              # 59 GB of bf16 weights: any 80 GB card
RUN_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")       # fine-tuned runs live in /cache/runs/<name>
RUN = os.environ.get("JEVANY_RUN", "").strip()            # empty = the released adapter
if RUN and not RUN_NAME.fullmatch(RUN):
    raise ValueError(f"JEVANY_RUN={RUN!r} is not a run name (lowercase letters, digits, . _ -)")
CHECKPOINT = f"/cache/runs/{RUN}/model" if RUN else f"{ADAPTER}@{ADAPTER_REV}"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(f"jevany[serve,multimodal] @ git+https://github.com/SimpleJev/JevAny@{JEVANY_REV}",
                 "huggingface_hub[hf_xet]", "fastapi[standard]")
    .env({"HF_HOME": "/cache/hf", "HF_XET_HIGH_PERFORMANCE": "1"})
)
weights = modal.Volume.from_name("jevany-weights", create_if_missing=True)
app = modal.App("glimmer-decide-jevany")


@app.function(image=image, volumes={"/cache": weights}, timeout=3600, cpu=4.0)
def download() -> None:
    """Fetch the ~60 GB base and the adapter into the Volume on a CPU container (cents), so the
    GPU never idles through the download on its first start."""
    from huggingface_hub import snapshot_download

    snapshot_download(BASE, revision=BASE_REV)
    snapshot_download(ADAPTER, revision=ADAPTER_REV)
    weights.commit()
    print("JevAny weights are in the Volume.")


class DecideRequest(BaseModel):
    state: dict[str, str] | str
    instructions: str = Field(min_length=1, max_length=2_000)
    criteria: dict[str, str] = Field(min_length=2, max_length=64)
    image: str | None = Field(default=None, max_length=12_000_000)   # data URL


@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/cache": weights},
    # The run name travels into the container as an env var, so CHECKPOINT resolves the same there.
    secrets=[modal.Secret.from_name("glimmer-decide-auth"),
             *([modal.Secret.from_dict({"JEVANY_RUN": RUN})] if RUN else [])],
    memory=16384,              # the bf16 base passes through host RAM (~60 GB) on its way to the GPU
    scaledown_window=300,      # idle 5 min -> container stops, billing stops
    max_containers=1,          # never fan out to several GPUs by accident
    startup_timeout=1800,      # covers a first start that still has to download the weights
    timeout=600,
)
class JevAnyService:
    @modal.enter()
    def load(self):
        import tempfile
        import threading

        import torch
        from jevany.checkpoint import LoadOptions
        from jevany.runtime import JevModel
        from PIL import Image

        if RUN and not os.path.isfile(os.path.join(CHECKPOINT, "head.pt")):
            raise RuntimeError(f"JEVANY_RUN={RUN}: no such run in the jevany-weights Volume "
                               f"(runs/{RUN}/model); train it first or clear JEVANY_RUN in .env")
        # bf16 and unmerged: JevAny's default merges the LoRA in fp32, which first loads a 120 GB
        # base that no 80 GB card holds. Unmerged bf16 is a little slower and fits.
        self.model = JevModel.from_pretrained(CHECKPOINT, device="cuda",
                                              options=LoadOptions(dtype=torch.bfloat16, merge=False))
        weights.commit()           # keep anything this load had to fetch (when download() was skipped)
        self.media = tempfile.mkdtemp(prefix="jevany-media-")
        self.lock = threading.Lock()
        # Warm the text and the vision paths so the first camera frame is fast.
        warm = os.path.join(self.media, "warmup.jpg")
        Image.new("RGB", (1024, 768), (127, 127, 127)).save(warm, "JPEG")
        options = {"left": "the first option", "right": "the second option"}
        self.answer({"note": "warm-up"}, "Pick one.", options, None)
        self.answer({"note": "warm-up"}, "Pick one.", options, warm)
        os.unlink(warm)

    def answer(self, state, instructions: str, criteria: dict[str, str], image_path: str | None) -> dict:
        request = {"state": state, "questions": {"action": {
            "type": "choice", "instructions": instructions, "criteria": criteria}}}
        if image_path:
            request["media"] = [{"type": "image", "uri": image_path}]   # in-process: a trusted local file
        with self.lock:            # one model per container; keep calls serial
            out = self.model(request)
        ans = out["answers"]["action"]
        return {"choice": ans["choice"], "probabilities": ans["probabilities"],
                "latency_ms": out.get("latency_ms"), "run": RUN or None}

    @modal.asgi_app()
    def web(self):
        import base64
        import binascii
        import hmac
        import io
        import uuid

        from fastapi import FastAPI, Header, HTTPException
        from PIL import Image, UnidentifiedImageError

        api = FastAPI()
        token = os.environ["ENCODER_TOKEN"]

        def check(authorization: str | None) -> None:
            if not authorization or not hmac.compare_digest(authorization, f"Bearer {token}"):
                raise HTTPException(401, "bad token")

        @api.get("/health")
        def health(authorization: str | None = Header(None)):
            check(authorization)
            return {"ok": True, "model": ADAPTER, "run": RUN or None, "base": BASE, "gpu": GPU}

        @api.post("/decide")
        def decide(req: DecideRequest, authorization: str | None = Header(None)):
            check(authorization)
            path = None
            if req.image is not None:
                if not req.image.startswith("data:image/") or "," not in req.image:
                    raise HTTPException(400, "image must be a data URL")
                try:
                    raw = base64.b64decode(req.image.split(",", 1)[1], validate=True)
                    frame = Image.open(io.BytesIO(raw)).convert("RGB")
                except (binascii.Error, ValueError, UnidentifiedImageError, OSError):
                    raise HTTPException(400, "image could not be decoded") from None
                path = os.path.join(self.media, f"{uuid.uuid4().hex}.jpg")
                frame.save(path, "JPEG", quality=92)
            try:
                return self.answer(req.state, req.instructions, req.criteria, path)
            except ValueError as e:
                raise HTTPException(422, str(e)) from None
            finally:
                if path:
                    os.unlink(path)

        return api
