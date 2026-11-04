"""MetaEncoder embedding service on Modal (one 80 GB GPU, scales to zero).

Deploy:   modal deploy deploy/modal_metaencoder.py
Needs two Modal secrets (deploy/setup_modal.py creates them from local files):
  huggingface          HF_TOKEN      read token of an account that accepted facebook/meta-encoder's terms
  glimmer-decide-auth  ENCODER_TOKEN shared bearer token; the same value goes to ENCODER_TOKEN in .env

Serving follows the vLLM recipe on the model card: pooling runner, last-token pooling,
L2-normalised embeddings, the repo's vllm_metaencoder.py for prompt building.
"""
import os

import modal
from pydantic import BaseModel, Field

MODEL_REPO = "facebook/meta-encoder"
WEIGHTS_DIR = "/cache/meta-encoder"
GPU = os.environ.get("ENCODER_GPU") or "H100"   # any 80 GB card: H100, A100-80GB

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Versions pinned by the model card; newer torchvision changes image resampling.
    .pip_install("vllm==0.23.0", "transformers==5.15.1", "torchvision<0.27",
                 "huggingface_hub[hf_xet]", "pillow", "numpy", "fastapi[standard]")
    .env({"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "HF_XET_HIGH_PERFORMANCE": "1"})
)
weights = modal.Volume.from_name("metaencoder-weights", create_if_missing=True)
app = modal.App("glimmer-decide-metaencoder")


class Item(BaseModel):
    text: str = Field(default="", max_length=20_000)
    image: str | None = Field(default=None, max_length=12_000_000)   # data URL


class EmbedRequest(BaseModel):
    items: list[Item] = Field(min_length=1, max_length=64)


@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/cache": weights},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("glimmer-decide-auth")],
    scaledown_window=300,      # idle 5 min -> container stops, billing stops
    max_containers=1,          # never fan out to several GPUs by accident
    startup_timeout=1800,      # first start downloads ~62 GB of weights
    timeout=600,
)
class MetaEncoderService:
    @modal.enter()
    def load(self):
        import sys
        import threading

        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor
        from vllm import LLM
        from vllm.config import PoolerConfig

        path = snapshot_download(MODEL_REPO, local_dir=WEIGHTS_DIR)
        weights.commit()
        sys.path.insert(0, path)
        import vllm_metaencoder

        vllm_metaencoder.register()
        self.ve = vllm_metaencoder
        self.processor = AutoProcessor.from_pretrained(path)
        self.llm = LLM(
            model=path, runner="pooling", dtype="bfloat16", max_model_len=8192,
            enforce_eager=True, limit_mm_per_prompt={"image": 2}, attention_backend="FLASH_ATTN",
            pooler_config=PoolerConfig(pooling_type="LAST", use_activation=True),
        )
        self.lock = threading.Lock()
        # Warm the text and vision paths at startup: without it the first camera frame
        # after a cold start paid ~25 s of kernel JIT (seen 2026-10-28), the second 0.6 s.
        from PIL import Image

        frame = Image.new("RGB", (1024, 768), (127, 127, 127))
        self.llm.embed([self.ve.to_prompt(self.processor, {"text": "warmup"}),
                        self.ve.to_prompt(self.processor, {"text": "warmup", "image": frame})],
                       tokenization_kwargs=self.ve.TOKENIZATION_KWARGS)

    def _embed(self, items: list[Item]) -> list:
        import base64
        import io

        from PIL import Image

        prepared = []
        for it in items:
            d = {}
            if it.text:
                d["text"] = it.text
            if it.image:
                if "," not in it.image:
                    raise ValueError("image data URL is missing its comma separator")
                raw = base64.b64decode(it.image.split(",", 1)[1])
                d["image"] = Image.open(io.BytesIO(raw)).convert("RGB")
            prepared.append(d)
        prompts = [self.ve.to_prompt(self.processor, d) for d in prepared]
        with self.lock:   # one LLM engine per container; keep calls serial
            outs = self.llm.embed(prompts, tokenization_kwargs=self.ve.TOKENIZATION_KWARGS)
        return [o.outputs.embedding for o in outs]

    @modal.asgi_app()
    def web(self):
        import base64
        import hmac

        import numpy as np
        from fastapi import FastAPI, Header, HTTPException

        api = FastAPI()
        token = os.environ["ENCODER_TOKEN"]

        def check(authorization: str | None) -> None:
            if not authorization or not hmac.compare_digest(authorization, f"Bearer {token}"):
                raise HTTPException(401, "bad token")

        @api.get("/health")
        def health(authorization: str | None = Header(None)):
            check(authorization)
            return {"ok": True, "model": MODEL_REPO, "gpu": GPU}

        @api.post("/embed")
        def embed(req: EmbedRequest, authorization: str | None = Header(None)):
            check(authorization)
            if any(it.image and not it.image.startswith("data:image/") for it in req.items):
                raise HTTPException(400, "image must be a data URL")
            vecs = [np.asarray(v, dtype=np.float32) for v in self._embed(req.items)]
            return {"dim": int(vecs[0].shape[0]),
                    "embeddings": [base64.b64encode(v.tobytes()).decode() for v in vecs]}

        return api
