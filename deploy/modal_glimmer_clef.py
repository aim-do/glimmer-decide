"""Clef-style Glimmer: Clef's joint schema head + a fresh LoRA on Muse Glimmer 30B, trained on Modal.

  modal run deploy/modal_glimmer_clef.py --selftest     # CPU, ~1 min, cents: real Muse tokenizer and
                                                        #   image processor on the prompt layout, head math
  modal run deploy/modal_glimmer_clef.py --smoke        # GPU: one generated frame, 20 steps from empty weights
  modal run deploy/modal_glimmer_clef.py --run clef-v1  # every sample in data/captures.jsonl

Unlike deploy/modal_jevany_train.py nothing is warm-started: the LoRA starts at zero (B = 0, so the
backbone first answers exactly as the base model) and the head from its default initialisation. The
head is Cloudflare Clef's (train/clef_head.py, ported, Apache-2.0); the prompt is Clef's layout in
Muse's chat format (train/clef_format.py); the loss is cross-entropy + Brier score. The rows come
from the same saved samples as JevAny's (train/jevany_data.py), with the page's own text.

Runs land in the jevany-weights Volume (it already holds the base) under clef-runs/<name>/:
  data/         the rows and images
  model/        adapter_model.safetensors + adapter_config.json (LoRA), joint_head.safetensors +
                joint_head_config.json (head), clef_glimmer.json (base, layout, settings)
  summary.json  loss per record, decisions before/after, and a reload check of the saved files
and locally in eval/results/clef-<name>.json. Serving it in the page is not wired yet.

Cost: GPU billing only while it runs; the weights load twice (train, then a reload check from the
saved files). The smoke run took 1.4 min (~$0.1 on an H100, 2026-11-04: loss 3.16 -> 0.00, the
right answer 0.10 -> 1.00, reload identical). Hard 2 h timeout, no retries.
"""
import json
import sys
import time
from pathlib import Path
from typing import Any

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
if modal.is_local():
    from dotenv import load_dotenv

    load_dotenv(HERE.parent / ".env")       # JEVANY_GPU, the same as for the page's Start

import modal_jevany as svc  # noqa: E402  same image, Volume, GPU and pinned base as the JevAny service

from train.clef_format import HEAD_DEFAULTS  # noqa: E402

COMMON: dict[str, Any] = {"lora": 16, "lora_alpha": 32, "weight_decay": 0.01, "brier_weight": 1.0, "max_image_tokens": 512,
          "checkpointing": True, "seed": 0, "max_steps": 0, **HEAD_DEFAULTS}
SMOKE = {"epochs": 20, "lr": 1e-4, "head_lr": 1e-3, "accum": 1, "lora_dropout": 0.0}   # 1 record: 20 steps
DEFAULTS = {"epochs": 5, "lr": 1e-4, "head_lr": 3e-4, "accum": 8, "lora_dropout": 0.05}

app = modal.App("glimmer-decide-clef-train")
image = svc.image.add_local_python_source("modal_jevany", "train")


def write_dataset(rows: list[dict], images: dict[str, bytes], data: Path) -> None:
    (data / "images").mkdir(parents=True)
    for name, raw in images.items():
        (data / "images" / Path(name).name).write_bytes(raw)
    (data / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                      encoding="utf-8")


@app.function(image=image, volumes={"/cache": svc.weights}, cpu=4.0, memory=16384, timeout=900, retries=0)
def cpu_selftest(rows: list[dict], images: dict[str, bytes]) -> dict:
    """No GPU: Muse's real tokenizer and image processor on the Clef layout (special tokens, image
    tokens, spans), the model's forward signature, and the head + loss learning on random features."""
    import inspect
    import tempfile

    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor

    from train.clef_format import encode_record, row_to_record, safe
    from train.clef_head import JointSchemaHead, collate, decision_loss, encode_media

    report: dict = {}
    data = Path(tempfile.mkdtemp()) / "data"
    write_dataset(rows, images, data)
    processor = AutoProcessor.from_pretrained(svc.BASE, revision=svc.BASE_REV)
    tok = processor.tokenizer
    report["special_tokens"] = {s: tok(s, add_special_tokens=False).input_ids for s in
                                ("<|begin_of_text|>", "<|start|>", "<|message|>", "<|eot|>", processor.image_token)}
    record, paths, targets = row_to_record(rows[0], data)
    ids, media, aligned = encode_media(processor, paths, COMMON["max_image_tokens"])
    image_id = getattr(processor, "image_token_id", None) or tok.convert_tokens_to_ids(processor.image_token)
    report["media"] = {"tokens": len(ids), "image_tokens": ids.count(image_id),
                       "tensors": {k: list(v.shape) if torch.is_tensor(v) else type(v).__name__ for k, v in media.items()},
                       "token_aligned": sorted(aligned)}
    enc = encode_record(tok, record, media_ids=ids, media=media, token_aligned=aligned)
    q = enc.questions[0]
    report["layout"] = {"tokens": len(enc.input_ids), "starts_with_bos": enc.input_ids[0] == tok.bos_token_id,
                        "image_tokens_in_record": list(enc.input_ids).count(image_id),
                        "instruction": tok.decode(enc.input_ids[slice(*q.question_span)]),
                        "instruction_expected": safe(record["questions"][q.question_id]["instructions"]),
                        "options": [tok.decode(enc.input_ids[slice(*s)]) for s in q.option_spans[:3]],
                        "option_ids": list(q.option_ids), "head_of_prompt": tok.decode(enc.input_ids[:40])}
    config = AutoConfig.from_pretrained(svc.BASE, revision=svc.BASE_REV)
    inner = getattr(transformers, config.architectures[0].replace("ForConditionalGeneration", "Model"), None)
    params = set(inspect.signature(inner.forward).parameters) if inner else set()
    report["forward"] = {"class": getattr(inner, "__name__", None),
                         "media_passed": sorted(k for k in media if k in params),
                         "media_dropped": sorted(k for k in media if k not in params),
                         "token_aligned_passed": sorted(k for k in aligned if k in params)}
    torch.manual_seed(0)                       # head + loss on random 64-d "hidden states"
    hidden = 64
    head = JointSchemaHead(hidden_size=hidden, width=32, routing_layers=1, layers=1, heads=4, feedforward=64)
    batch = collate([enc], tok.pad_token_id or 0, "cpu")
    states = torch.randn(1, len(enc.input_ids), hidden)
    table = torch.randn(max(enc.input_ids) + 1, hidden)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    goal = torch.tensor(targets[0])
    losses = []
    for _ in range(30):
        logits = head(states, batch["input_ids"], batch["attention_mask"], batch["records"], table)[0]
        loss, _, _ = decision_loss(logits, [goal])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(round(loss.item(), 4))
    report["head_cpu_losses"] = losses[:3] + ["..."] + losses[-3:]
    return report


@app.function(image=image, gpu=svc.GPU, volumes={"/cache": svc.weights}, memory=32768, timeout=7200, retries=0)
def train_clef(rows: list[dict], images: dict[str, bytes], run: str, settings: dict) -> dict:
    from train.clef_head import train

    if not svc.RUN_NAME.fullmatch(run):
        raise ValueError(f"bad run name {run!r}")
    svc.weights.reload()
    root = Path("/cache/clef-runs") / run
    if root.exists():
        raise ValueError(f"run {run!r} already exists in the Volume; pass another --run")
    write_dataset(rows, images, root / "data")
    summary = train(rows, root / "data", root / "model", settings, base=svc.BASE, revision=svc.BASE_REV,
                    log=lambda line: print(line, flush=True))
    summary.update(run=run, gpu=svc.GPU, settings=settings, model=f"clef-runs/{run}/model")
    (root / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    svc.weights.commit()
    return summary


def settings_for(smoke: bool, **overrides) -> dict:
    """COMMON + the smoke or regular schedule; overrides that are None/0 keep the default."""
    out = {**COMMON, **(SMOKE if smoke else DEFAULTS), "probe": 8}
    out.update({k: v for k, v in overrides.items() if v})
    return out


def report(summary: dict) -> str:
    lines = [f"run {summary['run']} on {summary['gpu']}: {summary['records']} frame(s), "
             f"{summary['optimizer_steps']} optimizer steps, LoRA on {summary['lora_modules']} modules, "
             f"peak GPU {summary['peak_gpu_gb']} GB, time {summary['seconds']}",
             "loss per record: " + " ".join(f"{x:.3f}" for x in summary["losses"]),
             f"saved files decide like the trained model: max difference {summary['reload_max_diff']:.1e}"]
    for b, a in zip(summary["before"], summary["after"]):
        lines.append(f"  {b['goal'][:48]!r} correct={'/'.join(b['gold'])}: "
                     f"before {b['choice']} (p correct {b['p_gold']:.3f}) -> "
                     f"after {a['choice']} (p correct {a['p_gold']:.3f})")
    return "\n".join(lines)


@app.local_entrypoint()
def main(data: str = "data/captures.jsonl", run: str = "", smoke: bool = False, selftest: bool = False,
         epochs: int = 0, steps: int = 0, lr: float = 0.0, head_lr: float = 0.0, accum: int = 0,
         lora: int = 0, width: int = 0, layers: int = 0, probe: int = 0):
    """Unset numbers take the smoke or the regular defaults (SMOKE / DEFAULTS / COMMON above)."""
    from train.jevany_data import captures_dataset, smoke_dataset

    if smoke or selftest:
        ds = smoke_dataset()
    else:
        path = Path(data)
        if not path.is_file():
            sys.exit(f"{path} not found: label frames in the page first "
                     "(Save sample: tick every correct action, then Save frame + goal + label).")
        ds = captures_dataset(path)
    for why in ds.skipped:
        print("skipped", why)
    if not ds.rows:
        sys.exit("no usable labelled frames")
    if selftest:
        print(json.dumps(cpu_selftest.remote(ds.rows[:1], ds.images), indent=1, ensure_ascii=False))
        return
    settings = settings_for(smoke, epochs=epochs, max_steps=steps, lr=lr, head_lr=head_lr, accum=accum,
                            lora=lora, width=width, layers=layers, probe=probe)
    if lora:
        settings["lora_alpha"] = 2 * lora                  # keep alpha / r = 2
    run = run or time.strftime("%Y%m%d-%H%M%S") + ("-smoke" if smoke else "")
    if not svc.RUN_NAME.fullmatch(run):
        sys.exit(f"--run {run!r}: use lowercase letters, digits, '.', '_' or '-'")
    print(f"clef run {run}: {len(ds.rows)} frame(s), settings {settings}, GPU {svc.GPU}")
    summary = train_clef.remote(ds.rows, ds.images, run, settings)
    print(report(summary))
    out = HERE.parent / "eval" / "results" / f"clef-{run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"saved {out.relative_to(HERE.parent)}")
