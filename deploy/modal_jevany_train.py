"""Fine-tune JevAny on your labelled camera frames, on one Modal GPU.

  modal run deploy/modal_jevany_train.py --smoke        # one generated frame, 20 steps: proves training runs
  modal run deploy/modal_jevany_train.py --run desk-v1  # every labelled frame in data/captures.jsonl
  modal run deploy/modal_jevany_train.py --data data/other.jsonl --epochs 3 --lr 2e-5

What trains: the released adapter (SimpleJev/JevAny-Muse-Glimmer-30B-LoRA: LoRA r8 on the Muse
Glimmer 30B decoder plus the pointer head) is warm-started and trained further by JevAny's own
trainer (jevany.train in its `init_from` mode, as in the project's finetune recipe). The 60 GB base
stays frozen in bf16 on one 80 GB GPU; the LoRA and the head train. Rows carry the exact text the
decision service reads (server/jevany.py: training_row), so training and serving see the same input.

Each run is kept in the jevany-weights Volume under runs/<name>/:
  data/          the rows and images it trained on
  model/         the new adapter + head (a JevAny checkpoint)
  summary.json   loss per record and the decisions on the training frames before and after
and locally in eval/results/train-<name>.json. To serve it: JEVANY_RUN=<name> in .env, then Stop
and Start JevAny in the page.

Cost: the GPU bills only while this runs. It loads the weights three times (decide before, train,
decide after): ~1 min the first time, less after. The smoke run took 2.5 min (~$0.2 on an H100,
2026-11-04: scissors 0.88 -> none 1.00 on its frame). Hard 2 h timeout, no retries.
"""
import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if modal.is_local():
    # `modal run` from the repo: JEVANY_GPU comes from .env, the same as for the page's Start.
    from dotenv import load_dotenv

    load_dotenv(HERE.parent / ".env")

import modal_jevany as svc  # noqa: E402  same image, Volume, GPU and pinned weights as the service

# The released adapter's exact LoRA modules (its adapter_config.json): a warm start must match them.
LORA_MODULES = "down_proj,gate_proj,k_proj,o_proj,q_proj,up_proj,v_proj"
SMOKE = {"epochs": 20, "lr": 1e-4, "head_lr": 1e-4, "accum": 1}   # one record: 20 epochs = 20 steps
DEFAULTS = {"epochs": 3, "lr": 2e-5, "head_lr": 0.0, "accum": 8}  # head_lr 0 = same as lr

app = modal.App("glimmer-decide-jevany-train")
image = svc.image.add_local_python_source("modal_jevany")


def trainer_args(data: str, out: str, s: dict) -> list[str]:
    """jevany.train flags: the released checkpoint's architecture (head.pt: lora 8, head 256 with a
    256 residual MLP, rows branches, muse_vision, bf16 base) plus this run's schedule."""
    return ["--data", data, "--out", out,
            "--base", svc.BASE, "--base_revision", svc.BASE_REV,
            "--init_from", f"{svc.ADAPTER}@{svc.ADAPTER_REV}",
            "--multimodal", "--backbone_adapter", "muse_vision", "--branch_mode", "rows",
            "--decision_mode", "pointer", "--lora", "8", "--lora_target_modules", LORA_MODULES,
            "--lora_dropout", "0.0", "--head_dim", "256", "--head_residual_dim", "256",
            "--device", "cuda", "--dtype", "bf16", "--weights_dtype", "bf16", "--checkpointing", "1",
            "--batch", "1", "--accum", str(s["accum"]), "--epochs", str(s["epochs"]),
            "--max_steps", str(s["max_steps"]), "--lr", str(s["lr"]), "--head_lr", str(s["head_lr"]),
            # Keep the page's option set: no injected "none of these" or distractor options.
            "--p_none", "0", "--p_none_distract", "0", "--p_distract", "0", "--seed", "0"]


@app.function(image=image, gpu=svc.GPU, volumes={"/cache": svc.weights}, memory=32768,
              timeout=7200, retries=0)
def fine_tune(rows: list[dict], images: dict[str, bytes], run: str, settings: dict) -> dict:
    """Write the data set into the Volume, decide on its frames with the released adapter, train,
    decide again with the new one. Returns the summary (also saved as runs/<run>/summary.json)."""
    import gc

    import jevany.train as trainer
    import torch
    from jevany.checkpoint import LoadOptions
    from jevany.runtime import JevModel

    if not svc.RUN_NAME.fullmatch(run):
        raise ValueError(f"bad run name {run!r}")
    svc.weights.reload()
    root = Path("/cache/runs") / run
    if root.exists():
        raise ValueError(f"run {run!r} already exists in the Volume; pass another --run")
    data = root / "data"
    (data / "images").mkdir(parents=True)
    for name, raw in images.items():
        (data / "images" / Path(name).name).write_bytes(raw)
    (data / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                      encoding="utf-8")
    probe_rows = rows[: settings["probe"]]
    seconds: dict[str, float] = {}

    def decide_all(checkpoint: str, phase: str) -> list[dict]:
        t0 = time.time()
        model = JevModel.from_pretrained(checkpoint, device="cuda",
                                         options=LoadOptions(dtype=torch.bfloat16, merge=False))
        out = []
        for row in probe_rows:
            q = row["questions"]["action"]
            gold = list(q.get("target") or [q["label"]])
            request = {"state": row["state"],
                       "media": [{"type": "image", "uri": str(data / row["media"][0]["uri"])}],
                       "questions": {"action": {"type": "choice", "instructions": q["instructions"],
                                                "criteria": q["criteria"]}}}
            probs = model(request)["answers"]["action"]["probabilities"]
            top = sorted(probs.items(), key=lambda kv: -kv[1])
            out.append({"goal": row["state"]["user goal"], "gold": gold, "choice": top[0][0],
                        "p_gold": round(sum(probs.get(g, 0.0) for g in gold), 4),
                        "top3": [[k, round(v, 4)] for k, v in top[:3]]})
        del model
        gc.collect()
        torch.cuda.empty_cache()
        seconds[phase] = round(time.time() - t0, 1)
        print(f"[{phase}] decided {len(out)} frame(s) in {seconds[phase]} s; "
              f"GPU memory still held: {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
        return out

    before = decide_all(f"{svc.ADAPTER}@{svc.ADAPTER_REV}", "before")

    losses: list[float] = []
    original = trainer.batch_loss

    def logged(model, a, batch, *args, **kwargs):
        loss, terms = original(model, a, batch, *args, **kwargs)
        losses.append(round(terms["ce"] / max(len(batch), 1), 4))
        print(f"[train] record {len(losses)}: loss {losses[-1]:.4f}", flush=True)
        return loss, terms

    t0 = time.time()
    trainer.batch_loss = logged      # the trainer prints every 10 steps; this logs every record
    try:
        trainer.main(trainer_args(str(data / "train.jsonl"), str(root / "model"), settings))
    finally:
        trainer.batch_loss = original
    gc.collect()
    torch.cuda.empty_cache()
    seconds["train"] = round(time.time() - t0, 1)
    metrics = json.loads((root / "model" / "training_metrics.json").read_text(encoding="utf-8"))

    after = decide_all(str(root / "model"), "after")
    summary = {"run": run, "gpu": svc.GPU, "records": len(rows), "settings": settings,
               "optimizer_steps": metrics.get("optimizer_steps"),
               "peak_gpu_gb": round((metrics.get("peak_device_bytes") or 0) / 1e9, 1),
               "losses": losses, "before": before, "after": after, "seconds": seconds,
               "model": f"runs/{run}/model"}
    (root / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    svc.weights.commit()
    return summary


def report(summary: dict) -> str:
    lines = [f"run {summary['run']} on {summary['gpu']}: {summary['records']} frame(s), "
             f"{summary['optimizer_steps']} optimizer steps, peak GPU {summary['peak_gpu_gb']} GB, "
             f"time {summary['seconds']}",
             "loss per record: " + " ".join(f"{x:.3f}" for x in summary["losses"])]
    for b, a in zip(summary["before"], summary["after"]):
        lines.append(f"  {b['goal'][:48]!r} correct={'/'.join(b['gold'])}: "
                     f"before {b['choice']} (p correct {b['p_gold']:.3f}) -> "
                     f"after {a['choice']} (p correct {a['p_gold']:.3f})")
    return "\n".join(lines)


@app.local_entrypoint()
def main(data: str = "data/captures.jsonl", run: str = "", smoke: bool = False, epochs: int = 0,
         steps: int = 0, lr: float = 0.0, head_lr: float = -1.0, accum: int = 0, probe: int = 8):
    """Unset numbers take the smoke or the regular defaults (SMOKE / DEFAULTS above)."""
    root = HERE.parent
    sys.path.insert(0, str(root))
    from train.jevany_data import captures_dataset, smoke_dataset

    if smoke:
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
    base = SMOKE if smoke else DEFAULTS
    settings = {"epochs": epochs or base["epochs"], "max_steps": steps, "lr": lr or base["lr"],
                "head_lr": base["head_lr"] if head_lr < 0 else head_lr,
                "accum": accum or base["accum"], "probe": probe}
    run = run or time.strftime("%Y%m%d-%H%M%S") + ("-smoke" if smoke else "")
    if not svc.RUN_NAME.fullmatch(run):
        sys.exit(f"--run {run!r}: use lowercase letters, digits, '.', '_' or '-'")
    size = sum(len(b) for b in ds.images.values()) / 1e6
    print(f"run {run}: {len(ds.rows)} frame(s), {size:.1f} MB of images, settings {settings}, GPU {svc.GPU}")
    summary = fine_tune.remote(ds.rows, ds.images, run, settings)
    print(report(summary))
    out = root / "eval" / "results" / f"train-{run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"saved {out.relative_to(root)}; to serve it: JEVANY_RUN={run} in .env, then Stop and Start JevAny.")
