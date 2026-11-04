"""Clef-style Glimmer: Clef's joint schema head on Muse Glimmer 30B with a fresh LoRA, trained here.

EvidenceRoutingLayer and JointSchemaHead are ported from Cloudflare's Clef release
(huggingface.co/Cloudflare/clef, joint_schema_model.py at revision cfddda0), Apache License 2.0
(copy in train/LICENSE-clef). Their math is unchanged. What differs from Clef: the backbone is
Muse Glimmer 30B (hidden size 6656) instead of Qwen3.8-27B; it carries a fresh LoRA on its
decoder instead of Clef's merged weights; the prompt is in Muse's chat format (clef_format.py);
and the training loop, loss, save and load below are ours, since Clef's release ships inference
code only. Training starts from empty weights: LoRA B = 0 (the backbone answers as the base model
does), and the head from its default initialisation.

Needs torch, transformers and peft (the Modal image of deploy/modal_jevany.py) and a CUDA GPU.
"""
from __future__ import annotations

import gc
import inspect
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from .clef_format import FORMAT, HEAD_DEFAULTS, EncodedRecord, encode_record, row_to_record

# The decoder's attention and MLP projections only: the vision tower and projector stay frozen.
LORA_TARGETS = r".*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


# ---- Clef's head (ported, math unchanged) --------------------------------------------------------

class EvidenceRoutingLayer(torch.nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query_norm = torch.nn.LayerNorm(width)
        self.memory_norm = torch.nn.LayerNorm(width)
        self.attention = torch.nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.attention_dropout = torch.nn.Dropout(dropout)
        self.feedforward_norm = torch.nn.LayerNorm(width)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(width, feedforward), torch.nn.GELU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(feedforward, width), torch.nn.Dropout(dropout))

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_queries = self.query_norm(queries)
        routed, _ = self.attention(normalized_queries, self.memory_norm(memory), self.memory_norm(memory),
                                   need_weights=False)
        queries = queries + self.attention_dropout(routed)
        return queries + self.feedforward(self.feedforward_norm(queries))


class JointSchemaHead(torch.nn.Module):
    """Reads the backbone's final hidden states: each option's span (context), its tokens' output
    embeddings (lexical), the instruction span, the last token (global) and the whole sequence
    (memory); routes evidence from the sequence to every option, then scores all options of all
    questions jointly. Returns one logit vector per question."""

    def __init__(self, hidden_size: int, width: int, routing_layers: int, layers: int, heads: int,
                 feedforward: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_norm = torch.nn.LayerNorm(hidden_size)
        self.memory_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.question_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.global_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = torch.nn.Embedding(3, width)
        self.evidence_layers = torch.nn.ModuleList(
            [EvidenceRoutingLayer(width=width, heads=heads, feedforward=feedforward, dropout=dropout)
             for _ in range(routing_layers)])
        self.option_summary_norm = torch.nn.LayerNorm(width)
        self.layers = torch.nn.ModuleList(
            [torch.nn.TransformerDecoderLayer(d_model=width, nhead=heads, dim_feedforward=feedforward,
                                              dropout=dropout, activation="gelu", batch_first=True,
                                              norm_first=True)
             for _ in range(layers)])
        self.field_norm = torch.nn.LayerNorm(width)
        self.option_norm = torch.nn.LayerNorm(width)
        self.residual_scorer = torch.nn.Sequential(
            torch.nn.Linear(width * 4, width), torch.nn.GELU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(width, 1))
        self.prior_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.residual_gate = torch.nn.Parameter(torch.zeros(()))

    @staticmethod
    def _mean_span(values: torch.Tensor, span: tuple[int, int]) -> torch.Tensor:
        start, end = span
        return values[start:end].mean(dim=0)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                records: list[EncodedRecord], output_embedding_weight: torch.Tensor) -> list[list[torch.Tensor]]:
        results: list[list[torch.Tensor]] = []
        normalized_hidden = self.hidden_norm(hidden_states)
        for batch_index, record in enumerate(records):
            sequence_length = int(attention_mask[batch_index].sum().item())
            sequence_hidden = normalized_hidden[batch_index, :sequence_length]
            memory = self.memory_projection(sequence_hidden).unsqueeze(0)
            global_vector = sequence_hidden[-1]
            question_vectors = torch.stack([self._mean_span(sequence_hidden, q.question_span)
                                            for q in record.questions])
            type_ids = torch.tensor([q.question_type for q in record.questions], device=hidden_states.device)
            option_contexts, lexical_options, option_counts = [], [], []
            for question in record.questions:
                option_contexts.append(torch.stack([self._mean_span(sequence_hidden, span)
                                                    for span in question.option_spans]))
                lexical_options.append(torch.stack(
                    [output_embedding_weight[input_ids[batch_index, start:end]].mean(dim=0)
                     for start, end in question.option_spans]))
                option_counts.append(len(question.option_spans))

            option_queries = [self.option_context_projection(context) + self.option_lexical_projection(lexical)
                              + self.option_question_projection(question_vectors[index]).unsqueeze(0)
                              for index, (context, lexical) in enumerate(zip(option_contexts, lexical_options))]
            routed_options = torch.cat(option_queries, dim=0).unsqueeze(0)
            for layer in self.evidence_layers:
                routed_options = layer(routed_options, memory)
            split_options = list(torch.split(routed_options[0], option_counts, dim=0))

            base_fields = self.question_projection(question_vectors)
            option_summaries = []
            for field, options in zip(base_fields, split_options):
                weights = torch.softmax(torch.matmul(options, field) / math.sqrt(options.shape[-1]), dim=0)
                option_summaries.append(torch.sum(weights.unsqueeze(-1) * options, dim=0))
            fields = (base_fields + self.option_summary_norm(torch.stack(option_summaries))
                      + self.global_projection(global_vector).unsqueeze(0) + self.type_embedding(type_ids))
            fields = fields.unsqueeze(0)
            for layer in self.layers:
                fields = layer(fields, memory)
            fields = self.field_norm(fields[0])

            record_logits: list[torch.Tensor] = []
            for field, lexical, routed in zip(fields, lexical_options, split_options):
                anchor = functional.normalize(question_vectors[len(record_logits)] + global_vector, dim=-1)
                prior_scale = self.prior_logit_scale.clamp(max=math.log(100.0)).exp()
                prior = prior_scale * torch.matmul(functional.normalize(lexical, dim=-1), anchor)
                options = self.option_norm(routed)
                repeated_field = field.unsqueeze(0).expand_as(options)
                cosine = functional.cosine_similarity(repeated_field, options, dim=-1)
                features = torch.cat([repeated_field, options, repeated_field * options,
                                      torch.abs(repeated_field - options)], dim=-1)
                residual = self.residual_scorer(features).squeeze(-1)
                joint_scale = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
                record_logits.append(prior + torch.sigmoid(self.residual_gate) * (joint_scale * cosine + residual))
            results.append(record_logits)
        return results


# ---- backbone, batches, loss ---------------------------------------------------------------------

class ClefGlimmer(torch.nn.Module):
    """Muse Glimmer (with its LoRA) -> final hidden states -> the joint head."""

    def __init__(self, backbone, head: JointSchemaHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def _base(self):
        return self.backbone.get_base_model() if hasattr(self.backbone, "get_base_model") else self.backbone

    def forward(self, batch: dict) -> list[list[torch.Tensor]]:
        base = self._base()
        inner = base.model                                    # vision tower + decoder, no LM head
        model = inner if batch["media"] else getattr(inner, "language_model", inner)
        media = media_inputs(model, batch["media"])
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False,
                    return_dict=True, **media)
        return self.head(out.last_hidden_state, batch["input_ids"], batch["attention_mask"], batch["records"],
                         base.get_output_embeddings().weight)


def media_inputs(model, media: dict) -> dict:
    """The processor tensors this model's forward names explicitly (pixel values, image grid, ...).
    Anything else (metadata, extras) is left out instead of leaking into **kwargs."""
    params = inspect.signature(model.forward).parameters
    return {k: v for k, v in media.items() if torch.is_tensor(v) and k in params}


def encode_media(processor, paths: list[str], max_image_tokens: int = 512):
    """Images -> (expanded image token ids, processor tensors, per-token processor lists)."""
    if not paths:
        return [], {}, {}
    out = processor(text=[processor.image_token * len(paths)], images=list(paths), return_tensors="pt",
                    images_kwargs={"max_image_tokens": max(1, max_image_tokens // len(paths))})
    ids = out["input_ids"][0].tolist()
    drop = 1 if ids and ids[0] == processor.tokenizer.bos_token_id else 0   # our prefix has the BOS
    n = len(ids)
    media, aligned = {}, {}
    for key, value in out.items():
        if key in ("input_ids", "attention_mask"):
            continue
        if (torch.is_tensor(value) and value.dim() == 2 and value.shape[0] == 1 and value.shape[1] == n
                and not value.is_floating_point()):
            aligned[key] = value[0, drop:].tolist()
        else:
            media[key] = value
    return ids[drop:], media, aligned


def collate(records: list[EncodedRecord], pad_token_id: int, device) -> dict:
    length = max(len(r.input_ids) for r in records)
    input_ids = torch.full((len(records), length), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(records), length), dtype=torch.long, device=device)
    for i, r in enumerate(records):
        input_ids[i, :len(r.input_ids)] = torch.tensor(r.input_ids, device=device)
        attention_mask[i, :len(r.input_ids)] = 1
    media: dict = {}
    for key in sorted({k for r in records if r.media for k in r.media}):
        values = [r.media[key] for r in records if r.media and key in r.media]
        media[key] = torch.cat(values, dim=0).to(device) if torch.is_tensor(values[0]) else values[0]
    for key in sorted({k for r in records for k in r.token_aligned}):
        tokens = torch.zeros((len(records), length), dtype=torch.long, device=device)
        for i, r in enumerate(records):
            if key in r.token_aligned:
                values = torch.tensor(r.token_aligned[key], dtype=torch.long, device=device)
                tokens[i, r.media_offset:r.media_offset + len(values)] = values
        media[key] = tokens
    return {"input_ids": input_ids, "attention_mask": attention_mask, "records": records, "media": media}


def decision_loss(logits: list[torch.Tensor], targets: list[torch.Tensor], brier_weight: float = 1.0):
    """Cross-entropy against the target distribution plus brier_weight x the multi-class Brier score
    (sum of squared probability errors), averaged over questions: CE fits the answer, Brier keeps
    the confidence calibrated. Returns (loss, ce, brier)."""
    ce = brier = 0.0
    for z, t in zip(logits, targets):
        log_p = functional.log_softmax(z.float(), dim=-1)
        ce = ce - (t * log_p).sum()
        brier = brier + ((log_p.exp() - t) ** 2).sum()
    n = max(len(logits), 1)
    return (ce + brier_weight * brier) / n, ce / n, brier / n


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def free_gpu() -> float:
    gc.collect()
    torch.cuda.empty_cache()
    return round(torch.cuda.memory_allocated() / 1e9, 1)


# ---- load / decide -------------------------------------------------------------------------------

def load_base(base: str, revision: str):
    import transformers
    from transformers import AutoConfig, AutoProcessor

    config = AutoConfig.from_pretrained(base, revision=revision)
    model_class = getattr(transformers, config.architectures[0])    # MuseGlimmerForConditionalGeneration
    processor = AutoProcessor.from_pretrained(base, revision=revision)
    model = model_class.from_pretrained(base, revision=revision, dtype=torch.bfloat16, device_map={"": "cuda"})
    model.requires_grad_(False)
    model.config.use_cache = False
    text = getattr(config, "text_config", None) or config
    return model, processor, int(text.hidden_size)


def load(run_dir: str | Path):
    """A trained run (model/ written by train) -> (ClefGlimmer, processor, meta), ready to decide."""
    from peft import PeftModel
    from safetensors.torch import load_file

    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "clef_glimmer.json").read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise ValueError(f"{run_dir}: format {meta.get('format')!r}, this code reads {FORMAT!r}")
    model, processor, _ = load_base(meta["base"], meta["base_revision"])
    backbone = PeftModel.from_pretrained(model, str(run_dir))
    head = JointSchemaHead(**json.loads((run_dir / "joint_head_config.json").read_text(encoding="utf-8")))
    head.load_state_dict(load_file(str(run_dir / "joint_head.safetensors")), strict=True)
    return ClefGlimmer(backbone, head.to("cuda")).eval(), processor, meta


@torch.no_grad()
def decide(clef: ClefGlimmer, processor, examples: list[dict]) -> list[dict]:
    """examples: {"encoded", "targets", "record"} -> per question: choice, p of the right answer, top 3."""
    was_training = clef.training
    clef.eval()
    out = []
    for ex in examples:
        with amp():
            logits = clef(collate([ex["encoded"]], processor.tokenizer.pad_token_id, "cuda"))[0]
        state = ex["record"]["state"]
        goal = state.get("user goal", "") if isinstance(state, dict) else str(state)[:80]
        for q, z, t in zip(ex["encoded"].questions, logits, ex["targets"]):
            probs = dict(zip(q.option_ids, z.float().softmax(-1).tolist()))
            gold = [k for k, w in zip(q.option_ids, t.tolist()) if w > 0]
            top = sorted(probs.items(), key=lambda kv: -kv[1])
            out.append({"goal": goal, "gold": gold, "choice": top[0][0],
                        "p_gold": round(sum(probs[g] for g in gold), 4),
                        "top3": [[k, round(v, 4)] for k, v in top[:3]], "probs": probs})
    clef.train(was_training)
    return out


# ---- training ------------------------------------------------------------------------------------

def enable_checkpointing(model) -> str:
    """Recompute decoder activations in the backward pass (~10 GB less memory on 52 layers)."""
    kwargs = {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    decoder = getattr(getattr(model, "model", None), "language_model", None)
    for target, name in ((decoder, "decoder"), (model, "whole model")):
        if target is not None and hasattr(target, "gradient_checkpointing_enable"):
            try:
                target.gradient_checkpointing_enable(**kwargs)
                return name
            except (ValueError, AttributeError, TypeError):
                continue
    return "unavailable"


def train(rows: list[dict], data_dir: Path, out_dir: Path, settings: dict, *, base: str, revision: str,
          log=print) -> dict:
    """Fresh LoRA + fresh head, trained on the rows; saves the run into out_dir and checks that the
    saved files decide exactly like the model that was just trained. Returns the summary."""
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file

    seconds, t0 = {}, time.time()
    torch.manual_seed(settings["seed"])
    rng = random.Random(settings["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    model, processor, hidden = load_base(base, revision)
    lora = LoraConfig(r=settings["lora"], lora_alpha=settings["lora_alpha"], lora_dropout=settings["lora_dropout"],
                      target_modules=LORA_TARGETS, bias="none")
    backbone = get_peft_model(model, lora)                # LoRA B starts at zero: the base model, unchanged
    wrapped = [n for n, _ in backbone.named_modules() if n.endswith(".lora_A")]
    if not wrapped or any("vision" in n or "visual" in n for n in wrapped):
        raise RuntimeError(f"LoRA targets look wrong: {len(wrapped)} modules, e.g. {wrapped[:3]}")
    checkpointing = enable_checkpointing(backbone.get_base_model()) if settings["checkpointing"] else "off"
    head_config = {"hidden_size": hidden, **{k: settings[k] for k in HEAD_DEFAULTS}}
    head = JointSchemaHead(**head_config).to("cuda")      # fp32 master weights; autocast runs it in bf16
    clef = ClefGlimmer(backbone, head)
    lora_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = list(head.parameters())
    seconds["load"] = round(time.time() - t0, 1)
    log(f"[load] {seconds['load']} s; LoRA on {len(wrapped)} decoder modules, "
        f"{sum(p.numel() for p in lora_params) / 1e6:.1f}M LoRA + {sum(p.numel() for p in head_params) / 1e6:.1f}M "
        f"head parameters to train; gradient checkpointing: {checkpointing}; "
        f"GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    t = time.time()
    examples: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        record, images, targets = row_to_record(row, data_dir, f"r{i}")
        ids, media, aligned = encode_media(processor, images, settings["max_image_tokens"])
        encoded = encode_record(processor.tokenizer, record, media_ids=ids, media=media, token_aligned=aligned)
        examples.append({"encoded": encoded, "record": record,
                         "targets": [torch.tensor(x, device="cuda") for x in targets]})
    image_token = getattr(processor, "image_token_id", None) or processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    tokens = [len(ex["encoded"].input_ids) for ex in examples]
    image_tokens = [ex["encoded"].input_ids.count(image_token) for ex in examples]
    seconds["encode"] = round(time.time() - t, 1)
    log(f"[data] {len(examples)} records, tokens {min(tokens)}-{max(tokens)}, image tokens {min(image_tokens)}-{max(image_tokens)}")

    probe = examples[: settings["probe"]]
    t = time.time()
    before = decide(clef, processor, probe)
    seconds["before"] = round(time.time() - t, 1)

    accum, epochs = settings["accum"], settings["epochs"]
    planned = epochs * math.ceil(len(examples) / accum)
    steps = min(planned, settings["max_steps"]) if settings["max_steps"] else planned
    opt = torch.optim.AdamW([{"params": lora_params, "lr": settings["lr"]},
                             {"params": head_params, "lr": settings["head_lr"]}], weight_decay=settings["weight_decay"])
    if steps > 2:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[settings["lr"], settings["head_lr"]],
                                                    total_steps=steps, pct_start=max(0.1, 2 / steps))
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    log(f"[train] {steps} optimizer steps ({epochs} epochs x {len(examples)} records, accum {accum}); "
        f"lr {settings['lr']} LoRA / {settings['head_lr']} head; loss = CE + {settings['brier_weight']} x Brier")
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    clef.train()
    losses, step = [], 0
    for epoch in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for k, index in enumerate(order):
            ex = examples[index]
            group = min(accum, len(order) - (k // accum) * accum)
            with amp():
                logits = clef(collate([ex["encoded"]], processor.tokenizer.pad_token_id, "cuda"))[0]
            loss, ce, brier = decision_loss(logits, ex["targets"], settings["brier_weight"])
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            (loss / group).backward()
            losses.append(round(loss.item(), 4))
            log(f"[train] record {len(losses)} (epoch {epoch + 1}): loss {losses[-1]:.4f} "
                f"(ce {ce.item():.4f}, brier {brier.item():.4f})")
            if (k + 1) % accum == 0 or k + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step >= steps:
                    break
        if step >= steps:
            break
    seconds["train"] = round(time.time() - t, 1)
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 1)

    out_dir.mkdir(parents=True)
    backbone.save_pretrained(str(out_dir))                   # adapter_model.safetensors + adapter_config.json
    save_file({k: v.detach().float().cpu().contiguous() for k, v in head.state_dict().items()},
              str(out_dir / "joint_head.safetensors"))
    (out_dir / "joint_head_config.json").write_text(json.dumps(head_config, indent=1), encoding="utf-8")
    meta = {"format": FORMAT, "base": base, "base_revision": revision,
            "max_image_tokens": settings["max_image_tokens"],
            "lora": {"r": settings["lora"], "alpha": settings["lora_alpha"], "dropout": settings["lora_dropout"],
                     "targets": LORA_TARGETS},
            "training": {k: settings[k] for k in ("epochs", "lr", "head_lr", "accum", "brier_weight", "seed")},
            "optimizer_steps": step, "records": len(examples),
            "head_source": "Cloudflare/clef joint_schema_model.py @ cfddda0 (Apache-2.0)"}
    (out_dir / "clef_glimmer.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    t = time.time()
    after = decide(clef, processor, probe)
    seconds["after"] = round(time.time() - t, 1)

    del clef, head, backbone, model, opt, sched, lora_params, head_params, logits, loss, ce, brier
    held = free_gpu()
    t = time.time()
    reloaded, processor, _ = load(out_dir)                   # the serving path, from the saved files
    again = decide(reloaded, processor, probe)
    seconds["reload"] = round(time.time() - t, 1)
    drift = max((abs(a["probs"][k] - b["probs"][k]) for a, b in zip(after, again) for k in a["probs"]), default=0.0)
    del reloaded
    free_gpu()
    for item in before + after:
        item.pop("probs", None)
    seconds["total"] = round(time.time() - t0, 1)
    log(f"[done] reload from the saved files: max probability difference {drift:.2e} "
        f"(GPU held before reload: {held} GB)")
    return {"records": len(examples), "optimizer_steps": step, "planned_steps": planned, "losses": losses,
            "before": before, "after": after, "reload_max_diff": round(drift, 6), "peak_gpu_gb": peak,
            "tokens": [min(tokens), max(tokens)], "image_tokens": [min(image_tokens), max(image_tokens)],
            "lora_modules": len(wrapped), "seconds": seconds}
