"""Clef-style records for Muse Glimmer: the prompt layout and the token spans the joint head reads.

Pure Python over any Hugging Face style tokenizer, so it is tested without torch. The layout is
Clef's (Cloudflare/clef, joint_schema_model.py, Apache-2.0) moved into Muse Glimmer's own chat
format: Clef's Qwen tokens <|im_start|> / <|im_end|> / <think></think> become Muse's
<|start|>role<|message|> ... <|eot|>, and the answer goes straight to the user channel
("assistant to=user"), skipping the reasoning channel the way Clef's empty think block does:

  <|begin_of_text|><|start|>system<|message|>{SYSTEM_PROMPT}<|eot|>
  <|start|>user<|message|>STATE:\n[image tokens]\n{state}
  \n\nSCHEMA FIELDS:\n\nFIELD 1\nID: action\nTYPE: choice\nINSTRUCTION: {instructions}
  \nALLOWED OPTIONS:\nOPTION 1: {"description":...,"option_id":"binder clip"}\n ... END FIELD\n
  <|eot|><|start|>assistant to=user<|message|>JOINT SCHEMA DECISIONS:

The head pools the instruction span, every option span and the whole sequence (clef_head.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = ("Read the complete state and schema. Decide every field jointly. Each answer "
                 "must be exactly one of that field's allowed options.")          # Clef's, unchanged
PREFIX = "<|begin_of_text|><|start|>system<|message|>" + SYSTEM_PROMPT + "<|eot|><|start|>user<|message|>STATE:\n"
SUFFIX = "\n<|eot|><|start|>assistant to=user<|message|>JOINT SCHEMA DECISIONS:"
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
FORMAT = "clef-glimmer v1"      # bump when the layout changes: a trained head only fits its own layout
HEAD_DEFAULTS = {"width": 1024, "routing_layers": 2, "layers": 4, "heads": 16, "feedforward": 4096}  # Clef's release


def render(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False,
                                                           separators=(",", ":"), sort_keys=True)


def safe(text: str) -> str:
    """User text must not open chat-format tokens (a goal of "<|eot|>..." would end the turn)."""
    return text.replace("<|", "‹|")


def question_options(question: dict) -> list[tuple[str, Any]]:
    """Clef's option order: choice keys sorted, noul true/false, score by level."""
    kind = str(question["type"])
    if kind == "noul":
        criteria = {"true": "The proposition is true or the answer is yes.",
                    "false": "The proposition is false or the answer is no."}
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if kind == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    question_type: int
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: tuple[int, ...]
    questions: tuple[EncodedQuestion, ...]
    record_id: str
    media: Any = field(default=None, compare=False, repr=False)            # processor tensors
    media_offset: int = 0                                                  # where the image tokens start
    token_aligned: dict = field(default_factory=dict, compare=False, repr=False)  # per-token processor lists


def _ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False).input_ids)


def encode_record(tokenizer, record: dict, *, media_ids: list[int] | None = None, media: Any = None,
                  token_aligned: dict | None = None, max_length: int = 8192,
                  max_state_tokens: int = 1024) -> EncodedRecord:
    """record: {"state", "questions": {id: {type, instructions, criteria}}, "id"?}. media_ids are
    the processor's expanded image tokens for the record's images (clef_head.encode_media)."""
    schema = _ids(tokenizer, "\n\nSCHEMA FIELDS:\n")
    questions = []
    for index, (question_id, question) in enumerate(record["questions"].items()):
        if str(question["type"]) not in QUESTION_TYPES:
            raise ValueError(f"{question_id}: type must be noul, choice or score")
        schema += _ids(tokenizer, f"\nFIELD {index + 1}\nID: {safe(str(question_id))}\n"
                                  f"TYPE: {question['type']}\nINSTRUCTION: ")
        start = len(schema)
        schema += _ids(tokenizer, safe(render(question.get("instructions") or str(question_id))))
        span = (start, len(schema))
        schema += _ids(tokenizer, "\nALLOWED OPTIONS:\n")
        spans, option_ids = [], []
        for option_index, (option_id, description) in enumerate(question_options(question)):
            schema += _ids(tokenizer, f"OPTION {option_index + 1}: ")
            option_start = len(schema)
            semantics = {"option_id": option_id} | ({"description": description} if description is not None else {})
            schema += _ids(tokenizer, safe(render(semantics)))
            spans.append((option_start, len(schema)))
            option_ids.append(option_id)
            schema += _ids(tokenizer, "\n")
        if len(option_ids) < 2:
            raise ValueError(f"{question_id}: needs at least two options")
        schema += _ids(tokenizer, "END FIELD\n")
        questions.append(EncodedQuestion(str(question_id), QUESTION_TYPES[str(question["type"])],
                                         span, tuple(spans), tuple(option_ids)))
    if not questions:
        raise ValueError("a record needs at least one question")
    prefix = _ids(tokenizer, PREFIX)
    offset = len(prefix)
    if media_ids:
        prefix = prefix + list(media_ids) + _ids(tokenizer, "\n")
    suffix = _ids(tokenizer, SUFFIX)
    state = _ids(tokenizer, safe(render(record["state"])))[:max_state_tokens]
    fixed = len(prefix) + len(schema) + len(suffix)
    if fixed > max_length:
        raise ValueError(f"the schema needs {fixed} tokens before the state; the limit is {max_length}")
    state = state[: max_length - fixed]
    shift = len(prefix) + len(state)
    moved = tuple(EncodedQuestion(q.question_id, q.question_type,
                                  (q.question_span[0] + shift, q.question_span[1] + shift),
                                  tuple((a + shift, b + shift) for a, b in q.option_spans), q.option_ids)
                  for q in questions)
    return EncodedRecord(tuple(prefix + state + schema + suffix), moved, str(record.get("id", "record")),
                         media=media, media_offset=offset if media_ids else 0,
                         token_aligned=dict(token_aligned or {}))


def row_to_record(row: dict, data_dir: Path, record_id: str = "record") -> tuple[dict, list[str], list[list[float]]]:
    """A labelled training row (train/jevany_data.py, the same text the page sends) -> a Clef record
    without labels, its image files, and one target distribution per question in option order."""
    questions, targets = {}, []
    for question_id, q in row["questions"].items():
        clean = {k: v for k, v in q.items() if k not in ("label", "target", "src")}
        questions[question_id] = clean
        keys = [key for key, _ in question_options(clean)]
        label = q["label"]
        if clean["type"] == "noul":
            label = "true" if label in (True, 1, "true") else "false"
        weights = q.get("target") or {str(label): 1.0}
        if set(map(str, weights)) - set(keys):
            raise ValueError(f"{question_id}: label or target outside its options")
        total = float(sum(weights.values()))
        if total <= 0:
            raise ValueError(f"{question_id}: the target puts no weight on any option")
        targets.append([float(weights.get(key, 0.0)) / total for key in keys])
    images = [str((Path(data_dir) / m["uri"]).resolve()) for m in row.get("media", []) if m.get("type") == "image"]
    return {"id": record_id, "state": row["state"], "questions": questions}, images, targets
