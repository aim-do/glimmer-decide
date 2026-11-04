"""Evaluate a decider on a labeled set (default: data/captures.jsonl). No training involved.

Usage:
  python -m eval.run_eval --decider encoder   # MetaEncoder, frame + goal
  python -m eval.run_eval --decider glimmer   # generative Glimmer with reasoning
  python -m eval.run_eval --decider both      # both deciders, one output file each
  python -m eval.run_eval --decider encoder --blind   # goal only (camera ablation)
  python -m eval.run_eval --data data/my.jsonl --repeats 3

Each record: {"image": "captures/x.jpg", "domain": "desk", "goal": "...", "gold": ["TAKE_MARKER"]}
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import time
import uuid
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from server.config import load_settings  # noqa: E402
from server.decision import decide  # noqa: E402
from server.encoder import decide_encoder  # noqa: E402
from server.tasks import ASK_VIEW, NONE, load_domain  # noqa: E402

DECIDERS = {"encoder": decide_encoder, "glimmer": decide}


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def load_records(data_path: Path) -> tuple[list[dict], int]:
    """Parse captures.jsonl, skipping (and counting) malformed lines."""
    records, skipped = [], 0
    for line in data_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            skipped += 1
            continue
        if not isinstance(rec, dict) or not rec.get("gold"):
            skipped += 1
            continue
        records.append(rec)
    return records, skipped


def summarize(rows: list[dict]) -> dict:
    rows = [r for r in rows if not r.get("skipped")]   # data problems are not model errors
    n = len(rows)
    absent = [r for r in rows if r["gold"] == [NONE]]
    present = [r for r in rows if NONE not in r["gold"] and ASK_VIEW not in r["gold"]]
    return {
        "n": n,
        "accuracy": sum(r["correct"] for r in rows) / n if n else None,
        # The dangerous error: confidently picking an object when the right one is not there.
        "false_pick_when_absent": (sum(r["pred"] not in (NONE, ASK_VIEW, None) for r in absent) / len(absent)
                                   if absent else None),
        "accuracy_when_present": sum(r["correct"] for r in present) / len(present) if present else None,
        "ask_view_rate": sum(r["pred"] == ASK_VIEW for r in rows) / n if n else None,
        "parse_failures": sum(r["pred"] is None for r in rows),
        "mean_latency_ms": int(sum(r["latency_ms"] for r in rows) / n) if n else None,
        "pred_counts": dict(Counter(str(r["pred"]) for r in rows)),
    }


def run_one(settings, decider: str, records: list[dict], data_path: Path,
            blind: bool, repeats: int) -> Path:
    decide_fn = DECIDERS[decider]
    domains: dict[str, object] = {}
    rows = []
    for i, rec in enumerate(records):
        img_path = data_path.parent / rec["image"]
        if not blind and not img_path.exists():
            print(f"[skip] missing image {img_path}")
            rows.append({"i": i, "image": rec["image"], "goal": rec["goal"], "gold": rec["gold"],
                         "pred": None, "correct": False, "latency_ms": 0,
                         "error": "missing image file", "probs": None, "skipped": True})
            continue
        if rec["domain"] not in domains:
            domains[rec["domain"]] = load_domain(rec["domain"])
        domain = domains[rec["domain"]]
        image = None if blind else data_url(img_path)
        for _ in range(repeats):
            d = decide_fn(settings, domain, rec["goal"], image)
            row = {"i": i, "image": rec["image"], "goal": rec["goal"], "gold": rec["gold"],
                   "pred": d.action, "correct": d.action in rec["gold"],
                   "latency_ms": d.latency_ms, "error": d.error, "probs": d.probs}
            rows.append(row)
            mark = "ok " if row["correct"] else "ERR"
            print(f"[{mark}] {rec['goal'][:50]:50s} gold={rec['gold']} pred={d.action}")

    summary = summarize(rows)
    summary.update(decider=decider, blind=blind, mock=settings.mock, data=str(data_path),
                   model=("facebook/meta-encoder" if decider == "encoder" else settings.glimmer_model),
                   reasoning=(None if decider == "encoder" else settings.glimmer_reasoning),
                   encoder_style=(settings.encoder_style if decider == "encoder" else None))
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S') + "-" + uuid.uuid4().hex[:6]
    out = out_dir / f"{stamp}-{decider}{'-blind' if blind else ''}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"-> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "captures.jsonl"))
    ap.add_argument("--decider", choices=sorted(DECIDERS) + ["both"], default=None)
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-mock", action="store_true", help="accept MOCK=1 (numbers are fake)")
    args = ap.parse_args()

    settings = load_settings()
    if settings.mock and not args.allow_mock:
        raise SystemExit("MOCK=1: results would be fake. Unset MOCK or pass --allow-mock.")
    decider = args.decider or settings.decider
    data_path = Path(args.data)
    records, skipped = load_records(data_path)
    if skipped:
        print(f"skipped {skipped} malformed/unlabeled record(s) in {data_path}")
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit(f"no labeled records in {data_path}")

    for name in (["encoder", "glimmer"] if decider == "both" else [decider]):
        run_one(settings, name, records, data_path, args.blind, args.repeats)


if __name__ == "__main__":
    main()
