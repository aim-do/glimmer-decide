"""Evaluate base Glimmer on a labeled set (default: data/captures.jsonl).

Usage:
  python -m eval.run_eval                     # with images
  python -m eval.run_eval --blind             # same questions, no image (camera ablation)
  python -m eval.run_eval --data data/my.jsonl --repeats 3

Each record: {"image": "captures/x.jpg", "domain": "desk", "goal": "...", "gold": ["TAKE_MARKER"]}
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from server.config import load_settings  # noqa: E402
from server.decision import decide  # noqa: E402
from server.tasks import ASK_VIEW, NONE, load_domain  # noqa: E402


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def summarize(rows: list[dict]) -> dict:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "captures.jsonl"))
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    settings = load_settings()
    data_path = Path(args.data)
    records = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [r for r in records if r.get("gold")]
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit(f"no labeled records in {data_path}")

    rows = []
    for i, rec in enumerate(records):
        domain = load_domain(rec["domain"])
        image = None if args.blind else data_url(data_path.parent / rec["image"])
        for _ in range(args.repeats):
            d = decide(settings, domain, rec["goal"], image)
            row = {"i": i, "image": rec["image"], "goal": rec["goal"], "gold": rec["gold"],
                   "pred": d.action, "correct": d.action in rec["gold"],
                   "latency_ms": d.latency_ms, "error": d.error}
            rows.append(row)
            mark = "ok " if row["correct"] else "ERR"
            print(f"[{mark}] {rec['goal'][:50]:50s} gold={rec['gold']} pred={d.action}")

    summary = summarize(rows)
    summary.update(model=settings.glimmer_model, blind=args.blind, mock=settings.mock,
                   reasoning=settings.glimmer_reasoning, data=str(data_path))
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}{'-blind' if args.blind else ''}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
