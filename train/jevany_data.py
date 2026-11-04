"""Saved frames -> JevAny training rows, in the exact text the decision service reads.

The page's "Save sample" panel (tick every correct action, "Save frame + goal + label") appends
{"image": "captures/x.jpg", "domain": "desk", "goal": "...", "gold": ["TAKE_MARKER"]} to
data/captures.jsonl, the file the eval reads too.
Each line becomes one labelled JevAny record (server/jevany.py: training_row) whose image
travels with it as "images/<file>"; deploy/modal_jevany_train.py ships both to the GPU.
"""
from __future__ import annotations

import json
import struct
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from server.jevany import training_row
from server.tasks import load_domain

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}
SMOKE_GOAL = "Cut the paper along a straight line"


@dataclass
class Dataset:
    rows: list[dict] = field(default_factory=list)          # JevAny records; media uri "images/<file>"
    images: dict[str, bytes] = field(default_factory=dict)  # file name -> bytes
    skipped: list[str] = field(default_factory=list)        # one reason per unusable line


def from_records(numbered: Iterable[tuple[int, object]], read_image: Callable[[str], bytes]) -> Dataset:
    """Build rows from (line number, capture record) pairs. read_image(path from the record)
    returns the file's bytes. A bad line is skipped with its reason instead of failing the set."""
    ds = Dataset()
    domains: dict = {}
    uploaded: dict[str, str] = {}     # path in the record -> file name in images/
    for n, rec in numbered:
        try:
            if not isinstance(rec, dict):
                raise ValueError("not a JSON object")
            if not rec.get("gold"):
                raise ValueError("no correct action ticked")
            goal, rel = rec.get("goal"), rec.get("image")
            if not isinstance(goal, str) or not goal.strip():
                raise ValueError("no goal")
            if not isinstance(rel, str) or Path(rel).suffix.lower() not in IMAGE_TYPES:
                raise ValueError(f"not an image file: {rel!r}")
            domain_id = rec.get("domain") or "desk"
            if domain_id not in domains:
                domains[domain_id] = load_domain(domain_id)
            name = uploaded.get(rel)
            if name is None:
                name = Path(rel).name
                if name in ds.images:                     # same file name from another folder
                    name = f"{len(ds.images)}-{name}"
            row = training_row(domains[domain_id], goal, f"images/{name}", list(rec["gold"]))
            if rel not in uploaded:
                ds.images[name] = read_image(rel)
                uploaded[rel] = name
        except KeyError as e:
            ds.skipped.append(f"line {n}: unknown action {e}")
        except (ValueError, TypeError, OSError) as e:
            ds.skipped.append(f"line {n}: {e}")
        else:
            ds.rows.append(row)
    return ds


def captures_dataset(path: Path) -> Dataset:
    """data/captures.jsonl (or any file in its format); image paths are relative to its folder
    and must stay inside it."""
    base = path.resolve().parent

    def read(rel: str) -> bytes:
        file = (base / rel).resolve()
        if not file.is_relative_to(base):
            raise ValueError(f"image outside {base}: {rel}")
        return file.read_bytes()

    records = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append((n, json.loads(line)))
            except ValueError:
                records.append((n, None))     # reported as "not a JSON object"
    return from_records(records, read)


def smoke_png() -> bytes:
    """A grey frame with a red block (320x240) and no desk object in it, so for "cut the paper"
    the right answer is "none". The released adapter still picks scissors here (p=0.88 on
    2026-11-04): it goes by the goal text. One frame is enough to see training move that."""
    width, height = 320, 240
    grey, red = bytes((128, 128, 128)), bytes((200, 40, 40))
    plain = b"\x00" + grey * width
    block = b"\x00" + grey * 101 + red * 119 + grey * (width - 220)       # red where 100 < x < 220
    raw = b"".join(block if 80 < y < 160 else plain for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def smoke_dataset() -> Dataset:
    """One frame through the same path as real captures: the GPU smoke test's whole data set."""
    record = {"image": "smoke-frame.png", "domain": "desk", "goal": SMOKE_GOAL, "gold": ["NONE"]}
    return from_records([(1, record)], lambda rel: smoke_png())
