"""Task domains: the closed set of actions Glimmer may choose from.

A domain lists the objects that can appear on the table. Every object becomes a
TAKE_<ID> action; NONE and ASK_VIEW are always appended so the model can refuse
instead of guessing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DOMAINS_DIR = Path(__file__).resolve().parent.parent / "domains"

NONE = "NONE"
ASK_VIEW = "ASK_VIEW"


@dataclass(frozen=True)
class Action:
    id: str
    label: str        # shown to the model as the option text
    hint: str         # shown to the user when chosen
    concept: str | None = None  # short noun phrase for SAM 3.1 highlight


@dataclass(frozen=True)
class Domain:
    id: str
    title: str
    actions: tuple[Action, ...]

    def by_id(self, action_id: str) -> Action:
        for a in self.actions:
            if a.id == action_id:
                return a
        raise KeyError(action_id)

    def letters(self) -> list[tuple[str, Action]]:
        return [(chr(ord("A") + i), a) for i, a in enumerate(self.actions)]


def _special_actions() -> list[Action]:
    return [
        Action(NONE, "none of the visible objects fits the goal",
               "I don't see a suitable object in the frame."),
        Action(ASK_VIEW, "the view is unclear (blurred, occluded, cropped) - ask for another angle",
               "Show me the table from another angle, without occlusion."),
    ]


def load_domain(domain_id: str) -> Domain:
    path = DOMAINS_DIR / f"{domain_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    actions = [
        Action(
            id=f"TAKE_{o['id'].upper()}",
            label=f"use the {o['name']}",
            hint=o.get("hint") or f"Take the {o['name']}.",
            concept=o.get("concept") or o["name"],
        )
        for o in raw["objects"]
    ]
    if len(actions) + 2 > 26:
        raise ValueError("too many objects for letter options")
    return Domain(raw["id"], raw["title"], tuple(actions + _special_actions()))


def list_domains() -> list[dict]:
    out = []
    for p in sorted(DOMAINS_DIR.glob("*.json")):
        raw = json.loads(p.read_text(encoding="utf-8"))
        out.append({"id": raw["id"], "title": raw["title"],
                    "examples": raw.get("example_goals", [])})
    return out
