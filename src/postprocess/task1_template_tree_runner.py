#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent.parent
LOCAL_CACHE_DIR = PACKAGE_ROOT / "cache"

KEYS: Tuple[str, ...] = ("A", "B", "C", "D", "L", "W", "PUB", "RR2")
PROFILE_PATH_DEFAULTS = {
    "A": str(LOCAL_CACHE_DIR / "route_a.txt"),
    "B": str(LOCAL_CACHE_DIR / "route_b.txt"),
    "C": str(LOCAL_CACHE_DIR / "route_c.txt"),
    "D": str(LOCAL_CACHE_DIR / "route_d.txt"),
    "L": str(LOCAL_CACHE_DIR / "route_l.txt"),
    "W": str(LOCAL_CACHE_DIR / "route_w.txt"),
    "PUB": str(LOCAL_CACHE_DIR / "route_pub.txt"),
    "RR2": str(LOCAL_CACHE_DIR / "route_rr2.txt"),
}

TREE_POLICY: Dict[str, Dict[str, Any]] = {
    "Are the distances between the vertical markers labeled A–B and B–C equal?": {
        "type": "split",
        "feature": "A",
        "when0": {
            "type": "split",
            "feature": "B",
            "when0": {"type": "leaf", "action": "D"},
            "when1": {"type": "leaf", "action": "W"},
        },
        "when1": {
            "type": "split",
            "feature": "B",
            "when0": {"type": "leaf", "action": "D"},
            "when1": {"type": "leaf", "action": "C"},
        },
    },
    "Are the left white pentagon and the right black pentagon equal in size?": {
        "type": "split",
        "feature": "A",
        "when0": {"type": "leaf", "action": "C"},
        "when1": {
            "type": "split",
            "feature": "L",
            "when0": {"type": "leaf", "action": "A"},
            "when1": {"type": "leaf", "action": "C"},
        },
    },
    "Are the left white square and the right black square equal in size?": {
        "type": "split",
        "feature": "A",
        "when0": {
            "type": "split",
            "feature": "L",
            "when0": {"type": "leaf", "action": "W"},
            "when1": {"type": "leaf", "action": "B"},
        },
        "when1": {"type": "leaf", "action": "W"},
    },
    "Are the red and black solid diagonal lines aligned?": {"type": "leaf", "action": "B"},
    "Are the those red lines straight?": {"type": "leaf", "action": "B"},
    "Are the two black lines of equal length?": {
        "type": "split",
        "feature": "A",
        "when0": {"type": "leaf", "action": "W"},
        "when1": {
            "type": "split",
            "feature": "W",
            "when0": {"type": "leaf", "action": "RR2"},
            "when1": {"type": "leaf", "action": "MAJ"},
        },
    },
    "Are the two circles of the same color?": {
        "type": "split",
        "feature": "A",
        "when0": {
            "type": "split",
            "feature": "L",
            "when0": {"type": "leaf", "action": "W"},
            "when1": {"type": "leaf", "action": "A"},
        },
        "when1": {"type": "leaf", "action": "A"},
    },
    "Are the two circles the same size?": {
        "type": "split",
        "feature": "A",
        "when0": {"type": "leaf", "action": "W"},
        "when1": {"type": "leaf", "action": "C"},
    },
    "Are the two horizontal black lines of equal length?": {
        "type": "split",
        "feature": "A",
        "when0": {
            "type": "split",
            "feature": "C",
            "when0": {"type": "leaf", "action": "W"},
            "when1": {"type": "leaf", "action": "MAJ"},
        },
        "when1": {
            "type": "split",
            "feature": "L",
            "when0": {"type": "leaf", "action": "A"},
            "when1": {"type": "leaf", "action": "W"},
        },
    },
    "Are the two orange circles the same size?": {
        "type": "split",
        "feature": "A",
        "when0": {
            "type": "split",
            "feature": "L",
            "when0": {"type": "leaf", "action": "W"},
            "when1": {"type": "leaf", "action": "B"},
        },
        "when1": {
            "type": "split",
            "feature": "W",
            "when0": {"type": "leaf", "action": "D"},
            "when1": {"type": "leaf", "action": "C"},
        },
    },
    "Are the two rectangle the same color?": {
        "type": "split",
        "feature": "A",
        "when0": {"type": "leaf", "action": "L"},
        "when1": {
            "type": "split",
            "feature": "C",
            "when0": {"type": "leaf", "action": "W"},
            "when1": {"type": "leaf", "action": "L"},
        },
    },
    "Are the two small squares of the same color?": {"type": "leaf", "action": "W"},
    "Are the two solid circles the same size?": {"type": "leaf", "action": "B"},
    "Are the two vertical bands of the same color?": {
        "type": "split",
        "feature": "B",
        "when0": {"type": "leaf", "action": "D"},
        "when1": {
            "type": "split",
            "feature": "L",
            "when0": {"type": "leaf", "action": "B"},
            "when1": {"type": "leaf", "action": "A"},
        },
    },
    "Are the two vertical lines straight?": {"type": "leaf", "action": "B"},
    "Are those vertical columns parallel?": {
        "type": "split",
        "feature": "A",
        "when0": {
            "type": "split",
            "feature": "B",
            "when0": {"type": "leaf", "action": "A"},
            "when1": {"type": "leaf", "action": "W"},
        },
        "when1": {"type": "leaf", "action": "B"},
    },
    "Do the squares on the left and right have straight edges?": {"type": "leaf", "action": "A"},
    "Is there an boundary in between every adjecent regions?": {
        "type": "split",
        "feature": "B",
        "when0": {
            "type": "split",
            "feature": "D",
            "when0": {"type": "leaf", "action": "MAJ"},
            "when1": {"type": "leaf", "action": "B"},
        },
        "when1": {"type": "leaf", "action": "A"},
    },
}

GPT52_PER_QUESTION_POLICY: Dict[str, str] = {
    "Are the distances between the vertical markers labeled A–B and B–C equal?": "A",
    "Are the left white pentagon and the right black pentagon equal in size?": "A",
    "Are the left white square and the right black square equal in size?": "W",
    "Are the red and black solid diagonal lines aligned?": "C",
    "Are the those red lines straight?": "A",
    "Are the two black lines of equal length?": "W",
    "Are the two circles of the same color?": "W",
    "Are the two circles the same size?": "A",
    "Are the two horizontal black lines of equal length?": "W",
    "Are the two orange circles the same size?": "C",
    "Are the two rectangle the same color?": "C",
    "Are the two small squares of the same color?": "W",
    "Are the two solid circles the same size?": "L",
    "Are the two vertical bands of the same color?": "L",
    "Are the two vertical lines straight?": "A",
    "Are those vertical columns parallel?": "L",
    "Do the squares on the left and right have straight edges?": "A",
    "Is there an boundary in between every adjecent regions?": "B",
}


def _display_path(path: str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PACKAGE_ROOT))
    except Exception:
        pass
    parts = p.parts
    if len(parts) >= 3:
        return str(Path(*parts[-3:]))
    return p.name or str(path)



def _first_line(prompt: str) -> str:
    for line in str(prompt).replace("\r\n", "\n").split("\n"):
        text = line.strip()
        if text:
            return text
    return str(prompt).strip()


def _load_rows(input_csv: str) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    with open(input_csv, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        if "image_path" not in cols or "prompt" not in cols:
            raise ValueError("CSV must contain image_path and prompt columns")
        for pos, row in enumerate(rd):
            raw_idx = row.get("index")
            try:
                idx = int(raw_idx) if raw_idx not in (None, "") else pos
            except Exception:
                idx = pos
            rows.append((idx, str(row.get("prompt", ""))))
    return rows


def _load_pred(path: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    p = Path(path)
    if not p.exists():
        return out
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        sp = raw.strip().split()
        if len(sp) < 2:
            continue
        try:
            idx = int(sp[0])
            val = int(float(sp[1]))
        except Exception:
            continue
        out[idx] = 1 if val == 1 else 0
    return out


def _heuristic_action(question: str, vals: Dict[str, int]) -> str:
    q = (question or "").lower()

    # Semantic routing over cached-profile roles.
    if "boundary" in q or "adjecent regions" in q:
        return "B"
    if "vertical columns" in q and "parallel" in q:
        return "L"
    if "horizontal black lines" in q or "two black lines of equal length" in q:
        return "W"
    if "red and black solid diagonal lines aligned" in q:
        return "C"
    if "orange circles" in q:
        return "C"
    if "rectangle the same color" in q:
        return "C"
    if "small squares of the same color" in q:
        return "W"
    if "circles of the same color" in q:
        return "W"
    if "solid circles the same size" in q:
        return "L"
    if "vertical bands of the same color" in q:
        return "L"
    if "straight edges" in q:
        return "A"
    if "vertical lines straight" in q:
        return "A"
    if "those red lines straight" in q:
        return "A"
    if "distances between the vertical markers" in q:
        return "A"
    if "left white pentagon" in q:
        return "A"
    if "circles the same size" in q:
        return "A"
    if "left white square and the right black square equal in size" in q:
        return "W"
    return "A"



def _pick(action: str, vals: Dict[str, int]) -> int:
    if action == "MAJ":
        return 1 if sum(vals[k] for k in KEYS) >= 5 else 0
    return 1 if vals.get(action, 0) == 1 else 0


def _eval_tree(node: Dict[str, Any], vals: Dict[str, int]) -> str:
    if node.get("type") == "leaf":
        return str(node["action"])
    feat = str(node["feature"])
    branch = node["when1"] if vals.get(feat, 0) == 1 else node["when0"]
    return _eval_tree(branch, vals)


def run_template_tree(input_csv: str, output_txt: str, output_json: str, profile_paths: Dict[str, str]) -> None:
    rows = _load_rows(input_csv)
    pred_maps = {k: _load_pred(v) for k, v in profile_paths.items()}
    missing = [k for k, v in pred_maps.items() if not v]
    if missing:
        print(f"[WARN] empty/missing profile predictions: {missing}")

    pred: Dict[int, int] = {}
    action_usage: Dict[str, int] = {}
    policy_mode = (os.getenv("TREE_POLICY_MODE", "semantic_heuristic") or "semantic_heuristic").strip().lower()

    for idx, prompt in rows:
        q = _first_line(prompt)
        vals = {k: (1 if pred_maps.get(k, {}).get(idx, 0) == 1 else 0) for k in KEYS}
        if policy_mode == "per_question_gpt52":
            action = GPT52_PER_QUESTION_POLICY[q]
        elif policy_mode == "semantic_heuristic":
            action = _heuristic_action(q, vals)
        else:
            action = _eval_tree(TREE_POLICY[q], vals)
        pred[idx] = _pick(action, vals)
        action_usage[action] = action_usage.get(action, 0) + 1

    with open(output_txt, "w", encoding="utf-8") as f:
        for idx in sorted(pred.keys()):
            f.write(f"{idx} {pred[idx]}\n")

    info = {
        "model": "template-tree-v1-public",
        "parameters": {
            "profile_paths": {k: _display_path(v) for k, v in profile_paths.items()},
            "vote_keys": KEYS,
            "policy_mode": policy_mode,
            "tree_policy": TREE_POLICY,
            "gpt52_per_question_policy": GPT52_PER_QUESTION_POLICY,
            "semantic_heuristic": "template-aware routing",
            "action_usage": action_usage,
            "pipeline": "template decision tree over cached profile outputs",
            "no_minus_one": True,
        },
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    vals = list(pred.values())
    print(f"Results saved to {_display_path(output_txt)}")
    print(f"Model info saved to {_display_path(output_json)}")
    print(f"Total: {len(vals)} | Answer 0: {vals.count(0)} | Answer 1: {vals.count(1)} | Failed: 0")
    print(f"Action usage: {action_usage}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Task1 template-tree runner over cached profiles.")
    ap.add_argument("--input-csv", default=str(PACKAGE_ROOT / "_work" / "test_abs.csv"))
    ap.add_argument("--output-txt", default=str(PACKAGE_ROOT / "output" / "prediction.txt"))
    ap.add_argument("--output-json", default=str(PACKAGE_ROOT / "output" / "model.json"))
    ap.add_argument("--cache-a", default=PROFILE_PATH_DEFAULTS["A"])
    ap.add_argument("--cache-b", default=PROFILE_PATH_DEFAULTS["B"])
    ap.add_argument("--cache-c", default=PROFILE_PATH_DEFAULTS["C"])
    ap.add_argument("--cache-d", default=PROFILE_PATH_DEFAULTS["D"])
    ap.add_argument("--cache-l", default=PROFILE_PATH_DEFAULTS["L"])
    ap.add_argument("--cache-w", default=PROFILE_PATH_DEFAULTS["W"])
    ap.add_argument("--cache-pub", default=PROFILE_PATH_DEFAULTS["PUB"])
    ap.add_argument("--cache-rr2", default=PROFILE_PATH_DEFAULTS["RR2"])
    args = ap.parse_args()

    profile_paths = {
        "A": args.cache_a,
        "B": args.cache_b,
        "C": args.cache_c,
        "D": args.cache_d,
        "L": args.cache_l,
        "W": args.cache_w,
        "PUB": args.cache_pub,
        "RR2": args.cache_rr2,
    }
    run_template_tree(args.input_csv, args.output_txt, args.output_json, profile_paths)
